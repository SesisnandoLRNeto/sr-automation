"""Utilitários: config, prompts e wrapper de chamadas LLM."""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from dotenv import load_dotenv

from src.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_prompt(name: str, **kwargs) -> str:
    prompt_path = Path("prompts") / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template não encontrado: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    if kwargs:
        template = template.format(**kwargs)
    return template


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _call_provider(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int = 30,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    start = time.time()
    resp = requests.post(api_url, json=body, headers=headers, timeout=timeout)
    latency_ms = (time.time() - start) * 1000

    if resp.status_code == 429:
        raise RateLimitError(f"Rate limit: {resp.status_code}")
    resp.raise_for_status()

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "text": choice["message"]["content"].strip(),
        "tokens_input": usage.get("prompt_tokens", 0),
        "tokens_output": usage.get("completion_tokens", 0),
        "latency_ms": latency_ms,
    }


class RateLimitError(Exception):
    pass


# Contador de rate limits consecutivos do Groq
_groq_consecutive_failures = 0
_GROQ_SKIP_THRESHOLD = 3


def call_llm(
    prompt: str,
    config: dict,
    stage: str,
    audit_logger: AuditLogger,
    article_id: str = "",
    decision_label: Optional[str] = None,
) -> dict:
    """Chama LLM via Groq, com fallback para Together AI."""
    global _groq_consecutive_failures

    params = config["inference"][stage]
    temperature = params["temperature"]
    top_p = params["top_p"]
    max_tokens = params["max_tokens"]
    prompt_hash = _hash_prompt(prompt)

    groq_key = os.getenv("GROQ_API_KEY", "")
    together_key = os.getenv("TOGETHER_API_KEY", "")

    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    together_url = "https://api.together.xyz/v1/chat/completions"

    provider = "groq"
    result = None

    # Pular Groq se rate limit consecutivo atingiu o threshold
    skip_groq = _groq_consecutive_failures >= _GROQ_SKIP_THRESHOLD

    if skip_groq:
        logger.debug("Groq em rate limit — usando Together AI direto.")
    else:
        try:
            result = _call_provider(
                prompt, groq_url, groq_key,
                config["model"]["model_name"],
                temperature, top_p, max_tokens,
            )
            provider = "groq"
            _groq_consecutive_failures = 0
        except RateLimitError:
            _groq_consecutive_failures += 1
            if _groq_consecutive_failures < _GROQ_SKIP_THRESHOLD:
                logger.warning("Groq rate limit. Aguardando 60s antes de retry...")
                time.sleep(60)
                try:
                    result = _call_provider(
                        prompt, groq_url, groq_key,
                        config["model"]["model_name"],
                        temperature, top_p, max_tokens,
                    )
                    provider = "groq"
                    _groq_consecutive_failures = 0
                except Exception:
                    logger.warning("Groq retry falhou. Fallback para Together AI.")
                    _groq_consecutive_failures += 1
                    result = None
            else:
                logger.warning(
                    f"Groq rate limit ({_groq_consecutive_failures}x consecutivo). "
                    "Usando Together AI para as próximas chamadas."
                )
                result = None
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            logger.warning(f"Groq erro ({e}). Fallback para Together AI.")
            result = None

    # Fallback: Together AI
    if result is None:
        result = _call_provider(
            prompt, together_url, together_key,
            config["model"]["fallback_model"],
            temperature, top_p, max_tokens,
        )
        provider = "together"

    # Log
    audit_logger.log(
        module=stage,
        article_id=article_id,
        decision=decision_label or result["text"][:200],
        confidence=1.0,
        prompt_hash=prompt_hash,
        provider=provider,
        tokens_in=result["tokens_input"],
        tokens_out=result["tokens_output"],
        latency_ms=result["latency_ms"],
        raw_response=result["text"],
    )

    result["provider"] = provider

    # Rate limit Groq free tier (~30 req/min)
    if provider == "groq":
        time.sleep(2.1)

    return result
