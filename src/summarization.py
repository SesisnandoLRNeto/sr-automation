"""Sumarização estruturada (TL;DR) em 3 frases: problema, solução, achados."""

import json
import logging
import re
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.audit_logger import AuditLogger
from src.utils import call_llm, load_prompt

logger = logging.getLogger(__name__)


def _parse_summary(text: str) -> dict:
    """
    Tenta dividir a resposta em 3 frases estruturadas.
    Fallback: retorna texto raw.
    """
    parts = {"problem": "", "solution": "", "findings": ""}

    # Tentar split por "1.", "2.", "3."
    numbered = re.split(r"\d+\.\s*", text.strip())
    numbered = [s.strip() for s in numbered if s.strip()]

    if len(numbered) >= 3:
        parts["problem"] = numbered[0]
        parts["solution"] = numbered[1]
        parts["findings"] = numbered[2]
        return parts

    # Tentar split por newlines
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) >= 3:
        parts["problem"] = lines[0]
        parts["solution"] = lines[1]
        parts["findings"] = lines[2]
        return parts

    # Fallback: tudo como problem
    parts["problem"] = text.strip()
    return parts


def summarize_article(
    title: str,
    abstract: str,
    article_id: str,
    config: dict,
    audit_logger: AuditLogger,
) -> dict:
    """Gera TL;DR em 3 frases para um artigo."""
    prompt = load_prompt("summarization", title=title, abstract=abstract)

    response = call_llm(
        prompt=prompt,
        config=config,
        stage="summarization",
        audit_logger=audit_logger,
        article_id=article_id,
    )

    parsed = _parse_summary(response["text"])

    return {
        "article_id": article_id,
        "problem": parsed["problem"],
        "solution": parsed["solution"],
        "findings": parsed["findings"],
        "raw_response": response["text"],
        "tokens_used": response["tokens_input"] + response["tokens_output"],
        "latency_ms": response["latency_ms"],
        "provider": response["provider"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def summarize_corpus(
    triage_results: pd.DataFrame,
    corpus_df: pd.DataFrame,
    config: dict,
) -> list[dict]:
    """
    Sumariza artigos incluídos (decision=YES).
    Salva em outputs/summaries.jsonl.
    """
    included = triage_results[triage_results["decision"] == "YES"]
    logger.info(f"Artigos incluídos para sumarização: {len(included)}")

    output_path = config["paths"]["outputs"] + "summaries.jsonl"
    results = []
    audit = AuditLogger()

    try:
        for _, row in tqdm(included.iterrows(), total=len(included), desc="Sumarização"):
            art_id = str(row["article_id"])
            corpus_row = corpus_df[corpus_df["id"] == art_id]
            if corpus_row.empty:
                logger.warning(f"Artigo {art_id} não encontrado no corpus")
                continue

            cr = corpus_row.iloc[0]
            result = summarize_article(
                title=cr["title"],
                abstract=cr.get("abstract", ""),
                article_id=art_id,
                config=config,
                audit_logger=audit,
            )
            results.append(result)

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        logger.warning("Interrompido. Salvando resultados parciais...")
    finally:
        audit.close()

    logger.info(f"Sumarização concluída: {len(results)} artigos")
    return results
