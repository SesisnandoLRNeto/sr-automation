"""Extração estruturada de dados dos artigos incluídos (5 campos)."""

import json
import logging
import re
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.audit_logger import AuditLogger
from src.utils import call_llm, load_prompt

logger = logging.getLogger(__name__)

EXPECTED_FIELDS = [
    "study_objective",
    "methodology",
    "main_results",
    "conclusions_limitations",
    "sample_data",
]


def _parse_json_response(text: str) -> dict:
    """Parse JSON da resposta, com fallback para regex."""
    # Tentativa 1: json.loads direto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tentativa 2: extrair JSON via regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Falha total
    return {"parse_error": True, "raw": text}


def extract_data(
    title: str,
    abstract: str,
    article_id: str,
    config: dict,
    audit_logger: AuditLogger,
) -> dict:
    """Extrai 5 campos estruturados de um artigo."""
    prompt = load_prompt("extraction", title=title, abstract=abstract)

    response = call_llm(
        prompt=prompt,
        config=config,
        stage="extraction",
        audit_logger=audit_logger,
        article_id=article_id,
    )

    extracted = _parse_json_response(response["text"])

    # Validar campos
    if not extracted.get("parse_error"):
        for field in EXPECTED_FIELDS:
            if field not in extracted or not extracted[field]:
                extracted[field] = "NOT MENTIONED"

    extracted["article_id"] = article_id
    extracted["tokens_used"] = response["tokens_input"] + response["tokens_output"]
    extracted["latency_ms"] = response["latency_ms"]
    extracted["provider"] = response["provider"]
    extracted["timestamp"] = datetime.now(timezone.utc).isoformat()

    return extracted


def extract_corpus(
    triage_results: pd.DataFrame,
    corpus_df: pd.DataFrame,
    config: dict,
) -> list[dict]:
    """
    Executa extração nos artigos com decision=YES.
    Salva em outputs/extraction_results.jsonl.
    """
    included = triage_results[triage_results["decision"] == "YES"]
    logger.info(f"Artigos incluídos para extração: {len(included)}")

    output_path = config["paths"]["outputs"] + "extraction_results.jsonl"
    results = []
    parse_errors = 0
    audit = AuditLogger()

    try:
        for _, row in tqdm(included.iterrows(), total=len(included), desc="Extração"):
            art_id = str(row["article_id"])
            corpus_row = corpus_df[corpus_df["id"] == art_id]
            if corpus_row.empty:
                logger.warning(f"Artigo {art_id} não encontrado no corpus")
                continue

            cr = corpus_row.iloc[0]
            result = extract_data(
                title=cr["title"],
                abstract=cr.get("abstract", ""),
                article_id=art_id,
                config=config,
                audit_logger=audit,
            )
            results.append(result)

            if result.get("parse_error"):
                parse_errors += 1

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        logger.warning("Interrompido. Salvando resultados parciais...")
    finally:
        audit.close()

    logger.info(f"Extração concluída: {len(results)} artigos, {parse_errors} erros de parse")
    return results
