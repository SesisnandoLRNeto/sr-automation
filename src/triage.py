"""
Triagem automática zero-shot de artigos.
Ref: §3.1.2 (triagem), §3.5.6 Fase 3 itens 3-4.

Nível único: título + abstract combinados (confirmado com usuário).
Prompts em inglês (YES/NO).
"""

import json
import logging
import re
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.audit_logger import AuditLogger
from src.utils import call_llm, load_config, load_prompt

logger = logging.getLogger(__name__)


def _parse_decision(text: str) -> tuple[str, str, float]:
    """
    Parse robusto de YES/NO da resposta do LLM.
    Retorna (decision, justification, confidence).
    """
    clean = text.strip()

    # Verificar nos primeiros 10 caracteres
    first_part = clean[:10].upper()
    if "YES" in first_part:
        decision = "YES"
        confidence = 1.0
    elif "NO" in first_part:
        decision = "NO"
        confidence = 1.0
    else:
        # Buscar no texto completo
        upper = clean.upper()
        has_yes = bool(re.search(r"\bYES\b", upper))
        has_no = bool(re.search(r"\bNO\b", upper))

        if has_yes and not has_no:
            decision = "YES"
            confidence = 0.8
        elif has_no and not has_yes:
            decision = "NO"
            confidence = 0.8
        elif has_yes and has_no:
            # Ambíguo — usar primeira ocorrência
            yes_pos = upper.index("YES")
            no_pos = upper.index("NO")
            decision = "YES" if yes_pos < no_pos else "NO"
            confidence = 0.5
        else:
            decision = "UNCERTAIN"
            confidence = 0.5

    # Extrair justificativa (tudo após YES/NO)
    justification = re.sub(r"^(YES|NO|UNCERTAIN)[.,:\-\s]*", "", clean, flags=re.IGNORECASE).strip()

    return decision, justification, confidence


def triage_article(
    title: str,
    abstract: str,
    article_id: str,
    criteria: list[str],
    config: dict,
    audit_logger: AuditLogger,
) -> dict:
    """Classificação zero-shot de um artigo."""
    criteria_text = "\n".join(f"- {c}" for c in criteria)
    prompt = load_prompt("triage", criteria=criteria_text, title=title, abstract=abstract)

    response = call_llm(
        prompt=prompt,
        config=config,
        stage="triage",
        audit_logger=audit_logger,
        article_id=article_id,
    )

    decision, justification, confidence = _parse_decision(response["text"])

    return {
        "article_id": article_id,
        "decision": decision,
        "justification": justification,
        "confidence": confidence,
        "tokens_used": response["tokens_input"] + response["tokens_output"],
        "latency_ms": response["latency_ms"],
        "provider": response["provider"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def triage_corpus(corpus_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Executa triagem em todo o corpus.
    Salva resultados incrementais em outputs/triage_results.jsonl.
    """
    criteria = config["gold_standard"]["criteria"]
    output_path = config["paths"]["outputs"] + "triage_results.jsonl"

    results = []
    audit = AuditLogger()

    try:
        for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Triagem"):
            result = triage_article(
                title=row["title"],
                abstract=row.get("abstract", ""),
                article_id=str(row["id"]),
                criteria=criteria,
                config=config,
                audit_logger=audit,
            )
            results.append(result)

            # Append incremental (JSONL)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        logger.warning("Interrompido pelo usuário. Salvando resultados parciais...")
    finally:
        audit.close()

    # Sumário
    df = pd.DataFrame(results)
    if not df.empty:
        counts = df["decision"].value_counts()
        logger.info("=== Resultados da Triagem ===")
        for dec, cnt in counts.items():
            logger.info(f"  {dec}: {cnt}")
        if "latency_ms" in df.columns:
            logger.info(f"  Latência média: {df['latency_ms'].mean():.0f}ms")

    return df
