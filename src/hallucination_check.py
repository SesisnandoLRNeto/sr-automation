"""Verificação de alucinação por amostragem (30/módulo)."""

import json
import logging
import os
import random

import pandas as pd

logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_hallucination_sample(config: dict) -> pd.DataFrame:
    """
    Prepara amostra para verificação manual de alucinação.
    30 entradas de cada módulo (triage, extraction, summarization) = 90 total.
    Salva outputs/hallucination_sample.csv.
    """
    output_dir = config["paths"]["outputs"]
    sample_size = config["hallucination_check"]["sample_size_per_module"]
    corpus_path = config["paths"]["corpus"]

    # Carregar corpus para referência
    corpus_df = pd.read_csv(corpus_path)
    corpus_map = {}
    for _, row in corpus_df.iterrows():
        corpus_map[str(row["id"])] = {
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
        }

    rows = []

    # --- Triage ---
    triage_path = os.path.join(output_dir, "triage_results.jsonl")
    triage_records = _load_jsonl(triage_path)
    triage_sample = random.sample(triage_records, min(sample_size, len(triage_records)))

    for r in triage_sample:
        aid = r.get("article_id", "")
        source = corpus_map.get(aid, {})
        rows.append({
            "module": "triage",
            "article_id": aid,
            "claim": f"Decision: {r.get('decision', '')}. {r.get('justification', '')}",
            "source_text": f"Title: {source.get('title', '')} | Abstract: {source.get('abstract', '')}",
            "classification": "",  # Preenchido manualmente: GROUNDED | INFERRED | HALLUCINATED
        })

    # --- Extraction ---
    extraction_path = os.path.join(output_dir, "extraction_results.jsonl")
    extraction_records = _load_jsonl(extraction_path)
    extraction_sample = random.sample(extraction_records, min(sample_size, len(extraction_records)))

    for r in extraction_sample:
        aid = r.get("article_id", "")
        source = corpus_map.get(aid, {})
        # Cada campo extraído como uma claim separada
        for field in ["study_objective", "methodology", "main_results",
                       "conclusions_limitations", "sample_data"]:
            value = r.get(field, "")
            if value and value != "NOT MENTIONED":
                rows.append({
                    "module": "extraction",
                    "article_id": aid,
                    "claim": f"{field}: {value}",
                    "source_text": f"Title: {source.get('title', '')} | Abstract: {source.get('abstract', '')}",
                    "classification": "",
                })

    # --- Summarization ---
    summaries_path = os.path.join(output_dir, "summaries.jsonl")
    summary_records = _load_jsonl(summaries_path)
    summary_sample = random.sample(summary_records, min(sample_size, len(summary_records)))

    for r in summary_sample:
        aid = r.get("article_id", "")
        source = corpus_map.get(aid, {})
        for field in ["problem", "solution", "findings"]:
            value = r.get(field, "")
            if value:
                rows.append({
                    "module": "summarization",
                    "article_id": aid,
                    "claim": f"{field}: {value}",
                    "source_text": f"Title: {source.get('title', '')} | Abstract: {source.get('abstract', '')}",
                    "classification": "",
                })

    df = pd.DataFrame(rows)
    sample_path = os.path.join(output_dir, "hallucination_sample.csv")
    df.to_csv(sample_path, index=False, encoding="utf-8")

    logger.info(f"Amostra de alucinação salva em {sample_path}")
    logger.info(f"  Triage claims: {len([r for r in rows if r['module'] == 'triage'])}")
    logger.info(f"  Extraction claims: {len([r for r in rows if r['module'] == 'extraction'])}")
    logger.info(f"  Summarization claims: {len([r for r in rows if r['module'] == 'summarization'])}")
    logger.info(f"  Total claims: {len(rows)}")

    return df


def calculate_hallucination_rate(sample_path: str) -> dict:
    """Calcula taxa de alucinação após revisão manual."""
    df = pd.read_csv(sample_path)

    # Filtrar apenas linhas com classificação preenchida
    classified = df[df["classification"].notna() & (df["classification"] != "")]

    if classified.empty:
        logger.warning("Nenhuma classificação preenchida no hallucination_sample.csv!")
        return {"error": "Sem classificações. Preencha a coluna 'classification'."}

    total = len(classified)
    counts = classified["classification"].str.upper().value_counts().to_dict()

    grounded = counts.get("GROUNDED", 0)
    inferred = counts.get("INFERRED", 0)
    hallucinated = counts.get("HALLUCINATED", 0)

    hr = (hallucinated / total) * 100 if total > 0 else 0.0

    # Por módulo
    by_module = {}
    for module in ["triage", "extraction", "summarization"]:
        mod_df = classified[classified["module"] == module]
        if not mod_df.empty:
            mod_total = len(mod_df)
            mod_hall = len(mod_df[mod_df["classification"].str.upper() == "HALLUCINATED"])
            by_module[module] = round((mod_hall / mod_total) * 100, 2)
        else:
            by_module[module] = 0.0

    result = {
        "total_claims": total,
        "grounded": grounded,
        "inferred": inferred,
        "hallucinated": hallucinated,
        "hallucination_rate_pct": round(hr, 2),
        "by_module": by_module,
    }

    logger.info("=== Taxa de Alucinação ===")
    logger.info(f"  Total claims: {total}")
    logger.info(f"  Grounded: {grounded}")
    logger.info(f"  Inferred: {inferred}")
    logger.info(f"  Hallucinated: {hallucinated}")
    logger.info(f"  Hallucination rate: {hr:.2f}%")
    for mod, rate in by_module.items():
        logger.info(f"  {mod}: {rate:.2f}%")

    return result
