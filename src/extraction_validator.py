"""Spot-check de extração: valida dados extraídos contra o abstract original."""

import json
import logging
import os
import random

import pandas as pd

logger = logging.getLogger(__name__)

EXTRACTION_FIELDS = [
    "study_objective",
    "methodology",
    "main_results",
    "conclusions_limitations",
    "sample_data",
]


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_extractions(
    extraction_path: str,
    corpus_path: str,
    sample_size: int = 10,
) -> pd.DataFrame:
    """
    Prepara spot-check: sample_size artigos × 5 campos = 50 verificações.
    Salva outputs/extraction_validation.csv.
    """
    extractions = _load_jsonl(extraction_path)
    corpus_df = pd.read_csv(corpus_path)

    corpus_map = {}
    for _, row in corpus_df.iterrows():
        corpus_map[str(row["id"])] = row.get("abstract", "")

    # Filtrar extrações válidas (sem parse_error)
    valid = [e for e in extractions if not e.get("parse_error")]
    sample = random.sample(valid, min(sample_size, len(valid)))

    rows = []
    for ext in sample:
        aid = ext.get("article_id", "")
        abstract = corpus_map.get(aid, "")

        for field in EXTRACTION_FIELDS:
            value = ext.get(field, "")
            rows.append({
                "article_id": aid,
                "field": field,
                "extracted_value": value,
                "source_abstract": abstract,
                "error_type": "",   # CORRECT | HALLUCINATION | OMISSION | IMPRECISION
                "notes": "",        # Comentários do revisor
            })

    df = pd.DataFrame(rows)
    output_path = os.path.join(os.path.dirname(extraction_path), "extraction_validation.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"Validation sample salva em {output_path}")
    logger.info(f"  Artigos amostrados: {len(sample)}")
    logger.info(f"  Campos a validar: {len(rows)} ({len(sample)} × {len(EXTRACTION_FIELDS)})")

    return df


def calculate_extraction_accuracy(validation_path: str) -> dict:
    """
    Calcula acurácia da extração após revisão manual.
    """
    df = pd.read_csv(validation_path)
    classified = df[df["error_type"].notna() & (df["error_type"] != "")]

    if classified.empty:
        logger.warning("Nenhuma classificação preenchida em extraction_validation.csv!")
        return {"error": "Sem classificações. Preencha a coluna 'error_type'."}

    total = len(classified)
    counts = classified["error_type"].str.upper().value_counts().to_dict()

    correct = counts.get("CORRECT", 0)
    hallucination = counts.get("HALLUCINATION", 0)
    omission = counts.get("OMISSION", 0)
    imprecision = counts.get("IMPRECISION", 0)

    accuracy = (correct / total) * 100 if total > 0 else 0.0

    # Por campo
    by_field = {}
    for field in EXTRACTION_FIELDS:
        field_df = classified[classified["field"] == field]
        if not field_df.empty:
            fc = len(field_df[field_df["error_type"].str.upper() == "CORRECT"])
            fe = len(field_df) - fc
            by_field[field] = {"correct": fc, "errors": fe}

    result = {
        "total_fields": total,
        "correct": correct,
        "hallucination": hallucination,
        "omission": omission,
        "imprecision": imprecision,
        "accuracy_pct": round(accuracy, 2),
        "by_field": by_field,
    }

    logger.info("=== Acurácia da Extração ===")
    logger.info(f"  Total campos: {total}")
    logger.info(f"  Corretos: {correct} ({accuracy:.1f}%)")
    logger.info(f"  Alucinações: {hallucination}")
    logger.info(f"  Omissões: {omission}")
    logger.info(f"  Imprecisões: {imprecision}")

    return result
