"""Métricas de avaliação: recall, precision, F1, workload reduction, kappa."""

import json
import logging
import os

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def calculate_metrics(
    triage_results_path: str,
    gold_standard_path: str,
    config: dict,
) -> dict:
    """
    Calcula todas as métricas comparando triagem automática vs gold standard.
    Salva outputs/metrics.json e outputs/false_negatives_analysis.csv.
    """
    # Carregar dados
    triage_records = _load_jsonl(triage_results_path)
    triage_df = pd.DataFrame(triage_records)

    gold_df = pd.read_csv(gold_standard_path)

    # Merge por article_id / id
    gold_id_col = "id" if "id" in gold_df.columns else "article_id"
    merged = triage_df.merge(
        gold_df, left_on="article_id", right_on=gold_id_col, how="inner"
    )

    if merged.empty:
        logger.error("Nenhum artigo em comum entre triagem e gold standard!")
        return {}

    logger.info(f"Artigos matched: {len(merged)}")

    # Mapear decisões para binário: YES/INCLUDE → 1, NO/EXCLUDE → 0
    def to_binary(val):
        v = str(val).upper().strip()
        return 1 if v in ("YES", "INCLUDE", "1") else 0

    y_pred = merged["decision"].apply(to_binary).values
    y_true = merged["consensus"].apply(to_binary).values

    # Confusion matrix: labels=[1, 0] → [[TP, FN], [FP, TN]]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = cm[0][0], cm[0][1]
    fp, tn = cm[1][0], cm[1][1]

    # Métricas de classificação
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Workload Reduction
    total = len(merged)
    excluded_auto = int((y_pred == 0).sum())
    wr = (excluded_auto / total) * 100 if total > 0 else 0.0

    # Métricas operacionais (do audit_log)
    audit_path = config["paths"]["outputs"] + "audit_log.jsonl"
    total_time_s = 0.0
    total_tokens = 0
    if os.path.exists(audit_path):
        audit_records = _load_jsonl(audit_path)
        total_time_s = sum(r.get("latency_ms", 0) for r in audit_records) / 1000
        total_tokens = sum(
            r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in audit_records
        )

    manual_minutes = config["evaluation"]["manual_baseline_minutes_per_article"]
    manual_baseline_s = total * manual_minutes * 60
    time_reduction_pct = (
        (1 - total_time_s / manual_baseline_s) * 100 if manual_baseline_s > 0 else 0.0
    )

    # Concordância inter-anotador no gold standard
    kappa_gold = None
    if "reviewer_a" in gold_df.columns and "reviewer_b" in gold_df.columns:
        ra = gold_df["reviewer_a"].apply(to_binary).values
        rb = gold_df["reviewer_b"].apply(to_binary).values
        kappa_gold = float(cohen_kappa_score(ra, rb))

    # Recall target check
    recall_target = config["evaluation"]["recall_target"]

    # Falsos negativos
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_articles = merged[fn_mask][["article_id"]].copy()
    fn_ids = fn_articles["article_id"].tolist()

    # Montar resultado
    metrics = {
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "workload_reduction_pct": round(wr, 2),
        "recall_target_met": recall >= recall_target,
        "total_articles": total,
        "included_by_system": int((y_pred == 1).sum()),
        "included_in_gold": int((y_true == 1).sum()),
        "kappa_gold_standard": round(kappa_gold, 4) if kappa_gold is not None else None,
        "total_time_seconds": round(total_time_s, 2),
        "manual_baseline_seconds": round(manual_baseline_s, 2),
        "time_reduction_pct": round(time_reduction_pct, 2),
        "total_tokens": total_tokens,
        "false_negatives": fn_ids,
    }

    # Salvar metrics.json
    output_dir = config["paths"]["outputs"]
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas salvas em {metrics_path}")

    # Salvar false_negatives_analysis.csv
    if fn_ids:
        fn_detail = merged[fn_mask].copy()
        fn_cols = ["article_id"]
        if "title" in fn_detail.columns:
            fn_cols.append("title")
        if "decision" in fn_detail.columns:
            fn_cols.append("decision")
        if "justification" in fn_detail.columns:
            fn_cols.append("justification")
        if "consensus" in fn_detail.columns:
            fn_cols.append("consensus")

        fn_path = os.path.join(output_dir, "false_negatives_analysis.csv")
        fn_detail[fn_cols].to_csv(fn_path, index=False, encoding="utf-8")
        logger.info(f"Análise de falsos negativos salva em {fn_path}")

    # Print resumo
    logger.info("=== Métricas de Avaliação ===")
    logger.info(f"  Recall: {metrics['recall']:.4f} (alvo: ≥{recall_target})")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    logger.info(f"  F1: {metrics['f1_score']:.4f}")
    logger.info(f"  Workload Reduction: {metrics['workload_reduction_pct']:.1f}%")
    logger.info(f"  Recall target met: {metrics['recall_target_met']}")
    if kappa_gold is not None:
        logger.info(f"  Kappa gold standard: {kappa_gold:.4f}")
    logger.info(f"  Falsos negativos: {len(fn_ids)}")

    return metrics
