"""Validação cruzada interna: 3 runs com variações de prompt."""

import json
import logging
import os
from statistics import mean

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from src.audit_logger import AuditLogger
from src.triage import _parse_decision
from src.utils import call_llm, load_prompt

logger = logging.getLogger(__name__)

# Sinônimos para Run 2
SYNONYM_CRITERIA = [
    "Presents an AI-powered educational tool or adaptive learning system",
    "Uses a large language model as a core element of the system",
    "Reports experimental assessment or working prototype",
]


def _build_prompt_run1(criteria: list[str], title: str, abstract: str) -> str:
    criteria_text = "\n".join(f"- {c}" for c in criteria)
    return load_prompt("triage", criteria=criteria_text, title=title, abstract=abstract)


def _build_prompt_run2(title: str, abstract: str) -> str:
    criteria_text = "\n".join(f"- {c}" for c in SYNONYM_CRITERIA)
    return load_prompt("triage", criteria=criteria_text, title=title, abstract=abstract)


def _build_prompt_run3(criteria: list[str], title: str, abstract: str) -> str:
    """Ordem invertida: abstract antes de title no prompt."""
    criteria_text = "\n".join(f"- {c}" for c in criteria)
    prompt_template = (
        "You are an expert in systematic literature reviews.\n\n"
        "Inclusion criteria:\n{criteria}\n\n"
        "Article:\n"
        "Abstract: {abstract}\n"
        "Title: {title}\n\n"
        "Question: Should this article be included in the systematic review?\n"
        "Answer ONLY \"YES\" or \"NO\" and justify in one sentence."
    )
    return prompt_template.format(criteria=criteria_text, title=title, abstract=abstract)


def _run_triage(
    corpus_df: pd.DataFrame,
    config: dict,
    prompt_builder,
    run_name: str,
    output_path: str,
    force_together: bool = False,
) -> list[dict]:
    """Executa triagem com um prompt builder específico."""
    results = []
    audit = AuditLogger(output_path=f"outputs/audit_log_crossval_{run_name}.jsonl")

    try:
        for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc=f"CrossVal {run_name}"):
            prompt = prompt_builder(row["title"], row.get("abstract", ""))

            response = call_llm(
                prompt=prompt,
                config=config,
                stage="triage",
                audit_logger=audit,
                article_id=str(row["id"]),
                use_cache=False,
                force_together=force_together,
            )

            decision, justification, confidence = _parse_decision(response["text"])

            result = {
                "article_id": str(row["id"]),
                "decision": decision,
                "justification": justification,
                "confidence": confidence,
                "run": run_name,
            }
            results.append(result)

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        logger.warning(f"CrossVal {run_name} interrompido.")
    finally:
        audit.close()

    return results


def run_cross_validation(corpus_df: pd.DataFrame, config: dict) -> dict:
    """
    3 runs de triagem com variações de prompt. Calcula Cohen's Kappa entre pares.
    """
    criteria = config["gold_standard"]["criteria"]
    output_dir = config["paths"]["outputs"]
    os.makedirs(output_dir, exist_ok=True)
    force_together = config["cross_validation"].get("force_together", False)

    if force_together:
        logger.info("=== Cross-Validation: 3 runs (via Together AI) ===")
    else:
        logger.info("=== Cross-Validation: 3 runs ===")

    # Run 1: Prompt original
    logger.info("Run 1: Prompt original")
    run1 = _run_triage(
        corpus_df, config,
        lambda t, a: _build_prompt_run1(criteria, t, a),
        "run1",
        os.path.join(output_dir, "crossval_run1.jsonl"),
        force_together=force_together,
    )

    # Run 2: Sinônimos
    logger.info("Run 2: Critérios com sinônimos")
    run2 = _run_triage(
        corpus_df, config,
        lambda t, a: _build_prompt_run2(t, a),
        "run2",
        os.path.join(output_dir, "crossval_run2.jsonl"),
        force_together=force_together,
    )

    # Run 3: Ordem invertida
    logger.info("Run 3: Ordem invertida (abstract → title)")
    run3 = _run_triage(
        corpus_df, config,
        lambda t, a: _build_prompt_run3(criteria, t, a),
        "run3",
        os.path.join(output_dir, "crossval_run3.jsonl"),
        force_together=force_together,
    )

    # Alinhar por article_id
    def to_map(results):
        return {r["article_id"]: r["decision"] for r in results}

    m1, m2, m3 = to_map(run1), to_map(run2), to_map(run3)
    common_ids = sorted(set(m1.keys()) & set(m2.keys()) & set(m3.keys()))

    if not common_ids:
        logger.error("Nenhum artigo em comum entre os 3 runs!")
        return {}

    def to_binary(dec):
        return 1 if dec == "YES" else 0

    d1 = [to_binary(m1[aid]) for aid in common_ids]
    d2 = [to_binary(m2[aid]) for aid in common_ids]
    d3 = [to_binary(m3[aid]) for aid in common_ids]

    kappa_12 = float(cohen_kappa_score(d1, d2))
    kappa_13 = float(cohen_kappa_score(d1, d3))
    kappa_23 = float(cohen_kappa_score(d2, d3))
    kappa_mean = mean([kappa_12, kappa_13, kappa_23])

    # Agreement
    agreements = sum(1 for i in range(len(common_ids)) if d1[i] == d2[i] == d3[i])
    agreement_pct = (agreements / len(common_ids)) * 100

    # Disagreements
    disagreements = []
    for i, aid in enumerate(common_ids):
        if not (d1[i] == d2[i] == d3[i]):
            disagreements.append({
                "article_id": aid,
                "run1": m1[aid],
                "run2": m2[aid],
                "run3": m3[aid],
            })

    result = {
        "kappa_run1_run2": round(kappa_12, 4),
        "kappa_run1_run3": round(kappa_13, 4),
        "kappa_run2_run3": round(kappa_23, 4),
        "kappa_mean": round(kappa_mean, 4),
        "expected_kappa": config["cross_validation"]["expected_kappa"],
        "agreement_pct": round(agreement_pct, 2),
        "total_articles": len(common_ids),
        "disagreements": disagreements,
    }

    # Salvar
    cv_path = os.path.join(output_dir, "cross_validation.json")
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Cross-validation salvo em {cv_path}")

    logger.info("=== Resultados Cross-Validation ===")
    logger.info(f"  Kappa R1-R2: {kappa_12:.4f}")
    logger.info(f"  Kappa R1-R3: {kappa_13:.4f}")
    logger.info(f"  Kappa R2-R3: {kappa_23:.4f}")
    logger.info(f"  Kappa médio: {kappa_mean:.4f} (esperado: ≥{result['expected_kappa']})")
    logger.info(f"  Concordância: {agreement_pct:.1f}%")
    logger.info(f"  Disagreements: {len(disagreements)}")

    return result
