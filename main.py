"""Orquestrador CLI do pipeline de revisão sistemática."""

import argparse
import logging
import os
import sys
import time

import pandas as pd

from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def _load_corpus(config: dict) -> pd.DataFrame:
    path = config["paths"]["corpus"]
    if not os.path.exists(path):
        logger.error(f"Corpus não encontrado em {path}. Execute --step corpus primeiro.")
        sys.exit(1)
    return pd.read_csv(path)


def _load_triage_results(config: dict) -> pd.DataFrame:
    import json
    path = config["paths"]["outputs"] + "triage_results.jsonl"
    if not os.path.exists(path):
        logger.error(f"Triagem não encontrada em {path}. Execute --step triage primeiro.")
        sys.exit(1)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def step_corpus(config: dict) -> pd.DataFrame:
    from src.corpus_builder import build_corpus
    return build_corpus(config)


def step_index(config: dict, corpus_df: pd.DataFrame) -> None:
    from src.retrieval import SemanticRetriever
    retriever = SemanticRetriever(config)
    retriever.index_corpus(corpus_df)


def step_triage(config: dict, corpus_df: pd.DataFrame) -> pd.DataFrame:
    from src.triage import triage_corpus
    return triage_corpus(corpus_df, config)


def step_extract(config: dict, triage_df: pd.DataFrame, corpus_df: pd.DataFrame) -> None:
    from src.extraction import extract_corpus
    extract_corpus(triage_df, corpus_df, config)


def step_summarize(config: dict, triage_df: pd.DataFrame, corpus_df: pd.DataFrame) -> None:
    from src.summarization import summarize_corpus
    summarize_corpus(triage_df, corpus_df, config)


def step_metrics(config: dict) -> dict:
    from src.metrics import calculate_metrics
    triage_path = config["paths"]["outputs"] + "triage_results.jsonl"
    gold_path = config["paths"]["gold_standard"]
    if not os.path.exists(gold_path):
        logger.error(f"Gold standard não encontrado em {gold_path}.")
        logger.info("Preencha data/gold_standard.csv usando o template antes de executar métricas.")
        sys.exit(1)
    return calculate_metrics(triage_path, gold_path, config)


def step_crossval(config: dict, corpus_df: pd.DataFrame) -> dict:
    from src.cross_validation import run_cross_validation
    return run_cross_validation(corpus_df, config)


def step_hallcheck(config: dict) -> None:
    from src.hallucination_check import prepare_hallucination_sample
    prepare_hallucination_sample(config)


def step_validate(config: dict) -> None:
    from src.extraction_validator import validate_extractions
    extraction_path = config["paths"]["outputs"] + "extraction_results.jsonl"
    corpus_path = config["paths"]["corpus"]
    if not os.path.exists(extraction_path):
        logger.error("Extração não encontrada. Execute --step extract primeiro.")
        sys.exit(1)
    validate_extractions(extraction_path, corpus_path)


def step_report(config: dict) -> None:
    from src.report_generator import (
        generate_confusion_matrix_plot,
        generate_latex_tables,
        generate_metrics_bar_chart,
    )
    metrics_path = os.path.join(config["paths"]["outputs"], "metrics.json")
    crossval_path = os.path.join(config["paths"]["outputs"], "cross_validation.json")

    if not os.path.exists(metrics_path):
        logger.error("Métricas não encontradas. Execute --step metrics primeiro.")
        sys.exit(1)

    generate_latex_tables(metrics_path, crossval_path, config)
    generate_confusion_matrix_plot(metrics_path, config)
    generate_metrics_bar_chart(metrics_path, config)


def run_full_pipeline(config: dict) -> None:
    """Execução completa do pipeline."""
    start_time = time.time()

    # 1. Corpus
    corpus_path = config["paths"]["corpus"]
    if os.path.exists(corpus_path):
        logger.info(f"Corpus existente encontrado em {corpus_path}. Usando-o.")
        corpus_df = pd.read_csv(corpus_path)
    else:
        logger.info("Coletando corpus...")
        corpus_df = step_corpus(config)

    # 2. Indexação
    logger.info("Indexando corpus...")
    step_index(config, corpus_df)

    # 3. Triagem
    logger.info("Executando triagem...")
    triage_df = step_triage(config, corpus_df)

    # 4. Extração
    logger.info("Executando extração...")
    step_extract(config, triage_df, corpus_df)

    # 5. Sumarização
    logger.info("Executando sumarização...")
    step_summarize(config, triage_df, corpus_df)

    # 6. Métricas (se gold standard disponível)
    gold_path = config["paths"]["gold_standard"]
    if os.path.exists(gold_path):
        logger.info("Calculando métricas...")
        step_metrics(config)
    else:
        logger.info(
            f"Gold standard não encontrado em {gold_path}. "
            "Preencha e execute: python main.py --step metrics"
        )

    elapsed = time.time() - start_time

    # Resumo final
    included = len(triage_df[triage_df["decision"] == "YES"]) if not triage_df.empty else 0
    excluded = len(triage_df[triage_df["decision"] == "NO"]) if not triage_df.empty else 0
    total_tokens = triage_df["tokens_used"].sum() if "tokens_used" in triage_df.columns else 0

    logger.info("=" * 50)
    logger.info("PIPELINE CONCLUÍDO")
    logger.info("=" * 50)
    logger.info(f"  Total artigos: {len(corpus_df)}")
    logger.info(f"  Incluídos: {included}")
    logger.info(f"  Excluídos: {excluded}")
    logger.info(f"  Tempo total: {elapsed:.1f}s")
    logger.info(f"  Tokens (triagem): {total_tokens}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="SR-Automation: Pipeline automatizado de revisão sistemática"
    )
    parser.add_argument(
        "--step",
        choices=["corpus", "triage", "extract", "summarize",
                 "metrics", "crossval", "hallcheck", "validate", "report"],
        help="Executar apenas um passo específico do pipeline",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Caminho para o arquivo de configuração (default: config.yaml)",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging()

    if args.step is None:
        run_full_pipeline(config)
    elif args.step == "corpus":
        step_corpus(config)
    elif args.step == "triage":
        corpus_df = _load_corpus(config)
        step_triage(config, corpus_df)
    elif args.step == "extract":
        corpus_df = _load_corpus(config)
        triage_df = _load_triage_results(config)
        step_extract(config, triage_df, corpus_df)
    elif args.step == "summarize":
        corpus_df = _load_corpus(config)
        triage_df = _load_triage_results(config)
        step_summarize(config, triage_df, corpus_df)
    elif args.step == "metrics":
        step_metrics(config)
    elif args.step == "crossval":
        corpus_df = _load_corpus(config)
        step_crossval(config, corpus_df)
    elif args.step == "hallcheck":
        step_hallcheck(config)
    elif args.step == "validate":
        step_validate(config)
    elif args.step == "report":
        step_report(config)


if __name__ == "__main__":
    main()
