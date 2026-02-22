"""Geração de tabelas LaTeX e figuras para a dissertação."""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_latex_tables(
    metrics_path: str,
    crossval_path: str,
    config: dict,
) -> None:
    """
    Gera tabelas LaTeX prontas para o Capítulo 4.
    Salva outputs/latex_tables.tex.
    """
    output_dir = config["paths"]["outputs"]
    os.makedirs(output_dir, exist_ok=True)

    metrics = _load_json(metrics_path)
    crossval = _load_json(crossval_path) if os.path.exists(crossval_path) else {}

    recall_target = config["evaluation"]["recall_target"]
    lines = []

    # Tabela 1: Métricas de Classificação
    lines.append("% Tabela 1: Métricas de Classificação")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Métricas de desempenho da triagem automatizada}")
    lines.append("\\label{tab:metricas_classificacao}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append("\\textbf{Métrica} & \\textbf{Valor} & \\textbf{Alvo} \\\\")
    lines.append("\\hline")
    lines.append(f"Recall & {metrics['recall']:.4f} & $\\geq$ {recall_target} \\\\")
    lines.append(f"Precision & {metrics['precision']:.4f} & -- \\\\")
    lines.append(f"Specificity & {metrics['specificity']:.4f} & -- \\\\")
    lines.append(f"F1-Score & {metrics['f1_score']:.4f} & -- \\\\")
    lines.append(f"Workload Reduction & {metrics['workload_reduction_pct']:.1f}\\% & -- \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Tabela 2: Matriz de Confusão
    cm = metrics["confusion_matrix"]
    lines.append("% Tabela 2: Matriz de Confusão")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Matriz de confusão da triagem automatizada}")
    lines.append("\\label{tab:matriz_confusao}")
    lines.append("\\begin{tabular}{l|cc|c}")
    lines.append("\\hline")
    lines.append(" & \\textbf{Pred. Incluir} & \\textbf{Pred. Excluir} & \\textbf{Total} \\\\")
    lines.append("\\hline")
    total_pos = cm["tp"] + cm["fn"]
    total_neg = cm["fp"] + cm["tn"]
    lines.append(f"\\textbf{{Real Incluir}} & {cm['tp']} (VP) & {cm['fn']} (FN) & {total_pos} \\\\")
    lines.append(f"\\textbf{{Real Excluir}} & {cm['fp']} (FP) & {cm['tn']} (VN) & {total_neg} \\\\")
    lines.append("\\hline")
    total_pred_pos = cm["tp"] + cm["fp"]
    total_pred_neg = cm["fn"] + cm["tn"]
    total = total_pos + total_neg
    lines.append(f"\\textbf{{Total}} & {total_pred_pos} & {total_pred_neg} & {total} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Tabela 3: Cross-Validation
    if crossval:
        lines.append("% Tabela 3: Validação Cruzada Interna")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\caption{Concordância entre runs de validação cruzada (Cohen's Kappa)}")
        lines.append("\\label{tab:cross_validation}")
        lines.append("\\begin{tabular}{lc}")
        lines.append("\\hline")
        lines.append("\\textbf{Par de Runs} & \\textbf{Kappa} \\\\")
        lines.append("\\hline")
        lines.append(f"Run 1 -- Run 2 (sinônimos) & {crossval.get('kappa_run1_run2', 0):.4f} \\\\")
        lines.append(f"Run 1 -- Run 3 (ordem inv.) & {crossval.get('kappa_run1_run3', 0):.4f} \\\\")
        lines.append(f"Run 2 -- Run 3 & {crossval.get('kappa_run2_run3', 0):.4f} \\\\")
        lines.append("\\hline")
        lines.append(f"\\textbf{{Média}} & \\textbf{{{crossval.get('kappa_mean', 0):.4f}}} \\\\")
        lines.append(f"Concordância total & {crossval.get('agreement_pct', 0):.1f}\\% \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    # Tabela 4: Eficiência Operacional
    lines.append("% Tabela 4: Eficiência Operacional")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Comparação de eficiência: pipeline automatizado vs. processo manual}")
    lines.append("\\label{tab:eficiencia}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\hline")
    lines.append("\\textbf{Indicador} & \\textbf{Valor} \\\\")
    lines.append("\\hline")
    lines.append(f"Tempo total do pipeline (s) & {metrics['total_time_seconds']:.1f} \\\\")
    lines.append(f"Baseline manual (s) & {metrics['manual_baseline_seconds']:.1f} \\\\")
    lines.append(f"Redução de tempo & {metrics['time_reduction_pct']:.1f}\\% \\\\")
    lines.append(f"Tokens consumidos & {metrics['total_tokens']:,} \\\\")
    lines.append(f"Total de artigos & {metrics['total_articles']} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    tex_path = os.path.join(output_dir, "latex_tables.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Tabelas LaTeX salvas em {tex_path}")


def generate_confusion_matrix_plot(metrics_path: str, config: dict) -> None:
    """Gera heatmap da confusion matrix (300 DPI)."""
    metrics = _load_json(metrics_path)
    cm = metrics["confusion_matrix"]

    fig_dir = os.path.join(config["paths"]["outputs"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    matrix = np.array([[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]])
    labels = np.array([
        [f'VP\n{cm["tp"]}', f'FN\n{cm["fn"]}'],
        [f'FP\n{cm["fp"]}', f'VN\n{cm["tn"]}'],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=labels, fmt="", cmap="Blues",
        xticklabels=["Incluir", "Excluir"],
        yticklabels=["Incluir", "Excluir"],
        ax=ax,
    )
    ax.set_xlabel("Predição do Sistema")
    ax.set_ylabel("Gold Standard")
    ax.set_title("Matriz de Confusão — Triagem Automatizada")

    fig_path = os.path.join(fig_dir, "confusion_matrix.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix salva em {fig_path}")


def generate_metrics_bar_chart(metrics_path: str, config: dict) -> None:
    """
    Gera bar chart comparando métricas do sistema vs baselines da literatura.
    Baselines: Cao et al. (2025), Rayyan (valores aproximados da literatura).
    """
    metrics = _load_json(metrics_path)
    fig_dir = os.path.join(config["paths"]["outputs"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    categories = ["Recall", "Precision", "F1-Score"]
    system_vals = [metrics["recall"], metrics["precision"], metrics["f1_score"]]

    # Baselines da literatura (valores aproximados)
    cao_vals = [0.90, 0.65, 0.76]      # Cao et al., 2025 (otto-SR)
    rayyan_vals = [0.85, 0.70, 0.77]   # Rayyan (SVM semiautomático)

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, system_vals, width, label="SR-Automation (este trabalho)", color="#2196F3")
    ax.bar(x, cao_vals, width, label="Cao et al. (2025)", color="#FF9800")
    ax.bar(x + width, rayyan_vals, width, label="Rayyan", color="#4CAF50")

    ax.set_ylabel("Score")
    ax.set_title("Comparação de Métricas — Sistema vs Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Linha de recall target
    recall_target = config["evaluation"]["recall_target"]
    ax.axhline(y=recall_target, color="red", linestyle="--", alpha=0.5, label=f"Recall target ({recall_target})")

    fig_path = os.path.join(fig_dir, "metrics_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfico de métricas salvo em {fig_path}")
