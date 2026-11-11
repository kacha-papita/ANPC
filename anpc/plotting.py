"""All ten figures from the report (Figures 1-10).

Every figure is written to ``<output_dir>/figures`` at 200 dpi.  The style is
kept intentionally clean and grayscale-friendly, matching the report's figures.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

import networkx as nx
import numpy as np

from .config import Config
from .experiment import ExperimentResults
from .policies import POLICY_NAMES

# Consistent palette ---------------------------------------------------------
COLOUR_INITIAL = "#111111"
COLOUR_ADDITIONAL = "#9e9e9e"
COLOUR_PROTECTED = "#ffd600"
COLOUR_OPERATING = "#2e7d32"
COLOUR_EDGE = "#c8c8c8"

GREYS = ["#4d4d4d", "#666666", "#808080", "#999999", "#b3b3b3", "#cccccc"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

FAMILY_LABELS = {
    "random": "Random",
    "small_world": "Small-world",
    "scale_free": "Scale-free",
}


def _figures_dir(cfg: Config) -> str:
    path = os.path.join(cfg.output_dir, "figures")
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 1 - one small-world network after protection and propagation
# ----------------------------------------------------------------------
def plot_example_network(results: ExperimentResults, cfg: Config) -> str:
    ex = results.example
    graph: nx.Graph = ex["graph"]
    pos = ex["position"]
    initial = set(ex["initial_failed"])
    protected = set(ex["protected_proposed"])
    failed = set(ex["failed_proposed"])
    additional = failed - initial

    colours: Dict[int, str] = {}
    sizes: List[int] = []
    for node in graph.nodes():
        if node in initial:
            colours[node] = COLOUR_INITIAL
        elif node in additional:
            colours[node] = COLOUR_ADDITIONAL
        elif node in protected:
            colours[node] = COLOUR_PROTECTED
        else:
            colours[node] = COLOUR_OPERATING
        sizes.append(420 if node in protected else 220)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color=COLOUR_EDGE, width=0.9, alpha=0.8
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=[colours[v] for v in graph.nodes()],
        node_size=sizes,
        linewidths=1.0,
        edgecolors="black",
    )
    nx.draw_networkx_labels(
        graph, pos, ax=ax, font_size=6, font_color="white"
    )

    handles = [
        mpatches.Patch(color=COLOUR_INITIAL, label="Initial failures"),
        mpatches.Patch(color=COLOUR_ADDITIONAL, label="Additional failures"),
        mpatches.Patch(color=COLOUR_PROTECTED, label="Protected"),
        mpatches.Patch(color=COLOUR_OPERATING, label="Operating"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
    ax.set_title(
        "One small-world network after protection and failure propagation",
        fontsize=12,
        pad=10,
    )
    ax.set_axis_off()
    path = os.path.join(_figures_dir(cfg), "fig01_example_network.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 2 - five-stage search pipeline
# ----------------------------------------------------------------------
def plot_pipeline(cfg: Config) -> str:
    stages = [
        "1. Input: graph G, initial failures F$_0$, budget B",
        "2. Extract 7 node features; predict risk r(v)",
        "3. Screen  Q(v)=0.55r+0.28b+0.17d; keep 13",
        "4. For each candidate: 11 shared simulations $\\rightarrow$ J",
        "5. Add best node to P; repeat until |P| = B",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(stages) * 1.6 + 0.4)
    ax.set_axis_off()

    box_w, box_h = 8.0, 0.95
    for i, text in enumerate(stages):
        y = (len(stages) - i) * 1.6 - 1.0
        box = FancyBboxPatch(
            (1.0, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.4,
            edgecolor="#333333",
            facecolor="#f2f2f2",
        )
        ax.add_patch(box)
        ax.text(
            1.0 + box_w / 2,
            y + box_h / 2,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
        )
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(1.0 + box_w / 2, y - 0.12),
                xytext=(1.0 + box_w / 2, y + box_h + 0.12),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6),
            )
    # Feedback loop annotation.
    ax.annotate(
        "update priorities",
        xy=(9.15, 1.35),
        xytext=(9.15, len(stages) * 1.6 - 1.1),
        fontsize=8.5,
        color="#555555",
        rotation=90,
        va="center",
        ha="center",
    )
    ax.annotate(
        "",
        xy=(9.0, 1.35),
        xytext=(9.0, len(stages) * 1.6 - 1.1),
        arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.2,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.set_title("The five stages of the proposed search", fontsize=13, pad=8)
    path = os.path.join(_figures_dir(cfg), "fig02_pipeline.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 3 - ROC curve
# ----------------------------------------------------------------------
def plot_roc(results: ExperimentResults, cfg: Config) -> str:
    fpr, tpr = results.roc
    metrics = results.risk_metrics
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(
        fpr,
        tpr,
        color="#1f4e79",
        lw=2.2,
        label=f"ROC (AUC = {metrics.roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1.2,
            label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic — held-out graphs",
                 fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    path = os.path.join(_figures_dir(cfg), "fig03_roc.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 4 - confusion matrix
# ----------------------------------------------------------------------
def plot_confusion(results: ExperimentResults, cfg: Config) -> str:
    m = results.risk_metrics
    cm = np.array([[m.true_healthy, m.false_alarms],
                   [m.missed_failures, m.found_failures]])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{labels[i, j]}\n{cm[i, j]:,}",
                ha="center", va="center",
                fontsize=12,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted healthy", "Predicted failed"])
    ax.set_yticklabels(["Actual healthy", "Actual failed"])
    ax.set_title(
        f"Confusion matrix (threshold = {m.threshold:.2f})",
        fontsize=12, pad=10,
    )
    path = os.path.join(_figures_dir(cfg), "fig04_confusion_matrix.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 5 - average combined score at four protected nodes
# ----------------------------------------------------------------------
def plot_budget_bar(results: ExperimentResults, cfg: Config) -> str:
    values = results.survival_connectivity["J"].astype(float)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(values))
    bars = ax.bar(x, values.values, color=GREYS, edgecolor="black", width=0.68)
    for bar, value in zip(bars, values.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(values.index), rotation=18, ha="right")
    ax.set_ylabel("Mean combined score J")
    ax.set_title(
        f"Average combined score at budget {cfg.key_budget}",
        fontsize=12, pad=10,
    )
    low = float(values.min()) - 0.02
    high = float(values.max()) + 0.015
    ax.set_ylim(max(0.0, low), min(1.0, high))
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(_figures_dir(cfg), "fig05_budget_bar.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 6 - score versus budget
# ----------------------------------------------------------------------
def plot_budget_lines(results: ExperimentResults, cfg: Config) -> str:
    table = results.table3
    budgets = sorted(table.columns.tolist())
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for i, policy in enumerate(POLICY_NAMES):
        y = [float(table.loc[policy, b]) for b in budgets]
        ax.plot(
            budgets,
            y,
            marker=MARKERS[i],
            linestyle=LINESTYLES[i],
            color=GREYS[i],
            lw=1.8,
            markersize=7,
            label=policy,
        )
    ax.set_xlabel("Protection budget")
    ax.set_ylabel("Mean combined score J")
    ax.set_title("Mean combined score as the protection budget increases",
                 fontsize=12, pad=10)
    ax.set_xticks(budgets)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")
    path = os.path.join(_figures_dir(cfg), "fig06_budget_lines.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 7 - share in the largest operational component
# ----------------------------------------------------------------------
def plot_connectivity_bar(results: ExperimentResults, cfg: Config) -> str:
    values = results.survival_connectivity["C"].astype(float)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(values))
    bars = ax.bar(x, values.values, color=GREYS, edgecolor="black", width=0.68)
    for bar, value in zip(bars, values.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(values.index), rotation=18, ha="right")
    ax.set_ylabel("Largest-component fraction C")
    ax.set_title(
        f"Share of the network in the largest operating component "
        f"(budget {cfg.key_budget})",
        fontsize=12, pad=10,
    )
    low = float(values.min()) - 0.02
    high = float(values.max()) + 0.015
    ax.set_ylim(max(0.0, low), min(1.0, high))
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(_figures_dir(cfg), "fig07_connectivity_bar.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 8 - score by graph family at the key budget
# ----------------------------------------------------------------------
def plot_family_grouped(results: ExperimentResults, cfg: Config) -> str:
    table = results.family_table
    families = [f for f in cfg.families if f in table.columns]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    width = 0.8 / len(POLICY_NAMES)
    x = np.arange(len(families))
    hatches = ["", "//", "\\\\", "..", "xx", "++"]
    for i, policy in enumerate(POLICY_NAMES):
        values = [float(table.loc[policy, f]) for f in families]
        ax.bar(
            x + (i - (len(POLICY_NAMES) - 1) / 2) * width,
            values,
            width,
            label=policy,
            color=GREYS[i],
            edgecolor="black",
            hatch=hatches[i],
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in families])
    ax.set_ylabel("Mean combined score J")
    ax.set_title(
        f"Mean combined score at budget {cfg.key_budget}, by graph family",
        fontsize=12, pad=10,
    )
    low = float(table.min().min()) - 0.02
    high = float(table.max().max()) + 0.015
    ax.set_ylim(max(0.0, low), min(1.0, high))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    path = os.path.join(_figures_dir(cfg), "fig08_family_grouped.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 9 - selection-only time
# ----------------------------------------------------------------------
def plot_timing(results: ExperimentResults, cfg: Config) -> str:
    values = results.timing.astype(float)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(values))
    floor = 1e-3
    heights = [max(v, floor) for v in values.values]
    bars = ax.bar(x, heights, color=GREYS, edgecolor="black", width=0.68)
    for bar, value in zip(bars, values.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value, floor) * 1.12,
            f"{value:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(list(values.index), rotation=18, ha="right")
    ax.set_ylabel("Selection-only time (ms, log scale)")
    ax.set_title(
        f"Average selection-only time at budget {cfg.key_budget} "
        "(features and inference excluded)",
        fontsize=11.5, pad=10,
    )
    ax.grid(True, axis="y", alpha=0.3, which="both")
    path = os.path.join(_figures_dir(cfg), "fig09_timing.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Figure 10 - shortlist-size ablation
# ----------------------------------------------------------------------
def plot_ablation(results: ExperimentResults, cfg: Config) -> str:
    table = results.ablation.set_index("Configuration")
    sizes = [5, 9, 13, 25]
    labels = ["5", "9", "13", "25"]
    values = [
        float(table.loc[f"Risk-informed, {s} nodes", "Mean score"]) for s in sizes
    ]
    full_value = float(table.loc["No shortlist, full search", "Mean score"])
    structure_value = float(
        table.loc["Structure only, 13 nodes", "Mean score"]
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = list(range(len(sizes) + 1))
    y = values + [full_value]
    ax.plot(x, y, marker="o", color="#1f4e79", lw=2, markersize=7,
            label="Risk-informed shortlist")
    ax.scatter(x[-1], y[-1], marker="s", color="#1f4e79", s=90, zorder=5)
    ax.axhline(structure_value, linestyle="--", color="#b0413e", lw=1.8,
               label="Structure only, 13 nodes")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.0004, f"{yi:.4f}", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels + ["Full"])
    ax.set_xlabel("Shortlist size")
    ax.set_ylabel("Mean combined score J")
    ax.set_title("Effect of shortlist size on measured protection score",
                 fontsize=12, pad=10)
    all_values = y + [structure_value]
    pad = max(0.0008, (max(all_values) - min(all_values)) * 0.25)
    ax.set_ylim(min(all_values) - pad, max(all_values) + pad)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    path = os.path.join(_figures_dir(cfg), "fig10_ablation.png")
    _save(fig, path)
    return path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def plot_all(results: ExperimentResults, cfg: Config) -> List[str]:
    """Generate every figure and return the list of paths."""
    paths = [
        plot_example_network(results, cfg),
        plot_pipeline(cfg),
        plot_roc(results, cfg),
        plot_confusion(results, cfg),
        plot_budget_bar(results, cfg),
        plot_budget_lines(results, cfg),
        plot_connectivity_bar(results, cfg),
        plot_family_grouped(results, cfg),
        plot_timing(results, cfg),
        plot_ablation(results, cfg),
    ]
    return paths
