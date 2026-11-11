"""Write CSV outputs and a human-readable reproduction summary.

The summary places the reproduced values next to the values printed in the
report so the reader can see immediately how close the reproduction is.
Exact agreement is not guaranteed because the report does not publish its
random seeds; the pipeline reproduces the *analysis*, not a bit-for-bit trace.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .experiment import ExperimentResults
from .policies import POLICY_NAMES

REPORT_RISK = {
    "Test nodes": 2880,
    "Failure proportion": 0.1083,
    "Area under ROC": 0.755,
    "Average precision": 0.281,
    "Brier score": 0.181,
    "True healthy": 1918,
    "False alarms": 650,
    "Missed failures": 124,
    "Found failures": 188,
}

REPORT_TABLE3 = {
    "Random": {2: 0.834, 4: 0.847, 6: 0.862},
    "Highest degree": {2: 0.870, 4: 0.889, 6: 0.905},
    "Betweenness": {2: 0.866, 4: 0.886, 6: 0.902},
    "Risk ranking": {2: 0.874, 4: 0.913, 6: 0.937},
    "Full greedy": {2: 0.880, 4: 0.913, 6: 0.935},
    "Proposed shortlist": {2: 0.881, 4: 0.914, 6: 0.936},
}

REPORT_SURVIVAL_CONNECTIVITY = {
    "Proposed shortlist": {"S": 0.9158, "C": 0.9121},
    "Random": {"S": 0.8575, "C": 0.8368},
}

REPORT_FAMILY = {
    "random": 0.879,
    "small_world": 0.938,
    "scale_free": 0.925,
}

REPORT_TIMING = {"Full greedy": 44.73, "Proposed shortlist": 15.72}

REPORT_ABLATION = {
    "Risk-informed, 5 nodes": 0.9144,
    "Risk-informed, 9 nodes": 0.9135,
    "Risk-informed, 13 nodes": 0.9140,
    "Risk-informed, 25 nodes": 0.9135,
    "No shortlist, full search": 0.9133,
    "Structure only, 13 nodes": 0.9094,
}


def _markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    index_name = df.index.name if df.index.name else ""
    header = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for index, row in df.iterrows():
        cells = [str(index)]
        for column in df.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                number = float(value)
                if number.is_integer() and abs(number) >= 1:
                    cells.append(str(int(number)))
                else:
                    cells.append(floatfmt.format(number))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _save_csvs(results: ExperimentResults, cfg: Config) -> None:
    csv_dir = os.path.join(cfg.output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    results.per_graph.to_csv(os.path.join(csv_dir, "per_graph_results.csv"),
                             index=False)
    results.table3.to_csv(os.path.join(csv_dir, "table3_combined_score.csv"))
    results.survival_connectivity.to_csv(
        os.path.join(csv_dir, "survival_connectivity.csv")
    )
    results.family_table.to_csv(os.path.join(csv_dir, "family_breakdown.csv"))
    results.timing.to_csv(os.path.join(csv_dir, "selection_time.csv"))
    results.ablation.to_csv(os.path.join(csv_dir, "ablation.csv"), index=False)
    results.feature_importances.to_csv(
        os.path.join(csv_dir, "feature_importances.csv")
    )


def write_summary(results: ExperimentResults, cfg: Config) -> str:
    """Write ``summary.md`` and all CSV artefacts."""
    _save_csvs(results, cfg)

    lines = []
    lines.append("# Adaptive Node Protection — Reproduction Summary\n")
    lines.append(
        "This file is generated automatically by `run_all.py`. It reproduces "
        "the experiments of *Adaptive Node Protection for Preserving "
        "Connectivity During Cascading Network Failures*.\n"
    )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    lines.append("## Experimental configuration\n")
    lines.append(
        f"- Nodes per graph: **{cfg.n_nodes}**\n"
        f"- Families: **{', '.join(cfg.families)}** ({cfg.holdout_per_family} held-out graphs each)\n"
        f"- Initial failures: **{cfg.n_initial_failed}**\n"
        f"- Transmission probability: **{cfg.transmission_p}**\n"
        f"- Failure horizon: **{cfg.horizon}** rounds\n"
        f"- Budgets: **{list(cfg.budgets)}**\n"
        f"- Candidate simulations: **{cfg.n_candidate_scenarios}**\n"
        f"- Evaluation simulations: **{cfg.n_eval_scenarios}** per network\n"
        f"- Shortlist size: **{cfg.shortlist_size}**\n"
    )

    # ------------------------------------------------------------------
    # Table 2 - classifier
    # ------------------------------------------------------------------
    m = results.risk_metrics
    reproduced = {
        "Test nodes": m.test_nodes,
        "Failure proportion": m.failure_proportion,
        "Area under ROC": m.roc_auc,
        "Average precision": m.average_precision,
        "Brier score": m.brier,
        "True healthy": m.true_healthy,
        "False alarms": m.false_alarms,
        "Missed failures": m.missed_failures,
        "Found failures": m.found_failures,
    }
    rows = []
    for key, reported in REPORT_RISK.items():
        rows.append(
            {"Measure": key, "Report": reported, "Reproduced": reproduced[key]}
        )
    table2 = pd.DataFrame(rows).set_index("Measure")
    lines.append("## Table 2 — held-out classifier results\n")
    lines.append(_markdown_table(table2))
    lines.append(
        f"\n*Threshold = {m.threshold:.2f}; precision = {m.precision:.4f}; "
        f"recall = {m.recall:.4f}.*\n"
    )

    # ------------------------------------------------------------------
    # Table 3 - combined score
    # ------------------------------------------------------------------
    rows = []
    for policy in POLICY_NAMES:
        row: Dict[str, object] = {"Policy": policy}
        for budget in (2, 4, 6):
            row[f"Report @{budget}"] = REPORT_TABLE3.get(policy, {}).get(
                budget, "—"
            )
            if budget in results.table3.columns:
                row[f"Reproduced @{budget}"] = float(
                    results.table3.loc[policy, budget]
                )
            else:
                row[f"Reproduced @{budget}"] = "—"
        rows.append(row)
    table3 = pd.DataFrame(rows).set_index("Policy")
    lines.append("## Table 3 — mean combined score by budget\n")
    lines.append(_markdown_table(table3))
    baseline_rows = results.per_graph[results.per_graph["Budget"] == 0]
    if not baseline_rows.empty:
        baseline_score = float(baseline_rows["J"].mean())
        lines.append(
            f"\n*No-protection baseline (budget 0): "
            f"**{baseline_score:.4f}** (report: 0.821).*\n"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Survival / connectivity at the key budget
    # ------------------------------------------------------------------
    lines.append(f"## Survival and connectivity at budget {cfg.key_budget}\n")
    sc = results.survival_connectivity
    rows = []
    for policy in POLICY_NAMES:
        reported = REPORT_SURVIVAL_CONNECTIVITY.get(policy)
        rows.append(
            {
                "Policy": policy,
                "Repro S": float(sc.loc[policy, "S"]),
                "Repro C": float(sc.loc[policy, "C"]),
                "Report S": reported["S"] if reported else "—",
                "Report C": reported["C"] if reported else "—",
            }
        )
    lines.append(_markdown_table(pd.DataFrame(rows).set_index("Policy")))
    lines.append("")

    # ------------------------------------------------------------------
    # Family breakdown
    # ------------------------------------------------------------------
    lines.append(f"## Combined score by graph family at budget {cfg.key_budget}\n")
    ordered_families = [c for c in cfg.families if c in results.family_table.columns]
    fam = results.family_table[ordered_families].copy()
    fam.columns = [
        f"{c} (report {REPORT_FAMILY.get(c, '—')})" for c in fam.columns
    ]
    lines.append(_markdown_table(fam))
    lines.append("")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    lines.append(f"## Selection-only time at budget {cfg.key_budget}\n")
    rows = []
    for policy in POLICY_NAMES:
        reported = REPORT_TIMING.get(policy)
        rows.append(
            {
                "Policy": policy,
                "Reproduced (ms)": float(results.timing.loc[policy]),
                "Report (ms)": reported,
            }
        )
    lines.append(_markdown_table(pd.DataFrame(rows).set_index("Policy"), "{:.2f}"))
    repro_greedy = float(results.timing.loc["Full greedy"])
    repro_prop = float(results.timing.loc["Proposed shortlist"])
    ratio = repro_greedy / repro_prop if repro_prop > 0 else float("inf")
    lines.append(
        f"\nReproduced selection-time ratio (full greedy / proposed) = "
        f"**{ratio:.2f}** (report: 2.84).\n"
    )

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    lines.append("## Paired bootstrap over the 21 held-out graphs (budget 4)\n")
    br = results.bootstrap_random
    bg = results.bootstrap_greedy
    lines.append(
        f"- Proposed − Random: **{br.mean_difference:+.4f}** "
        f"[{br.ci_low:+.4f}, {br.ci_high:+.4f}] "
        f"(report: ~0.049 to 0.087)\n"
        f"- Proposed − Full greedy: **{bg.mean_difference:+.4f}** "
        f"[{bg.ci_low:+.4f}, {bg.ci_high:+.4f}] "
        f"(report: ~−0.002 to 0.004)\n"
    )

    # ------------------------------------------------------------------
    # Ablation
    # ------------------------------------------------------------------
    lines.append("## Table 4 — shortlist size and feature ablation (budget 4)\n")
    abl = results.ablation.copy()
    abl["Report"] = abl["Configuration"].map(REPORT_ABLATION)
    lines.append(_markdown_table(abl.set_index("Configuration")))
    lines.append("")

    # ------------------------------------------------------------------
    # Example network
    # ------------------------------------------------------------------
    ex = results.example
    lines.append("## Section 7.4 — one small-world network examined closely\n")
    lines.append(
        f"- Initial failures: **{ex['initial_failed']}** "
        f"(report: 24 and 31)\n"
        f"- Proposed protection set: **{ex['protected_proposed']}** "
        f"(report: 23, 25, 29, 32)\n"
        f"- Random protection set: **{ex['protected_random']}** "
        f"(report: 19, 26, 36, 48)\n"
        f"- Proposed kept **{ex['summary_proposed']['operational']}** nodes "
        f"operational, LCC **{ex['summary_proposed']['lcc']}**, "
        f"score **{ex['summary_proposed']['score']:.2f}** "
        f"(report: 48 / 0.96)\n"
        f"- Random kept **{ex['summary_random']['operational']}** nodes "
        f"operational, LCC **{ex['summary_random']['lcc']}**, "
        f"score **{ex['summary_random']['score']:.2f}** "
        f"(report: 40 / 0.80)\n"
    )

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------
    lines.append("## Random-forest feature importances\n")
    fi = results.feature_importances.sort_values(ascending=False)
    fi_df = fi.to_frame(name="Importance")
    lines.append(_markdown_table(fi_df, "{:.4f}"))
    lines.append("")

    # ------------------------------------------------------------------
    # Figures & reproducibility note
    # ------------------------------------------------------------------
    lines.append("## Generated figures\n")
    lines.append(
        "1. `fig01_example_network.png` — network after protection/propagation\n"
        "2. `fig02_pipeline.png` — five-stage search pipeline\n"
        "3. `fig03_roc.png` — ROC curve\n"
        "4. `fig04_confusion_matrix.png` — confusion matrix\n"
        "5. `fig05_budget_bar.png` — combined score at budget 4\n"
        "6. `fig06_budget_lines.png` — score vs budget\n"
        "7. `fig07_connectivity_bar.png` — largest-component share at budget 4\n"
        "8. `fig08_family_grouped.png` — score by graph family\n"
        "9. `fig09_timing.png` — selection-only time\n"
        "10. `fig10_ablation.png` — shortlist-size effect\n"
    )
    lines.append(
        "\n> **Reproducibility note.** The report does not publish its random "
        "seeds, so the reproduced numbers are statistically comparable rather "
        "than bit-for-bit identical. All randomness is driven by "
        "`Config.base_seed`, and raw per-graph results are saved in "
        "`results/csv/` so every table and figure can be regenerated.\n"
    )

    text = "\n".join(lines)
    path = os.path.join(cfg.output_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
