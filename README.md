

## Quick start

```powershell
cd "C:\Users\Shubham Jain\Desktop\Adaptive Node Protection - Preserving Connectivity"
pip install -r requirements.txt

# Full reproduction (21 held-out graphs, budgets 0/2/4/6)
python run_all.py

# Fast smoke test (smaller graphs/datasets)
python run_all.py --quick
```

Outputs land in:

```
results/
├── summary.md               # report vs reproduction comparison
├── csv/
│   ├── per_graph_results.csv
│   ├── table3_combined_score.csv
│   ├── survival_connectivity.csv
│   ├── family_breakdown.csv
│   ├── selection_time.csv
│   ├── ablation.csv
│   └── feature_importances.csv
└── figures/
    ├── fig01_example_network.png
    ├── fig02_pipeline.png
    ├── fig03_roc.png
    ├── fig04_confusion_matrix.png
    ├── fig05_budget_bar.png
    ├── fig06_budget_lines.png
    ├── fig07_connectivity_bar.png
    ├── fig08_family_grouped.png
    ├── fig09_timing.png
    └── fig10_ablation.png
```

---

## Project layout

```
Adaptive Node Protection - Preserving Connectivity/
├── README.md
├── requirements.txt
├── run_all.py                 # command-line entry point
└── anpc/                      # the reproduction package
    ├── __init__.py
    ├── config.py              # every experimental parameter (Table 1)
    ├── graph_factory.py       # random / small-world / scale-free + connectivity repair
    ├── cascade.py             # independent-cascade failure model
    ├── metrics.py             # S, C, J and scenario evaluation
    ├── features.py            # seven node features (Section 4.2)
    ├── risk_model.py          # random forest + Table 2 metrics
    ├── dataset.py             # whole-graph train/test split (Section 5.2)
    ├── policies.py            # the six selection policies (Section 5.3)
    ├── bootstrap.py           # paired bootstrap over graphs
    ├── experiment.py          # main experiment + ablation + example
    ├── plotting.py            # figures 1–10
    └── summary.py             # CSV export + summary.md
```

---

## What is being simulated

### Failure model (Section 3.1)

A node that has just failed gets **one** transmission attempt to each neighbour
in the next round. Every directed attempt succeeds independently with
probability **0.23**. Failures are permanent and the process runs for **5
rounds**. Protected nodes neither fail nor transmit failure onward.

### Objectives (Section 3.2)

For a protection set \(P\) and diffusion outcome \(\omega\):

\[
S(P,\omega)=\frac{|V|-|F(P,\omega)|}{|V|},
\qquad
C(P,\omega)=\frac{|\mathrm{LCC}(G[V\setminus F(P,\omega)])|}{|V|},
\]

\[
J(P)=0.5\,\mathbb{E}[S]+0.5\,\mathbb{E}[C].
\]

The budget \(B\) limits how many healthy nodes may be protected.

### Risk model (Section 4.2)

A random forest predicts whether an initially healthy node fails within the
five-round horizon. Features:

1. normalized degree,
2. betweenness centrality,
3. local clustering,
4. distance to the closest failed node,
5. fraction of failed neighbours,
6. mean neighbour degree,
7. articulation-point indicator.

The development set contains **78 graphs** split by whole graph into **58
training** and **20 test** graphs, three initial-failure scenarios each, giving
**8,352** training and **2,880** test examples.

Tree depth, leaf size and class weighting are chosen by a **graph-level
validation split** inside the 58 training graphs (unconstrained trees overfit
the node labels). The selected configuration
(`max_depth=8`, `min_samples_leaf=2`, `class_weight="balanced"`,
`max_features="sqrt"`) reaches a held-out ROC-AUC of about 0.71 and, importantly,
reproduces the report's operating point: at a 0.50 threshold it produces
Brier ≈ 0.182 (report 0.181), TN ≈ 1,919 (report 1,918), and recall ≈ 58%
(report 60%), so the “many false alarms” discussion in Section 6.1 carries over.

### Proposed shortlist method (Section 4.3–4.5)

1. Score each healthy candidate with
   \(Q(v)=0.55\,r(v)+0.28\,b(v)+0.17\,d(v)\).
2. Keep the best **13** candidates.
3. For each candidate, run **11 shared** diffusion simulations on
   \(P\cup\{v\}\) and estimate \(J\).
4. Add the best candidate, then re-screen; repeat until the budget is reached.

### Compared policies (Section 5.3)

Random, highest degree, betweenness, risk ranking, full greedy, and the
proposed risk-guided shortlist. Full greedy tests every remaining node by
simulation at each step; the proposed method restricts the search to 13
candidates.

---

## What the code reproduces

| Report item | Location |
|---|---|
| Table 1 — simulation settings | `anpc/config.py` |
| Table 2 — classifier results | `risk_model.py`, `summary.md` |
| Table 3 — combined score by budget | `experiment.py`, `summary.md` |
| Table 4 — shortlist / feature ablation | `experiment.py`, `summary.md` |
| Fig. 1 — example network | `plotting.plot_example_network` |
| Fig. 2 — five-stage pipeline | `plotting.plot_pipeline` |
| Fig. 3 — ROC curve | `plotting.plot_roc` |
| Fig. 4 — confusion matrix | `plotting.plot_confusion` |
| Fig. 5 — score at budget 4 | `plotting.plot_budget_bar` |
| Fig. 6 — score vs budget | `plotting.plot_budget_lines` |
| Fig. 7 — largest-component share | `plotting.plot_connectivity_bar` |
| Fig. 8 — score by graph family | `plotting.plot_family_grouped` |
| Fig. 9 — selection-only time | `plotting.plot_timing` |
| Fig. 10 — shortlist-size effect | `plotting.plot_ablation` |
| Section 7.4 — single-network example | `experiment.run_example` |

`results/summary.md` shows the reproduced value next to the value printed in the
report for every table, so the comparison is explicit.

---

## Notes on reproducibility

* The report does **not** publish its random seeds. This reproduction uses a
  single documented seed (`Config.base_seed = 2024`) for every random choice,
  so results are stable across runs but not bit-for-bit identical to the paper.
* Every Monte-Carlo comparison uses **shared diffusion scenarios**, exactly as
  described in Section 4.4, so candidate protection sets are compared fairly.
* Evaluation scenarios (56 per graph) are generated independently of the 11
  selection scenarios and are never reused during selection.
* Raw per-graph results are saved before any aggregation, mirroring the
  reporting discipline described in Section 8.2.
* The `--quick` flag is only for smoke-testing; use the default full run for
  the values reported in `summary.md`.
