# Adaptive Node Protection — Reproduction Summary

This file is generated automatically by `run_all.py`. It reproduces the experiments of *Adaptive Node Protection for Preserving Connectivity During Cascading Network Failures*.

## Experimental configuration

- Nodes per graph: **50**
- Families: **random, small_world, scale_free** (7 held-out graphs each)
- Initial failures: **2**
- Transmission probability: **0.23**
- Failure horizon: **5** rounds
- Budgets: **[0, 2, 4, 6]**
- Candidate simulations: **11**
- Evaluation simulations: **56** per network
- Shortlist size: **13**

## Table 2 — held-out classifier results

| Measure | Report | Reproduced |
|---|---|---|
| Test nodes | 2880 | 2880 |
| Failure proportion | 0.1083 | 0.1243 |
| Area under ROC | 0.7550 | 0.7097 |
| Average precision | 0.2810 | 0.2504 |
| Brier score | 0.1810 | 0.1817 |
| True healthy | 1918 | 1919 |
| False alarms | 650 | 603 |
| Missed failures | 124 | 150 |
| Found failures | 188 | 208 |

*Threshold = 0.50; precision = 0.2565; recall = 0.5810.*

## Table 3 — mean combined score by budget

| Policy | Report @2 | Reproduced @2 | Report @4 | Reproduced @4 | Report @6 | Reproduced @6 |
|---|---|---|---|---|---|---|
| Random | 0.8340 | 0.8334 | 0.8470 | 0.8572 | 0.8620 | 0.8638 |
| Highest degree | 0.8700 | 0.8731 | 0.8890 | 0.8953 | 0.9050 | 0.9032 |
| Betweenness | 0.8660 | 0.8709 | 0.8860 | 0.8880 | 0.9020 | 0.9007 |
| Risk ranking | 0.8740 | 0.8773 | 0.9130 | 0.9071 | 0.9370 | 0.9345 |
| Full greedy | 0.8800 | 0.8810 | 0.9130 | 0.9145 | 0.9350 | 0.9359 |
| Proposed shortlist | 0.8810 | 0.8829 | 0.9140 | 0.9153 | 0.9360 | 0.9352 |

*No-protection baseline (budget 0): **0.8230** (report: 0.821).*


## Survival and connectivity at budget 4

| Policy | Repro S | Repro C | Report S | Report C |
|---|---|---|---|---|
| Random | 0.8660 | 0.8484 | 0.8575 | 0.8368 |
| Highest degree | 0.8964 | 0.8941 | — | — |
| Betweenness | 0.8892 | 0.8869 | — | — |
| Risk ranking | 0.9115 | 0.9027 | — | — |
| Full greedy | 0.9155 | 0.9135 | — | — |
| Proposed shortlist | 0.9161 | 0.9144 | 0.9158 | 0.9121 |

## Combined score by graph family at budget 4

| Policy | random (report 0.879) | small_world (report 0.938) | scale_free (report 0.925) |
|---|---|---|---|
| Random | 0.8450 | 0.8845 | 0.8421 |
| Highest degree | 0.8796 | 0.8909 | 0.9153 |
| Betweenness | 0.8635 | 0.8896 | 0.9109 |
| Risk ranking | 0.9005 | 0.9190 | 0.9017 |
| Full greedy | 0.8968 | 0.9209 | 0.9257 |
| Proposed shortlist | 0.8957 | 0.9235 | 0.9267 |

## Selection-only time at budget 4

| Policy | Reproduced (ms) | Report (ms) |
|---|---|---|
| Random | 0.14 | nan |
| Highest degree | 0.10 | nan |
| Betweenness | 11.85 | nan |
| Risk ranking | 0.05 | nan |
| Full greedy | 1029.46 | 44.73 |
| Proposed shortlist | 304.65 | 15.72 |

Reproduced selection-time ratio (full greedy / proposed) = **3.38** (report: 2.84).

## Paired bootstrap over the 21 held-out graphs (budget 4)

- Proposed − Random: **+0.0581** [+0.0424, +0.0783] (report: ~0.049 to 0.087)
- Proposed − Full greedy: **+0.0008** [-0.0049, +0.0058] (report: ~−0.002 to 0.004)

## Table 4 — shortlist size and feature ablation (budget 4)

| Configuration | Mean score | Report |
|---|---|---|
| Risk-informed, 5 nodes | 0.9134 | 0.9144 |
| Risk-informed, 9 nodes | 0.9176 | 0.9135 |
| Risk-informed, 13 nodes | 0.9146 | 0.9140 |
| Risk-informed, 25 nodes | 0.9137 | 0.9135 |
| No shortlist, full search | 0.9124 | 0.9133 |
| Structure only, 13 nodes | 0.9091 | 0.9094 |

## Section 7.4 — one small-world network examined closely

- Initial failures: **[24, 31]** (report: 24 and 31)
- Proposed protection set: **[22, 25, 29, 32]** (report: 23, 25, 29, 32)
- Random protection set: **[4, 14, 16, 42]** (report: 19, 26, 36, 48)
- Proposed kept **48** nodes operational, LCC **48**, score **0.96** (report: 48 / 0.96)
- Random kept **44** nodes operational, LCC **44**, score **0.88** (report: 40 / 0.80)

## Random-forest feature importances

|  | Importance |
|---|---|
| distance_to_failed | 0.2989 |
| fraction_failed_neighbours | 0.1934 |
| mean_neighbour_degree | 0.1603 |
| betweenness | 0.1589 |
| normalized_degree | 0.1149 |
| clustering | 0.0641 |
| articulation_point | 0.0095 |

## Generated figures

1. `fig01_example_network.png` — network after protection/propagation
2. `fig02_pipeline.png` — five-stage search pipeline
3. `fig03_roc.png` — ROC curve
4. `fig04_confusion_matrix.png` — confusion matrix
5. `fig05_budget_bar.png` — combined score at budget 4
6. `fig06_budget_lines.png` — score vs budget
7. `fig07_connectivity_bar.png` — largest-component share at budget 4
8. `fig08_family_grouped.png` — score by graph family
9. `fig09_timing.png` — selection-only time
10. `fig10_ablation.png` — shortlist-size effect


> **Reproducibility note.** The report does not publish its random seeds, so the reproduced numbers are statistically comparable rather than bit-for-bit identical. All randomness is driven by `Config.base_seed`, and raw per-graph results are saved in `results/csv/` so every table and figure can be regenerated.
