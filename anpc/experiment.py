"""Main protection experiment and all aggregation steps (Sections 5 & 6).

This module ties the pieces together:

1. build the classifier's development dataset and train the risk model;
2. generate the 21 held-out graphs (7 per family);
3. evaluate the six selection policies at budgets 0, 2, 4, 6;
4. aggregate Table 3, the survival/connectivity split, the family breakdown
   and the selection-time comparison;
5. compute paired bootstrap intervals over whole graphs;
6. run the shortlist-size and feature ablation;
7. reproduce the single-network example from Section 7.4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .bootstrap import BootstrapResult, paired_bootstrap
from .cascade import generate_edge_randoms, generate_scenarios, independent_cascade
from .config import Config
from .dataset import build_development_dataset
from .features import FEATURE_NAMES, extract_features
from .graph_factory import generate_graph
from .metrics import (
    combined_score,
    connectivity_fraction,
    evaluate_on_scenarios,
    survival_fraction,
)
from .policies import (
    POLICY_NAMES,
    select_policy,
    select_proposed_shortlist,
    select_random,
)
from .risk_model import RiskMetrics, RiskModel

ABLATION_CONFIGS: Tuple[Tuple[str, int, bool], ...] = (
    ("Risk-informed, 5 nodes", 5, True),
    ("Risk-informed, 9 nodes", 9, True),
    ("Risk-informed, 13 nodes", 13, True),
    ("Risk-informed, 25 nodes", 25, True),
    ("No shortlist, full search", 0, True),
    ("Structure only, 13 nodes", 13, False),
)


@dataclass
class GraphCase:
    """Everything the experiment needs for one held-out graph."""

    index: int
    family: str
    graph: nx.Graph
    initial_failed: List[int]
    risk: Dict[int, float]
    eval_scenarios: list
    position: Dict[int, Tuple[float, float]]


@dataclass
class ExperimentResults:
    config: Config
    risk_metrics: RiskMetrics
    roc: Tuple[np.ndarray, np.ndarray]
    per_graph: pd.DataFrame
    table3: pd.DataFrame
    survival_connectivity: pd.DataFrame
    family_table: pd.DataFrame
    timing: pd.Series
    bootstrap_random: BootstrapResult
    bootstrap_greedy: BootstrapResult
    ablation: pd.DataFrame
    example: Dict[str, object]
    feature_importances: pd.Series
    holdout_cases: List[GraphCase] = field(default_factory=list)


# ----------------------------------------------------------------------
# Held-out graphs
# ----------------------------------------------------------------------
def build_holdout_cases(
    cfg: Config, model: RiskModel, rng: np.random.Generator
) -> List[GraphCase]:
    cases: List[GraphCase] = []
    index = 0
    for family in cfg.families:
        for _ in range(cfg.holdout_per_family):
            graph = generate_graph(
                family, cfg.n_nodes, cfg, seed=cfg.base_seed + 5000 + index
            )
            initial_failed = (
                rng.choice(
                    list(graph.nodes()), size=cfg.n_initial_failed, replace=False
                )
                .astype(int)
                .tolist()
            )
            features, node_ids = extract_features(graph, initial_failed)
            risk = dict(zip(node_ids, model.predict_risk(features).tolist()))
            eval_scenarios = generate_scenarios(graph, cfg.n_eval_scenarios, rng)
            position = nx.spring_layout(
                graph, seed=cfg.base_seed + index, scale=1.0
            )
            cases.append(
                GraphCase(
                    index=index,
                    family=family,
                    graph=graph,
                    initial_failed=initial_failed,
                    risk=risk,
                    eval_scenarios=eval_scenarios,
                    position=position,
                )
            )
            index += 1
    return cases


# ----------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------
def run_experiment(cfg: Config, log=print) -> ExperimentResults:
    rng = np.random.default_rng(cfg.base_seed)

    log("[1/5] Building development dataset ...")
    dataset = build_development_dataset(cfg, rng)
    log(
        f"      train={dataset.train_features.shape} "
        f"test={dataset.test_features.shape}"
    )

    log("[2/5] Training random-forest risk model ...")
    model = RiskModel(cfg).fit(dataset.train_features, dataset.train_labels)
    risk_metrics = model.evaluate(dataset.test_features, dataset.test_labels)
    test_probabilities = model.predict_risk(dataset.test_features)
    fpr, tpr, _ = roc_curve(dataset.test_labels, test_probabilities)
    log(
        f"      ROC-AUC={risk_metrics.roc_auc:.3f}  "
        f"AP={risk_metrics.average_precision:.3f}  "
        f"Brier={risk_metrics.brier:.3f}"
    )

    log("[3/5] Generating held-out graphs ...")
    cases = build_holdout_cases(cfg, model, rng)
    log(f"      {len(cases)} graphs ({cfg.holdout_per_family} per family)")

    log("[4/5] Running protection comparison ...")
    rows: List[Dict[str, object]] = []
    for case in cases:
        for policy in POLICY_NAMES:
            for budget in cfg.budgets:
                seed = (
                    cfg.base_seed * 13
                    + case.index * 1000
                    + budget * 37
                    + POLICY_NAMES.index(policy)
                )
                selection_rng = np.random.default_rng(seed)
                start = time.perf_counter()
                protected = select_policy(
                    policy,
                    case.graph,
                    case.initial_failed,
                    budget,
                    cfg,
                    selection_rng,
                    case.risk,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                survival, connectivity, score = evaluate_on_scenarios(
                    case.graph,
                    protected,
                    case.initial_failed,
                    case.eval_scenarios,
                    cfg.transmission_p,
                    cfg.horizon,
                )
                rows.append(
                    {
                        "CaseIndex": case.index,
                        "Family": case.family,
                        "Policy": policy,
                        "Budget": budget,
                        "S": survival,
                        "C": connectivity,
                        "J": score,
                        "SelTimeMs": elapsed_ms,
                        "Protected": "|".join(str(v) for v in sorted(protected)),
                    }
                )
        log(f"      graph {case.index + 1}/{len(cases)} ({case.family}) done")

    per_graph = pd.DataFrame(rows)

    table3 = (
        per_graph.pivot_table(
            index="Policy", columns="Budget", values="J", aggfunc="mean"
        )
        .reindex(POLICY_NAMES)
    )
    key = cfg.key_budget
    survival_connectivity = (
        per_graph[per_graph["Budget"] == key]
        .groupby("Policy")[["S", "C", "J"]]
        .mean()
        .reindex(POLICY_NAMES)
    )
    family_table = (
        per_graph[per_graph["Budget"] == key]
        .pivot_table(index="Policy", columns="Family", values="J", aggfunc="mean")
        .reindex(POLICY_NAMES)
    )
    timing = (
        per_graph[per_graph["Budget"] == key]
        .groupby("Policy")["SelTimeMs"]
        .mean()
        .reindex(POLICY_NAMES)
    )

    pivot = per_graph[per_graph["Budget"] == key].pivot_table(
        index="CaseIndex", columns="Policy", values="J"
    )
    bootstrap_random = paired_bootstrap(
        (pivot["Proposed shortlist"] - pivot["Random"]).values,
        cfg.n_bootstrap,
        cfg.ci_level,
        seed=cfg.base_seed,
    )
    bootstrap_greedy = paired_bootstrap(
        (pivot["Proposed shortlist"] - pivot["Full greedy"]).values,
        cfg.n_bootstrap,
        cfg.ci_level,
        seed=cfg.base_seed + 1,
    )
    log(
        f"      bootstrap vs random : {bootstrap_random.mean_difference:+.4f} "
        f"[{bootstrap_random.ci_low:+.4f}, {bootstrap_random.ci_high:+.4f}]"
    )
    log(
        f"      bootstrap vs greedy : {bootstrap_greedy.mean_difference:+.4f} "
        f"[{bootstrap_greedy.ci_low:+.4f}, {bootstrap_greedy.ci_high:+.4f}]"
    )

    log("[5/5] Running shortlist ablation and example network ...")
    ablation = run_ablation(cfg, cases, log=log)
    example = run_example(cfg, cases, model, log=log)
    feature_importances = pd.Series(
        model.feature_importances(), index=list(FEATURE_NAMES)
    )

    return ExperimentResults(
        config=cfg,
        risk_metrics=risk_metrics,
        roc=(fpr, tpr),
        per_graph=per_graph,
        table3=table3,
        survival_connectivity=survival_connectivity,
        family_table=family_table,
        timing=timing,
        bootstrap_random=bootstrap_random,
        bootstrap_greedy=bootstrap_greedy,
        ablation=ablation,
        example=example,
        feature_importances=feature_importances,
        holdout_cases=cases,
    )


# ----------------------------------------------------------------------
# Ablation (Section 6.6)
# ----------------------------------------------------------------------
def run_ablation(
    cfg: Config, cases: Sequence[GraphCase], log=print
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label, size, use_risk in ABLATION_CONFIGS:
        scores: List[float] = []
        for case in cases:
            seed = (
                cfg.base_seed
                + 4000
                + case.index * 97
                + (size if size is not None else 999)
                + (0 if use_risk else 1)
            )
            rng = np.random.default_rng(seed)
            protected = select_proposed_shortlist(
                case.graph,
                case.initial_failed,
                cfg.key_budget,
                cfg,
                rng,
                case.risk,
                shortlist_size=size,
                use_risk=use_risk,
            )
            _, _, score = evaluate_on_scenarios(
                case.graph,
                protected,
                case.initial_failed,
                case.eval_scenarios,
                cfg.transmission_p,
                cfg.horizon,
            )
            scores.append(score)
        mean_score = float(np.mean(scores))
        rows.append({"Configuration": label, "Mean score": mean_score})
        log(f"      {label}: {mean_score:.4f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Single-network example (Section 7.4)
# ----------------------------------------------------------------------
def run_example(
    cfg: Config, cases: Sequence[GraphCase], model: RiskModel, log=print
) -> Dict[str, object]:
    case = next(c for c in cases if c.family == "small_world")
    graph = case.graph

    # Use the report's illustrative initial failures when available.
    if all(node in graph for node in (24, 31)):
        initial_failed = [24, 31]
    else:  # pragma: no cover - 50-node graphs always contain 24 and 31
        initial_failed = list(case.initial_failed)

    features, node_ids = extract_features(graph, initial_failed)
    risk = dict(zip(node_ids, model.predict_risk(features).tolist()))

    proposed_rng = np.random.default_rng(cfg.base_seed + 24601)
    protected_proposed = select_proposed_shortlist(
        graph, initial_failed, cfg.key_budget, cfg, proposed_rng, risk
    )
    random_rng = np.random.default_rng(cfg.base_seed + 8675309)
    protected_random = select_random(
        graph, initial_failed, cfg.key_budget, cfg, random_rng, risk
    )

    # One independent evaluation trajectory, shared by both policies.
    scenario = generate_edge_randoms(
        graph, np.random.default_rng(cfg.base_seed + 31415)
    )
    failed_proposed = independent_cascade(
        graph,
        initial_failed,
        protected_proposed,
        cfg.transmission_p,
        cfg.horizon,
        scenario,
    )
    failed_random = independent_cascade(
        graph,
        initial_failed,
        protected_random,
        cfg.transmission_p,
        cfg.horizon,
        scenario,
    )

    def summarise(failed) -> Dict[str, float]:
        survival = survival_fraction(graph, failed)
        connectivity = connectivity_fraction(graph, failed)
        return {
            "survival": survival,
            "connectivity": connectivity,
            "score": combined_score(survival, connectivity),
            "operational": int(round(survival * graph.number_of_nodes())),
            "lcc": int(round(connectivity * graph.number_of_nodes())),
        }

    example = {
        "family": case.family,
        "graph": graph,
        "position": case.position,
        "initial_failed": sorted(initial_failed),
        "protected_proposed": sorted(protected_proposed),
        "protected_random": sorted(protected_random),
        "failed_proposed": failed_proposed,
        "failed_random": failed_random,
        "summary_proposed": summarise(failed_proposed),
        "summary_random": summarise(failed_random),
    }
    log(
        "      example proposed protects "
        f"{example['protected_proposed']} -> "
        f"{example['summary_proposed']['operational']} operational"
    )
    log(
        "      example random protects   "
        f"{example['protected_random']} -> "
        f"{example['summary_random']['operational']} operational"
    )
    return example
