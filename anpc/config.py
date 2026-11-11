"""Central configuration for every experiment in the reproduction.

The values mirror Table 1 and Section 5 of the report.  They are collected in a
single dataclass so that a reader can see every experimental knob in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------
    # Graph generation (Section 5.1)
    # ------------------------------------------------------------------
    n_nodes: int = 50
    families: Tuple[str, ...] = ("random", "small_world", "scale_free")
    random_edge_p: float = 0.094          # Random graphs: edge probability
    ws_k: int = 4                         # Small-world: neighbours per node
    ws_rewire_p: float = 0.22             # Small-world: rewiring probability
    ba_m: int = 2                         # Scale-free: edges per new node
    base_seed: int = 2024

    # ------------------------------------------------------------------
    # Independent-cascade failure model (Section 3.1)
    # ------------------------------------------------------------------
    n_initial_failed: int = 2
    transmission_p: float = 0.23
    horizon: int = 5                      # failure rounds

    # ------------------------------------------------------------------
    # Development data for the risk classifier (Section 5.2)
    # ------------------------------------------------------------------
    n_dev_graphs: int = 78
    n_train_graphs: int = 58
    n_test_graphs: int = 20
    scenarios_per_dev_graph: int = 3

    # ------------------------------------------------------------------
    # Held-out protection experiment (Section 5.2)
    # ------------------------------------------------------------------
    n_holdout_graphs: int = 21
    holdout_per_family: int = 7
    n_eval_scenarios: int = 56            # held-out diffusion outcomes
    n_candidate_scenarios: int = 11       # shared simulations during selection

    # ------------------------------------------------------------------
    # Protection budgets (Section 5.1)
    # ------------------------------------------------------------------
    budgets: Tuple[int, ...] = (0, 2, 4, 6)
    key_budget: int = 4

    # ------------------------------------------------------------------
    # Risk-guided shortlist (Section 4.3 - 4.5)
    # ------------------------------------------------------------------
    shortlist_size: int = 13
    risk_weight: float = 0.55
    betweenness_weight: float = 0.28
    degree_weight: float = 0.17

    # ------------------------------------------------------------------
    # Random-forest risk model (Section 4.2 / 6.1)
    # ------------------------------------------------------------------
    # Class-weighted trees were selected by graph-level validation.  Balanced
    # weights reproduce the report's operating point (many false alarms at a
    # 0.50 threshold) and its Brier score almost exactly.
    rf_n_estimators: int = 500
    rf_max_depth: Optional[int] = 8
    rf_min_samples_leaf: int = 2
    rf_max_features: str = "sqrt"
    rf_class_weight: Optional[str] = "balanced"
    rf_random_state: int = 42
    classification_threshold: float = 0.50

    # ------------------------------------------------------------------
    # Bootstrap (Section 6.2 / 7.3)
    # ------------------------------------------------------------------
    n_bootstrap: int = 2000
    ci_level: float = 0.95

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = "results"


def quick_config(cfg: Config) -> Config:
    """A small configuration used for smoke tests (not for the final numbers)."""
    return replace(
        cfg,
        n_dev_graphs=12,
        n_train_graphs=9,
        n_test_graphs=3,
        scenarios_per_dev_graph=3,
        holdout_per_family=2,
        n_holdout_graphs=6,
        n_eval_scenarios=12,
        n_candidate_scenarios=5,
        budgets=(0, 2, 4),
        shortlist_size=5,
        rf_n_estimators=60,
        n_bootstrap=300,
    )
