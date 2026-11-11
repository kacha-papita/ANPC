"""The six protection-selection policies (Section 5.3).

1. Random selection
2. Highest degree
3. Betweenness centrality
4. Risk ranking (classifier output only)
5. Full greedy (every remaining node tested by simulation)
6. Proposed risk-guided adaptive shortlist

All policies return a set of at most ``budget`` healthy, unprotected nodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import networkx as nx
import numpy as np

from .cascade import EdgeRandom, generate_scenarios, independent_cascade
from .config import Config
from .metrics import (
    combined_score,
    connectivity_fraction,
    survival_fraction,
)


def _eligible(g: nx.Graph, initial_failed: Sequence[int], protected: Set[int]) -> List[int]:
    failed = set(initial_failed)
    return [v for v in g.nodes() if v not in failed and v not in protected]


def _normalize(scores: Dict[int, float], candidates: Sequence[int]) -> Dict[int, float]:
    """Min-max normalize scores over the current candidate set."""
    values = np.asarray([scores.get(v, 0.0) for v in candidates], dtype=float)
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-12:
        return {v: 0.0 for v in candidates}
    return {v: (scores.get(v, 0.0) - low) / (high - low) for v in candidates}


def estimate_score(
    g: nx.Graph,
    protected: Set[int],
    initial_failed: Sequence[int],
    scenarios: Sequence[EdgeRandom],
    cfg: Config,
) -> float:
    """Estimate J(P) on a fixed set of shared diffusion scenarios."""
    survival_values: List[float] = []
    connectivity_values: List[float] = []
    for edge_random in scenarios:
        failed = independent_cascade(
            g,
            initial_failed,
            protected,
            cfg.transmission_p,
            cfg.horizon,
            edge_random,
        )
        survival_values.append(survival_fraction(g, failed))
        connectivity_values.append(connectivity_fraction(g, failed))
    return combined_score(
        float(np.mean(survival_values)), float(np.mean(connectivity_values))
    )


# ----------------------------------------------------------------------
# Simple ranking policies
# ----------------------------------------------------------------------
def select_random(
    g: nx.Graph,
    initial_failed: Sequence[int],
    budget: int,
    cfg: Config,
    rng: np.random.Generator,
    risk: Optional[Dict[int, float]] = None,
) -> Set[int]:
    candidates = _eligible(g, initial_failed, set())
    k = min(budget, len(candidates))
    if k <= 0:
        return set()
    return set(int(v) for v in rng.choice(candidates, size=k, replace=False))


def select_highest_degree(
    g: nx.Graph,
    initial_failed: Sequence[int],
    budget: int,
    cfg: Config,
    rng: np.random.Generator,
    risk: Optional[Dict[int, float]] = None,
) -> Set[int]:
    candidates = _eligible(g, initial_failed, set())
    ranked = sorted(candidates, key=lambda v: (-g.degree(v), v))
    return set(ranked[:budget])


def select_betweenness(
    g: nx.Graph,
    initial_failed: Sequence[int],
    budget: int,
    cfg: Config,
    rng: np.random.Generator,
    risk: Optional[Dict[int, float]] = None,
) -> Set[int]:
    candidates = _eligible(g, initial_failed, set())
    centrality = nx.betweenness_centrality(g, normalized=True)
    ranked = sorted(candidates, key=lambda v: (-centrality.get(v, 0.0), v))
    return set(ranked[:budget])


def select_risk_ranking(
    g: nx.Graph,
    initial_failed: Sequence[int],
    budget: int,
    cfg: Config,
    rng: np.random.Generator,
    risk: Optional[Dict[int, float]] = None,
) -> Set[int]:
    candidates = _eligible(g, initial_failed, set())
    risk = risk or {}
    ranked = sorted(candidates, key=lambda v: (-risk.get(v, 0.0), v))
    return set(ranked[:budget])


# ----------------------------------------------------------------------
# Simulation-based policies
# ----------------------------------------------------------------------
def select_full_greedy(
    g: nx.Graph,
    initial_failed: Sequence[int],
    budget: int,
    cfg: Config,
    rng: np.random.Generator,
    risk: Optional[Dict[int, float]] = None,
) -> Set[int]:
    """Test every remaining candidate through simulation at each step."""
    protected: Set[int] = set()
    for _ in range(budget):
        candidates = _eligible(g, initial_failed, protected)
        if not candidates:
            break
        scenarios = generate_scenarios(g, cfg.n_candidate_scenarios, rng)
        best_node: Optional[int] = None
        best_score = -1.0
        for node in candidates:
            score = estimate_score(
                g, protected | {node}, initial_failed, scenarios, cfg
            )
            if score > best_score:
                best_score = score
                best_node = node
        if best_node is None:  # pragma: no cover - defensive
            break
        protected.add(best_node)
    return protected


def select_proposed_shortlist(
    g: nx.Graph,
    initial_failed: Sequence[int],
    budget: int,
    cfg: Config,
    rng: np.random.Generator,
    risk: Optional[Dict[int, float]] = None,
    shortlist_size: Optional[int] = None,
    use_risk: bool = True,
) -> Set[int]:
    """Risk-guided adaptive shortlist selection (Algorithm 1).

    Parameters
    ----------
    shortlist_size : int or None
        Override for the ablation study.  ``None`` means "all candidates"
        (the full-search ablation).
    use_risk : bool
        When False, the predicted risk is removed from the screening score
        (the "structure only" ablation).
    """
    protected: Set[int] = set()
    risk = risk or {}
    betweenness = nx.betweenness_centrality(g, normalized=True)
    degree = dict(g.degree())
    # ``None`` -> configured size (13); ``0`` -> no shortlist (full search).
    size = cfg.shortlist_size if shortlist_size is None else shortlist_size

    for _ in range(budget):
        candidates = _eligible(g, initial_failed, protected)
        if not candidates:
            break

        if size <= 0 or size >= len(candidates):
            shortlist = candidates
        else:
            risk_norm = (
                _normalize(risk, candidates)
                if use_risk
                else {v: 0.0 for v in candidates}
            )
            betweenness_norm = _normalize(betweenness, candidates)
            degree_norm = _normalize(
                {v: float(degree[v]) for v in candidates}, candidates
            )
            screening = {
                v: (
                    cfg.risk_weight * risk_norm[v]
                    + cfg.betweenness_weight * betweenness_norm[v]
                    + cfg.degree_weight * degree_norm[v]
                )
                for v in candidates
            }
            ranked = sorted(candidates, key=lambda v: (-screening[v], v))
            shortlist = ranked[:size]

        scenarios = generate_scenarios(g, cfg.n_candidate_scenarios, rng)
        best_node: Optional[int] = None
        best_score = -1.0
        for node in shortlist:
            score = estimate_score(
                g, protected | {node}, initial_failed, scenarios, cfg
            )
            if score > best_score:
                best_score = score
                best_node = node
        if best_node is None:  # pragma: no cover - defensive
            break
        protected.add(best_node)

    return protected


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
POLICY_NAMES = (
    "Random",
    "Highest degree",
    "Betweenness",
    "Risk ranking",
    "Full greedy",
    "Proposed shortlist",
)

_POLICY_FUNCTIONS = {
    "Random": select_random,
    "Highest degree": select_highest_degree,
    "Betweenness": select_betweenness,
    "Risk ranking": select_risk_ranking,
    "Full greedy": select_full_greedy,
    "Proposed shortlist": select_proposed_shortlist,
}


def select_policy(
    name: str,
    g: nx.Graph,
    initial_failed: Sequence[int],
    budget: int,
    cfg: Config,
    rng: np.random.Generator,
    risk: Optional[Dict[int, float]] = None,
) -> Set[int]:
    """Dispatch to the requested policy by display name."""
    return _POLICY_FUNCTIONS[name](g, initial_failed, budget, cfg, rng, risk)
