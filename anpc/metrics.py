"""Survival and connectivity objectives (Section 3.2).

For a protection set P and a diffusion outcome omega:

    S(P, omega) = (|V| - |F(P, omega)|) / |V|          (survival fraction)
    C(P, omega) = |LCC(G[V \\ F(P, omega)])| / |V|      (connectivity fraction)
    J(P)        = 0.5 * E[S] + 0.5 * E[C]               (combined score)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from .cascade import EdgeRandom, independent_cascade


def survival_fraction(g: nx.Graph, failed: Iterable[int]) -> float:
    """Fraction of nodes that are still operating."""
    n = g.number_of_nodes()
    return (n - len(set(failed))) / n if n else 0.0


def largest_component_size(g: nx.Graph, failed: Iterable[int]) -> int:
    """Size of the largest connected component of the surviving nodes."""
    failed_set = set(failed)
    alive = [v for v in g.nodes() if v not in failed_set]
    if not alive:
        return 0
    subgraph = g.subgraph(alive)
    return max(len(component) for component in nx.connected_components(subgraph))


def connectivity_fraction(g: nx.Graph, failed: Iterable[int]) -> float:
    """Largest operating component divided by the original node count."""
    n = g.number_of_nodes()
    return largest_component_size(g, failed) / n if n else 0.0


def combined_score(survival: float, connectivity: float) -> float:
    """Equal-weight combination used throughout the report."""
    return 0.5 * survival + 0.5 * connectivity


def evaluate_on_scenarios(
    g: nx.Graph,
    protected: Iterable[int],
    initial_failed: Iterable[int],
    scenarios: Sequence[EdgeRandom],
    transmission_p: float,
    horizon: int,
) -> Tuple[float, float, float]:
    """Evaluate a protection set on a fixed list of diffusion scenarios.

    Returns ``(mean survival, mean connectivity, combined score)``.
    """
    survival_values: List[float] = []
    connectivity_values: List[float] = []

    for edge_random in scenarios:
        failed = independent_cascade(
            g, initial_failed, protected, transmission_p, horizon, edge_random
        )
        survival_values.append(survival_fraction(g, failed))
        connectivity_values.append(connectivity_fraction(g, failed))

    mean_survival = float(np.mean(survival_values)) if survival_values else 0.0
    mean_connectivity = (
        float(np.mean(connectivity_values)) if connectivity_values else 0.0
    )
    return mean_survival, mean_connectivity, combined_score(
        mean_survival, mean_connectivity
    )
