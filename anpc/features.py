"""Node features used by the risk classifier (Section 4.2).

Seven interpretable features are computed for every initially healthy node:

1. normalized degree
2. betweenness centrality
3. local clustering coefficient
4. normalized shortest-path distance to the closest failed node
5. fraction of failed neighbours
6. normalized mean neighbour degree
7. articulation-point indicator (on the healthy subgraph)

Features describe the initial moment only; no future failure information leaks
into the representation.
"""

from __future__ import annotations

from collections import deque
from typing import List, Set, Tuple

import networkx as nx
import numpy as np

FEATURE_NAMES: Tuple[str, ...] = (
    "normalized_degree",
    "betweenness",
    "clustering",
    "distance_to_failed",
    "fraction_failed_neighbours",
    "mean_neighbour_degree",
    "articulation_point",
)


def extract_features(
    g: nx.Graph, initial_failed: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """Return a feature matrix and the corresponding healthy node ids."""
    n = g.number_of_nodes()
    failed_set: Set[int] = set(initial_failed)
    healthy = [v for v in g.nodes() if v not in failed_set]

    degree = dict(g.degree())
    max_degree = max(degree.values()) if degree else 0

    betweenness = nx.betweenness_centrality(g, normalized=True)
    clustering = nx.clustering(g)

    # Multi-source BFS from the initially failed nodes.
    distance = {v: float("inf") for v in g.nodes()}
    queue: deque = deque()
    for failed_node in failed_set:
        distance[failed_node] = 0
        queue.append(failed_node)
    while queue:
        u = queue.popleft()
        for w in g.neighbors(u):
            if distance[w] == float("inf"):
                distance[w] = distance[u] + 1
                queue.append(w)
    finite = [d for d in distance.values() if d != float("inf")]
    max_distance = max(finite) if finite else 1

    healthy_subgraph = g.subgraph(healthy)
    articulation_points: Set[int] = (
        set(nx.articulation_points(healthy_subgraph))
        if healthy_subgraph.number_of_nodes() > 0
        else set()
    )

    rows: List[List[float]] = []
    node_ids: List[int] = []
    for v in healthy:
        neighbours = list(g.neighbors(v))
        deg = degree[v]
        normalized_degree = deg / (n - 1) if n > 1 else 0.0
        d = distance[v]
        normalized_distance = (
            (d / max_distance)
            if max_distance > 0 and d != float("inf")
            else 0.0
        )
        fraction_failed = (
            sum(1 for w in neighbours if w in failed_set) / deg if deg else 0.0
        )
        mean_neighbour_degree = (
            float(np.mean([degree[w] for w in neighbours])) / (n - 1)
            if neighbours
            else 0.0
        )
        rows.append(
            [
                normalized_degree,
                betweenness.get(v, 0.0),
                clustering.get(v, 0.0),
                normalized_distance,
                fraction_failed,
                mean_neighbour_degree,
                1.0 if v in articulation_points else 0.0,
            ]
        )
        node_ids.append(v)

    return np.asarray(rows, dtype=float), node_ids
