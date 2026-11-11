"""Independent-cascade failure model (Section 3.1).

A newly failed node gets one transmission attempt to each neighbour during the
next round.  Every directed attempt succeeds independently with probability
``p``.  Failures are permanent and the process runs for a fixed horizon.
Protected nodes neither fail nor transmit failure onward.

Randomness is represented as a pre-computed dictionary mapping every *directed*
edge to a uniform draw.  This makes Monte-Carlo comparisons reproducible: the
same dictionary can be re-used for several candidate protection sets, which is
the "shared diffusion scenario" idea from Section 4.4.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx
import numpy as np

EdgeRandom = Dict[Tuple[int, int], float]


def generate_edge_randoms(g: nx.Graph, rng: np.random.Generator) -> EdgeRandom:
    """Draw one independent uniform number for every directed edge."""
    random_by_edge: EdgeRandom = {}
    for u, v in g.edges():
        random_by_edge[(u, v)] = float(rng.random())
        random_by_edge[(v, u)] = float(rng.random())
    return random_by_edge


def generate_scenarios(
    g: nx.Graph, count: int, rng: np.random.Generator
) -> List[EdgeRandom]:
    """Generate ``count`` independent shared diffusion scenarios."""
    return [generate_edge_randoms(g, rng) for _ in range(count)]


def independent_cascade(
    g: nx.Graph,
    initial_failed: Iterable[int],
    protected: Iterable[int],
    transmission_p: float,
    horizon: int,
    edge_random: EdgeRandom,
) -> Set[int]:
    """Run the independent cascade and return the final failed set."""
    failed: Set[int] = set(initial_failed)
    frontier: Set[int] = set(initial_failed)
    protected_set: Set[int] = set(protected)

    for _ in range(horizon):
        if not frontier:
            break
        new_failures: Set[int] = set()
        for u in frontier:
            for v in g.neighbors(u):
                if v in failed or v in protected_set:
                    continue
                # Each directed attempt is decided by its own uniform draw.
                if edge_random.get((u, v), 1.0) < transmission_p:
                    new_failures.add(v)
        new_failures -= failed
        failed |= new_failures
        frontier = new_failures

    return failed
