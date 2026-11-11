"""Graph families used in the study (Section 5.1).

Three synthetic families are generated:

* ``random``      -- Erdos-Renyi G(n, p) with p = 0.094
* ``small_world`` -- Watts-Strogatz with k = 4 and rewiring p = 0.22
* ``scale_free``  -- Barabasi-Albert with m = 2

Disconnected graphs are repaired with additional joining edges so that every
network starts connected, exactly as described in the report.
"""

from __future__ import annotations

import networkx as nx

from .config import Config


def generate_graph(family: str, n: int, cfg: Config, seed: int) -> nx.Graph:
    """Generate one connected graph from the requested family."""
    if family == "random":
        g = nx.gnp_random_graph(n, cfg.random_edge_p, seed=seed)
    elif family == "small_world":
        g = nx.watts_strogatz_graph(n, cfg.ws_k, cfg.ws_rewire_p, seed=seed)
    elif family == "scale_free":
        g = nx.barabasi_albert_graph(n, cfg.ba_m, seed=seed)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown graph family: {family}")
    return ensure_connected(g)


def ensure_connected(g: nx.Graph) -> nx.Graph:
    """Add joining edges between components until the graph is connected."""
    g = g.copy()
    for _ in range(g.number_of_nodes()):
        if nx.is_connected(g):
            return g
        components = sorted(nx.connected_components(g), key=lambda c: min(c))
        # Join each component to the next one in a deterministic ring.
        representatives = [min(comp) for comp in components]
        for i in range(len(representatives)):
            a = representatives[i]
            b = representatives[(i + 1) % len(representatives)]
            if a != b and not g.has_edge(a, b):
                g.add_edge(a, b)
    return g


def graph_family_list(cfg: Config) -> list[str]:
    """Return the family for each of the 21 held-out graphs."""
    families: list[str] = []
    for family in cfg.families:
        families.extend([family] * cfg.holdout_per_family)
    return families
