"""Development dataset construction (Section 5.2).

78 generated graphs are split by *whole graph* into 58 training graphs and 20
test graphs.  Each graph contributes three initial-failure scenarios and the
two initially failed nodes are excluded from classification, giving:

    58 * 3 * 48 = 8,352 training examples
    20 * 3 * 48 = 2,880 test examples

Labels come from the unprotected independent cascade over the five-round
horizon.  Splitting by graph prevents node-level leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .cascade import generate_edge_randoms, independent_cascade
from .config import Config
from .features import extract_features
from .graph_factory import generate_graph


@dataclass
class DevelopmentDataset:
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    n_train_graphs: int
    n_test_graphs: int


def build_development_dataset(
    cfg: Config, rng: np.random.Generator
) -> DevelopmentDataset:
    """Generate the classifier's development dataset."""
    families = [cfg.families[i % len(cfg.families)] for i in range(cfg.n_dev_graphs)]
    graphs = [
        generate_graph(family, cfg.n_nodes, cfg, seed=cfg.base_seed + 1000 + i)
        for i, family in enumerate(families)
    ]

    # Deterministic whole-graph split.
    permutation = rng.permutation(cfg.n_dev_graphs)
    train_indices = set(int(i) for i in permutation[: cfg.n_train_graphs])

    train_features: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    test_features: List[np.ndarray] = []
    test_labels: List[np.ndarray] = []

    for graph_index, graph in enumerate(graphs):
        is_train = graph_index in train_indices
        for _ in range(cfg.scenarios_per_dev_graph):
            initial_failed = (
                rng.choice(
                    list(graph.nodes()), size=cfg.n_initial_failed, replace=False
                )
                .astype(int)
                .tolist()
            )
            edge_random = generate_edge_randoms(graph, rng)
            failed = independent_cascade(
                graph,
                initial_failed,
                protected=set(),
                transmission_p=cfg.transmission_p,
                horizon=cfg.horizon,
                edge_random=edge_random,
            )
            features, node_ids = extract_features(graph, initial_failed)
            labels = np.asarray(
                [1 if node in failed else 0 for node in node_ids], dtype=int
            )
            if is_train:
                train_features.append(features)
                train_labels.append(labels)
            else:
                test_features.append(features)
                test_labels.append(labels)

    return DevelopmentDataset(
        train_features=np.vstack(train_features),
        train_labels=np.concatenate(train_labels),
        test_features=np.vstack(test_features),
        test_labels=np.concatenate(test_labels),
        n_train_graphs=len(train_indices),
        n_test_graphs=cfg.n_dev_graphs - len(train_indices),
    )
