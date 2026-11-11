"""Paired bootstrap over whole graph instances (Sections 6.2 & 7.3).

The unit of resampling is a graph, not an individual diffusion run, because the
study only contains 21 independent network instances.  Resampling individual
runs would overstate the amount of independent evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class BootstrapResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    n_graphs: int
    n_bootstrap: int


def paired_bootstrap(
    differences: Sequence[float],
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap the mean of paired per-graph differences."""
    values = np.asarray(list(differences), dtype=float)
    n = len(values)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0, n_bootstrap)

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()

    alpha = (1.0 - ci_level) / 2.0
    return BootstrapResult(
        mean_difference=float(values.mean()),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1.0 - alpha)),
        n_graphs=n,
        n_bootstrap=n_bootstrap,
    )
