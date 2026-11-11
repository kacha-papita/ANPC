"""Reproduce every result in the Adaptive Node Protection report.

Usage
-----
    python run_all.py                # full reproduction (21 held-out graphs)
    python run_all.py --quick        # fast smoke test
    python run_all.py --output out   # custom output directory

Outputs
-------
    results/figures/fig01 ... fig10   # ten report figures
    results/csv/*.csv                 # raw per-graph data and tables
    results/summary.md                # report vs reproduction comparison
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

try:
    from anpc.config import Config, quick_config
    from anpc.experiment import run_experiment
    from anpc.plotting import plot_all
    from anpc.summary import write_summary
except ImportError as exc:  # pragma: no cover
    print(f"[error] could not import the reproduction package: {exc}")
    print("        run: pip install -r requirements.txt")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Adaptive Node Protection study",
    )
    parser.add_argument("--quick", action="store_true",
                        help="run a small smoke-test configuration")
    parser.add_argument("--output", type=str, default="results",
                        help="output directory (default: results)")
    parser.add_argument("--seed", type=int, default=2024,
                        help="base random seed (default: 2024)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = Config()
    if args.quick:
        cfg = quick_config(cfg)
    cfg = replace(cfg, base_seed=args.seed, output_dir=args.output)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "csv"), exist_ok=True)

    print("=" * 68)
    print(" Adaptive Node Protection — Preserving Connectivity")
    print(" Reproduction of the 21-graph simulation study")
    print("=" * 68)
    print(f" Mode        : {'quick smoke test' if args.quick else 'full'}")
    print(f" Base seed   : {cfg.base_seed}")
    print(f" Output      : {cfg.output_dir}")
    print("-" * 68)

    started = time.perf_counter()
    results = run_experiment(cfg)
    print("-" * 68)
    print(" Generating figures ...")
    figure_paths = plot_all(results, cfg)
    for path in figure_paths:
        print(f"   {path}")
    print(" Writing summary ...")
    summary_path = write_summary(results, cfg)
    elapsed = time.perf_counter() - started

    print("-" * 68)
    print(f" Done in {elapsed:.1f} s")
    print(f" Summary : {summary_path}")
    print(f" Figures : {os.path.join(cfg.output_dir, 'figures')}")
    print(f" CSV     : {os.path.join(cfg.output_dir, 'csv')}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
