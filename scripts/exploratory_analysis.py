#!/usr/bin/env python
"""Exploratory analyses A and B from the saved per-probe outcomes (CPU, amendment of 2026-11-20).

A: plain answers against the operating point per finding and macro on each test, their gap and
the ECE; the re-calibration curve (the operating threshold refitted on n = 8 ... 128 validation
images, 1,000 stratified draws, seed 0; n = 0 the plain answer); the lesion-mask audit at the
operating threshold next to the pre-registered plain-answer version. B: flips against E2 seed 0
(forgotten and gained, by finding, size stratum and mask), the Jaccard overlap of the forgotten
sets between arms and families, and the compression-identified exemplars; image-bootstrap 95 %
intervals (10,000 resamples, seed 0). Exploratory: outside H1-H4, no verdicts.

It reads every ``outcomes*.npz`` and ``audit_<split>.npz`` under the folders given (the attached
outcome hand-offs; by default everything under /kaggle/input) and writes ``results/exploratory/``
(CSV files and SUMMARY.md). It reruns from scratch as arms arrive.

Example (Kaggle CPU notebook, the outcome datasets attached):
    python scripts/exploratory_analysis.py --outcomes /kaggle/input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outcomes",
        type=Path,
        action="append",
        default=None,
        help="a folder to search for outcome files (repeatable; default /kaggle/input)",
    )
    parser.add_argument("--out", type=Path, default=REPO / "results" / "exploratory")
    parser.add_argument(
        "--draws", type=int, default=1000, help="draws per n (pre-registered 1,000)"
    )
    parser.add_argument(
        "--resamples", type=int, default=10_000, help="bootstrap resamples (pre-registered 10,000)"
    )
    args = parser.parse_args(argv)

    from medcompress.cli import resolve_config
    from medcompress.evaluation.exploratory import run_all
    from medcompress.evaluation.runner import resolve_settings
    from medcompress.registry import REGISTRY

    namespace = argparse.Namespace(config=None, exp="E2", override=[], data_root=None, handoffs=[])
    settings = resolve_settings(resolve_config(namespace))
    counts = run_all(
        args.outcomes or [Path("/kaggle/input")], args.out,
        order=settings.findings, size_thresholds=settings.thresholds,
        mask_findings=settings.mask_findings, n_bins=int(settings.n_bins), known=list(REGISTRY),
        draws=args.draws, resamples=args.resamples,
    )  # fmt: skip
    for name, rows in counts.items():
        print(f"{name}.csv: {rows} rows")
    print(f"wrote {args.out} (SUMMARY.md lists the outcomes found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
