#!/usr/bin/env python
"""Step 08: aggregate per-run results into summary CSVs and paper tables.

Computes seed means/SDs, paired bootstrap CIs and Holm-corrected tests.
Used by: all experiments (final step).

Example (from the repository root):
    python scripts/08_aggregate_results.py --out-dir results/summary --tables-dir paper/tables
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_stage  # noqa: E402
from medcompress.evaluation.statistics import aggregate_results  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0], experiment_args=False)
    parser.add_argument(
        "--results-root", type=Path, default=None, help="Defaults to paths.results_root."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/summary"))
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("paper/tables"),
        help="Where the LaTeX tables and numbers.tex go (relative to the working directory).",
    )
    args = parser.parse_args(argv)
    config = resolve_config(args)
    results_root = args.results_root or Path(config["paths"]["results_root"])
    return run_stage(
        "08_aggregate_results",
        aggregate_results,
        results_root=results_root,
        out_dir=args.out_dir,
        tables_dir=args.tables_dir,
    )


if __name__ == "__main__":
    main_guard(main)
