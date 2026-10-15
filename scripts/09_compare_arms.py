#!/usr/bin/env python
"""Step 09: paired comparisons on the test split (the evidence rows of H1-H4).

Reads the per-probe outcomes (outcomes_test.npz) of every run under paths.output_root,
averages them over seeds and writes results/comparisons/. Needs the frozen pre-registration.

Example (from the repository root):
    python scripts/09_compare_arms.py -o paths.output_root=/kaggle/working/outputs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_stage  # noqa: E402
from medcompress.evaluation.compare_stage import run_comparisons  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0], experiment_args=False)
    parser.add_argument(
        "--resamples", type=int, default=None, help="Defaults to evaluation.bootstrap_resamples."
    )
    args = parser.parse_args(argv)
    config = resolve_config(args)
    return run_stage("09_compare_arms", run_comparisons, config=config, n_resamples=args.resamples)


if __name__ == "__main__":
    main_guard(main)
