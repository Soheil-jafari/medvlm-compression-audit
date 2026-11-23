#!/usr/bin/env python
"""Finding-balanced recovery of one pruning arm at B2, seed 0, then its validation pass (GPU;
FBR cells (iii) and (iv)).

Secondary study of the amendment of 2026-11-22 (docs/decisions/fbr_study_2026-11-22.md): the
arm's pruned model before recovery is rebuilt from E2 seed 0 and its committed removal.json
(checked against the committed parameter count and removal_by_layer.csv), trained with FBR (or,
with ``--control``, the sampling-only recipe), merged, scored on validation (all suites) and on
the probe-answer pass. Rows go to ``<out>/<label>/fbr_metrics.csv`` with ``training.json``; arrays
to ``<output_root>/fbr/<label>/``. ``--trial N`` stops after N updates and writes the timing and
the estimate (``trial.json``) instead.

Examples (Kaggle, one T4; the shared data, the E2 seed 0 adapter, the teacher cache and the
E2 margin cache attached):
    python scripts/fbr_recover.py --exp E5 --variant magnitude --budget B2 --seed 0 \
        --margins /kaggle/input/.../fbr_e2_margins.json --trial 20
    python scripts/fbr_recover.py --exp E5 --variant activation --budget B2 --seed 0 \
        --margins /kaggle/input/.../fbr_e2_margins.json --control
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import (  # noqa: E402
    build_parser,
    main_guard,
    parse_stage_args,
    resolve_config,
    run_stage,
)
from medcompress.compression.fbr_stage import run_fbr_arm  # noqa: E402

SCRIPT = "scripts/05_prune_and_recover.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--margins", type=Path, required=True, help="fbr_e2_margins.json")
    parser.add_argument("--control", action="store_true", help="sampling-only (coefficient 0)")
    parser.add_argument("--trial", type=int, default=None, metavar="N",
                        help="stop after N updates and write the timing")  # fmt: skip
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <results_root>/exploratory/fbr")  # fmt: skip
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None)
    config = resolve_config(args)
    out = args.out or Path(config["paths"]["results_root"]) / "exploratory" / "fbr"
    return run_stage("fbr_recover", run_fbr_arm, config=config, arm=run,
                     margins_file=args.margins, out=out, control=args.control,
                     trial_steps=args.trial)  # fmt: skip


if __name__ == "__main__":
    main_guard(main)
