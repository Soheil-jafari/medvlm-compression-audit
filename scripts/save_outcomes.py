#!/usr/bin/env python
"""Score a run again and save its per-probe outcomes, without touching results/ (GPU).

Amendment of 2026-11-20 (exploratory additions A and B): the CPU analyses read the per-probe
outcomes of the validation and test splits, which earlier sessions did not always keep (RC-G2
exported only the pruned models), and the mask audit's margins, which the evaluation did not save
before 2026-11-20. This script loads the run's model as stage 06 does, scores one split
(``-o evaluation.split=val|test|test_unfiltered``) and writes ``outcomes*.npz`` to the output
root; with ``--audit-only`` it runs the mask audit alone and writes ``audit_<split>.npz``. It
never writes a results file: the recomputed rows are only compared with the committed ones (the
largest difference is printed), and a rerun audit must reproduce the saved plain-answer flips.

Examples (Kaggle, one T4; the run's model hand-off and the E2 adapter attached):
    python scripts/save_outcomes.py --exp E7 --budget B2 --seed 0 -o evaluation.split=val
    python scripts/save_outcomes.py --exp E7 --budget B2 --seed 0 -o evaluation.split=test \
        --audit-only
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
from medcompress.evaluation.runner import save_outcomes  # noqa: E402

SCRIPT = "scripts/06_evaluate.py"
"""The stage it repeats: only experiments that are evaluated take it."""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run the mask audit alone and write audit_<split>.npz",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing file")
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None, require_budget=True)
    config = resolve_config(args)
    return run_stage("save_outcomes", save_outcomes, config=config, run=run,
                     audit_only=args.audit_only, force=args.force)  # fmt: skip


if __name__ == "__main__":
    main_guard(main)
