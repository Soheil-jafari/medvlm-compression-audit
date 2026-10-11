#!/usr/bin/env python
"""Step 06: run the evaluation suite and write metrics.csv for one run.

Used by: all experiments.

Example:
    python scripts/06_evaluate.py --exp E7 --budget B2 --seed 0
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
from medcompress.evaluation.runner import run_evaluation  # noqa: E402

SCRIPT = "scripts/06_evaluate.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None, require_budget=True)
    config = resolve_config(args)
    return run_stage("06_evaluate", run_evaluation, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
