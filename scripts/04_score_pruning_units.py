#!/usr/bin/env python
"""Step 04: score prunable units with the experiment's criterion.

Scores are shared across budgets, so --budget is optional here.
Used by: E5 (magnitude / activation / taylor), E6, E7, A1-A6.

Example:
    python scripts/04_score_pruning_units.py --exp E7 --seed 0
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
from medcompress.compression.pruning.criteria.base import score_units_for_experiment  # noqa: E402

SCRIPT = "scripts/04_score_pruning_units.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(
        parser, argv, script=SCRIPT, default_exp=None, require_budget=False
    )
    config = resolve_config(args)
    return run_stage("04_score_pruning_units", score_units_for_experiment, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
