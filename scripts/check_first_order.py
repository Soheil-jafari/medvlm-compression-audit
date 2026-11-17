#!/usr/bin/env python
"""RC-G2's check of the first-order estimate: 30 random channel groups scored exactly.

Runs after stage 04 of E7 (it reads the run's ``unit_scores.csv``) on the same probes and
weights, and writes ``first_order_check.csv`` and the ``first_order_check:*`` rows (the
Spearman correlation; below 0.7 the agent stops and asks the team, D07).

Then (amendment of 2026-11-17, CPU, from the files alone): the slope and intercept of the exact
effect on the estimate with 95 % intervals, and E7's B2 selection again with the 30 exact
effects in place of their estimates (``first_order_check_b2.csv``: which of the 30 switch
between kept and removed; whether a KV-group or layer decision changes). ``--followup-only``
runs that part alone from the committed files, without a GPU.

Example:
    python scripts/check_first_order.py --exp E7 --seed 0
    python scripts/check_first_order.py --exp E7 --seed 0 --followup-only
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
from medcompress.compression.pruning.stage import (  # noqa: E402
    first_order_check_for_experiment,
    first_order_followup,
)

SCORING_SCRIPT = "scripts/04_score_pruning_units.py"
"""The check belongs to the scoring stage: only the experiments that score units take it."""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--followup-only",
        action="store_true",
        help="the fit and the B2 substitution from the committed files (CPU)",
    )
    args, run = parse_stage_args(
        parser, argv, script=SCORING_SCRIPT, default_exp="E7", require_budget=False
    )
    config = resolve_config(args)
    if args.followup_only:
        return run_stage("check_first_order", first_order_followup, config=config, run=run)
    return run_stage("check_first_order", first_order_check_for_experiment, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
