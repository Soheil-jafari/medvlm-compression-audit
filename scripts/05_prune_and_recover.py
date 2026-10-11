#!/usr/bin/env python
"""Step 05: remove lowest-scoring units to the measured budget, recover, then mask audit.

Used by: E5, E6, E7, A1-A6.

Example:
    python scripts/05_prune_and_recover.py --exp E7 --budget B2 --seed 0
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
from medcompress.compression.recovery import run_prune_and_recover  # noqa: E402

SCRIPT = "scripts/05_prune_and_recover.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None, require_budget=True)
    config = resolve_config(args)
    return run_stage("05_prune_and_recover", run_prune_and_recover, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
