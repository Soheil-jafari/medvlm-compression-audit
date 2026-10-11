#!/usr/bin/env python
"""Step 01: run the teacher once and cache answers + top-k probabilities to disk.

Used by: E0 (consumed by E2, E4 and the recovery of E5-E7, A1-A6).

Example:
    python scripts/01_cache_teacher_outputs.py --exp E0
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
from medcompress.models.teacher_cache import run_teacher_caching  # noqa: E402

SCRIPT = "scripts/01_cache_teacher_outputs.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp="E0", require_budget=True)
    config = resolve_config(args)
    return run_stage("01_cache_teacher_outputs", run_teacher_caching, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
