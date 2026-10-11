#!/usr/bin/env python
"""Step 07: measure parameters, file size, peak memory, latency, throughput and energy.

Used by: all experiments. Final numbers come from the rented timing GPU.

Example:
    python scripts/07_benchmark_efficiency.py --exp E7 --budget B2 --seed 0
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
from medcompress.evaluation.efficiency import run_efficiency_benchmark  # noqa: E402

SCRIPT = "scripts/07_benchmark_efficiency.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None, require_budget=True)
    config = resolve_config(args)
    return run_stage("07_benchmark_efficiency", run_efficiency_benchmark, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
