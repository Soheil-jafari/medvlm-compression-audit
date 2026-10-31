#!/usr/bin/env python
"""Step 01: run the teacher once and cache answers + top-k probabilities to disk.

Used by: E0 (consumed by E2, E4 and the recovery of E5-E7, A1-A6).

Example:
    python scripts/01_cache_teacher_outputs.py --exp E0
    python scripts/01_cache_teacher_outputs.py --exp E0 --stop-after-hours 10 \
        --resume-from /kaggle/input/<the stopped session's output>

It refuses to start before Gate 1 has passed (RC-E0a). With ``--stop-after-hours`` it stops
cleanly between two shards of 1,000 answers when the next one would not fit, and exits with
code 3: run it again with that session's output attached as ``--resume-from``.
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
from medcompress.config import RunContext  # noqa: E402
from medcompress.evaluation.image_reliance import require_gate1_pass  # noqa: E402
from medcompress.models.teacher_cache import CacheStopped, run_teacher_caching  # noqa: E402

SCRIPT = "scripts/01_cache_teacher_outputs.py"
RESUME_NEEDED = 3
"""The cache stopped between two shards for ``--stop-after-hours``: run it again to resume."""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--stop-after-hours",
        type=float,
        default=None,
        help="stop cleanly between two shards before this many hours",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="a stopped session's output (attached as input) to continue from",
    )
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp="E0", require_budget=True)
    config = resolve_config(args)

    def cache_after_gate1(config: dict, run: RunContext | None) -> None:
        """The cache only for a teacher that passed Gate 1 (checked before any download)."""
        require_gate1_pass(config)
        run_teacher_caching(config, run, stop_after_hours=args.stop_after_hours,
                            resume_from=args.resume_from)  # fmt: skip

    try:
        return run_stage("01_cache_teacher_outputs", cache_after_gate1, config=config, run=run)
    except CacheStopped as stop:
        print(f"STOPPED for a resume: {stop}. Run the same cell again in a new session with "
              "this version's output attached and --resume-from <its path>")  # fmt: skip
        return RESUME_NEEDED


if __name__ == "__main__":
    main_guard(main)
