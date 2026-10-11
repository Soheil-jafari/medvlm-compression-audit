#!/usr/bin/env python
"""Step 00: check that the teacher actually relies on the image (go/no-go gate).

Runs the blank-image and masking stress tests on the teacher before any caching.
Used by: E0.

Example:
    python scripts/00_check_teacher_image_reliance.py --exp E0
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
from medcompress.evaluation.image_reliance import check_teacher_image_reliance  # noqa: E402

SCRIPT = "scripts/00_check_teacher_image_reliance.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp="E0", require_budget=True)
    config = resolve_config(args)
    return run_stage(
        "00_check_teacher_image_reliance", check_teacher_image_reliance, config=config, run=run
    )


if __name__ == "__main__":
    main_guard(main)
