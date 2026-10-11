#!/usr/bin/env python
"""Step 03: post-training quantization (GPTQ / AWQ weight-only, W8A8).

Used by: E3 (variants w8, w4, w8a8), E8 (E7 + 4-bit), E9 (teacher 4-bit).

Example:
    python scripts/03_quantize.py --exp E3 --variant w4 --seed 0
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
from medcompress.compression.quantization.ptq import run_quantization  # noqa: E402

SCRIPT = "scripts/03_quantize.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None, require_budget=True)
    config = resolve_config(args)
    return run_stage("03_quantize", run_quantization, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
