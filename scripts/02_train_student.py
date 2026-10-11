#!/usr/bin/env python
"""Step 02: train the student (answer fine-tuning, offline KD or QAKD).

The mode comes from training.mode in the experiment config.
Used by: E1 (sft), E2 (offline_kd), E4 (qakd).

Example:
    python scripts/02_train_student.py --exp E2 --seed 0
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
from medcompress.compression.distillation.offline_kd import run_student_training  # noqa: E402

SCRIPT = "scripts/02_train_student.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None, require_budget=True)
    config = resolve_config(args)
    return run_stage("02_train_student", run_student_training, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
