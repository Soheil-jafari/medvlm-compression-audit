#!/usr/bin/env python
"""Step 02: train the student (answer fine-tuning, offline KD or QAKD).

The mode comes from training.mode in the experiment config.
Used by: E1 (sft), E2 (offline_kd), E4 (qakd).

E2 starts only once rule 3's quality part has been read from the committed validation rows
(E0's macro balanced accuracy at least E1 seed 0's + 0.02, or the teacher's one extension
done): if it fails, the teacher and the cache change first.

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
from medcompress.config import RunContext  # noqa: E402
from medcompress.models.teacher_training import require_teacher_quality  # noqa: E402

SCRIPT = "scripts/02_train_student.py"


def train(config: dict, run: RunContext | None) -> None:
    """E2 waits for rule 3's quality check (E0 against E1 seed 0); E1 and E4 do not."""
    if run is not None and run.experiment_id == "E2":
        require_teacher_quality(config)
    run_student_training(config, run)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None, require_budget=True)
    config = resolve_config(args)
    return run_stage("02_train_student", train, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
