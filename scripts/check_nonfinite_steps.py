#!/usr/bin/env python
"""Which optimizer steps of a recovery (or a training run) hold an image whose fp16 vision output
is non-finite (2026-11-16, CPU only, no model).

Rebuilds the run's batch order exactly as the training loop does (the fixed pool of
``data_subset`` examples drawn with the data seed, visited in the order of the run's seed, 16
examples per update) and marks every step that contains a training image listed by RC-CAL0's
student scan (``vision_scan_student.csv``). Prints those step indices, the sample ids in each (to
the console only: they are not committed), and the cumulative count at every 25th step, the
points where the training log prints its "non-finite steps" count, so the two can be compared.

Example (Kaggle, CPU; the shared data attached):
    python scripts/check_nonfinite_steps.py --exp E5 --variant magnitude --budget B2 --seed 0 \
        --scan vision_scan_student.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_context  # noqa: E402


def step_positions(n_examples: int, per_step: int, steps: int, order: list[int]) -> list[list[int]]:
    """The pool indices of every step's examples (the loop wraps around the pool)."""
    return [
        [order[((k - 1) * per_step + j) % n_examples] for j in range(per_step)]
        for k in range(1, steps + 1)
    ]


def flagged_steps(
    image_of: list[str], positions: list[list[int]], bad: set[str]
) -> dict[int, list[int]]:
    """``{step: [pool indices with a listed image]}`` for the steps that hold one (1-based)."""
    out: dict[int, list[int]] = {}
    for step, rows in enumerate(positions, 1):
        hits = [i for i in rows if image_of[i] in bad]
        if hits:
            out[step] = hits
    return out


def cumulative(steps: dict[int, list[int]], total: int, every: int = 25) -> list[tuple[int, int]]:
    """``(step, flagged steps so far)`` at every ``every``-th step and at the last one."""
    marks = sorted({*range(every, total + 1, every), total})
    return [(m, sum(1 for s in steps if s <= m)) for m in marks]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--scan", type=Path, required=True, help="vision_scan_student.csv (RC-CAL0)"
    )
    parser.add_argument("--section", choices=("recovery", "training"), default="recovery")
    args = parser.parse_args(argv)
    run = run_context(args, require_budget=False)
    assert run is not None
    config = resolve_config(args)

    from medcompress.compression.distillation.offline_kd import (
        data_seed,
        select_subset,
        training_pool,
    )
    from medcompress.compression.pruning.stage import load_train_samples

    section = dict(config[args.section])
    per_step = int(section["batch_size"]) * int(section["grad_accumulation"])
    steps = int(section["steps"])
    train = load_train_samples(config)
    pool = training_pool(train, section.get("data_subset"), data_seed(config))
    order = select_subset([s.sample_id for s in pool], None, run.seed)
    with args.scan.open(encoding="utf-8", newline="") as handle:
        listed = [r["sample_id"] for r in csv.DictReader(handle) if r["split"] == "train"]
    image_by_id = {s.sample_id: s.image_path for s in train}
    bad = {image_by_id[i] for i in listed if i in image_by_id}
    unknown = [i for i in listed if i not in image_by_id]
    image_of = [s.image_path for s in pool]
    rows_in_pool = sum(1 for p in image_of if p in bad)
    flagged = flagged_steps(image_of, step_positions(len(pool), per_step, steps, order), bad)
    print(f"{args.section} of {run.experiment_tag} seed {run.seed}: {len(pool)} pool examples, "
          f"{steps} steps of {per_step}; {len(bad)} listed training images ({len(unknown)} not "
          f"in the split), {rows_in_pool} pool rows show one of them")  # fmt: skip
    print(f"steps holding a listed image: {len(flagged)}: {sorted(flagged)}")
    for step, hits in sorted(flagged.items()):
        print(f"  step {step}: {[pool[i].sample_id for i in hits]}")
    print("cumulative, as the log prints it: "
          + ", ".join(f"step {m}: {n}" for m, n in cumulative(flagged, steps)))  # fmt: skip
    return 0


if __name__ == "__main__":
    main_guard(main)
