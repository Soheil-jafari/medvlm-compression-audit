#!/usr/bin/env python
"""Which examples a training run's skipped updates held (CPU, report only; writes nothing).

A step of 625 is 16 micro-batches of one example. The training visits the pool in a seeded
order (``select_subset`` with the run's seed) and keeps its place across resumes, so step ``k``
(1-based) holds the examples ``order[(k-1)*16 : k*16]``. For every skipped step of the run's
``train_history.json`` this prints those examples' features, and the same over all steps:

* ``differs``: the cached teacher answer is not the ground truth (stripped text compared; the
  training compares token ids, so this is close but not exact). Only these micro-batches run
  the second forward and backward of the split step;
* ``teacher_tokens``: the length of the cached teacher answer in tokens (exact: the KD batch);
* ``gt_words``: the length of the ground-truth answer in words (no tokenizer on CPU);
* which loss term was not finite at the step (``ce_losses`` / ``kd_losses`` of the history).

Which of the 16 micro-batches was the non-finite one is not recorded, so a step is compared
as a whole. Descriptive only; nothing is decided on it.

Example (Kaggle CPU notebook, e2-seed0-adapter, medcompress-data and e0-teacher-cache attached):
    python scripts/skipped_steps.py --exp E2 --seed 0 --data-root $DATA --from $CACHE \\
        --history "$(find $ADAPTER -name train_history.json | head -n 1)"
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_stage  # noqa: E402
from medcompress.config import ConfigError, RunContext  # noqa: E402


@dataclass(frozen=True)
class StepFeatures:
    """What the examples of one step look like."""

    step: int
    differs: int
    teacher_tokens_max: int
    teacher_tokens_mean: float
    gt_words_max: int


def step_indices(order: Sequence[int], step: int, per_micro: int, accumulation: int) -> list[int]:
    """Indices of the examples of ``step`` (1-based), as the training loop visits them."""
    per_step = per_micro * accumulation
    start = (step - 1) * per_step
    return [order[(start + i) % len(order)] for i in range(per_step)]


def features(step: int, examples: Sequence[tuple[bool, int, int]]) -> StepFeatures:
    """``examples``: (differs, teacher tokens, ground-truth words) of the step's examples."""
    tokens = [t for _, t, _ in examples]
    return StepFeatures(
        step=step,
        differs=sum(d for d, _, _ in examples),
        teacher_tokens_max=max(tokens),
        teacher_tokens_mean=statistics.fmean(tokens),
        gt_words_max=max(w for _, _, w in examples),
    )


def _not_finite(values: Sequence[float | None], step: int) -> bool:
    if step - 1 >= len(values):
        return False
    value = values[step - 1]
    return value is None or not math.isfinite(float(value))


def summary(rows: Sequence[StepFeatures]) -> str:
    """Means of the step features (one line)."""
    if not rows:
        return "no steps"
    return (
        f"differs {statistics.fmean(r.differs for r in rows):.2f}/16; teacher tokens max "
        f"{statistics.fmean(r.teacher_tokens_max for r in rows):.1f}, mean "
        f"{statistics.fmean(r.teacher_tokens_mean for r in rows):.1f}; ground-truth words max "
        f"{statistics.fmean(r.gt_words_max for r in rows):.1f}"
    )


def mid_rank(value: float, population: Sequence[float]) -> float:
    """Share of ``population`` below ``value``, ties counted half (0.5 on average by chance)."""
    below = sum(v < value for v in population)
    equal = sum(v == value for v in population)
    return (below + 0.5 * equal) / len(population)


def report(config: dict, run: RunContext | None, history: Path) -> None:
    from medcompress.compression.distillation.offline_kd import (
        TrainConfig,
        data_seed,
        examples_from_samples,
        select_subset,
        training_pool,
    )
    from medcompress.config import load_config
    from medcompress.data.prepared import load_prepared
    from medcompress.models.teacher_cache import load_pinned_teacher_cache

    if run is None:
        raise ConfigError("--exp is required")
    if not Path(history).is_file():
        raise ConfigError(f"{history} is not there: attach the run's adapter hand-off")
    data = json.loads(Path(history).read_text(encoding="utf-8"))
    cfg = TrainConfig.from_config(config["training"])
    paths = config["paths"]
    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    train = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))["train"]
    records = {
        r.sample_id: r
        for r in load_pinned_teacher_cache(config, Path(paths["cache_root"]) / "teacher")
    }
    pool = training_pool(train, cfg.data_subset, data_seed(config))
    examples = examples_from_samples(pool, Path(paths["data_root"]), records=records)
    order = select_subset([e.sample_id for e in examples], None, run.seed)

    def describe(index: int) -> tuple[bool, int, int]:
        ex = examples[index]
        differs = ex.teacher.answer_text.strip() != ex.answer.strip()
        return differs, len(ex.teacher.answer_token_ids), len(ex.answer.split())

    every = [
        features(k, [describe(i) for i in step_indices(order, k, cfg.batch_size,
                                                        cfg.grad_accumulation)])
        for k in range(1, int(data["steps"]) + 1)
    ]  # fmt: skip
    skipped = [int(k) for k in data.get("skipped_steps", [])]
    rows = [every[k - 1] for k in skipped]
    ce, kd = data.get("ce_losses", []), data.get("kd_losses", [])
    share = statistics.fmean(describe(i)[0] for i in range(len(examples)))
    print(f"{len(examples)} examples; {share:.3f} with a teacher answer that differs; "
          f"{len(skipped)} skipped steps of {len(every)}")  # fmt: skip
    print("step  differs  teacher_tokens(max/mean)  gt_words_max  not finite")
    for r in rows:
        terms = [n for n, v in (("ce", ce), ("kd", kd)) if _not_finite(v, r.step)]
        print(f"{r.step:4d}  {r.differs:2d}/16    {r.teacher_tokens_max:4d} / "
              f"{r.teacher_tokens_mean:5.1f}            {r.gt_words_max:4d}          "
              f"{'+'.join(terms) or '-'}")  # fmt: skip
    print(f"skipped steps: {summary(rows)}")
    print(f"all steps:     {summary(every)}")
    if rows:
        tokens = [r.teacher_tokens_max for r in every]
        ranks = [mid_rank(r.teacher_tokens_max, tokens) for r in rows]
        diffs = [r.differs for r in every]
        diff_ranks = [mid_rank(r.differs, diffs) for r in rows]
        print(f"mid-rank among all steps, mean over the skipped steps: longest teacher answer "
              f"{statistics.fmean(ranks):.2f}, differs {statistics.fmean(diff_ranks):.2f} "
              "(about 0.5 = no clustering)")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--history", type=Path, required=True, help="the run's train_history.json")
    args = parser.parse_args(argv)
    if args.exp is None:
        parser.error("--exp is required")
    run = RunContext(args.exp, seed=args.seed, budget=args.budget, variant=args.variant)
    config = resolve_config(args)
    return run_stage("skipped_steps", report, config=config, run=run, history=args.history)


if __name__ == "__main__":
    main_guard(main)
