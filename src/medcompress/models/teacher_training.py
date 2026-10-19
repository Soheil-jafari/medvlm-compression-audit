"""Our own teacher adapter: feasibility (RC-TT0), rule 3 and the training plan (RC-TT1).

D01 amendment of 2026-10-18: the teacher is Qwen2.5-VL-7B-Instruct at its pinned revision with a
LoRA adapter trained on our training split only, at image side 672. RC-TT0 measures one LoRA
step of the 7B in two setups, each with and without gradient checkpointing:

* ``fp16_2gpu``: the fp16 base split over two T4s (``device_map="auto"``);
* ``nf4_1gpu``: QLoRA, the base in 4-bit NF4 on one T4.

**Rule 3** (pre-registered before any validation result) then fixes the setup and the size:

* setup (a) ``fp16_2gpu`` unless it runs out of memory or is more than ``slower_factor`` (2x)
  slower per example than (b) ``nf4_1gpu``; within a setup the faster variant that ran;
* ``N`` = the largest multiple of 1,000 training examples whose projected time, from the
  warm step time, is at most 15 GPU hours. RC-TT0 recorded both readings of a T4 x2 hour
  (counted twice, and once); measured on 2026-10-19, it counts **once** against the Kaggle
  quota, so RC-TT1 uses that reading (:func:`t4x2_counts_once`). The cap is one pass over the
  training split's QA rows, applied by the training run, which knows the split.

The LoRA targets are the published adapter's (``adapter_config.json``, D01 of 2026-10-19),
read on Kaggle and applied inside the language model's decoder layers
(:func:`published_target_names`, ``compression/lora.py: layer_regex``).

The examples are taken in one seeded order stratified by question type (:func:`teacher_order`),
so the N examples of any size, and the +50 % extension of the quality rule, are prefixes of
the same order.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT = "RC-TT0"
SETUPS: tuple[str, ...] = ("fp16_2gpu", "nf4_1gpu")
SETUP_GPUS = {"fp16_2gpu": 2, "nf4_1gpu": 1}
FEASIBILITY_STEPS: tuple[str, ...] = (
    "lora_config",
    "fp16_2gpu",
    "fp16_2gpu_gc",
    "nf4_1gpu",
    "nf4_1gpu_gc",
)
DEFAULT_CSV = Path("results/smoke/RC-TT0/teacher_training.csv")
NF4 = {"bits": 4, "quant_type": "nf4", "compute_dtype": "float16", "double_quant": True}
RESUME_DIR = "E0/teacher_resume"
ADAPTER_DIR = "E0/teacher_adapter"
EXTENSION_RESUME_DIR = "E0/teacher_resume_extension"


class RuleError(ValueError):
    """Rule 3 cannot be applied (no setup ran, or the result file lacks its rows)."""


@dataclass(frozen=True)
class SetupResult:
    """One measured setup: did one LoRA step run, and how long did the warm step take."""

    setup: str
    gradient_checkpointing: bool
    ok: bool
    seconds_per_example: float | None
    peak_gpu_bytes: float | None

    @property
    def name(self) -> str:
        return self.setup + ("_gc" if self.gradient_checkpointing else "")


@dataclass(frozen=True)
class TeacherPlan:
    """What rule 3 decided, read back from RC-TT0's result file by the training run."""

    setup: str
    gradient_checkpointing: bool
    seconds_per_example: float
    n_examples: int
    n_examples_if_t4x2_counts_once: int
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_from_published: bool

    @property
    def gpus(self) -> int:
        return SETUP_GPUS[self.setup]


# ------------------------------------------------------------------ rule 3


def setup_results(values: Mapping[str, float]) -> list[SetupResult]:
    """The four measured setups from RC-TT0's rows (``image_sides.lora_step_rows`` names)."""
    out = []
    for setup in SETUPS:
        for gc in (False, True):
            label = "lora_gc_step" if gc else "lora_step"
            ok = values.get(f"{setup}_{label}_ok_mb1") == 1.0
            seconds = values.get(f"{setup}_{label}_seconds_mb1")
            out.append(
                SetupResult(
                    setup, gc, ok, float(seconds) if ok and seconds is not None else None,
                    values.get(f"{setup}_peak_gpu_bytes_{label}_mb1"),
                )
            )  # fmt: skip
    return out


def choose_setup(results: Sequence[SetupResult], *, slower_factor: float = 2.0) -> SetupResult:
    """Rule 3's setup: (a) fp16 over 2 x T4 unless it failed or is > ``slower_factor`` slower."""

    def best(setup: str) -> SetupResult | None:
        ran = [r for r in results if r.setup == setup and r.ok and r.seconds_per_example]
        return min(
            ran, key=lambda r: (r.seconds_per_example, r.gradient_checkpointing), default=None
        )

    a, b = best("fp16_2gpu"), best("nf4_1gpu")
    if a is not None and (
        b is None or a.seconds_per_example <= slower_factor * b.seconds_per_example
    ):  # type: ignore[operator]
        return a
    if b is not None:
        return b
    raise RuleError("no setup ran one LoRA step (all out of memory or failed): see RC-TT0's log")


def rule_examples(
    seconds_per_example: float, gpus_counted: int, *, budget_gpu_hours: float, step: int
) -> int:
    """The largest multiple of ``step`` examples within ``budget_gpu_hours`` (0 if none fits)."""
    if seconds_per_example <= 0 or gpus_counted < 1:
        raise RuleError("the step time and the number of GPUs must be positive")
    fits = budget_gpu_hours * 3600 / (seconds_per_example * gpus_counted)
    return int(fits // step) * step


def rule_rows(
    values: Mapping[str, float], settings: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], SetupResult, int, int]:
    """Apply rule 3 to RC-TT0's rows; return the rule rows, the chosen setup and both N."""
    rule = settings["rule"]
    chosen = choose_setup(setup_results(values), slower_factor=float(rule["slower_factor"]))
    gpus = SETUP_GPUS[chosen.setup]
    budget, step = float(rule["budget_gpu_hours"]), int(rule["example_step"])
    counted = gpus if rule.get("t4x2_counts_twice", True) else 1
    seconds = float(chosen.seconds_per_example)  # type: ignore[arg-type]
    n = rule_examples(seconds, counted, budget_gpu_hours=budget, step=step)
    n_once = rule_examples(seconds, 1, budget_gpu_hours=budget, step=step)
    if n < step:
        raise RuleError(f"not even {step} examples fit {budget} GPU hours at {seconds:.2f} s each")
    wall = n * seconds / 3600
    rows = [
        ("rule_setup_fp16_2gpu", float(chosen.setup == "fp16_2gpu")),
        ("rule_gradient_checkpointing", float(chosen.gradient_checkpointing)),
        ("rule_seconds_per_example", seconds),
        ("rule_gpus", float(gpus)),
        ("rule_gpus_counted", float(counted)),
        ("rule_n_examples", float(n)),
        ("rule_n_examples_if_t4x2_counts_once", float(n_once)),
        ("rule_projected_wall_hours", wall),
        ("rule_projected_gpu_hours", wall * counted),
        ("rule_budget_gpu_hours", budget),
    ]
    return [_row(m, v) for m, v in rows], chosen, n, n_once


def _row(metric: str, value: float) -> dict[str, Any]:
    return {"experiment": EXPERIMENT, "budget": "", "seed": 0, "metric": metric, "value": value}


def read_values(path: Path) -> dict[str, float]:
    """``{metric: value}`` of RC-TT0's result file."""
    path = Path(path)
    if not path.is_file():
        raise RuleError(f"{path} is missing: run RC-TT0 first and commit its result file")
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["metric"]: float(r["value"]) for r in csv.DictReader(handle) if r["value"] != ""}


def read_plan(path: Path, *, t4x2_counts_once: bool = False) -> TeacherPlan:
    """The teacher plan from RC-TT0's result file: rule 3's rows and the LoRA settings."""
    values = read_values(path)
    needed = (
        "rule_setup_fp16_2gpu", "rule_gradient_checkpointing", "rule_seconds_per_example",
        "rule_n_examples", "rule_n_examples_if_t4x2_counts_once", "lora_rank", "lora_alpha",
        "lora_dropout", "lora_from_published",
    )  # fmt: skip
    missing = [k for k in needed if k not in values]
    if missing:
        raise RuleError(f"{path} lacks {missing}: RC-TT0 did not finish its rule step")
    key = "rule_n_examples_if_t4x2_counts_once" if t4x2_counts_once else "rule_n_examples"
    return TeacherPlan(
        setup="fp16_2gpu" if values["rule_setup_fp16_2gpu"] == 1.0 else "nf4_1gpu",
        gradient_checkpointing=values["rule_gradient_checkpointing"] == 1.0,
        seconds_per_example=values["rule_seconds_per_example"],
        n_examples=int(values[key]),
        n_examples_if_t4x2_counts_once=int(values["rule_n_examples_if_t4x2_counts_once"]),
        lora_rank=int(values["lora_rank"]),
        lora_alpha=float(values["lora_alpha"]),
        lora_dropout=float(values["lora_dropout"]),
        lora_from_published=values["lora_from_published"] == 1.0,
    )


# ------------------------------------------------------------------ the published adapter config


LM_TARGETS = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def published_target_names(spec: Any) -> tuple[str, ...]:
    """The leaf module names in the published ``target_modules`` of ``adapter_config.json``.

    - ``"all-linear"`` -> ``("all-linear",)``;
    - a list -> the last path component of each entry (``model.layers.0.self_attn.q_proj`` ->
      ``q_proj``), sorted and unique;
    - a regular expression (as ms-swift writes one, for example
      ``^(model)(?!...).*\\.(down_proj|...|v_proj)$``) -> the names of its last group of plain
      alternatives.

    Raises :class:`RuleError` if no names can be read: a person then decides (D01).
    """
    if isinstance(spec, str) and spec.strip() == "all-linear":
        return ("all-linear",)
    if isinstance(spec, str):
        for group in reversed(re.findall(r"\(([^()]*)\)", spec)):
            names = group.removeprefix("?:").split("|")
            if names and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in names):
                return tuple(sorted(set(names)))
        raise RuleError(f"no module names could be read from target_modules {spec!r}")
    if isinstance(spec, list | tuple) and spec:
        return tuple(sorted({str(n).rsplit(".", 1)[-1] for n in spec}))
    raise RuleError(f"target_modules {spec!r} names no module")


def adapted_counts(linear_names: Iterable[str], regex: str, names: Sequence[str]) -> dict[str, int]:
    """How many linear layers of the model each target name adapts (0: the name matches none).

    ``linear_names`` are the full names of the model's linear layers, ``regex`` the one passed
    to peft (``compression/lora.py: lora_target_regex``).
    """
    pattern = re.compile(regex)
    counts = {name: 0 for name in names}
    for full in linear_names:
        if pattern.fullmatch(full):
            leaf = full.rsplit(".", 1)[-1]
            counts[leaf] = counts.get(leaf, 0) + 1
    return counts


def t4x2_counts_once(settings: Mapping[str, Any]) -> bool:
    """Whether a T4 x2 hour counts once against the Kaggle quota (``rule.t4x2_counts_twice``)."""
    return not bool(settings["rule"].get("t4x2_counts_twice", True))


def lora_settings(
    published: Mapping[str, Any] | None, default: Mapping[str, Any]
) -> dict[str, Any]:
    """Rank, alpha and dropout: the published adapter's if it can be read, else D06's.

    ``published`` is the published ``adapter_config.json`` (``None`` if it could not be read).
    ``targets_match`` says whether its target modules are the seven projections of the language
    model (``all-linear``, a list or a regular expression over them), D06's targets
    (``decoder_attention`` + ``decoder_mlp``, the vision tower frozen).
    """
    if not published:
        return {
            "rank": int(default["rank"]),
            "alpha": float(default["alpha"]),
            "dropout": float(default["dropout"]),
            "from_published": 0.0,
            "targets_match": -1.0,  # not known
        }
    try:
        names = published_target_names(published.get("target_modules"))
    except RuleError:
        names = ()
    match = names == ("all-linear",) or set(names) == LM_TARGETS
    return {
        "rank": int(published.get("r", default["rank"])),
        "alpha": float(published.get("lora_alpha", default["alpha"])),
        "dropout": float(published.get("lora_dropout", default["dropout"])),
        "from_published": 1.0,
        "targets_match": float(bool(match)),
    }


# ------------------------------------------------------------------ the examples


def question_type(sample: Any) -> str:
    """The stratum of a training example: its question classes, joined (``none`` if empty)."""
    classes = (getattr(sample, "metadata", None) or {}).get("question_class") or ()
    return "|".join(sorted(str(c) for c in classes)) or "none"


def teacher_order(train: Sequence[Any], seed: int) -> list[Any]:
    """All training examples in one seeded order, stratified by question type.

    Within each type the examples are shuffled with ``seed``. The k-th example of a type with
    n examples gets the key k / n, and the order is by key (types in a seeded order on ties).
    So the first example of every type comes first, rare types included, and any prefix of
    the order holds the types about in proportion. ``N`` examples are the first N; the +50 %
    extension of the quality rule is the next N / 2.
    """
    by_type: dict[str, list[Any]] = {}
    for sample in sorted(train, key=lambda s: s.sample_id):
        by_type.setdefault(question_type(sample), []).append(sample)
    rng = random.Random(seed)
    types = sorted(by_type)
    rng.shuffle(types)
    rank = {t: i for i, t in enumerate(types)}
    keyed = []
    for name in types:
        members = by_type[name]
        rng.shuffle(members)
        n = len(members)
        keyed += [(k / n, rank[name], k, s) for k, s in enumerate(members)]
    keyed.sort(key=lambda item: item[:3])
    return [item[3] for item in keyed]


def training_size(plan: TeacherPlan, train_rows: int) -> int:
    """N of rule 3, capped at one pass over the training split's QA rows."""
    return min(plan.n_examples, int(train_rows))


def extension_size(n: int, fraction: float = 0.5) -> int:
    """The examples of the one extension the quality rule allows: +50 % of N (rounded up)."""
    return int(math.ceil(n * fraction))


def checkpoint_every(seconds_per_example: float, per_update: int, minutes: float) -> int:
    """Optimizer steps between two saves so a save comes about every ``minutes`` (an estimate)."""
    return max(1, int(round(minutes * 60 / (seconds_per_example * per_update))))


def train_settings(plan: TeacherPlan, settings: Mapping[str, Any], n: int) -> dict[str, Any]:
    """The ``TrainConfig`` fields of the teacher run (answers only, no distillation)."""
    per_update = int(settings["batch_size"]) * int(settings["grad_accumulation"])
    steps = int(math.ceil(n / per_update))
    return {
        "mode": "sft",
        "lora": {
            "rank": plan.lora_rank, "alpha": plan.lora_alpha, "dropout": plan.lora_dropout,
            "target_modules": (
                settings["target_modules"]
                if isinstance(settings["target_modules"], str)  # "published"
                else list(settings["target_modules"])
            ),
        },
        "steps": steps,
        "learning_rate": float(settings["learning_rate"]),
        "batch_size": int(settings["batch_size"]),
        "grad_accumulation": int(settings["grad_accumulation"]),
        "kd_weight": 0.0,
        "kd_temperature": 1.0,
        "data_subset": n,
        "warmup_steps": int(math.ceil(float(settings["warmup_fraction"]) * steps)),
        "lr_schedule": str(settings["lr_schedule"]),
        "weight_decay": float(settings["weight_decay"]),
        "max_grad_norm": float(settings["max_grad_norm"]),
        "gradient_checkpointing": plan.gradient_checkpointing,
        "checkpoint_every": checkpoint_every(
            plan.seconds_per_example, per_update, float(settings["checkpoint_minutes"])
        ),
    }  # fmt: skip


# ------------------------------------------------------------------ resume


def find_resume_states(roots: Iterable[Path], folder_name: str = "teacher_resume") -> list[Path]:
    """Resume folders (``.../<folder_name>`` holding ``current/progress.json``) under ``roots``.

    Sorted by the step they reached, the furthest first, so a re-run continues from the
    latest checkpoint, also one in a previous notebook version's output attached as input.
    """
    found = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for progress in root.rglob(f"{folder_name}/*/progress.json"):
            if progress.parent.name not in ("current", "old"):
                continue
            try:
                step = int(json.loads(progress.read_text(encoding="utf-8")).get("step", 0))
            except (OSError, ValueError):
                continue
            found.append((step, progress.parent.parent))
    found.sort(key=lambda item: (-item[0], str(item[1])))
    unique: list[Path] = []
    for _, folder in found:
        if folder not in unique:
            unique.append(folder)
    return unique


def saved_step(folder: Path) -> int:
    """The step a resume folder reached (0 if it holds none)."""
    for name in ("current", "old"):
        progress = Path(folder) / name / "progress.json"
        if progress.is_file():
            return int(json.loads(progress.read_text(encoding="utf-8")).get("step", 0))
    return 0
