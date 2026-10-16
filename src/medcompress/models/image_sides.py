"""Measurements for the choice of the image side (run card RC-S0b), on synthetic images.

The wrapper only **shrinks** an image whose longest side exceeds ``inputs.image_longest_side``
(``models/hf.py``), so the 448 x 336 synthetic images of RC-M1 and RC-S0 look the same at every
larger side. RC-S0b therefore draws the same synthetic images (same generator, same seed) at a
size above every measured side, so that each side scales them down: at 448 they become
448 x 336, the RC-S0 size.

Every measurement is one **step** at one side, and every (side, step) runs in a child process
of its own (``scripts/measure_image_sides.py child``), so the GPU memory of one step is fully
returned to the system before the next starts. The first RC-S0b run (2026-10-16) measured the
steps one after the other in one process and a second model copy stayed on the GPU, which made
every later step fail at about 15.4 GB. The steps:

* ``score``: the scoring measurements of :func:`medcompress.models.smoke.measure_model`
  (seconds per probe, peak memory, generation speed) and the visual tokens per image;
* ``lora_mb<b>``: two LoRA optimizer steps of the student at micro-batch ``b`` on a freshly
  loaded model, the first to warm up (it includes attaching the adapter), the second timed;
* ``lora_gc_mb<b>``: the same with gradient checkpointing (a memory fallback, measured only).

Any failure of a step, an out-of-memory error included, is a result: the step's ``..._ok`` row
is 0, the peak memory reached before the failure is written when it is known, and the run goes
on with the next step and side. Rows are tagged ``RC-S0b:side<N>``; when a step is run again,
its earlier rows for that side are replaced, so a failed rerun leaves no stale timing behind.

Nothing here downloads data; the models come from the hub cache of the session.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PATCH_PIXELS = 28
"""Pixels per side of one visual token of Qwen2.5-VL (14-pixel patches merged 2 x 2)."""
DEFAULT_SIDES: tuple[int, ...] = (448, 560, 672)
STUDENT_STEPS: tuple[str, ...] = (
    "score",
    "lora_mb1",
    "lora_mb2",
    "lora_mb4",
    "lora_gc_mb1",
    "lora_gc_mb2",
    "lora_gc_mb4",
)
"""Micro-batch 4 is the D06 setting; 2 and 1 are the fallbacks of the D06 amendment proposal;
the ``gc`` steps measure gradient checkpointing, which the teacher adapter was trained with."""
TEACHER_STEPS: tuple[str, ...] = ("score",)
DEFAULT_CHILD_TIMEOUT = 1800.0
"""Seconds a child may run before it is stopped and its step recorded as failed (an estimate
well above the RC-M1 teacher download and load, about 2 minutes)."""
EXPERIMENT_PREFIX = "RC-S0b"
SETTINGS_KEYWORDS: tuple[str, ...] = (
    "pixel",
    "image",
    "resolution",
    "resize",
    "patch",
    "longest",
    "shortest",
)
NOT_STATED = "not stated"
REVISION_NOT_PINNED = "revision not pinned"

Row = dict[str, Any]
Runner = Callable[..., Any]


# ------------------------------------------------------------------ sides and images


def check_sides(sides: Iterable[int]) -> tuple[int, ...]:
    """The sides, sorted; each must be a positive multiple of 28 (whole visual tokens)."""
    out = tuple(sorted({int(s) for s in sides}))
    bad = [s for s in out if s <= 0 or s % PATCH_PIXELS]
    if not out or bad:
        raise ValueError(f"image sides must be positive multiples of {PATCH_PIXELS}, got {bad}")
    return out


def source_size(sides: Sequence[int]) -> tuple[int, int]:
    """Size of the synthetic images: twice the largest side, 4:3 like RC-S0's 448 x 336.

    Every measured side then resizes them **down**; 1344 x 1008 gives exactly 448 x 336,
    560 x 420 and 672 x 504 for the default sides.
    """
    width = 2 * max(check_sides(sides))
    return width, width * 3 // 4


def side_tag(side: int) -> str:
    """``RC-S0b:side448``: the experiment label of the rows of one side."""
    return f"{EXPERIMENT_PREFIX}:side{int(side)}"


def tag_rows(rows: Iterable[Mapping[str, Any]], side: int) -> list[Row]:
    """Rows in the results schema, labelled with their side."""
    return [{**dict(r), "experiment": side_tag(side), "budget": "", "seed": 0} for r in rows]


def _row(metric: str, value: float) -> Row:
    return {"experiment": "", "budget": "", "seed": 0, "metric": metric, "value": float(value)}


# ------------------------------------------------------------------ steps and row names


@dataclass(frozen=True)
class Step:
    """One measurement: ``score``, ``lora_mb<b>`` or ``lora_gc_mb<b>``."""

    name: str
    kind: str
    micro_batch: int = 0
    gradient_checkpointing: bool = False

    @property
    def label(self) -> str:
        """``scoring``, ``lora_step`` or ``lora_gc_step``: the middle of the row names."""
        if self.kind == "score":
            return "scoring"
        return "lora_gc_step" if self.gradient_checkpointing else "lora_step"


_LORA_STEP = re.compile(r"lora(_gc)?_mb(\d+)")


def parse_step(name: str) -> Step:
    """The :class:`Step` of a step name; raises ``ValueError`` for an unknown name."""
    if name == "score":
        return Step("score", "score")
    match = _LORA_STEP.fullmatch(name)
    if match is None or int(match.group(2)) < 1:
        raise ValueError(f"unknown step {name!r}: use score, lora_mb<b> or lora_gc_mb<b>")
    return Step(name, "lora", int(match.group(2)), match.group(1) is not None)


def check_steps(names: Iterable[str], part: str) -> tuple[str, ...]:
    """The steps in the given order without repeats; the teacher has only ``score``."""
    out = tuple(dict.fromkeys(parse_step(n).name for n in names))
    if not out:
        raise ValueError("no steps given")
    if part == "teacher" and out != ("score",):
        raise ValueError("the teacher has only the step 'score'")
    return out


def ok_metric(prefix: str, step: Step) -> str:
    """``student_scoring_ok``, ``student_lora_step_ok_mb4``, ``student_lora_gc_step_ok_mb4``."""
    if step.kind == "score":
        return f"{prefix}_scoring_ok"
    return f"{prefix}_{step.label}_ok_mb{step.micro_batch}"


def peak_metric(prefix: str, step: Step) -> str:
    """The step's peak-memory row (for a failed step, the peak reached before the failure)."""
    if step.kind == "score":
        return f"{prefix}_peak_gpu_bytes_scoring"
    return f"{prefix}_peak_gpu_bytes_{step.label}_mb{step.micro_batch}"


def seconds_metric(prefix: str, step: Step, *, first: bool = False) -> str:
    """The timed (second) LoRA step; with ``first``, the warm-up step (adapter attach included)."""
    part = "first_seconds" if first else "seconds"
    return f"{prefix}_{step.label}_{part}_mb{step.micro_batch}"


_LORA_METRIC = re.compile(
    r"(?:peak_gpu_bytes_)?(lora(?:_gc)?)_step_(?:ok_|seconds_|first_seconds_)?mb(\d+)"
)


def step_of_metric(prefix: str, metric: str) -> str | None:
    """The step that writes ``metric`` (``None`` if the row is not of this prefix)."""
    if not metric.startswith(f"{prefix}_"):
        return None
    match = _LORA_METRIC.fullmatch(metric[len(prefix) + 1 :])
    return f"{match.group(1)}_mb{match.group(2)}" if match else "score"


def failure_rows(prefix: str, step: Step, peak: float | None) -> list[Row]:
    """The rows of a failed step: ``..._ok`` = 0 and the peak if it is known."""
    rows = [_row(ok_metric(prefix, step), 0.0)]
    if peak is not None:
        rows.append(_row(peak_metric(prefix, step), peak))
    return rows


def step_ok(rows: Sequence[Mapping[str, Any]], prefix: str, step: Step) -> bool:
    """True if the rows hold the step's ``..._ok`` row with the value 1."""
    name = ok_metric(prefix, step)
    return any(r["metric"] == name and float(r["value"]) == 1.0 for r in rows)


# ------------------------------------------------------------------ memory


def safe_peak() -> float | None:
    """The peak GPU bytes since the last reset, or ``None`` without CUDA or if it cannot be read."""
    from medcompress.models.smoke import cuda_available, peak_gpu_bytes

    try:
        if not cuda_available():
            return None
        peak = peak_gpu_bytes()
    except Exception:  # noqa: BLE001 - a broken CUDA state must not hide the step's result
        return None
    return None if peak is None else float(peak)


def free_memory() -> None:
    """Drop Python references and empty the CUDA cache (after a model or an out-of-memory)."""
    gc.collect()
    from medcompress.models.smoke import cuda_available

    if cuda_available():
        import torch

        torch.cuda.empty_cache()


# ------------------------------------------------------------------ the LoRA step


def lora_step_rows(
    build: Callable[[], Any],
    examples: Sequence[Any],
    micro_batch: int,
    prefix: str,
    *,
    lora: Mapping[str, Any],
    learning_rate: float,
    gradient_checkpointing: bool = False,
    train: Callable[..., Any] | None = None,
    seed: int = 0,
) -> list[Row]:
    """Two LoRA optimizer steps at micro-batch ``micro_batch`` on a freshly built model.

    ``build()`` loads the model; ``train`` defaults to ``offline_kd.train_lora``. The first step
    warms up and is written as ``..._first_seconds_mb<b>``: the wall time of the whole call,
    which includes attaching the adapter, creating the optimizer and the first CUDA kernels.
    The second, on the same model and adapter, is the timed step ``..._seconds_mb<b>``: the
    optimizer step alone (``TrainResult.seconds``), measured as RC-S0's ``r0_lora_step_seconds``
    on a model that had scored first (0.74 s for the 3B at side 448). Any exception (loading
    and CUDA out of memory included) is recorded as ``..._ok_mb<b>`` = 0 and the message is
    printed. When CUDA is present the peak memory over both steps is written too (for a failed
    step, the peak reached before the failure).
    """
    from medcompress.models.smoke import reset_peak, synchronize

    if micro_batch < 1 or micro_batch > len(examples):
        raise ValueError(f"micro-batch {micro_batch} needs 1..{len(examples)} examples")
    if train is None:
        from medcompress.compression.distillation.offline_kd import train_lora as train
    name = f"lora_gc_mb{micro_batch}" if gradient_checkpointing else f"lora_mb{micro_batch}"
    step = parse_step(name)
    config = _train_config(lora, learning_rate, micro_batch, gradient_checkpointing)
    batch = list(examples[:micro_batch])
    model: Any = None
    times: list[float] = []
    peaks: list[float] = []
    ok = 0.0
    try:
        model = build()
        losses: list[float] = []
        for _ in range(2):  # warm-up step, then the timed step
            reset_peak()
            start = time.perf_counter()
            result = train(model, batch, config, seed=seed)
            synchronize()
            wall = time.perf_counter() - start
            loop = getattr(result, "seconds", None)  # the optimizer step alone, as in RC-S0
            times.append(wall if not times or not isinstance(loop, int | float) else float(loop))
            losses += list(getattr(result, "losses", []) or [])[:1]
            peak = safe_peak()
            if peak is not None:
                peaks.append(peak)
        ok = float(len(losses) == 2 and all(math.isfinite(float(v)) for v in losses))
    except Exception as exc:  # noqa: BLE001 - out of memory is the finding, not a crash
        print(f"{prefix} {name} failed: {type(exc).__name__}: {exc}")
        peak = safe_peak()
        if peak is not None:
            peaks.append(peak)
    rows = [_row(ok_metric(prefix, step), ok)]
    if ok == 1.0:
        rows.append(_row(seconds_metric(prefix, step, first=True), times[0]))
        rows.append(_row(seconds_metric(prefix, step), times[1]))
    if peaks:
        rows.append(_row(peak_metric(prefix, step), max(peaks)))
    del model
    free_memory()
    return rows


def _train_config(
    lora: Mapping[str, Any], learning_rate: float, micro_batch: int, gradient_checkpointing: bool
) -> Any:
    """One optimizer step of the D06 recipe (answers only) at the given micro-batch."""
    from medcompress.compression.distillation.offline_kd import TrainConfig
    from medcompress.compression.lora import LoRAConfig

    return TrainConfig(
        mode="sft",
        lora=LoRAConfig(
            int(lora["rank"]),
            float(lora["alpha"]),
            float(lora["dropout"]),
            tuple(lora["target_modules"]),
        ),
        steps=1,
        learning_rate=float(learning_rate),
        batch_size=int(micro_batch),
        grad_accumulation=1,
        kd_weight=0.0,
        kd_temperature=1.0,
        data_subset="all",
        extra={"gradient_checkpointing": True} if gradient_checkpointing else {},
    )


# ------------------------------------------------------------------ children and the CSV


def write_child_result(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """What a child hands back to the parent: its rows, as JSON (written atomically)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"rows": [dict(r) for r in rows]}), encoding="utf-8")
    tmp.replace(path)


def read_child_result(path: Path) -> list[Row] | None:
    """The rows a child wrote, or ``None`` if it wrote nothing readable."""
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return None
    return rows


def run_child(
    command: Sequence[str],
    result_file: Path,
    prefix: str,
    step: Step,
    *,
    timeout: float | None = DEFAULT_CHILD_TIMEOUT,
    run: Runner = subprocess.run,
) -> list[Row]:
    """Run one step in a child process and return its rows.

    A child that ends without its rows (killed, crashed, timed out) or without the step's
    ``..._ok`` row is recorded as a failed step with no peak (the parent cannot read it).
    """
    result_file = Path(result_file)
    result_file.unlink(missing_ok=True)
    code: int | None = None
    try:
        code = run(list(command), timeout=timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        print(f"{prefix} {step.name}: child stopped after {timeout} s")
    rows = read_child_result(result_file)
    if rows is None or not any(r.get("metric") == ok_metric(prefix, step) for r in rows):
        print(f"{prefix} {step.name}: child ended (exit code {code}) without its rows; failed")
        return failure_rows(prefix, step, None)
    if code not in (0, None):
        print(f"{prefix} {step.name}: child wrote its rows, then exited with code {code}")
    return rows


def write_step_rows(
    rows: Sequence[Mapping[str, Any]], target: Path, side: int, step: Step, prefix: str
) -> None:
    """Write one step's rows for one side, replacing that step's earlier rows for the side."""
    from medcompress.utils.io import append_results

    target = Path(target)
    tag = side_tag(side)
    kept: list[dict[str, Any]] = []
    if target.exists():
        with target.open("r", encoding="utf-8", newline="") as handle:
            for old in csv.DictReader(handle):
                same = old["experiment"] == tag and (
                    step_of_metric(prefix, old["metric"]) == step.name
                )
                if not same:
                    kept.append(old)
        target.unlink()
    append_results(kept + tag_rows(rows, side), target)


def measure_sides(
    prefix: str,
    sides: Sequence[int],
    steps: Sequence[str],
    target: Path,
    command_for: Callable[[int, Step, Path], Sequence[str]],
    workdir: Path,
    *,
    timeout: float | None = DEFAULT_CHILD_TIMEOUT,
    run: Runner = subprocess.run,
) -> list[tuple[int, str, bool]]:
    """Run every (side, step) as its own child; write its rows as soon as it is done.

    ``command_for(side, step, result_file)`` builds the child's command line. Returns
    (side, step, ok) per child, in the order run.
    """
    outcomes: list[tuple[int, str, bool]] = []
    for side in check_sides(sides):
        for name in steps:
            step = parse_step(name)
            result_file = Path(workdir) / f"{prefix}_side{side}_{step.name}.json"
            print(f"== {prefix} side {side} step {step.name}", flush=True)
            sys.stderr.flush()
            start = time.perf_counter()
            rows = run_child(
                command_for(side, step, result_file), result_file, prefix, step,
                timeout=timeout, run=run,
            )  # fmt: skip
            write_step_rows(rows, target, side, step, prefix)
            ok = step_ok(rows, prefix, step)
            outcomes.append((side, step.name, ok))
            elapsed = time.perf_counter() - start
            print(
                f"{prefix} side {side} {step.name}: ok={int(ok)}, {len(rows)} rows, {elapsed:.0f} s"
            )
    return outcomes


# ------------------------------------------------------------------ the settings printout


def matching_lines(text: str, keywords: Sequence[str] = SETTINGS_KEYWORDS) -> list[str]:
    """The lines of ``text`` that mention an image setting (case-insensitive)."""
    return [
        line.rstrip()
        for line in text.splitlines()
        if any(word in line.lower() for word in keywords)
    ]


def report_lines(name: str, lines: Sequence[str]) -> list[str]:
    """What the card's settings step prints for one source: the lines, or "not stated"."""
    return [f"{name}: {line}" for line in lines] or [f"{name}: {NOT_STATED}"]
