"""Measurements for the choice of the image side (run card RC-S0b), on synthetic images.

The wrapper only **shrinks** an image whose longest side exceeds ``inputs.image_longest_side``
(``models/hf.py``), so the 448 x 336 synthetic images of RC-M1 and RC-S0 look the same at every
larger side. RC-S0b therefore draws the same synthetic images (same generator, same seed) at a
size above every measured side, so that each side scales them down: at 448 they become
448 x 336, the RC-S0 size. Per side it records, in rows tagged ``RC-S0b:side<N>``:

* the scoring measurements of :func:`medcompress.models.smoke.measure_model` (seconds per
  probe, peak memory, generation speed) and the visual tokens per image;
* for the student, one LoRA optimizer step at each micro-batch (D06 trains with 4), each on a
  freshly loaded model. An out-of-memory error is a result: it is caught and written as
  ``<prefix>_lora_step_ok_mb<b>`` = 0, and the run goes on with the next measurement.

Nothing here downloads data; the models come from the hub cache of the session.
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

PATCH_PIXELS = 28
"""Pixels per side of one visual token of Qwen2.5-VL (14-pixel patches merged 2 x 2)."""
DEFAULT_SIDES: tuple[int, ...] = (448, 560, 672)
DEFAULT_MICRO_BATCHES: tuple[int, ...] = (1, 2, 4)
"""1 and 4 (the D06 micro-batch) are asked for; 2 is the fallback D06 amendment would use."""
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

Row = dict[str, Any]


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


def free_memory() -> None:
    """Drop Python references and empty the CUDA cache (after a model or an out-of-memory)."""
    gc.collect()
    from medcompress.models.smoke import cuda_available

    if cuda_available():
        import torch

        torch.cuda.empty_cache()


def lora_step_rows(
    build: Callable[[], Any],
    examples: Sequence[Any],
    micro_batch: int,
    prefix: str,
    *,
    lora: Mapping[str, Any],
    learning_rate: float,
    train: Callable[..., Any] | None = None,
    seed: int = 0,
) -> list[Row]:
    """One LoRA optimizer step at micro-batch ``micro_batch`` on a freshly built model.

    ``build()`` loads the model; ``train`` defaults to ``offline_kd.train_lora``. Any exception
    (CUDA out of memory included) is recorded as ``<prefix>_lora_step_ok_mb<b>`` = 0 and the
    message is printed; the model is released either way. When CUDA is present the peak memory
    of the step is written too (for a failed step, the peak reached before the failure).
    """
    from medcompress.models.smoke import cuda_available, peak_gpu_bytes, reset_peak

    if micro_batch < 1 or micro_batch > len(examples):
        raise ValueError(f"micro-batch {micro_batch} needs 1..{len(examples)} examples")
    if train is None:
        from medcompress.compression.distillation.offline_kd import train_lora as train
    config = _train_config(lora, learning_rate, micro_batch)
    tag = f"mb{micro_batch}"
    model = build()
    reset_peak()
    ok, seconds = 0.0, math.nan
    try:
        start = time.perf_counter()
        result = train(model, list(examples[:micro_batch]), config, seed=seed)
        seconds = time.perf_counter() - start
        losses = list(getattr(result, "losses", []) or [])
        ok = float(bool(losses) and math.isfinite(float(losses[0])))
    except Exception as exc:  # noqa: BLE001 - out of memory is the finding, not a crash
        print(
            f"{prefix} LoRA step at micro-batch {micro_batch} failed: {type(exc).__name__}: {exc}"
        )
    rows = [_row(f"{prefix}_lora_step_ok_{tag}", ok)]
    if ok == 1.0:
        rows.append(_row(f"{prefix}_lora_step_seconds_{tag}", seconds))
    peak = peak_gpu_bytes() if cuda_available() else None
    if peak is not None:
        rows.append(_row(f"{prefix}_peak_gpu_bytes_lora_step_{tag}", peak))
    del model
    free_memory()
    return rows


def _train_config(lora: Mapping[str, Any], learning_rate: float, micro_batch: int) -> Any:
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
    )


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
