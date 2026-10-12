"""Smoke-test measurements for run card RC-M1 (and the feasibility run RC-S0).

Everything here uses synthetic images and generic probe questions, so no dataset is
needed to find out whether a model loads, scores yes/no probes in one forward pass,
stays finite in fp16 and gets smaller in measured memory when a layer is removed.
GPU numbers are recorded only when CUDA is available.
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from medcompress.data.probes import format_probe_question

if TYPE_CHECKING:
    from medcompress.models.base import VLMWrapper

SMOKE_EXPERIMENT = "RC-M1"
DEFAULT_TEMPLATE = "Is there a {finding} in this image? Answer yes or no."
DEFAULT_FINDINGS = ("polyp", "ulcer", "bleeding", "instrument")

Row = dict[str, Any]


def synthetic_images(n: int, *, size: tuple[int, int] = (448, 336), seed: int = 0) -> list[Any]:
    """Return ``n`` reproducible RGB images (smooth gradient plus noise); not medical data."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    width, height = size
    xs = np.linspace(0, 1, width)[None, :, None]
    ys = np.linspace(0, 1, height)[:, None, None]
    images = []
    for i in range(n):
        base = 0.5 * xs * np.array([1.0, 0.4 + 0.1 * i, 0.2]) + 0.5 * ys * np.array([0.3, 0.6, 1.0])
        noise = 0.15 * rng.random((height, width, 3))
        array = np.clip(base + noise, 0, 1)
        images.append(Image.fromarray((array * 255).astype("uint8")))
    return images


def probe_questions(
    findings: Sequence[str] = DEFAULT_FINDINGS, template: str = DEFAULT_TEMPLATE
) -> list[str]:
    """Fill the probe template once per finding."""
    return [format_probe_question(template, f) for f in findings]


# ------------------------------------------------------------------ GPU helpers


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def cuda_available() -> bool:
    """True when torch is installed and sees a CUDA device."""
    torch = _torch()
    return bool(torch is not None and torch.cuda.is_available())


def synchronize() -> None:
    """Wait for queued GPU work so timings and memory readings are complete."""
    torch = _torch()
    if cuda_available():
        torch.cuda.synchronize()


def reset_peak() -> None:
    """Reset the peak-memory counters of all visible GPUs."""
    torch = _torch()
    if cuda_available():
        synchronize()
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)


def peak_gpu_bytes() -> int | None:
    """Peak allocated bytes summed over all visible GPUs since the last reset."""
    torch = _torch()
    if not cuda_available():
        return None
    synchronize()
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))


def allocated_gpu_bytes() -> int | None:
    """Currently allocated bytes summed over all visible GPUs, after freeing caches."""
    torch = _torch()
    if not cuda_available():
        return None
    gc.collect()
    torch.cuda.empty_cache()
    synchronize()
    return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))


def _row(metric: str, value: float | int | bool) -> Row:
    return {
        "experiment": SMOKE_EXPERIMENT,
        "budget": "",
        "seed": 0,
        "metric": metric,
        "value": float(value),
    }


# -------------------------------------------------------------------- measures


def measure_model(
    model: VLMWrapper,
    images: Sequence[Any],
    questions: Sequence[str],
    prefix: str,
    *,
    generate_tokens: int = 16,
    repeats: int = 2,
) -> tuple[list[Row], list[str]]:
    """Score the probes, time scoring and generation, and check the values are usable.

    Returns the result rows and the generated texts (print them in the run report;
    they are not written to the CSV).
    """
    if len(images) != len(questions):
        raise ValueError("need one image per question")
    rows: list[Row] = [_row(f"{prefix}_params", model.num_parameters())]
    allocated = allocated_gpu_bytes()
    if allocated is not None:
        rows.append(_row(f"{prefix}_allocated_gpu_bytes_after_load", allocated))
    reset_peak()
    model.score_yes_no(images[:1], questions[:1])  # warm-up (kernels, caches)
    synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        scores = model.score_yes_no(images, questions)
    synchronize()
    seconds = (time.perf_counter() - start) / (repeats * len(questions))
    finite = all(math.isfinite(s.logp_yes) and math.isfinite(s.logp_no) for s in scores)
    mass = [math.exp(s.logp_yes) + math.exp(s.logp_no) for s in scores]
    rows += [
        _row(f"{prefix}_scoring_seconds_per_probe", seconds),
        _row(f"{prefix}_scores_finite", finite),
        _row(f"{prefix}_yes_no_probability_mass", sum(mass) / len(mass)),
        _row(
            f"{prefix}_predicted_yes_fraction",
            sum(s.predicted == "yes" for s in scores) / len(scores),
        ),
    ]
    peak = peak_gpu_bytes()
    if peak is not None:
        rows.append(_row(f"{prefix}_peak_gpu_bytes_scoring", peak))
    reset_peak()
    start = time.perf_counter()
    outputs = model.generate(images[:1], questions[:1], max_new_tokens=generate_tokens)
    synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = max(1, len(outputs[0].token_ids))
    rows.append(_row(f"{prefix}_generate_tokens_per_second", new_tokens / elapsed))
    peak = peak_gpu_bytes()
    if peak is not None:
        rows.append(_row(f"{prefix}_peak_gpu_bytes_generate", peak))
    return rows, [o.text for o in outputs]


def measure_layer_removal(
    model: VLMWrapper,
    layer_index: int,
    images: Sequence[Any],
    questions: Sequence[str],
    prefix: str,
    *,
    bytes_per_parameter: int = 2,
) -> list[Row]:
    """Remove one decoder layer physically and record the parameter and memory drop.

    The memory drop is compared with the removed layer's parameter bytes; a masked
    (not removed) layer would show no drop, which is what this check guards against.
    """
    from medcompress.compression.pruning.apply import remove_units, verify_physical_removal

    layers = [u for u in model.list_prunable_units() if u.unit_type.value == "decoder_layer"]
    unit = next((u for u in layers if u.layer_index == layer_index), None)
    if unit is None:
        raise ValueError(f"no decoder layer {layer_index}")
    before_scores = model.score_yes_no(images, questions)
    before_params = model.num_parameters()
    before_bytes = allocated_gpu_bytes()
    remove_units(model, [unit])
    report = verify_physical_removal(before_params, model, [unit])
    after_bytes = allocated_gpu_bytes()
    after_scores = model.score_yes_no(images, questions)
    delta = max(
        abs(a.logp_yes - b.logp_yes) for a, b in zip(before_scores, after_scores, strict=True)
    )
    rows = [
        _row(f"{prefix}_removed_layer_parameters", unit.num_params),
        _row(f"{prefix}_parameter_drop_matches", report["ok"]),
        _row(f"{prefix}_scores_change_after_removal", delta > 0),
        _row(
            f"{prefix}_scores_finite_after_removal",
            all(math.isfinite(s.logp_yes) for s in after_scores),
        ),
    ]
    if before_bytes is not None and after_bytes is not None:
        expected = unit.num_params * bytes_per_parameter
        rows += [
            _row(f"{prefix}_removal_expected_drop_bytes", expected),
            _row(f"{prefix}_removal_measured_drop_bytes", before_bytes - after_bytes),
        ]
    return rows
