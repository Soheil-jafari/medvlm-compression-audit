"""Smoke test of the quantization toolchain on a GPU (run card RC-Q0).

Answers, on the platform the arms will run on, before any real quantization run:

1. which GPTQModel kernel classes build a module from our codes and reproduce them exactly
   (``kernels.build_gptq_linear`` checks every module numerically) and how fast each is, for
   4 and 8 bits;
2. whether the student can be quantized end to end with GPTQ (``ptq.gptq_quantize_decoder``)
   and LLM.int8, passes the kernel check, still answers the probes, and how much memory the
   quantized model needs compared with 16 bit.

Calibration and probes use synthetic images (``models/smoke.py``): this checks the tools, not
the quality of the result, so no dataset is needed. Rows go to ``results/smoke/RC-Q0.csv``
with experiment ``RC-Q0``.
"""

from __future__ import annotations

import importlib
import math
import pkgutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from medcompress.compression.quantization import grid

SMOKE_EXPERIMENT = "RC-Q0"


def row(metric: str, value: float | int | bool) -> dict[str, Any]:
    """One results row of the smoke test."""
    return {
        "experiment": SMOKE_EXPERIMENT,
        "budget": "",
        "seed": 0,
        "metric": metric,
        "value": float(value),
    }


def discover_kernel_classes() -> list[str]:
    """Dotted paths of GPTQModel's quantized-linear classes that declare supported bit widths.

    Read from the installed package (``gptqmodel.nn_modules.qlinear``), so nothing about its
    internal layout is assumed; classes that fail to import are skipped.
    """
    package = importlib.import_module("gptqmodel.nn_modules.qlinear")
    found: set[str] = set()
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # noqa: BLE001 - optional kernels may need missing extensions
            continue
        for name, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and obj.__module__ == module.__name__
                and hasattr(obj, "SUPPORTS_BITS")
                and not name.startswith("_")
            ):
                found.add(f"{obj.__module__}.{name}")
    return sorted(found)


def try_kernel(
    kernel_class: str,
    bits: int,
    *,
    in_features: int = 2048,
    out_features: int = 2048,
    group_size: int = 128,
    repeats: int = 20,
    device: Any = None,
) -> dict[str, Any]:
    """Build one kernel module from random codes, check it and time a batch-1 forward.

    Returns ``{"ok": bool, "ms": float, "error": str}``; ``ms`` is NaN when it failed.
    """
    import torch

    from medcompress.compression.quantization.kernels import build_gptq_linear

    device = device or torch.device("cuda")
    rng = np.random.default_rng(bits)
    codes, scales = grid.round_to_nearest(
        rng.normal(size=(out_features, in_features)), bits, group_size
    )
    try:
        module = build_gptq_linear(
            codes,
            scales,
            None,
            bits=bits,
            group_size=group_size,
            kernel_class=kernel_class,
            dtype=torch.float16,
            device=device,
        )
        x = torch.randn(1, in_features, dtype=torch.float16, device=device)
        with torch.no_grad():
            for _ in range(3):
                module(x)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(repeats):
                module(x)
            torch.cuda.synchronize()
        return {"ok": True, "ms": 1000 * (time.perf_counter() - started) / repeats, "error": ""}
    except Exception as exc:  # noqa: BLE001 - every failure is a finding of the smoke test
        return {"ok": False, "ms": math.nan, "error": f"{type(exc).__name__}: {exc}"[:300]}


def choose_kernel_class(rows: Sequence[Mapping[str, Any]], classes: Sequence[str]) -> str | None:
    """The class RC-Q0 picks: the fastest at w4 among those that passed **both** w4 and w8.

    ``rows`` are the :func:`kernel_rows`; ``classes`` the dotted paths they were made from.
    Every GPTQ recipe (E3 ``w4`` and ``w8``, E8, E9) loads the same class, so a class that
    fails one of the two widths is never chosen. Ties are broken by name. None if no class
    passed both.
    """
    values = {str(r["metric"]): float(r["value"]) for r in rows}
    shorts = {dotted.rsplit(".", 1)[-1]: dotted for dotted in reversed(classes)}
    passed = [
        short
        for short in shorts
        if values.get(f"kernel_ok:{short}:w4") == 1.0
        and values.get(f"kernel_ok:{short}:w8") == 1.0
        and f"kernel_ms:{short}:w4" in values
    ]
    if not passed:
        return None
    fastest = min(passed, key=lambda short: (values[f"kernel_ms:{short}:w4"], short))
    return shorts[fastest]


def kernel_rows(classes: Sequence[str], log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """``kernel_ok:<class>:w<bits>`` and ``kernel_ms:<class>:w<bits>`` rows for every class."""
    rows = []
    for dotted in classes:
        short = dotted.rsplit(".", 1)[-1]
        for bits in grid.SUPPORTED_BITS:
            result = try_kernel(dotted, bits)
            log(f"{dotted} w{bits}: {'ok' if result['ok'] else result['error']}")
            rows.append(row(f"kernel_ok:{short}:w{bits}", result["ok"]))
            if result["ok"]:
                rows.append(row(f"kernel_ms:{short}:w{bits}", result["ms"]))
    return rows


def answers_agree(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Fraction of probes whose predicted answer is the same in two score lists."""
    if len(a) != len(b) or not a:
        raise ValueError("need two score lists of the same, non-zero length")
    return float(np.mean([x.predicted == y.predicted for x, y in zip(a, b, strict=True)]))


def write_synthetic_calibration(
    folder: Path, images: Sequence[Any], questions: Sequence[str]
) -> list[Any]:
    """Save synthetic images and return VQASample rows (answer "yes") for GPTQ calibration."""
    from medcompress.data.base import VQASample

    folder.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, (image, question) in enumerate(zip(images, questions, strict=True)):
        name = f"calibration_{i}.png"
        image.save(folder / name)
        samples.append(VQASample(f"smoke{i}", "synthetic", name, question, "yes"))
    return samples
