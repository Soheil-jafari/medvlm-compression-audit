#!/usr/bin/env python
"""Inspect linear inputs and GPTQ Hessian precision on a few E3 calibration examples."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_context  # noqa: E402


def layer_range(value: str) -> range:
    """Parse a zero-based layer index or an inclusive range such as 0-2."""
    try:
        first, _, last = value.partition("-")
        start, stop = int(first), int(last or first)
        if 0 <= start <= stop:
            return range(start, stop + 1)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use a layer index or an inclusive range such as 0-2")


def report_input(prefix: str, name: str, inputs: Any) -> None:
    """Print input statistics and the finiteness of this batch's Hessian in each dtype."""
    import torch

    if inputs.shape[0] == 0:
        print(f"{prefix} {name}: no input tokens")
        return
    x = inputs.float()
    finite = torch.isfinite(x)
    nonfinite = x.numel() - int(finite.sum())
    values = x[finite] if nonfinite else x
    low, high = torch.aminmax(values) if values.numel() else (float("nan"),) * 2

    def finite_hessian(dtype: Any) -> bool:
        with torch.autocast(device_type=x.device.type, enabled=False):
            scaled = x.to(dtype) * (2.0 / x.shape[0]) ** 0.5
            hessian = scaled.T @ scaled
            return all(torch.isfinite(chunk).all().item() for chunk in hessian.split(256))

    print(
        f"{prefix} {name}: dtype={inputs.dtype}, min={float(low):.6g}, max={float(high):.6g}, "
        f"absmax={max(abs(float(low)), abs(float(high))):.6g}, "
        f"non-finite inputs={nonfinite}, "
        f"Hessian finite float16={finite_hessian(torch.float16)}, "
        f"float32={finite_hessian(torch.float32)}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Load the selected E2 adapter and inspect the prefix of E3's calibration subset."""
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--layers", type=layer_range, default=range(3))
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args(argv)
    if args.exp != "E3" or args.variant not in {"w4", "w8"}:
        parser.error("use --exp E3 with --variant w4 or w8")

    import torch

    from medcompress.compression.pruning.stage import load_train_samples
    from medcompress.compression.pruning.surgery import find_decoder_stack
    from medcompress.compression.quantization.ptq import (
        PTQRecipe,
        calibration_batches,
        collect_hessians,
        load_parent,
    )
    from medcompress.data.splits import make_calibration_subset

    if torch.cuda.device_count() != 1:
        parser.error("expose exactly one GPU with CUDA_VISIBLE_DEVICES=0")
    run = run_context(args)
    assert run is not None
    config = resolve_config(args)
    recipe = PTQRecipe.from_config(config, run.variant)
    assert recipe.calibration_size is not None
    if not 1 <= args.examples <= recipe.calibration_size:
        parser.error(f"--examples must be between 1 and {recipe.calibration_size}")
    train = load_train_samples(config)
    samples = make_calibration_subset(train, recipe.calibration_size, seed=run.seed)
    model = load_parent(config, run, int(config["inputs"]["image_longest_side"]))
    if args.layers.stop > len(find_decoder_stack(model.module).layers):
        parser.error("--layers includes an index outside the decoder")
    batches = calibration_batches(
        model, samples[: args.examples], Path(config["paths"]["data_root"])
    )
    print("Calibration batches on the unquantized E2 parent (no weights are changed).")
    for layer in args.layers:
        for i, batch in enumerate(batches, start=1):
            collect_hessians(
                model,
                layer,
                [batch],
                recipe.targets,
                inspect_inputs=partial(report_input, f"layer {layer} batch {i}"),
            )
    return 0


if __name__ == "__main__":
    main_guard(main)
