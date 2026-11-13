#!/usr/bin/env python
"""Trace GPTQ's layer inputs through the production capture path, batch by batch (E3).

Runs exactly what stage 03 runs: the same parent (``ptq.load_parent``), the same calibration
subset and batches (``make_calibration_subset``, ``calibration_batches``: all of the recipe's
examples by default) and the same capture (``collect_hessians``). For every batch it prints, at
each step of that forward, whether the tensor is finite and its absolute maximum:
``pixel_values``, the vision tower's output, ``inputs_embeds`` and ``position_ids`` (what the
language model receives), the decoder layer-0 input and the captured input of each target.
For the first batch, and the first batch with a non-finite value, it also runs the model's own
full forward on the same batch and prints where the two first differ (they must be equal).
With ``--gptq`` it then rounds the chosen layers' matrices from the production Hessians, as
stage 03 does (the model's weights are not changed).

Exit code 1 if any captured input is not finite, the two forwards differ, or GPTQ fails.

Example (Kaggle, one T4, the shared data and e2-seed0-adapter attached):
    python scripts/check_gptq_inputs.py --exp E3 --variant w4 --seed 0 --layers 0 --gptq
"""

from __future__ import annotations

import argparse
import sys
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


def describe(tensor: Any) -> tuple[bool, float]:
    """``(finite, absmax over the finite values)`` of a tensor."""
    import torch

    x = tensor.detach().float()
    finite = torch.isfinite(x)
    values = x[finite]
    return bool(finite.all()), float(values.abs().max()) if values.numel() else float("nan")


class Trace:
    """Hooks that keep each step's tensor of one forward (vision output, the language model's
    ``inputs_embeds`` / ``position_ids``, the layer-0 input, the captured target inputs)."""

    def __init__(self, module: Any, layers: Any) -> None:
        self.steps: dict[str, Any] = {}
        self.handles: list[Any] = []
        inner = getattr(module, "model", module)
        visual = getattr(inner, "visual", None)
        language = getattr(inner, "language_model", None)
        if visual is not None:
            self.handles.append(visual.register_forward_hook(self._vision))
        if language is not None:
            self.handles.append(
                language.register_forward_pre_hook(self._language, with_kwargs=True)
            )
        self.handles.append(layers[0].register_forward_pre_hook(self._layer0, with_kwargs=True))

    def put(self, name: str, tensor: Any) -> None:
        if tensor is not None and hasattr(tensor, "detach"):
            self.steps[name] = tensor.detach().clone()

    def _vision(self, _m: Any, _args: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else getattr(output, "pooler_output", output)
        self.put("vision tower output", value)

    def _language(self, _m: Any, args: Any, kwargs: Any) -> None:
        self.put("inputs_embeds", kwargs.get("inputs_embeds"))
        self.put("position_ids", kwargs.get("position_ids"))

    def _layer0(self, _m: Any, args: Any, kwargs: Any) -> None:
        self.put("layer-0 input", args[0] if args else kwargs.get("hidden_states"))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def first_difference(ours: dict[str, Any], theirs: dict[str, Any]) -> str | None:
    """The first step whose tensors differ (None if all are equal)."""
    import torch

    for name, tensor in ours.items():
        other = theirs.get(name)
        if (
            other is None
            or other.shape != tensor.shape
            or not torch.equal(
                torch.nan_to_num(other.float(), nan=12345.0),
                torch.nan_to_num(tensor.float(), nan=12345.0),
            )
        ):
            return name
    return None


def own_forward_difference(
    module: Any, layers: Any, layer: int, batch: Any, targets: Any, production: Trace
) -> str | None:
    """Run the model's own full forward on ``batch`` with the same hooks and return the first
    step where it differs from the production capture (None if every step is equal)."""
    import torch

    from medcompress.compression.quantization.ptq import PARENT_OF

    own = Trace(module, layers)
    own.put("pixel_values", batch.get("pixel_values"))
    hooks = []
    for name in targets:
        linear = getattr(getattr(layers[layer], PARENT_OF[name]), name)

        def hook(_m: Any, args: Any, name: str = name) -> None:
            own.put(f"captured {name} input", args[0].reshape(-1, args[0].shape[-1]))

        hooks.append(linear.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            device = module.device
            module(**{k: v.to(device) for k, v in batch.items()}, use_cache=False)
    finally:
        own.remove()
        for handle in hooks:
            handle.remove()
    return first_difference(production.steps, own.steps)


def main(argv: list[str] | None = None) -> int:
    """Trace the production capture of E3's calibration batches (see the module docstring)."""
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--layers", type=layer_range, default=range(1))
    parser.add_argument("--examples", type=int, default=None, help="default: all of them")
    parser.add_argument("--gptq", action="store_true", help="round the layers' matrices too")
    args = parser.parse_args(argv)
    if args.exp != "E3" or args.variant not in {"w4", "w8"}:
        parser.error("use --exp E3 with --variant w4 or w8")

    import torch

    from medcompress.compression.pruning.stage import load_train_samples
    from medcompress.compression.pruning.surgery import find_decoder_stack
    from medcompress.compression.quantization.gptq import gptq_torch
    from medcompress.compression.quantization.ptq import (
        PARENT_OF,
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
    count = args.examples or recipe.calibration_size
    if not 1 <= count <= recipe.calibration_size:
        parser.error(f"--examples must be between 1 and {recipe.calibration_size}")
    train = load_train_samples(config)
    samples = make_calibration_subset(train, recipe.calibration_size, seed=run.seed)[:count]
    model = load_parent(config, run, int(config["inputs"]["image_longest_side"]))
    module = model.module
    layers = find_decoder_stack(module).layers
    if args.layers.stop > len(layers):
        parser.error("--layers includes an index outside the decoder")
    batches = calibration_batches(model, samples, Path(config["paths"]["data_root"]))
    ids = [s.sample_id for s in samples]
    print(f"{len(batches)} calibration batches, the production capture (stage 03's path); "
          f"parent {run.experiment_tag}'s weights are not changed", flush=True)  # fmt: skip
    nonfinite: list[str] = []
    differences = 0
    compared_bad = False
    layer = args.layers.start
    for index, batch in enumerate(batches):
        trace = Trace(module, layers)
        trace.put("pixel_values", batch.get("pixel_values"))

        def captured(name: str, inputs: Any, trace: Trace = trace) -> None:
            trace.put(f"captured {name} input", inputs)

        try:
            collect_hessians(model, layer, [batch], recipe.targets, inspect_inputs=captured)
        finally:
            trace.remove()
        stats = {k: describe(v) for k, v in trace.steps.items()}
        bad = [k for k, (finite, _m) in stats.items() if not finite]
        line = ", ".join(
            f"{k} {'ok' if f else 'NON-FINITE'} {m:.4g}" for k, (f, m) in stats.items()
        )
        print(f"batch {index + 1} ({ids[index]}): {line}", flush=True)
        if bad:
            nonfinite.append(ids[index])
            print(f"batch {index + 1} ({ids[index]}): first non-finite step: {bad[0]}", flush=True)
        if index == 0 or (bad and not compared_bad):
            compared_bad = compared_bad or bool(bad)
            differ = own_forward_difference(module, layers, layer, batch, recipe.targets, trace)
            verdict = "equal at every step" if differ is None else f"first differ at: {differ}"
            print(f"batch {index + 1}: production capture vs the model's own forward: {verdict}")
            differences += differ is not None
        del trace
    print(f"non-finite batches: {len(nonfinite)} of {len(batches)}: {nonfinite[:10]}", flush=True)
    problems = len(nonfinite) + differences
    if args.gptq:
        for layer in args.layers:
            try:
                hessians = collect_hessians(model, layer, batches, recipe.targets, labels=ids)
                for name in recipe.targets:
                    linear = getattr(getattr(layers[layer], PARENT_OF[name]), name)
                    result = gptq_torch(
                        linear.weight, hessians[name], bits=recipe.weight_bits,
                        group_size=recipe.group_size, damping=recipe.damping,
                        block_size=recipe.block_size, matrix=f"layer{layer}.{name}",
                    )  # fmt: skip
                    print(f"GPTQ layer {layer} {name}: ok (output error {result.output_error:.4g}, "
                          f"damping retries {result.damping_retries})", flush=True)  # fmt: skip
            except (ValueError, RuntimeError) as exc:
                print(f"GPTQ layer {layer}: FAILED: {exc}", flush=True)
                problems += 1
    return 1 if problems else 0


if __name__ == "__main__":
    main_guard(main)
