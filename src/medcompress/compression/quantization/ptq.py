"""Post-training quantization of the decoder: E3 (from E2), E8 (from E7) and E9 (the teacher).

Recipes (``configs/quantization.yaml``, D08):

* ``w4`` / ``w8``: GPTQ (``gptq.py``) with 4 or 8 bits and groups of 128 input columns, on
  the decoder's linear layers only; calibrated on the same 128 training images as the pruning
  criteria (``data/splits.make_calibration_subset`` with the run seed).
* ``w8a8``: LLM.int8 from bitsandbytes (8-bit weights and activations, 16-bit outliers); it
  needs no calibration and is applied when the model is loaded.

GPTQ runs on the model in memory, layer by layer: the inputs of each decoder layer's linear
layers are collected on the calibration examples (image, question and ground-truth answer in
the chat template), its matrices are rounded, the rounded weights are written back, and the
next layer is calibrated on the outputs of the already rounded layers (the sequential scheme of
the GPTQ paper). Working in memory is what lets E8 quantize a pruned model, whose layers have
different widths and cannot be reloaded with ``from_pretrained``.

A quantized checkpoint stores our codes and scales (``quantized.npz``, packed with
``grid.pack_codes``) and a manifest saying which model it starts from. Loading rebuilds the
parent model and swaps every target layer for a kernel module (``kernels.py``); the kernel
class is a setting (``quantization.kernel_class``) chosen by the smoke test RC-Q0.

torch is imported only inside the functions that need it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from medcompress.compression.quantization import grid
from medcompress.compression.quantization.gptq import DEFAULT_BLOCK_SIZE, DEFAULT_DAMPING
from medcompress.compression.quantization.kernels import (
    DECODER_TARGETS,
    LLM_INT8_THRESHOLD,
    PARENT_OF,
)
from medcompress.config import ConfigError, find_todos

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

PTQ_BACKENDS: tuple[str, ...] = ("gptq", "llm_int8")
QUANTIZED_SUBDIR = "quantized"
MANIFEST_NAME = "quant_manifest.json"
TENSORS_NAME = "quantized.npz"
FORMAT_VERSION = 1
REFERENCE_KERNEL = "reference"
"""``quantization.kernel_class`` value that loads :class:`ReferenceQuantLinear` (tests, CPU);
such a model fails the low-precision kernel check and is never reported as quantized."""
_SHARED_INPUT = {
    "q_proj": "attn_in",
    "k_proj": "attn_in",
    "v_proj": "attn_in",
    "o_proj": "attn_out",
    "gate_proj": "mlp_in",
    "up_proj": "mlp_in",
    "down_proj": "mlp_act",
}


def _open(value: Any) -> bool:
    return value is None or bool(find_todos({"v": value}))


# ----------------------------------------------------------------------- settings


@dataclass(frozen=True)
class PTQRecipe:
    """One quantization recipe, resolved for a run."""

    name: str
    backend: str
    weight_bits: int
    activation_bits: int | None
    group_size: int | None
    targets: tuple[str, ...]
    calibration_size: int | None
    damping: float = DEFAULT_DAMPING
    block_size: int = DEFAULT_BLOCK_SIZE
    llm_int8_threshold: float = LLM_INT8_THRESHOLD

    @classmethod
    def from_config(cls, config: Mapping[str, Any], variant: str | None = None) -> PTQRecipe:
        """Read the recipe named by ``quantization.recipe`` (``by_variant``: the run's variant).

        Raises:
            ConfigError: the recipe is unknown, or a value it needs is missing or ``TODO``.
        """
        section = config.get("quantization") or {}
        name = section.get("recipe")
        if name == "by_variant":
            name = variant
        if _open(name):
            raise ConfigError("settings still open: quantization.recipe")
        recipes = section.get("recipes") or {}
        if name not in recipes:
            raise ConfigError(f"unknown quantization recipe {name!r}; known: {sorted(recipes)}")
        recipe = recipes[name]
        backend = recipe.get("backend")
        if backend not in PTQ_BACKENDS:
            raise ConfigError(
                f"quantization backend must be one of {PTQ_BACKENDS}, got {backend!r}"
            )
        open_keys = []
        bits = recipe.get("weight_bits")
        group = recipe.get("group_size")
        size = (section.get("calibration") or {}).get("size")
        if _open(bits):
            open_keys.append(f"quantization.recipes.{name}.weight_bits")
        if backend == "gptq":
            if _open(group):
                open_keys.append(f"quantization.recipes.{name}.group_size")
            if _open(size):
                open_keys.append("quantization.calibration.size")
        if open_keys:
            raise ConfigError(f"settings still open: {', '.join(open_keys)}")
        targets = tuple(section.get("targets") or DECODER_TARGETS)
        unknown = sorted(set(targets) - set(DECODER_TARGETS))
        if unknown:
            raise ConfigError(f"unknown quantization targets {unknown}; known: {DECODER_TARGETS}")
        if backend == "gptq" and int(bits) not in grid.SUPPORTED_BITS:
            raise ConfigError(f"GPTQ bits must be one of {grid.SUPPORTED_BITS}")
        return cls(
            name=str(name),
            backend=str(backend),
            weight_bits=int(bits),
            activation_bits=recipe.get("activation_bits"),
            group_size=int(group) if backend == "gptq" else None,
            targets=targets,
            calibration_size=int(size) if backend == "gptq" else None,
            damping=float(recipe.get("damping", DEFAULT_DAMPING)),
            block_size=int(recipe.get("block_size", DEFAULT_BLOCK_SIZE)),
            llm_int8_threshold=float(recipe.get("llm_int8_threshold", LLM_INT8_THRESHOLD)),
        )


def kernel_class_from_config(config: Mapping[str, Any]) -> str:
    """``quantization.kernel_class``: a dotted GPTQModel class, or ``"reference"``."""
    value = (config.get("quantization") or {}).get("kernel_class")
    if _open(value):
        raise ConfigError("settings still open: quantization.kernel_class (set after RC-Q0)")
    return str(value)


def quantized_dir(config: Mapping[str, Any], run: RunContext) -> Path:
    """``<checkpoint_root>/<ID>[/<variant>]/<budget|nobudget>/seed<k>/quantized``."""
    tag = run.experiment_tag.replace(":", "/")
    base = Path(config["paths"]["checkpoint_root"]) / tag / (run.budget or "nobudget")
    return base / f"seed{run.seed}" / QUANTIZED_SUBDIR


# --------------------------------------------------------------------- checkpoint


@dataclass(frozen=True)
class QuantizedMatrix:
    """The codes ``(out, in)`` and scales ``(out, n_groups)`` of one decoder matrix."""

    key: str
    codes: np.ndarray
    scales: np.ndarray
    bits: int
    group_size: int

    def dequantized(self) -> np.ndarray:
        """``(q - z) * s`` as float64."""
        return grid.dequantize(self.codes, self.scales, self.bits, self.group_size)


def save_quantized(
    records: Mapping[str, QuantizedMatrix],
    recipe: PTQRecipe,
    directory: Path | str,
    *,
    start_from: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``quantized.npz`` (packed codes, 16-bit scales) and the manifest; return the folder.

    ``start_from`` names the parent model (experiment, budget, seed) that loading rebuilds.
    A recipe applied at load time (LLM.int8) has no records; only the manifest is written.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, rec in sorted(records.items()):
        arrays[f"{key}.qweight"] = grid.pack_codes(rec.codes, rec.bits)
        arrays[f"{key}.scales"] = rec.scales.astype(np.float16)
        arrays[f"{key}.in_features"] = np.array(rec.codes.shape[1], dtype=np.int64)
    if arrays:
        np.savez(out / TENSORS_NAME, **arrays)
    manifest = {
        "format_version": FORMAT_VERSION,
        "recipe": asdict(recipe),
        "start_from": dict(start_from),
        "matrices": sorted(records),
        "quantized_weight_bytes": int(
            sum(grid.packed_bytes(*r.codes.shape, r.bits, r.group_size) for r in records.values())
        ),
        **dict(extra or {}),
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def read_quantized(directory: Path | str) -> tuple[dict[str, Any], dict[str, QuantizedMatrix]]:
    """Return the manifest and the records of a quantized checkpoint."""
    folder = Path(directory)
    manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{folder}: unsupported format_version {manifest.get('format_version')!r}")
    recipe = manifest["recipe"]
    records: dict[str, QuantizedMatrix] = {}
    if manifest["matrices"]:
        with np.load(folder / TENSORS_NAME, allow_pickle=False) as data:
            for key in manifest["matrices"]:
                in_features = int(data[f"{key}.in_features"])
                codes = grid.unpack_codes(
                    data[f"{key}.qweight"], recipe["weight_bits"], in_features
                )
                records[key] = QuantizedMatrix(
                    key,
                    codes,
                    data[f"{key}.scales"].astype(np.float64),
                    int(recipe["weight_bits"]),
                    int(recipe["group_size"]),
                )
    return manifest, records


# --------------------------------------------------------------------- calibration


def calibration_batches(
    model: VLMWrapper, samples: Sequence[VQASample], data_root: Path
) -> list[Any]:
    """One model input per calibration example: image, question and ground-truth answer.

    Built with the training batch code (chat template, answer tokens and the stop token
    appended), one example per batch, so no padding enters the Hessians.
    """
    from medcompress.compression.distillation.batching import (
        assemble,
        ground_truth_ids,
        prompt_inputs,
    )

    out = []
    pad = model.tokenizer.pad_token_id  # type: ignore[attr-defined]
    device = model.module.device  # type: ignore[attr-defined]
    for sample in samples:
        prompt = prompt_inputs(model, Path(data_root) / sample.image_path, sample.question)  # type: ignore[arg-type]
        batch = assemble([prompt], [ground_truth_ids(model, sample.answer)], pad)  # type: ignore[arg-type]
        out.append({k: v.to(device) for k, v in batch.inputs.items()})
    return out


class _StopForward(Exception):
    """Raised by a hook to end a forward pass once the layer of interest has run."""


def collect_hessians(
    model: VLMWrapper | Any,
    layer_index: int,
    batches: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> dict[str, Any]:
    """``{target name: H}`` for one decoder layer: ``H = 2 / n * sum x x^T`` of its inputs.

    Targets that read the same input (q/k/v; gate/up) share one Hessian. The forward pass is
    cut after the layer, so calibrating layer ``i`` costs ``i + 1`` layers per example.
    """
    import torch

    from medcompress.compression.pruning.surgery import find_decoder_stack

    module = getattr(model, "module", model)
    layer = find_decoder_stack(module).layers[layer_index]
    parents = {"self_attn": layer.self_attn, "mlp": layer.mlp}
    sources: dict[str, str] = {}
    for name in names:
        sources.setdefault(_SHARED_INPUT[name], name)
    sums: dict[str, Any] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(source: str) -> Any:
        def hook(_module: Any, args: tuple) -> None:
            x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
            n_old = counts.get(source, 0)
            total = n_old + x.shape[0]
            if source not in sums:
                sums[source] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
            sums[source] *= n_old / total
            x = (2.0 / total) ** 0.5 * x
            sums[source] += x.T @ x
            counts[source] = total

        return hook

    for source, name in sources.items():
        linear = getattr(parents[PARENT_OF[name]], name)
        handles.append(linear.register_forward_pre_hook(make_hook(source)))

    def stop(_module: Any, _args: tuple, _output: Any) -> None:
        raise _StopForward

    handles.append(layer.register_forward_hook(stop))
    try:
        with torch.inference_mode():
            for batch in batches:
                try:
                    module(**batch, use_cache=False)
                except _StopForward:
                    pass
    finally:
        for handle in handles:
            handle.remove()
    return {name: sums[_SHARED_INPUT[name]] for name in names}


def gptq_quantize_decoder(
    model: VLMWrapper | Any,
    batches: Sequence[Mapping[str, Any]],
    recipe: PTQRecipe,
    *,
    log: Any = print,
) -> dict[str, QuantizedMatrix]:
    """Round every target matrix of the decoder with GPTQ, layer after layer (in place).

    The rounded weights replace the original ones in the model (still 16-bit tensors), so
    each layer is calibrated on the outputs of the rounded layers before it. Returns the codes
    and scales per matrix; :func:`install_quantized` then swaps in the kernel modules.
    """
    import torch

    from medcompress.compression.pruning.surgery import find_decoder_stack
    from medcompress.compression.quantization.gptq import gptq_torch
    from medcompress.compression.quantization.kernels import target_path

    if recipe.backend != "gptq" or recipe.group_size is None:
        raise ValueError("gptq_quantize_decoder needs a GPTQ recipe")
    module = getattr(model, "module", model)
    stack = find_decoder_stack(module)
    records: dict[str, QuantizedMatrix] = {}
    for i, layer in enumerate(stack.layers):
        hessians = collect_hessians(model, i, batches, recipe.targets)
        for name in recipe.targets:
            linear = getattr(getattr(layer, PARENT_OF[name]), name)
            result = gptq_torch(
                linear.weight,
                hessians[name],
                bits=recipe.weight_bits,
                group_size=recipe.group_size,
                damping=recipe.damping,
                block_size=recipe.block_size,
            )
            with torch.no_grad():
                rounded = torch.as_tensor(result.dequantized(), dtype=linear.weight.dtype)
                linear.weight.copy_(rounded.to(linear.weight.device))
            key = target_path(i, name)
            records[key] = QuantizedMatrix(
                key, result.codes, result.scales, recipe.weight_bits, recipe.group_size
            )
        del hessians
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"GPTQ: layer {i + 1}/{len(stack.layers)} done")
    return records


def install_quantized(
    model: VLMWrapper | Any,
    records: Mapping[str, QuantizedMatrix],
    recipe: PTQRecipe,
    *,
    kernel_class: str,
) -> None:
    """Replace every target linear layer by a kernel module holding its codes and scales."""
    from medcompress.compression.quantization.kernels import (
        build_gptq_linear,
        build_reference_linear,
        decoder_targets,
    )

    targets = decoder_targets(model, recipe.targets)
    missing = [key for key, _, _ in targets if key not in records]
    if missing:
        raise ValueError(f"the checkpoint lacks {len(missing)} matrices, e.g. {missing[:3]}")
    for key, parent, attr in targets:
        old, rec = getattr(parent, attr), records[key]
        if tuple(old.weight.shape) != rec.codes.shape:
            raise ValueError(f"{key}: shape {tuple(old.weight.shape)} != {rec.codes.shape}")
        kwargs = {
            "bits": rec.bits,
            "group_size": rec.group_size,
            "dtype": old.weight.dtype,
            "device": old.weight.device,
        }
        if kernel_class == REFERENCE_KERNEL:
            new = build_reference_linear(rec.codes, rec.scales, old.bias, **kwargs)
        else:
            new = build_gptq_linear(
                rec.codes, rec.scales, old.bias, kernel_class=kernel_class, **kwargs
            )
        setattr(parent, attr, new)


def apply_llm_int8(model: VLMWrapper | Any, recipe: PTQRecipe) -> None:
    """Replace every target linear layer by a bitsandbytes LLM.int8 layer (needs a GPU)."""
    from medcompress.compression.quantization.kernels import build_int8_linear, decoder_targets

    for _key, parent, attr in decoder_targets(model, recipe.targets):
        setattr(parent, attr, build_int8_linear(getattr(parent, attr), recipe.llm_int8_threshold))


def load_quantized(model: VLMWrapper | Any, directory: Path | str, *, kernel_class: str) -> Any:
    """Turn the parent ``model`` into the quantized model saved in ``directory``."""
    manifest, records = read_quantized(directory)
    recipe = PTQRecipe(**{**manifest["recipe"], "targets": tuple(manifest["recipe"]["targets"])})
    if recipe.backend == "llm_int8":
        apply_llm_int8(model, recipe)
    else:
        install_quantized(model, records, recipe, kernel_class=kernel_class)
    return model


# ---------------------------------------------------------------------- the stage


def parent_of(run: RunContext) -> dict[str, Any]:
    """The model a quantized run starts from: E3 <- E2, E8 <- E7 (same budget), E9 <- E0,
    E4 <- the pretrained student (its codes come from quantization-aware training)."""
    if run.experiment_id == "E4":
        return {"experiment": "pretrained", "budget": None, "seed": run.seed}
    if run.experiment_id == "E3":
        return {"experiment": "E2", "budget": None, "seed": run.seed}
    if run.experiment_id == "E8":
        if run.budget is None:
            raise ConfigError("E8 belongs to a budget (--budget)")
        return {"experiment": "E7", "budget": run.budget, "seed": run.seed}
    if run.experiment_id == "E9":
        return {"experiment": "E0", "budget": None, "seed": 0}
    raise ValueError(f"{run.experiment_id} is not a post-training quantization arm")


def load_parent(config: Mapping[str, Any], run: RunContext, side: int) -> VLMWrapper:
    """Load the 16-bit model that ``run`` quantizes (adapters merged, pruning replayed)."""
    from medcompress.config import RunContext as Run
    from medcompress.config import load_config
    from medcompress.evaluation.runner import load_trained_student

    parent = parent_of(run)
    if parent["experiment"] == "pretrained":
        from medcompress.models.loading import ModelSpec, load_model

        student = config.get("model", {}).get("config")
        if _open(student):
            raise ConfigError("settings still open: model.config (the student)")
        spec = ModelSpec.from_config(load_config(student))
        return load_model(spec, allow_candidate=True, image_longest_side=side)
    if parent["experiment"] == "E0":
        from medcompress.models.loading import ModelSpec, load_model

        spec = ModelSpec.from_config(load_config(config["components"]["teacher"]))
        return load_model(spec, image_longest_side=side)
    student = load_trained_student(config, "E2", run.seed, side)
    if parent["experiment"] == "E2":
        return student
    from medcompress.compression.pruning.checkpoint import load_pruned
    from medcompress.compression.recovery import pruned_dir

    return load_pruned(student, pruned_dir(config, Run("E7", seed=run.seed, budget=run.budget)))


def load_run_quantized(config: Mapping[str, Any], run: RunContext, side: int) -> VLMWrapper:
    """The quantized model of ``run`` for evaluation and timing (runner, benchmark)."""
    model = load_parent(config, run, side)
    return load_quantized(
        model, quantized_dir(config, run), kernel_class=kernel_class_from_config(config)
    )


def run_quantization(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    model: VLMWrapper | None = None,
    train: Sequence[VQASample] | None = None,
) -> None:
    """Stage entry point for ``scripts/03_quantize.py``.

    GPTQ recipes: loads the parent model (``model`` replaces it in tests), draws the 128
    calibration images with the run seed, rounds the decoder layer by layer and saves the codes
    to :func:`quantized_dir`. LLM.int8: writes only the manifest (it is applied at load time).
    Writes ``quantization_*`` rows to the run's ``metrics.csv``.

    Used by: E3, E8, E9.
    """
    from medcompress.utils.io import upsert_results

    if run is None:
        raise ConfigError("--exp is required")
    recipe = PTQRecipe.from_config(config, run.variant)
    parent = parent_of(run)
    paths = config["paths"]
    rows: dict[str, float] = {}
    extra: dict[str, Any] = {}
    records: dict[str, QuantizedMatrix] = {}
    if recipe.backend == "gptq":
        side = config.get("inputs", {}).get("image_longest_side")
        if model is None and _open(side):
            raise ConfigError("settings still open: inputs.image_longest_side")
        if train is None:
            from medcompress.compression.pruning.stage import load_train_samples

            train = load_train_samples(config)
        from medcompress.data.splits import make_calibration_subset

        assert recipe.calibration_size is not None
        calibration = make_calibration_subset(train, recipe.calibration_size, seed=run.seed)
        if model is None:
            model = load_parent(config, run, int(side))
        started = time.perf_counter()
        batches = calibration_batches(model, calibration, Path(paths["data_root"]))
        records = gptq_quantize_decoder(model, batches, recipe)
        rows["quantization_seconds"] = time.perf_counter() - started
        rows["quantization_calibration_images"] = float(len(calibration))
        extra["calibration_sample_ids"] = [s.sample_id for s in calibration]
    directory = save_quantized(
        records, recipe, quantized_dir(config, run), start_from=parent, extra=extra
    )
    rows["quantized_matrices"] = float(len(records))
    if records:
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        rows["quantized_weight_bytes"] = float(manifest["quantized_weight_bytes"])
    rows["quantization_weight_bits"] = float(recipe.weight_bits)
    target = run.run_dir(paths["results_root"]) / "metrics.csv"
    upsert_results(
        [
            {
                "experiment": run.experiment_tag,
                "budget": run.budget or "",
                "seed": run.seed,
                "metric": name,
                "value": value,
            }
            for name, value in rows.items()
        ],
        target,
    )
    print(
        f"quantized {run.experiment_tag} with {recipe.name} ({recipe.backend}); saved {directory}"
    )
