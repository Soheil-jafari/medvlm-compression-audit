"""Model loading from model configs (``configs/models/*.yaml``).

Weights come from a hub ID or a local path named in the config. A model is only
loaded when its source is known and its revision is pinned, so a later run cannot
silently pick up different weights. Candidate students that are still being compared
carry a ``sha_seen`` revision and may be loaded with ``allow_candidate=True``; the
chosen teacher and the final student carry a pinned ``revision``.

torch, transformers and peft are imported only inside the functions that need them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.config import CANDIDATE_STATUS, TODO_MARKER, ConfigError

if TYPE_CHECKING:
    from medcompress.models.base import VLMWrapper

DEFAULT_DTYPE = "float16"  # the T4 has no bf16

# Families that are treated as shipping their own model code until the feasibility
# run (RC-S0) shows that the installed transformers loads them natively. Loading
# such a family runs code from the model repository, so it needs an explicit opt-in
# (allow_remote_code=True) after the code has been read. The list is conservative:
# removing a family from it is a decision to be recorded, adding one is always safe.
REMOTE_CODE_FAMILIES = frozenset({"ovis2_5", "fastvlm", "hulu_med", "sail_vl2", "internvl3_5"})


class ModelConfigError(ConfigError):
    """Raised when a model config cannot be loaded as it stands."""


def _is_open(value: Any) -> bool:
    """True for a missing field or a placeholder starting with ``TODO``."""
    return value is None or (isinstance(value, str) and value.strip().startswith(TODO_MARKER))


@dataclass(frozen=True)
class ModelSpec:
    """Parsed ``model`` section of a model config."""

    name: str
    role: str  # "teacher" or "student"
    family: str
    source: str  # local path or hub ID
    revision: str | None  # pinned commit; None while only a candidate
    sha_seen: str | None  # commit read when the candidate was checked
    adapter: str | None
    adapter_revision: str | None
    dtype: str
    status: str
    tokenizer_family: str | None
    vocab_size: int | None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ModelSpec:
        """Build a ModelSpec from a loaded model config (with or without the ``model`` key).

        A missing or ``TODO`` dtype becomes ``float16``, the only practical choice on a T4.
        """
        section = config.get("model", config)
        if not isinstance(section, Mapping):
            raise ModelConfigError("the 'model' section must be a mapping")
        arch = section.get("architecture") or {}

        def text(key: str, default: str | None = None) -> str | None:
            value = section.get(key)
            return default if _is_open(value) else str(value)

        name = text("name")
        role = text("role")
        family = text("family")
        source = text("source")
        missing = [
            key
            for key, value in (
                ("name", name),
                ("role", role),
                ("family", family),
                ("source", source),
            )
            if value is None
        ]
        if missing:
            raise ModelConfigError(f"model config has no value for: {', '.join(missing)}")
        if role not in {"teacher", "student"}:
            raise ModelConfigError(f"role must be 'teacher' or 'student', got {role!r}")
        vocab = arch.get("vocab_size")
        return cls(
            name=str(name),
            role=str(role),
            family=str(family),
            source=str(source),
            revision=text("revision"),
            sha_seen=text("sha_seen"),
            adapter=text("adapter"),
            adapter_revision=text("adapter_revision"),
            dtype=str(text("dtype", DEFAULT_DTYPE)),
            status=str(section.get("status", CANDIDATE_STATUS)),
            tokenizer_family=text("tokenizer_family") or _opt(arch.get("tokenizer_family")),
            vocab_size=int(vocab) if isinstance(vocab, int) else None,
        )

    @property
    def is_candidate(self) -> bool:
        return self.status.strip().upper().startswith("CANDIDATE")

    @property
    def resolved_revision(self) -> str | None:
        """The revision to load: the pinned one, otherwise the one seen for a candidate."""
        return self.revision or self.sha_seen

    @property
    def needs_remote_code(self) -> bool:
        return self.family in REMOTE_CODE_FAMILIES


def _opt(value: Any) -> str | None:
    return None if _is_open(value) else str(value)


def check_loadable(spec: ModelSpec, *, allow_candidate: bool = False) -> None:
    """Raise :class:`ModelConfigError` unless the spec names weights that can be loaded.

    A candidate (not yet decided) is refused unless ``allow_candidate`` is set. A model
    with no pinned or seen revision is always refused, and a local path is exempt.
    """
    if spec.is_candidate and not allow_candidate:
        raise ModelConfigError(
            f"{spec.name} is still a candidate; pass allow_candidate=True only in the "
            "student-selection runs"
        )
    is_local = Path(spec.source).expanduser().exists()
    if not is_local and spec.resolved_revision is None:
        raise ModelConfigError(
            f"{spec.name} has no pinned revision; refusing to load unpinned weights"
        )
    if spec.adapter and not spec.adapter_revision and not Path(spec.adapter).expanduser().exists():
        raise ModelConfigError(  # a local adapter (our teacher's): checked before any download
            f"the adapter of {spec.name} is not at {spec.adapter}: attach the RC-TT1 hand-off "
            "(the teacher adapter) with --from <path>"
        )


_QUANTIZATION_KEYS = {"bits", "quant_type", "compute_dtype", "double_quant"}


def _quantization_config(quantization: Mapping[str, Any], torch_module: Any) -> Any:
    """Translate ``{"bits": 4|8, ...}`` into a bitsandbytes config."""
    unknown = set(quantization) - _QUANTIZATION_KEYS
    if unknown:
        raise ModelConfigError(f"unknown quantization keys: {sorted(unknown)}")
    from transformers import BitsAndBytesConfig

    bits = int(quantization.get("bits", 4))
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if bits == 4:
        compute = getattr(torch_module, str(quantization.get("compute_dtype", DEFAULT_DTYPE)))
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quantization.get("double_quant", False)),
            bnb_4bit_compute_dtype=compute,
        )
    raise ModelConfigError(f"bits must be 4 or 8, got {bits}")


GPU_CLEAN_LIMIT = 3 * 2**29
"""Memory a GPU may already have in use when a model is loaded (1.5 GiB: this process's CUDA
context and allocator slack). More means a model of an earlier stage, or another process, is
still there, and the new model would be offloaded or run out of memory (RC-G2, 2026-11-13)."""
GPU_RESERVE = 3 * 2**29
"""Room kept free on every GPU when a model is split over several (activations, workspaces)."""


class GPUPlacementError(RuntimeError):
    """A model load would not sit entirely on the GPU(s): a dirty GPU, or offloaded tensors.
    A run whose model is offloaded is invalid (its memory and latency would be wrong)."""


def gpu_memory_check(
    what: str,
    *,
    cuda: bool | None = None,
    mem_info: Callable[[int], tuple[int, int]] | None = None,
    count: int | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Free this process's cached GPU memory, print every GPU's free memory, and refuse to load
    ``what`` when a GPU already has more than :data:`GPU_CLEAN_LIMIT` in use."""
    if cuda is None or mem_info is None or count is None:
        try:
            import torch
        except ImportError:
            return
        cuda = torch.cuda.is_available() if cuda is None else cuda
        if not cuda:
            return
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        mem_info = mem_info or torch.cuda.mem_get_info
        count = torch.cuda.device_count() if count is None else count
    if not cuda:
        return
    dirty = []
    for index in range(count):
        free, total = mem_info(index)
        gib = (free / 2**30, total / 2**30)
        log(f"[{what}] GPU{index}: {gib[0]:.2f} of {gib[1]:.2f} GiB free before loading")
        if total - free > GPU_CLEAN_LIMIT:
            dirty.append(f"GPU{index} has {(total - free) / 2**30:.2f} GiB in use")
    if dirty:
        raise GPUPlacementError(
            f"{what}: {', '.join(dirty)} before the load (a model of an earlier stage or another "
            "process is still there); refusing to load, the model would be offloaded"
        )


def explicit_device_map(
    device_map: Any,
    spec: ModelSpec | None = None,
    *,
    cuda: bool | None = None,
    count: int | None = None,
    planner: Callable[[], dict[str, int]] | None = None,
) -> Any:
    """The device map a load uses: never "auto". One GPU: ``{"": 0}``; several: an explicit
    balanced plan of every module (``placement.plan_for_model``); a given mapping is kept but
    may name GPUs only. Without CUDA (the tests) the given map is kept.

    Raises:
        GPUPlacementError: the map names the CPU or the disk while a GPU is there.
    """
    if cuda is None or count is None:
        import torch

        cuda = torch.cuda.is_available() if cuda is None else cuda
        count = torch.cuda.device_count() if count is None else count
    if not cuda:
        return device_map
    if isinstance(device_map, Mapping):
        bad = {k: v for k, v in device_map.items() if str(v) in ("cpu", "disk", "meta")}
        if bad:
            raise GPUPlacementError(f"the device map puts modules off the GPU: {bad}")
        return dict(device_map)
    if device_map not in (None, "auto", "balanced"):
        raise GPUPlacementError(f"device_map {device_map!r} while a GPU is there: use the GPU")
    if count == 1:
        return {"": 0}
    if planner is None:
        from medcompress.compression.quantization.placement import plan_for_model

        if spec is None:
            raise GPUPlacementError("a model split over several GPUs needs its spec to be planned")
        return plan_for_model(spec, reserve=GPU_RESERVE)
    return planner()


def require_on_gpu(module: Any, what: str, *, cuda: bool | None = None) -> None:
    """Raise :class:`GPUPlacementError` if any parameter is on the CPU or the meta device, any
    buffer is on the meta device, or the device map names the CPU or the disk (an offload).
    Without CUDA (the tests) nothing is checked unless ``cuda=True``."""
    if cuda is None:
        try:
            import torch
        except ImportError:
            return
        cuda = torch.cuda.is_available()
    if not cuda:
        return
    off = [f"{n} on {p.device}" for n, p in module.named_parameters()
           if p.device.type in ("cpu", "meta")]  # fmt: skip
    off += [f"{n} (buffer) on meta" for n, b in module.named_buffers() if b.device.type == "meta"]
    mapped = getattr(module, "hf_device_map", None) or {}
    off += [f"{n} mapped to {d}" for n, d in mapped.items() if str(d) in ("cpu", "disk")]
    if off:
        raise GPUPlacementError(
            f"{what}: {len(off)} tensors are not on the GPU (offloaded), e.g. {off[:3]}; a run "
            "that offloads is invalid (its memory and latency would be wrong)"
        )


PEAKS_ENV = "MEDCOMPRESS_REPORT_PEAKS"
"""Set to ``1`` (RC-SMOKE) to print each load's and each stage's peak GPU memory."""
_STAGE_PEAK: dict[int, int] = {}


def _reporting() -> Any:
    """torch when peaks are reported and CUDA is there, else None."""
    import os

    if os.environ.get(PEAKS_ENV) != "1":
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def _keep_stage_peak(torch: Any) -> None:
    for i in range(torch.cuda.device_count()):
        _STAGE_PEAK[i] = max(_STAGE_PEAK.get(i, 0), int(torch.cuda.max_memory_allocated(i)))


def peak_before_load() -> None:
    """Keep the stage's peak so far, then reset the counters (this load's own peak follows)."""
    torch = _reporting()
    if torch is not None:
        _keep_stage_peak(torch)
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def peak_after_load(name: str) -> None:
    """Print ``MEDCOMPRESS_LOAD_PEAK <name> gpu0=<bytes> ...`` (the load's own peak)."""
    torch = _reporting()
    if torch is not None:
        sizes = " ".join(
            f"gpu{i}={int(torch.cuda.max_memory_allocated(i))}"
            for i in range(torch.cuda.device_count())
        )
        print(f"MEDCOMPRESS_LOAD_PEAK {name.replace(' ', '_')} {sizes}", flush=True)


def print_stage_peak(stage: str) -> None:
    """Print ``MEDCOMPRESS_STAGE_PEAK <stage> gpu0=<bytes> ...`` (the whole stage's peak)."""
    torch = _reporting()
    if torch is not None:
        _keep_stage_peak(torch)
        sizes = " ".join(f"gpu{i}={b}" for i, b in sorted(_STAGE_PEAK.items()))
        print(f"MEDCOMPRESS_STAGE_PEAK {stage} {sizes}", flush=True)


def release_model(model: Any) -> None:
    """Free a loaded model's GPU memory now: move every parameter and buffer to the meta device,
    then collect garbage and empty the CUDA cache (2026-11-13).

    Moving the tensors frees their storage even if a stray reference to the module object
    survives somewhere (RC-G2 session 1: after ``del base`` the student's 7.2 GiB stayed on GPU 0
    and the next load of the same stage was refused). The model cannot be used afterwards.
    """
    import gc

    module = getattr(model, "module", model)
    try:
        import torch
    except ImportError:  # pragma: no cover - a model implies torch
        return
    with torch.no_grad():
        module.to("meta")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(
    spec: ModelSpec,
    *,
    device_map: str | Mapping[str, Any] | None = "auto",
    quantization: Mapping[str, Any] | None = None,
    allow_candidate: bool = False,
    allow_remote_code: bool = False,
    image_longest_side: int | None = None,
    max_pixels: int | None = None,
    batch_size: int = 8,
    mlp_channel_group_size: int = 128,
    max_memory: Mapping[Any, Any] | None = None,
) -> VLMWrapper:
    """Load a teacher or student as a :class:`VLMWrapper`.

    The adapter named in the spec (if any) is attached. It is merged into the weights
    so the returned model has no adapter layers, except when the weights are
    quantized: merging into 4-bit or 8-bit weights is lossy, so the adapter stays
    separate there.

    Args:
        device_map: passed to ``from_pretrained`` (``"auto"`` splits across GPUs).
        quantization: ``{"bits": 4 or 8, ...}`` for bitsandbytes loading, or None.
        allow_candidate: accept a model whose status is still a candidate.
        allow_remote_code: run model code from the repository (needed by some families).
        image_longest_side: resize images so their longest side is at most this.
        max_pixels: forwarded to the image processor of families that support it.
        max_memory: ``{gpu: bytes}`` caps for the weights of a split model (passed to
            ``from_pretrained``); E9 keeps room on each GPU for GPTQ (2026-11-12).

    Used by: every experiment.
    """
    check_loadable(spec, allow_candidate=allow_candidate)
    if spec.needs_remote_code and not allow_remote_code:
        raise ModelConfigError(
            f"{spec.name} ({spec.family}) needs code from its model repository; "
            "pass allow_remote_code=True after reading that code"
        )
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from medcompress.models.hf import HFVLMWrapper

    dtype = getattr(torch, spec.dtype, None)
    if dtype is None:
        raise ModelConfigError(f"unknown dtype {spec.dtype!r}")
    revision = spec.resolved_revision
    common: dict[str, Any] = {"trust_remote_code": allow_remote_code}
    if revision is not None and not Path(spec.source).expanduser().exists():
        common["revision"] = revision
    gpu_memory_check(spec.name)  # a clean GPU before every load (2026-11-13)
    peak_before_load()
    device_map = explicit_device_map(device_map, spec)  # never "auto": no silent offload
    model_kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device_map, **common}
    if max_memory is not None:
        model_kwargs["max_memory"] = dict(max_memory)
    if quantization:
        model_kwargs["quantization_config"] = _quantization_config(quantization, torch)
    processor_kwargs = dict(common)
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    module = AutoModelForImageTextToText.from_pretrained(spec.source, **model_kwargs)
    processor = AutoProcessor.from_pretrained(spec.source, **processor_kwargs)
    wrapper = HFVLMWrapper(
        module,
        processor,
        spec,
        image_longest_side=image_longest_side,
        batch_size=batch_size,
        mlp_channel_group_size=mlp_channel_group_size,
    )
    require_on_gpu(module, spec.name)  # before the adapter: an offload is named, not hit
    if spec.adapter:
        adapter_kwargs = {"revision": spec.adapter_revision} if spec.adapter_revision else {}
        wrapper = load_adapter(wrapper, spec.adapter, merge=not quantization, **adapter_kwargs)
        require_on_gpu(wrapper.module, f"{spec.name} with its adapter")
    peak_after_load(spec.name)
    return wrapper


def remap_adapter_key(key: str, module_names: Sequence[str]) -> str:
    """Rename one LoRA state-dict key so it points at a module of the current model.

    Adapters saved with an older transformers version name modules differently
    (``model.layers.0...`` instead of ``model.language_model.layers.0...``). The key
    ``base_model.model.<path>.lora_A.weight`` is matched on the longest tail of
    ``<path>`` that picks out exactly one module in ``module_names``. A key whose
    module already exists is returned unchanged.

    Raises:
        ValueError: if no module, or more than one, matches.
    """
    prefix = "base_model.model."
    if not key.startswith(prefix) or ".lora_" not in key:
        raise ValueError(f"not a LoRA key: {key}")
    path, rest = key[len(prefix) :].split(".lora_", 1)
    names = set(module_names)
    if path in names:
        return key
    parts = path.split(".")
    for start in range(1, len(parts)):
        tail = ".".join(parts[start:])
        matches = [n for n in names if n == tail or n.endswith("." + tail)]
        if len(matches) == 1:
            return f"{prefix}{matches[0]}.lora_{rest}"
        if len(matches) > 1:
            raise ValueError(f"{key}: {len(matches)} modules end with {tail!r}")
    raise ValueError(f"{key}: no module of this model matches")


_VISION_PARTS = ("visual", "vision")


def untrained_adapter_modules(module: Any) -> list[str]:
    """Names of LoRA modules outside the vision tower whose ``lora_B`` weights are all zero.

    A trained adapter has non-zero ``lora_B`` matrices. All zeros after loading means
    the adapter file did not reach that module (for instance because the module
    names in the file no longer match the model), and the adapter would do nothing
    there. Vision-tower modules are skipped: a language adapter has no weights for them.
    """
    import re

    pattern = re.compile(r"\.lora_B\.[^.]+$")
    untrained = []
    for name, sub in module.named_modules():
        if pattern.search(name) and not any(part in name for part in _VISION_PARTS):
            weight = getattr(sub, "weight", None)
            if weight is not None and float(weight.detach().abs().sum()) == 0.0:
                untrained.append(name)
    return untrained


def _reload_with_renamed_keys(
    peft_model: Any, adapter: Path | str, kwargs: Mapping[str, Any]
) -> None:
    """Load the adapter file again with its module names mapped onto this model."""
    import re

    from peft import set_peft_model_state_dict
    from peft.utils import load_peft_weights

    state = load_peft_weights(str(adapter), device="cpu", **kwargs)
    targets = [
        name for name, sub in peft_model.base_model.model.named_modules() if hasattr(sub, "lora_A")
    ]
    renamed = {remap_adapter_key(key, targets): value for key, value in state.items()}
    # the file stores keys without the adapter name; peft expects exactly that form
    renamed = {re.sub(r"\.lora_([AB])\.default\.", r".lora_\1.", k): v for k, v in renamed.items()}
    set_peft_model_state_dict(peft_model, renamed)


def load_adapter(
    model: VLMWrapper,
    adapter: Path | str,
    *,
    merge: bool = False,
    revision: str | None = None,
    check_trained: bool = True,
) -> VLMWrapper:
    """Attach a LoRA adapter (a published domain adapter, or an E1/E2 training result).

    With ``merge=True`` the adapter is folded into the base weights and the adapter
    layers are removed. Merging into 4-bit or 8-bit weights is refused: it is lossy.
    With ``check_trained`` (the default) loading fails if a language-model adapter
    module is still all zero, so a silently ignored adapter cannot go unnoticed.
    """
    from peft import PeftModel

    module = getattr(model, "module", None)
    if module is None:
        raise TypeError("load_adapter needs a wrapper with a 'module' attribute")
    if merge and getattr(module, "quantization_method", None) is not None:
        raise ModelConfigError("refusing to merge an adapter into quantized weights")
    kwargs = {"revision": revision} if revision else {}
    peft_model = PeftModel.from_pretrained(module, str(adapter), **kwargs)
    if check_trained and untrained_adapter_modules(peft_model):
        _reload_with_renamed_keys(peft_model, adapter, kwargs)
        untrained = untrained_adapter_modules(peft_model)
        if untrained:
            raise ModelConfigError(
                f"adapter {adapter} left {len(untrained)} language-model modules untrained "
                f"(first: {untrained[0]}); its weights did not load"
            )
    model.module = peft_model.merge_and_unload() if merge else peft_model  # type: ignore[attr-defined]
    return model


def check_shared_tokenizer(teacher: VLMWrapper, student: VLMWrapper) -> None:
    """Raise ValueError if teacher and student tokenizers differ.

    Used by: E2, E4 (distillation), recovery for E5-E7 and A1-A6.
    """
    a, b = teacher.tokenizer_fingerprint(), student.tokenizer_fingerprint()
    if a != b:
        raise ValueError(
            "teacher and student tokenizers differ, so cached teacher token "
            f"probabilities cannot be used for distillation ({a[:18]}... vs {b[:18]}...)"
        )
