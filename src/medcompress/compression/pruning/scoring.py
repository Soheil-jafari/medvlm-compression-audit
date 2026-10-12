"""Machinery shared by the pruning criteria: ablating a unit and estimating its effect.

Two ways to ask "how much does the model depend on unit j" are provided:

* **Exact ablation** (``ablate``): the unit is switched off with a forward hook (the layer
  becomes the identity, a GQA group's attention output or an MLP channel group's activation
  is zeroed) and the objective is measured again. For the Qwen decoder this equals removing
  the unit physically, so the score is the effect of the pruning step itself.
* **First-order estimate** (``UnitGates``): every unit gets a gate ``g_j = 1`` multiplying its
  output. The derivative of an objective with respect to ``g_j`` at ``g_j = 1`` is the first-order
  change of the objective when the unit is switched off, and one backward pass gives it for
  every unit at once. It costs about three forward passes in total instead of one pass per
  unit, and is accurate for small units (an MLP channel group) and rough for large ones (a
  layer).

Both give an estimate of the same quantity, ``objective(model) - objective(model without j)``.
"""

from __future__ import annotations

import contextlib
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.compression.pruning import surgery
from medcompress.compression.pruning.units import PrunableUnit, UnitType

if TYPE_CHECKING:
    from medcompress.data.base import VQASample
    from medcompress.data.probes import FindingProbe
    from medcompress.evaluation.image_reliance import ImageProbe
    from medcompress.models.base import VLMWrapper, YesNoScore

ABLATION_METHODS: tuple[str, ...] = ("exact", "first_order")


# ---------------------------------------------------------------------- ablation


@contextlib.contextmanager
def ablate(model: VLMWrapper | Any, unit: PrunableUnit) -> Iterator[None]:
    """Switch one unit off while the context is open (scoring only; pruning deletes weights).

    Raises:
        NotImplementedError: vision-encoder units are not wired.
    """
    module = getattr(model, "module", model)
    if unit.unit_type is UnitType.DECODER_LAYER:
        context = surgery.skip_layer(module, unit.layer_index)
    elif unit.unit_type is UnitType.ATTENTION_HEAD:
        if unit.index is None:
            raise ValueError(f"{unit.unit_id} has no group index")
        context = surgery.zero_kv_group(module, unit.layer_index, unit.index)
    elif unit.unit_type is UnitType.MLP_CHANNEL_GROUP:
        if unit.channel_start is None or unit.channel_count is None:
            raise ValueError(f"{unit.unit_id} has no channel range")
        channels = range(unit.channel_start, unit.channel_start + unit.channel_count)
        context = surgery.zero_mlp_channels(module, unit.layer_index, list(channels))
    else:
        raise NotImplementedError("vision-block ablation is off by default and not wired")
    with context:
        yield


# ------------------------------------------------------------------ first order


@contextlib.contextmanager
def parameters_frozen(module: Any) -> Iterator[None]:
    """Stop the weights from collecting gradients (only the unit gates should)."""
    flags = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


class UnitGates:
    """Gate variables on a set of units, applied through forward hooks.

    Gates are fp32 leaf tensors that start at 1. After a backward pass,
    :meth:`gradients` returns ``d objective / d gate`` for every unit. :meth:`set_value`
    changes a gate (used to check a gradient against a finite difference, or to switch a
    unit off with value 0).
    """

    def __init__(self, model: VLMWrapper | Any, units: Sequence[PrunableUnit]) -> None:
        import torch

        self.module = getattr(model, "module", model)
        self.stack = surgery.find_decoder_stack(self.module)
        device = next(self.module.parameters()).device
        self.units = list(units)
        self._slot: dict[str, tuple[Any, int]] = {}
        n_layers = len(self.stack.layers)
        self._layer_gate: Any = None
        self._kv_gate: dict[int, Any] = {}
        self._mlp_gate: dict[int, Any] = {}
        self._mlp_sizes: dict[int, list[int]] = {}
        types = {u.unit_type for u in self.units}
        if UnitType.DECODER_LAYER in types:
            self._layer_gate = torch.ones(n_layers, device=device, requires_grad=True)
        mlp_units: dict[int, list[PrunableUnit]] = {}
        for unit in self.units:
            if unit.unit_type is UnitType.DECODER_LAYER:
                self._slot[unit.unit_id] = (self._layer_gate, unit.layer_index)
            elif unit.unit_type is UnitType.ATTENTION_HEAD:
                if unit.layer_index not in self._kv_gate:
                    layer = self.stack.layers[unit.layer_index]
                    shape = surgery.layer_shape(layer, self.stack.config)
                    self._kv_gate[unit.layer_index] = torch.ones(
                        shape["num_kv_heads"], device=device, requires_grad=True
                    )
                self._slot[unit.unit_id] = (self._kv_gate[unit.layer_index], int(unit.index or 0))
            elif unit.unit_type is UnitType.MLP_CHANNEL_GROUP:
                mlp_units.setdefault(unit.layer_index, []).append(unit)
            else:
                raise NotImplementedError("vision-block gates are not wired")
        for layer_index, members in mlp_units.items():
            members.sort(key=lambda u: u.channel_start or 0)
            width = surgery.layer_shape(self.stack.layers[layer_index], self.stack.config)[
                "intermediate"
            ]
            start = 0
            for u in members:
                if u.channel_start != start or not u.channel_count:
                    raise ValueError(
                        f"layer {layer_index}: channel groups must tile the MLP width "
                        f"({u.unit_id} starts at {u.channel_start}, expected {start})"
                    )
                start += u.channel_count
            if start != width:
                raise ValueError(f"layer {layer_index}: channel groups cover {start} of {width}")
            gate = torch.ones(len(members), device=device, requires_grad=True)
            self._mlp_gate[layer_index] = gate
            self._mlp_sizes[layer_index] = [int(u.channel_count or 0) for u in members]
            for position, u in enumerate(members):
                self._slot[u.unit_id] = (gate, position)

    # ------------------------------------------------------------------ values

    def set_value(self, unit_id: str, value: float) -> None:
        """Set the gate of one unit."""
        import torch

        tensor, index = self._slot[unit_id]
        with torch.no_grad():
            tensor[index] = value

    def reset(self) -> None:
        """Gates back to 1 and gradients cleared."""
        import torch

        for tensor in self._tensors():
            with torch.no_grad():
                tensor.fill_(1.0)
            tensor.grad = None

    def _tensors(self) -> list[Any]:
        tensors = [self._layer_gate] if self._layer_gate is not None else []
        return tensors + list(self._kv_gate.values()) + list(self._mlp_gate.values())

    def gradients(self) -> dict[str, float]:
        """``{unit_id: d objective / d gate}`` after one or more backward passes."""
        out = {}
        for unit in self.units:
            tensor, index = self._slot[unit.unit_id]
            out[unit.unit_id] = 0.0 if tensor.grad is None else float(tensor.grad[index])
        return out

    # ------------------------------------------------------------------- hooks

    @contextlib.contextmanager
    def installed(self) -> Iterator[UnitGates]:
        """Attach the gates to the model for the duration of the context."""
        handles = []
        stack = self.stack
        if self._layer_gate is not None:
            for index, layer in enumerate(stack.layers):
                handles.append(
                    layer.register_forward_hook(self._layer_hook(index), with_kwargs=True)
                )
        for index, gate in self._kv_gate.items():
            layer = stack.layers[index]
            shape = surgery.layer_shape(layer, stack.config)
            width = shape["group_ratio"] * shape["head_dim"]
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    self._kv_hook(gate, shape["num_kv_heads"], width)
                )
            )
        for index, gate in self._mlp_gate.items():
            sizes = self._mlp_sizes[index]
            handles.append(
                stack.layers[index].mlp.down_proj.register_forward_pre_hook(
                    self._mlp_hook(gate, sizes)
                )
            )
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def _layer_hook(self, index: int) -> Callable[..., Any]:
        gates = self._layer_gate

        def hook(_module: Any, args: tuple, kwargs: dict, output: Any) -> Any:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            out = output[0] if isinstance(output, tuple) else output
            mixed = hidden + gates[index].to(out.dtype) * (out - hidden)
            return (mixed, *output[1:]) if isinstance(output, tuple) else mixed

        return hook

    @staticmethod
    def _kv_hook(gate: Any, n_groups: int, width: int) -> Callable[..., Any]:
        def pre_hook(_module: Any, args: tuple) -> tuple:
            x = args[0]
            shaped = x.reshape(*x.shape[:-1], n_groups, width)
            shaped = shaped * gate.to(x.dtype)[:, None]
            return (shaped.reshape(x.shape), *args[1:])

        return pre_hook

    @staticmethod
    def _mlp_hook(gate: Any, sizes: Sequence[int]) -> Callable[..., Any]:
        import torch

        counts = torch.tensor(list(sizes), device=gate.device)

        def pre_hook(_module: Any, args: tuple) -> tuple:
            x = args[0]
            per_channel = torch.repeat_interleave(gate, counts).to(x.dtype)
            return (x * per_channel, *args[1:])

        return pre_hook


def default_loss_scale(module: Any) -> float:
    """1 for fp32 and 1024 for half precision (the backward pass would otherwise underflow)."""
    import torch

    dtype = next(module.parameters()).dtype
    return 1.0 if dtype == torch.float32 else 1024.0


def accumulate_gate_gradients(
    model: VLMWrapper | Any,
    units: Sequence[PrunableUnit],
    steps: Sequence[Callable[[], Any]],
    *,
    loss_scale: float | None = None,
) -> dict[str, float]:
    """Sum the gate gradients of the objectives returned by ``steps``.

    Each element of ``steps`` runs one forward pass with the gates installed and returns a
    scalar tensor (the objective, to be differentiated). Half-precision models use a loss
    scale that is lowered by a factor of 16 whenever a gradient is not finite.

    Raises:
        ValueError: gradients stay non-finite or all zero at scale 1.
    """
    module = getattr(model, "module", model)
    scale = default_loss_scale(module) if loss_scale is None else float(loss_scale)
    while True:
        gates = UnitGates(model, units)
        with parameters_frozen(module), gates.installed():
            for step in steps:
                (step() * scale).backward()
        grads = gates.gradients()
        if all(math.isfinite(g) for g in grads.values()):
            if scale > 1.0 and not any(grads.values()):
                scale = max(1.0, scale / 16.0)  # nothing came through: try a smaller scale
                continue
            if not any(grads.values()):
                raise ValueError(
                    "every gate gradient is zero; the objective does not reach the units"
                )
            return {k: v / scale for k, v in grads.items()}
        if scale <= 1.0:
            raise ValueError("gate gradients are not finite even without loss scaling")
        scale = max(1.0, scale / 16.0)


# ---------------------------------------------------------------------- probes


def to_finding_probes(probes: Sequence[ImageProbe]) -> list[FindingProbe]:
    """Probes in the form used by the criterion's Delta (answer ``"yes"`` or ``"no"``)."""
    from medcompress.data.probes import FindingProbe

    return [
        FindingProbe(p.sample_id, p.image_path, p.finding, p.question, "yes" if p.present else "no")
        for p in probes
    ]


def balance_probes(
    probes: Sequence[ImageProbe], negatives_per_positive: float | None, seed: int
) -> list[ImageProbe]:
    """Keep all positive probes of a finding and at most ``k`` negatives per positive.

    ``None`` keeps everything. The negatives are a seeded random subset, so the same
    calibration images and seed give the same probe set for every criterion variant.
    """
    if negatives_per_positive is None:
        return list(probes)
    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must not be negative")
    by_finding: dict[str, list[ImageProbe]] = {}
    for probe in probes:
        by_finding.setdefault(probe.finding, []).append(probe)
    keep: set[int] = set()
    for finding in sorted(by_finding):
        group = by_finding[finding]
        positives = [p for p in group if p.present]
        negatives = [p for p in group if not p.present]
        limit = math.ceil(negatives_per_positive * len(positives))
        chosen = random.Random(f"{seed}:{finding}").sample(negatives, min(limit, len(negatives)))
        keep.update(id(p) for p in positives + chosen)
    return [p for p in probes if id(p) in keep]


def score_probe_images(
    model: VLMWrapper, probes: Sequence[FindingProbe], data_root: Path | str
) -> list[YesNoScore]:
    """Yes/no scores of every probe on its image (one forward pass per probe)."""
    root = Path(data_root)
    return model.score_yes_no([root / p.image_path for p in probes], [p.question for p in probes])


@dataclass(frozen=True)
class ProbeObjective:
    """The clinically weighted probe objective ``sum_c w_c * mean_x log p(correct answer)``.

    ``coefficients[i]`` is ``w_c / n_c`` for probe ``i`` of finding ``c``, so a weighted sum of
    the per-probe log-probabilities is the objective. Findings without probes carry nothing.
    """

    probes: tuple[FindingProbe, ...]
    coefficients: tuple[float, ...]

    @classmethod
    def build(cls, probes: Sequence[FindingProbe], weights: Mapping[str, float]) -> ProbeObjective:
        counts: dict[str, int] = {}
        for probe in probes:
            if probe.finding not in weights:
                raise ValueError(f"No weight for finding {probe.finding!r}")
            counts[probe.finding] = counts.get(probe.finding, 0) + 1
        return cls(tuple(probes), tuple(weights[p.finding] / counts[p.finding] for p in probes))


def probe_objective_steps(
    model: Any,
    objective: ProbeObjective,
    data_root: Path | str,
    batch_size: int,
) -> list[Callable[[], Any]]:
    """Forward passes (one per batch) whose sum is the probe objective, with gradients."""
    import torch

    root = Path(data_root)
    steps: list[Callable[[], Any]] = []
    for start in range(0, len(objective.probes), batch_size):
        part = objective.probes[start : start + batch_size]
        coef = objective.coefficients[start : start + batch_size]

        def step(part: Sequence[FindingProbe] = part, coef: Sequence[float] = coef) -> Any:
            yes, no = model.yes_no_logprobs(
                [root / p.image_path for p in part], [p.question for p in part], grad=True
            )
            is_yes = torch.tensor([p.answer == "yes" for p in part], device=yes.device)
            correct = torch.where(is_yes, yes, no)
            return (correct * torch.tensor(list(coef), device=yes.device)).sum()

        steps.append(step)
    return steps


def calibration_images(calibration: Sequence[VQASample]) -> list[str]:
    """Distinct image paths of the calibration rows, in order."""
    seen: dict[str, None] = {}
    for sample in calibration:
        seen.setdefault(sample.image_path)
    return list(seen)
