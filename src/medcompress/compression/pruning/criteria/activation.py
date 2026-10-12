"""Activation criterion: Wanda-style importance, summed over the weights of a unit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from medcompress.compression.pruning import surgery
from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion
from medcompress.compression.pruning.criteria.common import unit_slices

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

STATISTICS: tuple[str, ...] = ("attn_in", "attn_out", "mlp_in", "mlp_act")


def collect_input_norms(
    model: Any, calibration: Sequence[VQASample], data_root: Any
) -> dict[int, dict[str, Any]]:
    """L2 norm over all calibration tokens of every input feature of the decoder matrices.

    Returns ``{layer_index: {statistic: vector}}`` with the statistics ``attn_in`` (input of
    q/k/v), ``attn_out`` (input of the output projection), ``mlp_in`` (input of gate/up) and
    ``mlp_act`` (input of the down projection). Each calibration row is one forward pass
    on its image and question.
    """
    from pathlib import Path

    import torch

    module = model.module
    stack = surgery.find_decoder_stack(module)
    sums: dict[int, dict[str, Any]] = {i: {} for i in range(len(stack.layers))}

    def collector(layer: int, name: str) -> Any:
        def pre_hook(_module: Any, args: tuple) -> None:
            x = args[0].detach().float()
            total = x.pow(2).sum(dim=tuple(range(x.dim() - 1)))
            sums[layer][name] = sums[layer].get(name, 0) + total

        return pre_hook

    handles = []
    for i, layer in enumerate(stack.layers):
        handles += [
            layer.self_attn.q_proj.register_forward_pre_hook(collector(i, "attn_in")),
            layer.self_attn.o_proj.register_forward_pre_hook(collector(i, "attn_out")),
            layer.mlp.gate_proj.register_forward_pre_hook(collector(i, "mlp_in")),
            layer.mlp.down_proj.register_forward_pre_hook(collector(i, "mlp_act")),
        ]
    try:
        with torch.inference_mode():
            for sample in calibration:
                inputs = model.build_inputs(
                    [Path(data_root) / sample.image_path], [sample.question]
                )
                module(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    return {i: {k: v.sqrt() for k, v in stats.items()} for i, stats in sums.items()}


@register_criterion("activation")
class ActivationCriterion(PruningCriterion):
    """Wanda-style structured importance. Used by: E5.

    Wanda scores a weight by ``|W_ij| * ||X_j||_2``, the weight times the norm of the input
    feature it multiplies over the calibration tokens. A unit's score is the sum of that
    quantity over the weights deleted with it (norm parameters, which multiply no input, count
    with their absolute value).
    """

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return ``{unit_id: sum |w| * ||x||}`` over the unit's weights."""
        import torch

        if not calibration:
            raise ValueError("the activation criterion needs calibration samples")
        norms = collect_input_norms(model, calibration, self.data_root)
        out: dict[str, float] = {}
        with torch.no_grad():
            for unit in units:
                total = 0.0
                for part in unit_slices(model, unit):
                    weight = part.weight.detach().abs().float()
                    if part.stat is None:
                        total += float(weight.sum())
                        continue
                    norm = norms[unit.layer_index][part.stat]
                    if part.columns is not None:
                        norm = norm[part.columns]
                    total += float((weight * norm.to(weight.device)[None, :]).sum())
                    if part.bias is not None:
                        total += float(part.bias.detach().abs().float().sum())
                out[unit.unit_id] = total
        return out
