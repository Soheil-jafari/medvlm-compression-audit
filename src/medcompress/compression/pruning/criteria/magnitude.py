"""Magnitude criterion: importance from the weights each unit holds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion
from medcompress.compression.pruning.criteria.common import unit_slices

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


@register_criterion("magnitude")
class MagnitudeCriterion(PruningCriterion):
    """Score each unit by the L1 norm of the weights that are deleted with it. Used by: E5.

    Data-free. The score is a sum over weights, so it grows with the size of the unit;
    dividing by the parameter count (``pruning.score_normalisation: per_parameter``) turns it
    into the mean absolute weight.
    """

    needs_calibration = False

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return ``{unit_id: sum |w|}`` over the weights and biases of the unit."""
        import torch

        out: dict[str, float] = {}
        with torch.no_grad():
            for unit in units:
                total = 0.0
                for part in unit_slices(model, unit):
                    total += float(part.weight.detach().abs().float().sum())
                    if part.bias is not None:
                        total += float(part.bias.detach().abs().float().sum())
                out[unit.unit_id] = total
        return out
