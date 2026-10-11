"""Magnitude criterion: importance from weight norms of each unit's parameters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


@register_criterion("magnitude")
class MagnitudeCriterion(PruningCriterion):
    """Score each unit by the norm of its weights (data-free). Used by: E5:magnitude."""

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return a weight-norm score per unit, normalised per unit type (norm TBD)."""
        raise NotImplementedError("TODO: weight-norm importance per unit (E5:magnitude).")
