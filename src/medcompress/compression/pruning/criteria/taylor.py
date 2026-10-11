"""Gradient (first-order Taylor) criterion: |sum of weight x gradient| per unit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


@register_criterion("taylor")
class TaylorCriterion(PruningCriterion):
    """First-order Taylor importance of the answer loss. Used by: E5:taylor."""

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return Taylor importance per unit from the next-token loss on calibration QA."""
        raise NotImplementedError("TODO: first-order Taylor importance (E5:taylor).")
