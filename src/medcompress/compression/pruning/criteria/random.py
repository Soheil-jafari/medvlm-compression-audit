"""Random removal: lower-bound control at the same budget."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


@register_criterion("random")
class RandomCriterion(PruningCriterion):
    """Seeded random scores over the same unit set as E7. Used by: A3."""

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return i.i.d. uniform scores drawn with the run seed (same units as E7)."""
        raise NotImplementedError("TODO: seeded random scores (A3).")
