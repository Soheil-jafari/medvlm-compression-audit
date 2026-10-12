"""Random criterion: seeded random scores on the same units (ablation A3)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


@register_criterion("random")
class RandomCriterion(PruningCriterion):
    """Score each unit with a seeded uniform random number. Used by: A3.

    The score of a unit depends only on the seed and the unit id, so the same seed gives
    the same scores for any subset of units, and the model and the data are never used.
    """

    needs_calibration = False

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return ``{unit_id: u}`` with ``u`` uniform in [0, 1) from ``Random(seed:unit_id)``."""
        return {u.unit_id: random.Random(f"{self.seed}:{u.unit_id}").random() for u in units}
