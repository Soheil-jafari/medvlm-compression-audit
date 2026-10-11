"""Output-similarity layer pruning (Khan et al. style, arXiv:2507.21976).

Scores each decoder layer by how much it changes its input hidden states on
calibration data (layers whose output is most similar to their input are least
important). TODO: check Khan et al. for the exact similarity measure and whether
layers are removed one-shot or iteratively before implementing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


@register_criterion("output_similarity")
class OutputSimilarityCriterion(PruningCriterion):
    """Layer-level criterion; only ``decoder_layer`` units are valid. Used by: E6."""

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return 1 - similarity(layer input, layer output) per decoder layer."""
        raise NotImplementedError("TODO: input/output similarity per decoder layer (E6).")
