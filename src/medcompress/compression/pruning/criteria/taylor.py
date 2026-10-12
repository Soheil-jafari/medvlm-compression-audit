"""Taylor criterion: first-order effect of removing a unit on the answer loss."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from medcompress.compression.pruning import scoring
from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


@register_criterion("taylor")
class TaylorCriterion(PruningCriterion):
    """First-order Taylor importance with the ordinary answer loss. Used by: E5.

    Each unit has a gate multiplying its output. For every calibration row the
    cross-entropy of the reference answer is differentiated with respect to the gates; the
    importance of a unit is the mean over rows of ``|dL/dg|``, the first-order change of the
    loss when the unit is removed. It is the generic counterpart of the clinical criterion
    (same machinery, but the loss of the whole answer instead of the finding probes).
    """

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return ``{unit_id: mean_rows |dL/dg|}`` on the calibration rows."""
        from medcompress.compression.distillation import batching
        from medcompress.compression.distillation.losses import answer_ce_loss
        from medcompress.compression.distillation.offline_kd import forward_answer_logits

        if not calibration:
            raise ValueError("the Taylor criterion needs calibration samples")
        scale = (self.config.get("pruning", {}).get("taylor") or {}).get("loss_scale")
        pad = model.tokenizer.pad_token_id
        device = model.module.device
        total = {u.unit_id: 0.0 for u in units}
        for sample in calibration:
            prompt = batching.prompt_inputs(
                model, self.data_root / Path(sample.image_path), sample.question
            )
            ids = batching.ground_truth_ids(model, sample.answer)
            batch = batching.move(batching.assemble([prompt], [ids], pad), device)

            def step(batch: batching.Batch = batch) -> object:
                return answer_ce_loss(forward_answer_logits(model.module, batch), batch.targets)

            grads = scoring.accumulate_gate_gradients(model, units, [step], loss_scale=scale)
            for key, value in grads.items():
                total[key] += abs(value) / len(calibration)
        return total
