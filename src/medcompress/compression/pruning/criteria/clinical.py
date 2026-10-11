"""Clinically weighted structured-pruning criterion (OUR METHOD; experiments E7, E8, A1-A6).

Goal
----
Rank prunable units by how much the model's ability to answer fixed clinical
finding probes depends on them, weighting rare and clinically important findings
more, so that structured pruning preserves small and rare findings that average
accuracy hides.

Definitions
-----------
U       Candidate units j: decoder layers, attention head groups (respecting
        grouped KV heads), MLP channel groups; vision encoder blocks optional;
        the projector is always kept. See ``compression/pruning/units.py``.
C       Finding list from ``configs/domain/<domain>.yaml`` (domain undecided).
X_cal   Calibration images drawn from the training split
        (``data/splits.make_calibration_subset``); size N (A4: 32/64/128/256).
q_c     Fixed probe question for finding c, from the domain probe template, e.g.
        "Is there a <finding> in this image? Answer yes or no."
y(c,x)  Correct answer ("yes"/"no") for finding c on image x, from labels.
theta   Parameters of the E2 student (the starting point of every pruning arm).
theta\\j Same model with unit j ablated (zeroed / bypassed for scoring only;
        the final model removes units physically, see ``apply.py``).
p(a | x, q)  Model probability of answer token(s) a after image x and question q,
        read from a single forward pass (``VLMWrapper.score_yes_no``); no free
        generation.

Score
-----
    I_j = sum_{c in C} w_c * (1 / |X_cal|) * sum_{x in X_cal} Delta_j(c, x)

Delta variants (A5 compares them):
    "logprob_drop" (default):
        Delta_j(c, x) = log p_theta(y(c,x) | x, q_c) - log p_{theta\\j}(y(c,x) | x, q_c)
        i.e. the drop in log-probability of the CORRECT answer when j is ablated.
    "kl" (alternative):
        Delta_j(c, x) = KL( p_theta(. | x, q_c) || p_{theta\\j}(. | x, q_c) )
        i.e. change in prediction regardless of correctness.
        TODO: decide whether the distribution is over {yes, no} renormalised or
        over the full vocabulary at the answer position.

Weights (from ``configs/preregistration/finding_weights.yaml``, frozen before
any E7 run):
    w_c = min(1 / prevalence_c, cap) * tier_multiplier(tier_c)
    prevalence_c: positive rate of finding c in the training split.
    cap:          upper bound on the inverse-prevalence factor (pre-registered).
    tier:         clinical importance tier of c (pre-registered table).
    TODO: decide and pre-register whether weights are normalised (e.g. to sum to |C|).

Higher I_j means more important. Units are removed in ascending order of I_j.
TODO: decide whether negative Delta values (ablation helps) are kept or clipped at 0.

Procedure
---------
1. Probe:    build probes for every (x in X_cal, c in C) (``data/probes.py``) and
             record baseline log-probabilities with theta.
2. Ablate:   for each unit j, ablate j and re-score all probes (one forward pass
             per batch; cache baseline outputs).
3. Score:    compute I_j as above.
4. Rank:     sort units by I_j ascending (deterministic tie-breaking).
5. Remove:   physically remove lowest-scoring units until the MEASURED budget
             (B1/B2/B3) is met (``budgets.search_matched_budget``), respecting
             structural constraints (``units.validate_removal_set``).
6. Recover:  shared recovery training, identical for every pruning arm
             (``compression/recovery.py``: LoRA + offline KD from cached teacher
             outputs, same data subset and steps).
7. Audit:    mask audit (``evaluation/image_reliance.mask_audit``): does the
             pruned model still change its answer when the lesion is masked more
             than when an area-matched random region is masked?

Ablation hooks (config keys under ``pruning.clinical``)
-------------------------------------------------------
A1  ``weighting: uniform``          -> w_c = 1 for all c.
A2  ``weighting: rarity_only``      -> w_c = min(1 / prevalence_c, cap).
A3  ``pruning.criterion: random``   -> handled by ``criteria/random.py`` on the
                                       same unit set.
A4  ``calibration_size: 32|64|128|256`` -> nested calibration subsets.
A5  ``delta: logprob_drop | kl``    -> Delta definition.
A6  ``units.types``                 -> [decoder_layer] vs
                                       [decoder_layer, attention_head, mlp_channel_group].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.data.probes import FindingProbe
    from medcompress.models.base import VLMWrapper, YesNoScore

DELTA_VARIANTS: tuple[str, ...] = ("logprob_drop", "kl")
WEIGHTING_SCHEMES: tuple[str, ...] = ("clinical", "uniform", "rarity_only")


@dataclass(frozen=True)
class FindingWeight:
    """One row of the pre-registered weight table."""

    finding: str
    prevalence: float
    importance_tier: str
    cap: float
    final_weight: float


def load_finding_weights(table: Mapping[str, object]) -> list[FindingWeight]:
    """Parse ``configs/preregistration/finding_weights.yaml``.

    Must refuse a table whose ``frozen_on`` is still ``TODO`` when running a
    final (non-debug) experiment, so weights cannot be tuned after seeing results.

    Used by: E7, E8, A2, A4-A6.
    """
    raise NotImplementedError("TODO: parse and validate the frozen weight table.")


def compute_weights(
    findings: Sequence[str],
    table: Sequence[FindingWeight],
    scheme: str = "clinical",
) -> dict[str, float]:
    """Return w_c per finding for a weighting scheme.

    ``clinical``:    min(1 / prevalence_c, cap) * tier_multiplier(tier_c)  (E7)
    ``uniform``:     1                                                      (A1)
    ``rarity_only``: min(1 / prevalence_c, cap)                             (A2)

    Must check that the recomputed clinical weights match ``final_weight`` in the
    frozen table.
    """
    raise NotImplementedError("TODO: compute w_c for clinical / uniform / rarity_only.")


def delta(
    baseline: Sequence[YesNoScore],
    ablated: Sequence[YesNoScore],
    probes: Sequence[FindingProbe],
    variant: str = "logprob_drop",
) -> dict[tuple[str, str], float]:
    """Return Delta_j(c, x) keyed by (finding, sample_id) for one ablated unit.

    ``logprob_drop``: log p_theta(y) - log p_{theta\\j}(y) for the correct answer y.
    ``kl``:           KL(p_theta || p_{theta\\j}) over the answer distribution (A5).
    """
    raise NotImplementedError("TODO: Delta_j(c, x) for logprob_drop / kl (E7, A5).")


def aggregate_importance(
    deltas: Mapping[tuple[str, str], float], weights: Mapping[str, float]
) -> float:
    """Return I_j = sum_c w_c * mean_x Delta_j(c, x) for one unit."""
    raise NotImplementedError("TODO: weighted sum over findings of mean Delta over images.")


@register_criterion("clinical")
class ClinicalCriterion(PruningCriterion):
    """Clinically weighted probe-based importance (see module docstring).

    Config keys (``pruning.clinical``): ``delta``, ``weighting``, ``weights_table``,
    ``calibration_size``, ``probe_template`` (from the domain config),
    ``ablation_method``, ``batch_size``.

    Used by: E7 (all budgets), E8 (via E7), A1, A2, A4, A5, A6.
    """

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Run steps 1-3 of the procedure and return ``{unit_id: I_j}``."""
        raise NotImplementedError(
            "TODO: probe -> ablate each unit -> Delta -> weighted importance I_j (E7, A1-A6)."
        )

    def ablate(self, model: VLMWrapper, unit: PrunableUnit) -> VLMWrapper:
        """Temporarily disable one unit for scoring (context manager or copy).

        Ablation (zeroing outputs / bypassing a layer) is only used for scoring;
        the final model removes units physically.
        """
        raise NotImplementedError("TODO: reversible ablation of a single unit for scoring.")
