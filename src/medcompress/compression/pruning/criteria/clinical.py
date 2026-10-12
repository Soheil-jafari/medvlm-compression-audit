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
        Implemented over {yes, no} renormalised, the only distribution the
        ``YesNoScore`` interface exposes (proposal for the pre-registration).

Weights (from ``configs/preregistration/finding_weights.yaml``, frozen before
any E7 run):
    w_c = min(1 / prevalence_c, cap) * tier_multiplier(tier_c)
    prevalence_c: positive rate of finding c in the training split.
    cap:          upper bound on the inverse-prevalence factor (pre-registered).
    tier:         clinical importance tier of c (pre-registered table).
    Normalisation (``none`` or ``sum_to_n``) is a parameter of ``compute_weights``;
    the choice is pre-registered.

Higher I_j means more important. Units are removed in ascending order of I_j.
Negative Delta values (ablation helps) are kept by default; clipping at 0 is a
parameter of ``aggregate_importance`` and the choice is pre-registered.

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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.data.probes import FindingProbe
    from medcompress.models.base import VLMWrapper, YesNoScore

DELTA_VARIANTS: tuple[str, ...] = ("logprob_drop", "kl")
WEIGHTING_SCHEMES: tuple[str, ...] = ("clinical", "uniform", "rarity_only")
NORMALISATIONS: tuple[str, ...] = ("none", "sum_to_n")


@dataclass(frozen=True)
class FindingWeight:
    """One row of the pre-registered weight table."""

    finding: str
    prevalence: float
    importance_tier: str
    cap: float
    final_weight: float


def _is_todo(value: Any) -> bool:
    return value is None or str(value).strip().upper().startswith("TODO")


def load_finding_weights(
    table: Mapping[str, object], *, require_frozen: bool = True
) -> list[FindingWeight]:
    """Parse ``configs/preregistration/finding_weights.yaml``.

    Refuses a table whose ``preregistration.frozen_on`` is still ``TODO`` unless
    ``require_frozen=False`` (debug runs only), so weights cannot be tuned after
    seeing results. Every row must have numeric ``prevalence`` in [0, 1],
    positive ``cap`` and non-negative ``final_weight``.

    Used by: E7, E8, A2, A4-A6.
    """
    prereg = table.get("preregistration") or {}
    if require_frozen and _is_todo(prereg.get("frozen_on")):  # type: ignore[union-attr]
        raise ValueError("Finding weights are not frozen (preregistration.frozen_on is TODO).")
    rows = table.get("findings") or []
    weights: list[FindingWeight] = []
    for row in rows:  # type: ignore[union-attr]
        if any(
            _is_todo(row.get(k))
            for k in ("finding", "prevalence", "importance_tier", "cap", "final_weight")
        ):
            raise ValueError(f"Incomplete weight row: {row}")
        weight = FindingWeight(
            finding=str(row["finding"]),
            prevalence=float(row["prevalence"]),
            importance_tier=str(row["importance_tier"]),
            cap=float(row["cap"]),
            final_weight=float(row["final_weight"]),
        )
        if not 0.0 <= weight.prevalence <= 1.0 or weight.cap <= 0 or weight.final_weight < 0:
            raise ValueError(f"Invalid weight row: {row}")
        weights.append(weight)
    if not weights:
        raise ValueError("The weight table has no findings.")
    if len({w.finding for w in weights}) != len(weights):
        raise ValueError("Duplicate findings in the weight table.")
    return weights


def load_tier_multipliers(table: Mapping[str, object]) -> dict[str, float]:
    """Return ``{tier: multiplier}`` from the ``weighting.tiers`` block of the table."""
    tiers = (table.get("weighting") or {}).get("tiers") or []  # type: ignore[union-attr]
    out: dict[str, float] = {}
    for row in tiers:
        if _is_todo(row.get("tier")) or _is_todo(row.get("multiplier")):
            raise ValueError(f"Incomplete tier row: {row}")
        out[str(row["tier"])] = float(row["multiplier"])
    return out


def rarity_factor(prevalence: float, cap: float) -> float:
    """Return ``min(1 / prevalence, cap)``; a prevalence of 0 gives the cap."""
    return cap if prevalence <= 0 else min(1.0 / prevalence, cap)


def compute_weights(
    findings: Sequence[str],
    table: Sequence[FindingWeight],
    scheme: str = "clinical",
    *,
    tier_multipliers: Mapping[str, float] | None = None,
    normalisation: str = "none",
    rel_tol: float = 1e-6,
) -> dict[str, float]:
    """Return w_c per finding for a weighting scheme.

    ``clinical``:    min(1 / prevalence_c, cap) * tier_multiplier(tier_c)  (E7)
    ``uniform``:     1                                                      (A1)
    ``rarity_only``: min(1 / prevalence_c, cap)                             (A2)

    For ``clinical`` the frozen ``final_weight`` is used; when ``tier_multipliers``
    are given the weight is recomputed and must match ``final_weight`` (so the
    table cannot drift from its formula). ``normalisation="sum_to_n"`` rescales the
    weights to sum to the number of findings (pre-registered choice).
    """
    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(f"scheme must be one of {WEIGHTING_SCHEMES}")
    if normalisation not in NORMALISATIONS:
        raise ValueError(f"normalisation must be one of {NORMALISATIONS}")
    rows = {w.finding: w for w in table}
    missing = [f for f in findings if f not in rows]
    if missing:
        raise ValueError(f"Findings missing from the weight table: {missing}")
    weights: dict[str, float] = {}
    for finding in findings:
        row = rows[finding]
        if scheme == "uniform":
            weights[finding] = 1.0
        elif scheme == "rarity_only":
            weights[finding] = rarity_factor(row.prevalence, row.cap)
        else:
            if tier_multipliers is not None:
                if row.importance_tier not in tier_multipliers:
                    raise ValueError(f"Unknown tier {row.importance_tier!r} for {finding}")
                recomputed = (
                    rarity_factor(row.prevalence, row.cap) * tier_multipliers[row.importance_tier]
                )
                if not math.isclose(recomputed, row.final_weight, rel_tol=rel_tol):
                    raise ValueError(
                        f"final_weight of {finding} ({row.final_weight}) does not match the "
                        f"formula ({recomputed})"
                    )
            weights[finding] = row.final_weight
    if normalisation == "sum_to_n":
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Cannot normalise weights that sum to zero.")
        weights = {f: w * len(weights) / total for f, w in weights.items()}
    return weights


def _log_softmax_pair(a: float, b: float) -> tuple[float, float]:
    top = max(a, b)
    log_norm = top + math.log(math.exp(a - top) + math.exp(b - top))
    return a - log_norm, b - log_norm


def delta(
    baseline: Sequence[YesNoScore],
    ablated: Sequence[YesNoScore],
    probes: Sequence[FindingProbe],
    variant: str = "logprob_drop",
) -> dict[tuple[str, str], float]:
    """Return Delta_j(c, x) keyed by (finding, sample_id) for one ablated unit.

    ``logprob_drop``: log p_theta(y) - log p_{theta\\j}(y) for the correct answer y,
                      using the model's own answer log-probabilities.
    ``kl``:           KL(p_theta || p_{theta\\j}) over the answer distribution
                      renormalised over {yes, no} (A5). This resolves the open choice
                      in the module docstring for the ``YesNoScore`` interface, which
                      only exposes the two answer log-probabilities.
    """
    if variant not in DELTA_VARIANTS:
        raise ValueError(f"variant must be one of {DELTA_VARIANTS}")
    if not len(baseline) == len(ablated) == len(probes):
        raise ValueError("baseline, ablated and probes must be aligned and of equal length.")
    out: dict[tuple[str, str], float] = {}
    for base, abl, probe in zip(baseline, ablated, probes, strict=True):
        key = (probe.finding, probe.sample_id)
        if key in out:
            raise ValueError(f"Duplicate probe {key}")
        if variant == "logprob_drop":
            if probe.answer not in ("yes", "no"):
                raise ValueError(f"Probe answer must be yes/no, got {probe.answer!r}")
            if probe.answer == "yes":
                out[key] = base.logp_yes - abl.logp_yes
            else:
                out[key] = base.logp_no - abl.logp_no
        else:
            p_yes, p_no = _log_softmax_pair(base.logp_yes, base.logp_no)
            q_yes, q_no = _log_softmax_pair(abl.logp_yes, abl.logp_no)
            out[key] = math.exp(p_yes) * (p_yes - q_yes) + math.exp(p_no) * (p_no - q_no)
    return out


def aggregate_importance(
    deltas: Mapping[tuple[str, str], float],
    weights: Mapping[str, float],
    *,
    clip_negative: bool = False,
) -> float:
    """Return I_j = sum_c w_c * mean_x Delta_j(c, x) for one unit.

    Findings without probes contribute nothing. ``clip_negative=True`` clips
    negative Delta values (ablation helps) at 0 before averaging; the choice is
    pre-registered (open point in the module docstring).
    """
    per_finding: dict[str, list[float]] = {}
    for (finding, _sample), value in deltas.items():
        if finding not in weights:
            raise ValueError(f"No weight for finding {finding!r}")
        per_finding.setdefault(finding, []).append(max(value, 0.0) if clip_negative else value)
    return float(sum(weights[f] * (sum(values) / len(values)) for f, values in per_finding.items()))


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
