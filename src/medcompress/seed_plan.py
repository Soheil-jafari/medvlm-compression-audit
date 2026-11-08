"""The seed plan (pre-registration section 11, amendment of 2026-11-08, Soheil and Amir).

* **Main-claim tier, 3 seeds (0, 1, 2):** E2, E7 at B2 and the comparator at B2. H1-H3 are
  confirmatory on this tier only.
* **Secondary, 3 seeds:** A1 (uniform finding weights) at B2, for the pre-specified secondary
  comparison E7 against A1 (amendment of 2026-11-08, item 8). Not part of any confirmatory
  verdict. It re-weights E7's per-finding effects (no scoring of its own) and runs seeds 1 and
  2 under the same futility rule as the main tier.
* **Everything else, seed 0 only** (exploratory, descriptive): the other E5 / E6 arms, every E3
  variant, E4, E8, E9, every arm at B1 and B3, the ablations A1-A6 and the training-length
  check. H4 (the ranking of E3-E8) is exploratory, on seed 0. E1 keeps the 3 seeds it ran
  before the amendment (the control of E2).
* **Never 2 seeds** for an arm: the majority answer of the McNemar tests is "right in more than
  half of the seeds", and with 2 seeds a 1-1 split counts as wrong.
* **The comparator** is chosen on validation, seed 0, at B2, among
  :data:`COMPARATOR_CANDIDATES`, by :func:`rank_key`: (i) ``op_small_finding_recall``; ties:
  (ii) the recall at the operating point over every mask-positive polyp
  (``op_mask_positive_recall``); ties: (iii) the mean over the small polyps of the probe margin
  minus the arm's own ``op_threshold:polyp`` (``op_small_margin_mean``); then (iv) the name.
  Only the chosen arm runs seeds 1 and 2.
* **Futility-only stopping rule:** after seed 0, seeds 1 and 2 of the main tier run unless E7
  at B2 ranks below the comparator by the same ordering (:func:`futility_stop`). Then the study
  stops there and reports seed 0 as exploratory. Stopping only for futility can only remove
  chances to claim an effect, so it does not raise the false-positive rate.

``scripts/choose_comparator.py`` applies the ordering to the committed validation rows and
writes :data:`CHOICE_FILE`; the run cards of seeds 1 and 2 read it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SEEDS: tuple[int, ...] = (0, 1, 2)
EXPLORATORY_SEEDS: tuple[int, ...] = (0,)
PRIMARY_BUDGET = "B2"
OURS = "E7"
REFERENCE = "E2"
CONTROL = "E1"
"""E1 ran its 3 seeds before the amendment; they are kept (the control of E2)."""
COMPARATOR_CANDIDATES: tuple[str, ...] = ("E5:magnitude", "E5:activation", "E5:taylor", "E6")
SECONDARY_ARMS: tuple[tuple[str, str], ...] = (("A1", PRIMARY_BUDGET),)
"""Arms with 3 seeds for the pre-specified secondary comparison (E7 against A1 at B2)."""
MAIN = "main"
SECONDARY = "secondary"
EXPLORATORY = "exploratory"

SMALL_RECALL = "op_small_finding_recall"
MASK_RECALL = "op_mask_positive_recall"
SMALL_MARGIN = "op_small_margin_mean"
STRATUM_RECALL = "op_recall_stratum:"
STRATUM_COUNT = "n_stratum:"
CHOICE_FILE = Path("seed_plan") / "comparator_B2.json"
"""Under the results root: the comparator, the ordering and the futility decision."""


class SeedPlanError(ValueError):
    """A run or a comparison that the seed plan does not allow (for example 2 seeds)."""


@dataclass(frozen=True)
class RankScore:
    """One arm's place in the ordering of item 4 (validation, seed 0, B2)."""

    label: str
    small_recall: float
    mask_recall: float
    small_margin: float


def _desc(value: float) -> float:
    """Higher first; a missing value sorts last."""
    return -value if not math.isnan(value) else math.inf


def rank_key(score: RankScore) -> tuple[float, float, float, str]:
    """The ordering of item 4: (i) small recall, (ii) mask-positive recall, (iii) the small
    polyps' mean margin over the threshold, all higher first, then (iv) the name."""
    return (
        _desc(score.small_recall),
        _desc(score.mask_recall),
        _desc(score.small_margin),
        score.label,
    )


def rank(scores: Iterable[RankScore]) -> list[RankScore]:
    """The arms best first."""
    return sorted(scores, key=rank_key)


def _complete(score: RankScore) -> None:
    missing = [
        name
        for name, value in (
            (SMALL_RECALL, score.small_recall),
            (MASK_RECALL, score.mask_recall),
            (SMALL_MARGIN, score.small_margin),
        )
        if math.isnan(value)
    ]
    if missing:
        raise SeedPlanError(
            f"{score.label}: validation {', '.join(missing)} missing at seed 0 "
            "(evaluate it again, or add the rows from its saved outcomes with "
            "scripts/rescore_probes.py)"
        )


def choose_comparator(scores: Mapping[str, RankScore]) -> RankScore:
    """The comparator: the first of :data:`COMPARATOR_CANDIDATES` by :func:`rank_key`.

    Every candidate must have all three criteria, so a tie is never left to a missing value.
    """
    absent = [label for label in COMPARATOR_CANDIDATES if label not in scores]
    if absent:
        raise SeedPlanError(f"no validation rows at seed 0, B2 for {absent}")
    candidates = [scores[label] for label in COMPARATOR_CANDIDATES]
    for score in candidates:
        _complete(score)
    return rank(candidates)[0]


def futility_stop(ours: RankScore, comparator: RankScore) -> bool:
    """True when E7 at B2 ranks below the comparator by :func:`rank_key` (the study then
    stops after seed 0). A full tie on (i)-(iii) goes to the name, where ``E7`` comes after
    every candidate, so it stops too (the rule as written)."""
    _complete(ours)
    return rank_key(ours) > rank_key(comparator)


def decided_by(first: RankScore, second: RankScore) -> str:
    """Which criterion of item 4 puts ``first`` before ``second`` (for the record)."""
    a, b = rank_key(first), rank_key(second)
    for name, x, y in zip(("(i)", "(ii)", "(iii)", "(iv) name"), a, b, strict=True):
        if x != y:
            return name
    return "identical"


# ------------------------------------------------------------------ from the results rows


def scores_from_rows(
    rows: Any, labels: Sequence[str], *, budget: str = PRIMARY_BUDGET, seed: int = 0
) -> dict[str, RankScore]:
    """The :class:`RankScore` of every label with validation rows at ``budget``, ``seed``.

    ``rows`` is ``aggregate.load_results`` (long schema with ``split``). The mask-positive
    recall falls back to the strata rows (``op_recall_stratum:<s>`` weighted by
    ``n_stratum:<s>``, the same probes) when its own row is not there.
    """
    if rows.empty:
        return {}
    data = rows[
        (rows["split"].fillna("") == "val")
        & (rows["budget"].fillna("") == budget)
        & (rows["seed"].astype(int) == seed)
    ]
    out = {}
    for label in labels:
        values = dict(
            zip(
                data.loc[data["experiment"] == label, "metric"],
                data.loc[data["experiment"] == label, "value"].astype(float),
                strict=True,
            )
        )
        if not values:
            continue
        out[label] = RankScore(
            label=label,
            small_recall=values.get(SMALL_RECALL, math.nan),
            mask_recall=values.get(MASK_RECALL, _strata_recall(values)),
            small_margin=values.get(SMALL_MARGIN, math.nan),
        )
    return out


def _strata_recall(values: Mapping[str, float]) -> float:
    hits = total = 0.0
    for name, recall in values.items():
        if name.startswith(STRATUM_RECALL):
            count = values.get(STRATUM_COUNT + name[len(STRATUM_RECALL) :])
            if count is None:
                return math.nan
            hits, total = hits + recall * count, total + count
    return hits / total if total else math.nan


# ---------------------------------------------------------------------- the tiers


def is_main(label: str, budget: str, comparator: str | None) -> bool:
    """Whether ``label`` at ``budget`` (``""`` = none) is in the main-claim tier."""
    if label == REFERENCE and not budget:
        return True
    if budget != PRIMARY_BUDGET:
        return False
    return label == OURS or (comparator is not None and label == comparator)


def is_secondary(label: str, budget: str) -> bool:
    """Whether ``label`` at ``budget`` is a 3-seed arm of the secondary comparison (A1 at B2)."""
    return (label, budget) in SECONDARY_ARMS


def tier(label: str, budget: str, comparator: str | None) -> str:
    """:data:`MAIN`, :data:`SECONDARY` or :data:`EXPLORATORY`."""
    if is_main(label, budget, comparator):
        return MAIN
    return SECONDARY if is_secondary(label, budget) else EXPLORATORY


def planned_seeds(label: str, budget: str, comparator: str | None) -> tuple[int, ...]:
    """The seeds the plan runs: 3 for the main tier (and E1, already run), seed 0 otherwise.

    Before the comparator is chosen (``None``) the candidates have seed 0 only.
    """
    if label == CONTROL and not budget:
        return SEEDS
    three = is_main(label, budget, comparator) or is_secondary(label, budget)
    return SEEDS if three else EXPLORATORY_SEEDS


def check_seed_count(name: str, seeds: Sequence[int]) -> None:
    """Refuse 2 seeds (see the module docstring): an arm has 1 or 3."""
    if len(set(seeds)) == 2:
        raise SeedPlanError(
            f"{name} has 2 seeds {sorted(set(seeds))}: the plan allows 1 (seed 0) or 3; with 2 "
            "the majority answer (right in more than half) counts a 1-1 split as wrong"
        )


# ------------------------------------------------------------------------ the record


def choice_record(scores: Mapping[str, RankScore], ours: RankScore | None) -> dict[str, Any]:
    """The comparator, the ordering and the futility decision, as written to the file."""
    best = choose_comparator(scores)
    ordered = rank(scores[label] for label in COMPARATOR_CANDIDATES)
    record: dict[str, Any] = {
        "budget": PRIMARY_BUDGET,
        "seed": 0,
        "split": "val",
        "comparator": best.label,
        "decided_by": decided_by(ordered[0], ordered[1]),
        "ranking": [asdict(s) for s in ordered],
    }
    if ours is not None:
        stop = futility_stop(ours, best)
        record["E7"] = asdict(ours)
        record["futility_stop"] = stop
        record["seeds_1_2"] = (
            "stop: E7 ranks below the comparator; seed 0 is reported as exploratory"
            if stop
            else "run: E2, E7, the comparator and A1 (secondary) at seeds 1 and 2"
        )
    return record


def write_choice(record: Mapping[str, Any], results_root: Path) -> Path:
    """Write the record to :data:`CHOICE_FILE` under ``results_root`` and return the path."""
    path = Path(results_root) / CHOICE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_choice(results_root: Path) -> dict[str, Any] | None:
    """The committed choice, or ``None`` before seed 0's validation rows exist."""
    path = Path(results_root) / CHOICE_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
