"""Human labels for near-duplicate pairs, and the joining rule they choose (D02 amendment 3).

1. **Sample** (:func:`labelling_sample`): about 300 candidate pairs, seed 20261012, from a
   partition of the candidates into strata (:data:`STRATA`), each sampled at random without
   replacement. The border zones of the earlier measures are oversampled (pairs within 4 dHash
   bits, crop correlation 0.50-0.80, the run-3 sheet pairs), and the cosine range is covered in
   equal-width bins, so the rare high-cosine pairs are oversampled too. Every pair has a weight
   (stratum size / pairs drawn from it) for estimates over all candidates.
2. **Sheets and template**: the pairs side by side, numbered in a random order, with no score
   (:func:`write_labelling_files`), and a CSV template ``pair_id, rater_1, rater_2`` with the
   labels ``same_scene``, ``different``, ``unsure``. Two team members label independently.
3. **Scores** (:func:`score_labels`): Cohen's kappa; for every rule of
   ``candidates.rule_family()``, the sensitivity and precision for "same scene" with Wilson 95 %
   intervals, a pair being positive when **both** raters call it ``same_scene``.
4. **Choice, fixed in advance** (:func:`choose_rule`): the rule with the highest sensitivity
   among those with precision >= 0.5 on the labelled pairs, whose largest group over all
   images stays within the chaining guard (10 %); ties go to the higher precision, then the
   simpler rule (one view before a crop combination), then the views in the order full,
   centre, max, then the lower threshold (it catches more copies the labels did not show).

Recorded with D02 amendment 3 on 2026-10-19, after RC-D1-embed and before any label exists:

5. **Top-up** (:func:`topup_sample`): if fewer than 30 labelled pairs lie within +-0.03 of the
   chosen cosine threshold (in the chosen view), 100 more candidate pairs are drawn from that
   band, seed 20261013, leaving out the pairs already labelled; both raters label them and the
   rule is chosen again, once, on all labels by the same criterion.
6. **Disagreements** (:func:`adjudicate`): the third team member decides every pair the two
   raters label differently, and every pair of a labelling round whose Cohen's kappa is below
   0.40. A decided pair is positive only if the third member calls it ``same_scene``.
7. **Unsure** (:func:`truth_from`): primary, ``unsure`` counts as not ``same_scene``. In a
   sensitivity analysis it counts as not ``same_scene`` for precision and as ``same_scene``
   for sensitivity (the less favourable reading of each).
8. **Blinding** (:func:`add_commitment`, :func:`check_commitments`): the raters see no score;
   each commits the SHA-256 of their own label file to the repository before the labels are
   discussed (the commit order is in the history), and the scoring step checks the files
   against these commitments.

Only aggregate numbers are committed. The sheets, the pair list and the filled label files
name or show images: they stay private.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medcompress.data.candidates import RULE_VIEWS, CandidateTable, JoinRule
from medcompress.utils.bundles import file_sha256

LABELS: tuple[str, ...] = ("same_scene", "different", "unsure")
SAMPLE_SEED = 20261012
MIN_PRECISION = 0.5
PAIRS_PER_SHEET = 10
LABEL_THUMB = (480, 384)
"""Each image of a labelling sheet fits this box: large enough to judge mucosa and edges."""
COSINE_BIN_EDGES: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                                       0.95, 1.01)  # fmt: skip
TOPUP_SEED = 20261013
TOPUP_BAND = 0.03
"""Half-width of the band around the chosen cosine threshold (the top-up, step 5)."""
TOPUP_MIN_IN_BAND = 30
"""A top-up is drawn if fewer labelled pairs than this lie in the band."""
TOPUP_PAIRS = 100
TOPUP_STRATUM = "topup_band"
BAND_TOLERANCE = 1e-6
"""The cosines are float32: a pair on the band's edge counts as inside."""
ADJUDICATE_ALL_BELOW_KAPPA = 0.40
"""If a round's Cohen's kappa (three labels) is below this, the third member decides every pair
of that round, not only the disagreements."""
ROLES: tuple[str, ...] = ("rater_1", "rater_2", "rater_3")
ROUNDS: tuple[str, ...] = ("first", "topup")
COMMITMENTS_FILE = Path("docs/runs/RC-D1-label_commitments.csv")
COMMITMENT_FIELDS: tuple[str, ...] = ("round", "role", "file", "sha256", "labels")


def _cosine_strata() -> list[tuple[str, int]]:
    names = [("cos_below_0.50", 12)]
    for lo, hi in zip(COSINE_BIN_EDGES[:-1], COSINE_BIN_EDGES[1:], strict=True):
        names.append((f"cos_{lo:.2f}_{min(hi, 1.0):.2f}", 12))
    return names


STRATA: tuple[tuple[str, int | None], ...] = (
    ("run3_band", None),  # every pair
    ("dhash_0_2", 20),
    ("dhash_3_4", 20),
    ("crop_0.50_0.65", 25),
    ("crop_0.65_0.80", 25),
    ("crop_0.80_1.00", 20),
    *_cosine_strata(),
)
"""The partition of the candidates, in the order a pair is assigned (the first that fits), and
how many pairs each gives (``None``: all). About 300 pairs in all: up to 60 run-3 pairs, 110 from
the dHash and crop border zones, 132 from the 11 cosine bins (the larger of the two cosines)."""


def assign_strata(table: CandidateTable) -> np.ndarray:
    """The stratum of every candidate pair (the first of :data:`STRATA` that fits)."""
    out = np.empty(len(table), dtype=object)
    cos = table.cos_max
    for k in range(len(table)):
        bits, crop, c = int(table.bits[k]), float(table.crop[k]), float(cos[k])
        if table.sources[k] & 16:
            out[k] = "run3_band"
        elif bits <= 2:
            out[k] = "dhash_0_2"
        elif bits <= 4:
            out[k] = "dhash_3_4"
        elif 0.50 <= crop < 0.65:
            out[k] = "crop_0.50_0.65"
        elif 0.65 <= crop < 0.80:
            out[k] = "crop_0.65_0.80"
        elif crop >= 0.80:
            out[k] = "crop_0.80_1.00"
        elif c < COSINE_BIN_EDGES[0]:
            out[k] = "cos_below_0.50"
        else:
            j = int(np.searchsorted(COSINE_BIN_EDGES, c, side="right")) - 1
            lo, hi = COSINE_BIN_EDGES[j], COSINE_BIN_EDGES[j + 1]
            out[k] = f"cos_{lo:.2f}_{min(hi, 1.0):.2f}"
    return out


@dataclass(frozen=True)
class LabelPair:
    """One pair to label: its number on the sheets, its images and scores (private)."""

    pair_id: int
    index: int
    image_a: str
    image_b: str
    stratum: str
    weight: float
    bits: int
    crop: float
    cos_full: float
    cos_centre: float


def labelling_sample(table: CandidateTable, *, seed: int = SAMPLE_SEED) -> list[LabelPair]:
    """The stratified sample of :data:`STRATA`, numbered 1..n in a seeded random order."""
    strata = assign_strata(table)
    rng = np.random.default_rng(seed)
    chosen: list[tuple[int, str, float]] = []
    for name, quota in STRATA:
        members = np.flatnonzero(strata == name)
        if len(members) == 0:
            continue
        take = len(members) if quota is None else min(quota, len(members))
        picked = np.sort(rng.choice(members, size=take, replace=False))
        chosen += [(int(k), name, len(members) / take) for k in picked]
    order = rng.permutation(len(chosen))
    pairs = []
    for pair_id, position in enumerate(order, start=1):
        k, name, weight = chosen[position]
        pairs.append(
            LabelPair(
                pair_id, k, table.images[table.a[k]], table.images[table.b[k]], name, weight,
                int(table.bits[k]), float(table.crop[k]), float(table.cos_full[k]),
                float(table.cos_centre[k]),
            )
        )  # fmt: skip
    return sorted(pairs, key=lambda p: p.pair_id)


def stratum_sizes(table: CandidateTable) -> dict[str, int]:
    """How many candidate pairs each stratum holds."""
    strata = assign_strata(table)
    return {name: int(np.sum(strata == name)) for name, _ in STRATA}


# ------------------------------------------------------------------ the top-up (step 5)


def in_band(cos: np.ndarray, threshold: float, band: float = TOPUP_BAND) -> np.ndarray:
    """Which cosines lie within ``threshold +- band`` (edges included)."""
    return np.abs(np.asarray(cos, dtype=np.float64) - threshold) <= band + BAND_TOLERANCE


def labelled_in_band(pairs: Sequence[LabelPair], rule: JoinRule, band: float = TOPUP_BAND) -> int:
    """How many labelled pairs lie within ``band`` of the rule's threshold, in its view."""
    return int(in_band(pair_cosines(pairs, rule.view), rule.threshold, band).sum())


@dataclass(frozen=True)
class TopUp:
    """The top-up of D02 amendment 3 (step 5 above) for one chosen rule.

    ``in_band_labelled``: labelled pairs in the band; ``in_band_unlabelled``: candidate pairs
    in the band not labelled yet (what the top-up draws from); ``pairs``: the pairs drawn
    (none if the top-up is not needed), numbered after the labelled ones, each weighted
    ``in_band_unlabelled / drawn`` within the band."""

    rule: JoinRule
    band: float
    min_in_band: int
    in_band_labelled: int
    in_band_unlabelled: int
    pairs: tuple[LabelPair, ...]

    @property
    def needed(self) -> bool:
        return self.in_band_labelled < self.min_in_band


def topup_sample(
    table: CandidateTable,
    labelled: Sequence[LabelPair],
    rule: JoinRule,
    *,
    band: float = TOPUP_BAND,
    min_in_band: int = TOPUP_MIN_IN_BAND,
    size: int = TOPUP_PAIRS,
    seed: int = TOPUP_SEED,
) -> TopUp:
    """If fewer than ``min_in_band`` labelled pairs lie within ``band`` of the rule's cosine
    threshold (in its view), draw ``size`` candidate pairs from that band at random (``seed``),
    leaving out the pairs already labelled (by their two images). Only the candidate pairs and
    their scores are used: no label, and no new embedding."""
    count = labelled_in_band(labelled, rule, band)
    taken = {tuple(sorted((p.image_a, p.image_b))) for p in labelled}
    members = [
        int(k) for k in np.flatnonzero(in_band(table.cosine(rule.view), rule.threshold, band))
        if tuple(sorted((table.images[table.a[k]], table.images[table.b[k]]))) not in taken
    ]  # fmt: skip
    if count >= min_in_band or not members:
        return TopUp(rule, band, min_in_band, count, len(members), ())
    rng = np.random.default_rng(seed)
    take = min(size, len(members))
    picked = np.sort(rng.choice(np.array(members), size=take, replace=False))
    order = rng.permutation(take)
    start = max((p.pair_id for p in labelled), default=0) + 1
    pairs = []
    for offset, position in enumerate(order):
        k = int(picked[position])
        pairs.append(
            LabelPair(
                start + offset, k, table.images[table.a[k]], table.images[table.b[k]],
                TOPUP_STRATUM, len(members) / take, int(table.bits[k]), float(table.crop[k]),
                float(table.cos_full[k]), float(table.cos_centre[k]),
            )
        )  # fmt: skip
    return TopUp(rule, band, min_in_band, count, len(members), tuple(pairs))


PAIR_FIELDS = ("pair_id", "image_a", "image_b", "stratum", "weight", "bits", "crop",
               "cos_full", "cos_centre")  # fmt: skip


def write_labelling_files(
    pairs: Sequence[LabelPair], data_root: Path, out_dir: Path
) -> dict[str, Path]:
    """The private pair list (with the scores, for the scoring step), the pairs' images
    without scores (for the raters), the label template and the numbered sheets (no scores)."""
    from medcompress.config import repo_root
    from medcompress.data.contact_sheet import render_pairs

    out_dir = Path(out_dir).resolve()
    if out_dir.is_relative_to(repo_root().resolve()):
        raise ValueError(f"the labelling files name images; write them outside the repo: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_list = out_dir / "label_pairs.csv"
    with pair_list.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAIR_FIELDS)
        for p in pairs:
            writer.writerow([p.pair_id, p.image_a, p.image_b, p.stratum, f"{p.weight:.6f}",
                             p.bits, f"{p.crop:.6f}", f"{p.cos_full:.6f}",
                             f"{p.cos_centre:.6f}"])  # fmt: skip
    images = out_dir / "label_images.csv"  # for the raters: which images, never the scores
    with images.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "image_a", "image_b"])
        for p in pairs:
            writer.writerow([p.pair_id, p.image_a, p.image_b])
    template = out_dir / "labels_template.csv"
    with template.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "rater_1", "rater_2"])
        for p in pairs:
            writer.writerow([p.pair_id, "", ""])
    sheets = out_dir / "label_sheets"
    for start in range(0, len(pairs), PAIRS_PER_SHEET):
        chunk = pairs[start : start + PAIRS_PER_SHEET]
        cells = tuple(
            (str(Path(data_root) / p.image_a), str(Path(data_root) / p.image_b),
             f"pair {p.pair_id}")
            for p in chunk
        )  # fmt: skip
        title = f"Pairs {chunk[0].pair_id} to {chunk[-1].pair_id}: same scene, different, unsure?"
        name = f"pairs_{chunk[0].pair_id:03d}_{chunk[-1].pair_id:03d}.png"
        render_pairs(
            ((title, cells),), sheets / name, crop_fraction=None, columns=1, numbered=False,
            thumb=LABEL_THUMB,
        )  # fmt: skip
    return {"pairs": pair_list, "images": images, "template": template, "sheets": sheets}


def read_pairs(path: Path) -> list[LabelPair]:
    """The private pair list written by :func:`write_labelling_files`."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            LabelPair(int(r["pair_id"]), -1, r["image_a"], r["image_b"], r["stratum"],
                      float(r["weight"]), int(r["bits"]), float(r["crop"]),
                      float(r["cos_full"]), float(r["cos_centre"]))
            for r in csv.DictReader(handle)
        ]  # fmt: skip


# ------------------------------------------------------------------ reading the labels


class LabelError(ValueError):
    """The label files are incomplete or hold an unknown label."""


def _label(value: str) -> str:
    text = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text not in LABELS:
        raise LabelError(f"unknown label {value!r}; use one of {LABELS}")
    return text


def read_labels(pair_ids: Sequence[int], *files: Path) -> tuple[dict[int, str], dict[int, str]]:
    """``(rater_1, rater_2)`` labels by pair id, from one file with both columns filled or
    from two copies of the template, each with its own column filled. Every pair needs both."""
    first: dict[int, str] = {}
    second: dict[int, str] = {}
    for path in files:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pid = int(row["pair_id"])
                for column, target in (("rater_1", first), ("rater_2", second)):
                    value = (row.get(column) or "").strip()
                    if value:
                        if pid in target and target[pid] != _label(value):
                            raise LabelError(f"pair {pid}: two different {column} labels")
                        target[pid] = _label(value)
    for name, labels in (("rater_1", first), ("rater_2", second)):
        missing = [p for p in pair_ids if p not in labels]
        if missing:
            raise LabelError(f"{name} has no label for {len(missing)} pairs (first: {missing[0]})")
    return first, second


# ------------------------------------------------------------------ statistics


def cohen_kappa(first: Sequence[str], second: Sequence[str], categories: Sequence[str]) -> float:
    """Cohen's kappa of two raters over ``categories`` (nan if chance agreement is 1)."""
    n = len(first)
    if n == 0:
        return math.nan
    observed = sum(1 for x, y in zip(first, second, strict=True) if x == y) / n
    expected = sum(
        (sum(1 for x in first if x == c) / n) * (sum(1 for y in second if y == c) / n)
        for c in categories
    )
    return math.nan if expected >= 1 else (observed - expected) / (1 - expected)


def wilson(successes: int, total: int, z: float = 1.959964) -> tuple[float, float, float]:
    """The proportion and its Wilson 95 % interval (nan if ``total`` is 0)."""
    if total == 0:
        return math.nan, math.nan, math.nan
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return p, max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class RuleScore:
    """How one rule does on the labelled pairs."""

    rule: JoinRule
    joined: int
    true_positive: int
    positives: int
    sensitivity: tuple[float, float, float]
    precision: tuple[float, float, float]


def score_rule(
    rule: JoinRule,
    pairs: Sequence[LabelPair],
    truth: np.ndarray,
    *,
    sensitivity_truth: np.ndarray | None = None,
) -> RuleScore:
    """Sensitivity and precision of ``rule`` on the labelled pairs (Wilson intervals).
    ``sensitivity_truth``: other positives for the sensitivity only (the unsure analysis);
    ``true_positive`` and ``positives`` then count those."""
    joined = _joins(rule, pairs)
    tp = int(np.sum(joined & truth))
    positives = truth if sensitivity_truth is None else sensitivity_truth
    found = int(np.sum(joined & positives))
    return RuleScore(
        rule, int(joined.sum()), found, int(positives.sum()),
        wilson(found, int(positives.sum())), wilson(tp, int(joined.sum())),
    )  # fmt: skip


def pair_cosines(pairs: Sequence[LabelPair], view: str) -> np.ndarray:
    """The cosine of each labelled pair in ``view`` (``full``, ``centre`` or ``max``)."""
    full = np.array([p.cos_full for p in pairs], dtype=np.float64)
    centre = np.array([p.cos_centre for p in pairs], dtype=np.float64)
    return {"full": full, "centre": centre, "max": np.maximum(full, centre)}[view]


def _joins(rule: JoinRule, pairs: Sequence[LabelPair]) -> np.ndarray:
    joined = pair_cosines(pairs, rule.view) >= rule.threshold
    if rule.crop_threshold is not None:
        joined |= np.array([p.crop for p in pairs]) >= rule.crop_threshold
    return joined


def weighted_scores(
    rule: JoinRule,
    pairs: Sequence[LabelPair],
    truth: np.ndarray,
    *,
    resamples: int = 1000,
    seed: int = SAMPLE_SEED,
) -> dict[str, tuple[float, float, float]]:
    """Sensitivity and precision over **all** candidate pairs, weighting each labelled pair by
    its stratum's size over the pairs drawn from it, with stratified bootstrap 95 % intervals."""
    joined = _joins(rule, pairs)
    weights = np.array([p.weight for p in pairs])
    strata = np.array([p.stratum for p in pairs])

    def estimate(idx: np.ndarray) -> tuple[float, float]:
        w, j, t = weights[idx], joined[idx], truth[idx]
        tp = float(np.sum(w * (j & t)))
        pos, sel = float(np.sum(w * t)), float(np.sum(w * j))
        return (tp / pos if pos else math.nan), (tp / sel if sel else math.nan)

    point = estimate(np.arange(len(pairs)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    draws = np.array([
        estimate(np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups]))
        for _ in range(resamples)
    ])  # fmt: skip
    out = {}
    for k, name in enumerate(("sensitivity", "precision")):
        column = draws[:, k][np.isfinite(draws[:, k])]
        low, high = np.percentile(column, [2.5, 97.5]) if len(column) else (math.nan, math.nan)
        out[name] = (point[k], float(low), float(high))
    return out


def choose_rule(
    scores: Sequence[RuleScore],
    guard: Callable[[JoinRule], tuple[bool, int]],
    *,
    min_precision: float = MIN_PRECISION,
) -> tuple[RuleScore | None, list[tuple[JoinRule, bool, int]]]:
    """The rule fixed in advance (module docstring). ``guard(rule)`` gives (within the
    chaining guard, largest group) over all images; it is asked in order of preference until
    a rule passes. Returns the chosen score (``None`` if no rule qualifies) and the guard
    checks made."""
    eligible = [s for s in scores if s.joined > 0 and s.precision[0] >= min_precision]
    eligible.sort(
        key=lambda s: (
            -s.sensitivity[0] if not math.isnan(s.sensitivity[0]) else 0.0,
            -s.precision[0],
            s.rule.complexity,
            RULE_VIEWS.index(s.rule.view),
            s.rule.threshold,
            s.rule.crop_threshold or 0.0,
        )
    )
    checks: list[tuple[JoinRule, bool, int]] = []
    for score in eligible:
        ok, largest = guard(score.rule)
        checks.append((score.rule, ok, largest))
        if ok:
            return score, checks
    return None, checks


def truth_of(first: Mapping[int, str], second: Mapping[int, str], ids: Sequence[int]) -> np.ndarray:
    """A pair is positive when both raters call it ``same_scene`` (before any adjudication)."""
    return np.array([first[i] == "same_scene" and second[i] == "same_scene" for i in ids])


# ------------------------------------------------------------------ disagreements (step 6)


def read_third(*files: Path) -> dict[int, str]:
    """The third member's labels by pair id (the ``rater_3`` column of any of ``files``)."""
    third: dict[int, str] = {}
    for path in files:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = (row.get("rater_3") or "").strip()
                if value:
                    pid = int(row["pair_id"])
                    if pid in third and third[pid] != _label(value):
                        raise LabelError(f"pair {pid}: two different rater_3 labels")
                    third[pid] = _label(value)
    return third


@dataclass(frozen=True)
class Adjudication:
    """One labelling round after the third member: ``final`` is the label each pair counts
    with (the raters' common label, or the third member's), ``required`` the pairs the third
    member decides, ``missing`` those without a third label yet, ``kappa`` the round's Cohen's
    kappa before adjudication and ``all_pairs`` whether it was below 0.40 (every pair decided)."""

    ids: tuple[int, ...]
    kappa: float
    all_pairs: bool
    required: tuple[int, ...]
    missing: tuple[int, ...]
    final: Mapping[int, str]


def adjudicate(
    first: Mapping[int, str],
    second: Mapping[int, str],
    third: Mapping[int, str],
    ids: Sequence[int],
    *,
    below: float = ADJUDICATE_ALL_BELOW_KAPPA,
) -> Adjudication:
    """Step 6 (module docstring) for one round of pairs ``ids``. A third label on a pair that
    is not to be decided is refused: the third member labels only the pairs asked for."""
    kappa = cohen_kappa([first[i] for i in ids], [second[i] for i in ids], LABELS)
    all_pairs = not math.isnan(kappa) and kappa < below
    required = tuple(ids) if all_pairs else tuple(i for i in ids if first[i] != second[i])
    asked = set(required)
    extra = sorted(i for i in ids if i in third and i not in asked)
    if extra:
        raise LabelError(
            f"rater_3 labelled {len(extra)} pairs that are not to be decided (first: {extra[0]})"
        )
    missing = tuple(i for i in required if i not in third)
    final = {i: (third[i] if i in asked and i in third else first[i]) for i in ids}
    return Adjudication(tuple(ids), kappa, all_pairs, required, missing, final)


def truth_from(
    final: Mapping[int, str], ids: Sequence[int], *, unsure_positive: bool = False
) -> np.ndarray:
    """Primary: a pair is positive only if its label is ``same_scene`` (both raters, or the
    third member). ``unsure_positive``: ``unsure`` counts as positive too (the sensitivity
    side of the unsure analysis)."""
    positive = ("same_scene", "unsure") if unsure_positive else ("same_scene",)
    return np.array([final[i] in positive for i in ids], dtype=bool)


def write_adjudication_template(ids: Sequence[int], out_dir: Path) -> Path:
    """The pairs the third member decides: ``pair_id, rater_3`` (no score, no rater's label).
    It is derived from the labels, so it stays private."""
    from medcompress.config import repo_root

    out_dir = Path(out_dir).resolve()
    if out_dir.is_relative_to(repo_root().resolve()):
        raise ValueError(f"the adjudication list comes from the labels; keep it private: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adjudication_template.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "rater_3"])
        for pid in sorted(ids):
            writer.writerow([pid, ""])
    return path


# ------------------------------------------------------------------ blinding (step 8)


def check_label_file(path: Path, role: str) -> int:
    """A rater's own file: only the ``role`` column is filled, with known labels. Returns how
    many labels it holds."""
    if role not in ROLES:
        raise LabelError(f"unknown role {role!r}; use one of {ROLES}")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if "pair_id" not in columns or role not in columns:
            raise LabelError(f"{path} needs the columns pair_id and {role}")
        filled = 0
        for row in reader:
            for other in ROLES:
                if other != role and (row.get(other) or "").strip():
                    raise LabelError(f"{path} holds {other} labels too: each rater commits their "
                                     "own file, with only their own column filled")  # fmt: skip
            if (row.get(role) or "").strip():
                _label(row[role])
                filled += 1
    if filled == 0:
        raise LabelError(f"{path} holds no {role} label")
    return filled


def read_commitments(path: Path) -> list[dict[str, str]]:
    """The committed label-file hashes (empty if the file does not exist)."""
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_commitment(commitments: Path, path: Path, *, role: str, round_: str) -> dict[str, str]:
    """Check ``path`` as ``role``'s file of round ``round_`` and add its SHA-256 to
    ``commitments``. One file per role and round: a second one is refused."""
    if round_ not in ROUNDS:
        raise LabelError(f"unknown round {round_!r}; use one of {ROUNDS}")
    labels = check_label_file(path, role)
    rows = read_commitments(commitments)
    if any(r["round"] == round_ and r["role"] == role for r in rows):
        raise LabelError(f"{role} already committed a file for round {round_}: a new file needs "
                         "a recorded decision")  # fmt: skip
    row = {"round": round_, "role": role, "file": Path(path).name, "sha256": file_sha256(path),
           "labels": str(labels)}  # fmt: skip
    commitments = Path(commitments)
    commitments.parent.mkdir(parents=True, exist_ok=True)
    new = not commitments.is_file() or not commitments.read_text(encoding="utf-8").strip()
    with commitments.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMMITMENT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return row


def check_commitments(files: Sequence[Path], commitments: Path) -> list[dict[str, str]]:
    """Every label file must match a committed SHA-256; returns the matching rows, in order."""
    rows = read_commitments(commitments)
    by_hash = {r["sha256"]: r for r in rows}
    matched = []
    for path in files:
        digest = file_sha256(path)
        if digest not in by_hash:
            raise LabelError(f"{Path(path).name} (SHA-256 {digest[:12]}...) is not in "
                             f"{commitments}: it was changed after it was committed, or never "
                             "committed (D02 amendment 3, blinding)")  # fmt: skip
        matched.append(by_hash[digest])
    return matched
