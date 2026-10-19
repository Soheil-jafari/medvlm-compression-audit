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

LABELS: tuple[str, ...] = ("same_scene", "different", "unsure")
SAMPLE_SEED = 20261012
MIN_PRECISION = 0.5
PAIRS_PER_SHEET = 10
LABEL_THUMB = (480, 384)
"""Each image of a labelling sheet fits this box: large enough to judge mucosa and edges."""
COSINE_BIN_EDGES: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                                       0.95, 1.01)  # fmt: skip


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


def score_rule(rule: JoinRule, pairs: Sequence[LabelPair], truth: np.ndarray) -> RuleScore:
    """Sensitivity and precision of ``rule`` on the labelled pairs (Wilson intervals)."""
    joined = _joins(rule, pairs)
    tp = int(np.sum(joined & truth))
    return RuleScore(
        rule, int(joined.sum()), tp, int(truth.sum()),
        wilson(tp, int(truth.sum())), wilson(tp, int(joined.sum())),
    )  # fmt: skip


def _joins(rule: JoinRule, pairs: Sequence[LabelPair]) -> np.ndarray:
    full = np.array([p.cos_full for p in pairs])
    centre = np.array([p.cos_centre for p in pairs])
    cos = {"full": full, "centre": centre, "max": np.maximum(full, centre)}[rule.view]
    joined = cos >= rule.threshold
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
    """A pair is positive when both raters call it ``same_scene``."""
    return np.array([first[i] == "same_scene" and second[i] == "same_scene" for i in ids])
