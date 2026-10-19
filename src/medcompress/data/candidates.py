"""Candidate pairs and joining rules for near-duplicates (D02 amendment 3 of 2026-10-19).

The labelling step needs pairs that might be copies, from every measure that can find them:

* ``knn_full`` / ``knn_centre``: for every image, its ``k`` nearest neighbours by the cosine of
  each embedding view (:mod:`medcompress.data.embeddings`);
* ``dhash``: every pair within ``dhash_bits`` of the 64-bit dHash (the earlier candidates);
* ``crop_knn``: for every image, its ``crop_k`` highest crop correlations at any dHash
  distance (RC-D1 run 3 found copies at 14-32 bits this way);
* ``run3_band``: the pairs of RC-D1 run 3's final contact sheet, reproduced from the same data,
  code and seed.

Each pair carries its dHash distance, crop correlation and both cosines. A joining rule
(:class:`JoinRule`) is a threshold on one cosine view, optionally OR a crop-correlation level;
it is applied to **all** pairs of images, not only to the candidates
(:func:`joined_pairs_all`), so no copy above the threshold is missed for want of a candidate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medcompress.data.dedup import (
    CropSettings,
    _popcount,
    _to_uint64,
    candidate_pairs,
    cross_similarity_matrix,
)
from medcompress.data.embeddings import knn_pairs, pair_cosines, pairs_at_or_above

SOURCE_BITS: Mapping[str, int] = {
    "knn_full": 1, "knn_centre": 2, "dhash": 4, "crop_knn": 8, "run3_band": 16,
}  # fmt: skip
VIEW_CODES: Mapping[str, int] = {"full": 1, "centre": 2, "max": 3, "crop": 4}
"""How a rule's view is written as a number in the results files."""


@dataclass(frozen=True)
class CandidateTable:
    """Candidate pairs (``a[k] < b[k]``, indices into ``images``) and their four scores."""

    images: tuple[str, ...]
    a: np.ndarray
    b: np.ndarray
    bits: np.ndarray
    crop: np.ndarray
    cos_full: np.ndarray
    cos_centre: np.ndarray
    sources: np.ndarray

    def __len__(self) -> int:
        return len(self.a)

    @property
    def cos_max(self) -> np.ndarray:
        """The larger of the two cosines of each pair."""
        return np.maximum(self.cos_full, self.cos_centre)

    def cosine(self, view: str) -> np.ndarray:
        """The cosine of each pair in ``view`` (``full``, ``centre`` or ``max``)."""
        return {"full": self.cos_full, "centre": self.cos_centre, "max": self.cos_max}[view]

    def has(self, source: str) -> np.ndarray:
        """Which pairs came (also) from ``source``."""
        return (self.sources & SOURCE_BITS[source]) > 0

    def save(self, path: Path) -> Path:
        """Write the table (private: image names inside)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, images=np.array(self.images), a=self.a, b=self.b, bits=self.bits,
            crop=self.crop, cos_full=self.cos_full, cos_centre=self.cos_centre,
            sources=self.sources,
        )  # fmt: skip
        return path

    @classmethod
    def load(cls, path: Path) -> CandidateTable:
        """Read a table written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                images=tuple(str(x) for x in data["images"]),
                **{k: data[k] for k in ("a", "b", "bits", "crop", "cos_full", "cos_centre",
                                        "sources")},
            )  # fmt: skip


@dataclass(frozen=True)
class CropPairs:
    """Every pair (any dHash distance) whose crop correlation is at least ``floor``."""

    a: np.ndarray
    b: np.ndarray
    value: np.ndarray
    floor: float

    def save(self, path: Path) -> Path:
        np.savez(path, a=self.a, b=self.b, value=self.value, floor=np.array(self.floor))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> CropPairs:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["a"], data["b"], data["value"], float(data["floor"]))


def crop_pass(
    signatures: Sequence[np.ndarray],
    crop: CropSettings,
    *,
    k: int,
    floor: float,
    wanted: Mapping[int, Iterable[int]] | None = None,
    block: int = 256,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float], CropPairs]:
    """One pass over all pairs of crop correlations (in row blocks, float32 matrix products).

    Returns the pairs in which one image is among the other's ``k`` highest correlations,
    the correlation of those pairs and of the pairs ``wanted`` lists (``{i: [j, ...]}``,
    i < j), and every pair at or above ``floor``.
    """
    n = len(signatures)
    knn: set[tuple[int, int]] = set()
    values: dict[tuple[int, int], float] = {}
    above_a, above_b, above_v = [], [], []
    wanted = wanted or {}
    for start in range(0, n, block):
        rows = list(range(start, min(start + block, n)))
        matrix = cross_similarity_matrix([signatures[i] for i in rows], list(signatures), crop)
        for r, i in enumerate(rows):
            line = matrix[r].copy()
            line[i] = -np.inf
            for j in np.argsort(-line, kind="stable")[: min(k, n - 1)]:
                pair = (min(i, int(j)), max(i, int(j)))
                knn.add(pair)
                values.setdefault(pair, float(line[j]))
            for j in wanted.get(i, ()):
                values.setdefault((i, j), float(matrix[r, j]))
            cols = np.nonzero(line[i + 1 :] >= floor)[0] + i + 1
            above_a.append(np.full(len(cols), i, dtype=np.int64))
            above_b.append(cols.astype(np.int64))
            above_v.append(line[cols].astype(np.float32))
    pairs = CropPairs(
        np.concatenate(above_a) if above_a else np.zeros(0, np.int64),
        np.concatenate(above_b) if above_b else np.zeros(0, np.int64),
        np.concatenate(above_v) if above_v else np.zeros(0, np.float32),
        float(floor),
    )
    return knn, values, pairs


def build_candidates(
    images: Sequence[str],
    fingerprints: Mapping[str, str],
    signatures: Mapping[str, np.ndarray],
    embeddings: Mapping[str, np.ndarray],
    crop: CropSettings,
    *,
    knn_k: int,
    dhash_bits: int,
    crop_k: int,
    crop_floor: float,
    extra: Iterable[tuple[str, str]] = (),
) -> tuple[CandidateTable, CropPairs]:
    """The candidate table (every source, all four scores) and the crop pairs above
    ``crop_floor``. ``images`` fixes the order; ``extra`` are the run-3 band pairs."""
    images = list(images)
    index = {p: i for i, p in enumerate(images)}
    found: dict[tuple[int, int], int] = {}

    def add(pairs: Iterable[tuple[int, int]], source: str) -> None:
        for i, j in pairs:
            pair = (min(i, j), max(i, j))
            found[pair] = found.get(pair, 0) | SOURCE_BITS[source]

    add(knn_pairs(embeddings["full"], knn_k), "knn_full")
    add(knn_pairs(embeddings["centre"], knn_k), "knn_centre")
    add(((index[a], index[b]) for a, b, _ in
         candidate_pairs(fingerprints, threshold=dhash_bits)), "dhash")  # fmt: skip
    add(((index[a], index[b]) for a, b in extra if a in index and b in index), "run3_band")
    wanted: dict[int, list[int]] = {}
    for i, j in found:
        wanted.setdefault(i, []).append(j)
    knn, values, above = crop_pass(
        [signatures[p] for p in images], crop, k=crop_k, floor=crop_floor, wanted=wanted
    )
    add(knn, "crop_knn")
    ordered = sorted(found)
    a = np.array([i for i, _ in ordered], dtype=np.int64)
    b = np.array([j for _, j in ordered], dtype=np.int64)
    hashes = _to_uint64([fingerprints[p] for p in images])
    table = CandidateTable(
        images=tuple(images),
        a=a,
        b=b,
        bits=_popcount(hashes[a] ^ hashes[b]).astype(np.int16) if len(a) else np.zeros(0, np.int16),
        crop=np.array([values[p] for p in ordered], dtype=np.float32),
        cos_full=pair_cosines(embeddings["full"], a, b),
        cos_centre=pair_cosines(embeddings["centre"], a, b),
        sources=np.array([found[p] for p in ordered], dtype=np.int16),
    )
    return table, above


# ------------------------------------------------------------------ rules


@dataclass(frozen=True)
class JoinRule:
    """Join two images if their cosine in ``view`` is at least ``threshold``, or (when
    ``crop_threshold`` is set) their crop correlation is at least ``crop_threshold``.

    ``view == "crop"`` is the earlier rule of D02 amendment 2 (dHash candidates within 12 bits
    with a crop correlation of at least ``threshold``), kept to reproduce RC-D1 run 3.
    """

    view: str
    threshold: float
    crop_threshold: float | None = None

    @property
    def label(self) -> str:
        """``centre:0.91`` or ``max:0.88:or_crop:0.65`` (used in the result metric names)."""
        text = f"{self.view}:{self.threshold:.2f}"
        if self.crop_threshold is not None:
            text += f":or_crop:{self.crop_threshold:.2f}"
        return text

    @property
    def complexity(self) -> int:
        """0 for one cosine view, 1 with the crop part (simpler rules win ties)."""
        return int(self.crop_threshold is not None)

    def joins(self, table: CandidateTable) -> np.ndarray:
        """Which candidate pairs the rule joins."""
        if self.view == "crop":
            return (table.bits <= 12) & (table.crop >= self.threshold)
        joined = table.cosine(self.view) >= self.threshold
        if self.crop_threshold is not None:
            joined |= table.crop >= self.crop_threshold
        return joined

    def code(self) -> dict[str, float]:
        """The rule as numbers (results files): view code, threshold, crop threshold or -1."""
        return {
            "view_code": float(VIEW_CODES[self.view]),
            "threshold": float(self.threshold),
            "crop_threshold": -1.0 if self.crop_threshold is None else float(self.crop_threshold),
        }

    @classmethod
    def from_code(cls, view_code: float, threshold: float, crop_threshold: float) -> JoinRule:
        """The inverse of :meth:`code`."""
        view = {v: k for k, v in VIEW_CODES.items()}[int(round(view_code))]
        crop = None if crop_threshold < 0 else round(float(crop_threshold), 4)
        return cls(view, round(float(threshold), 4), crop)


RULE_VIEWS: tuple[str, ...] = ("full", "centre", "max")
RULE_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.01 * k, 2) for k in range(50))
RULE_CROP_LEVELS: tuple[float, ...] = (0.65, 0.80)


def rule_family() -> list[JoinRule]:
    """Every candidate rule, fixed in advance (D02 amendment 3): each cosine view (full,
    centre, the larger of the two) at 0.50 to 0.99 in steps of 0.01, alone and OR a crop
    correlation of at least 0.65 or 0.80."""
    rules = []
    for view in RULE_VIEWS:
        for t in RULE_THRESHOLDS:
            rules.append(JoinRule(view, t))
            rules += [JoinRule(view, t, c) for c in RULE_CROP_LEVELS]
    return rules


def joined_pairs_all(
    rule: JoinRule, embeddings: Mapping[str, np.ndarray], crop_pairs: CropPairs | None
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of images (i < j) the rule joins, over **all** pairs: exact."""
    if rule.view == "crop":
        raise ValueError("the crop rule of amendment 2 is applied by dedup.two_stage_groups")
    views = ("full", "centre") if rule.view == "max" else (rule.view,)
    parts_a, parts_b = [], []
    for view in views:
        a, b, _ = pairs_at_or_above(embeddings[view], rule.threshold)
        parts_a.append(a)
        parts_b.append(b)
    if rule.crop_threshold is not None:
        if crop_pairs is None or crop_pairs.floor > rule.crop_threshold:
            raise ValueError("the crop pairs do not reach down to the rule's crop level")
        keep = crop_pairs.value >= rule.crop_threshold
        parts_a.append(crop_pairs.a[keep])
        parts_b.append(crop_pairs.b[keep])
    a, b = np.concatenate(parts_a), np.concatenate(parts_b)
    if len(a) == 0:
        return a, b
    unique = np.unique(np.stack([a, b], axis=1), axis=0)
    return unique[:, 0], unique[:, 1]


def component_groups(images: Sequence[str], a: np.ndarray, b: np.ndarray) -> dict[str, str]:
    """``{image: group id}`` from joined index pairs: connected components, the group id
    being the smallest image key of the component (as ``dedup.union_groups``)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    order = sorted(range(len(images)), key=lambda i: images[i])
    rank = np.empty(len(images), dtype=np.int64)
    rank[order] = np.arange(len(images))
    n = len(images)
    graph = coo_matrix((np.ones(len(a)), (rank[a], rank[b])), shape=(n, n)) if n else None
    if graph is None:
        return {}
    _, labels = connected_components(graph, directed=False)
    smallest: dict[int, int] = {}
    for position, label in enumerate(labels):  # positions are in key order
        smallest.setdefault(int(label), position)
    return {images[i]: images[order[smallest[int(labels[rank[i]])]]] for i in range(n)}


def largest_group(groups: Mapping[str, str]) -> int:
    """The number of images in the largest group."""
    sizes: dict[str, int] = {}
    for g in groups.values():
        sizes[g] = sizes.get(g, 0) + 1
    return max(sizes.values(), default=0)
