"""Near-duplicate detection between training images and evaluation sets.

Public endoscopy and fundus datasets often reuse the same source images. Before
any external-test number is reported, external images that are near-duplicates
of training images are removed. The same fingerprints link Kvasir-SEG masks to
Kvasir-VQA-x1 images (D02). The fingerprint method and threshold are fixed in the
preregistration; removed counts go into the data card.

Fingerprint: difference hash (dHash) of a grey-scale image resized to 9 x 8, giving
64 bits as 16 hex characters; distance = number of differing bits (Hamming).

Two-stage grouping (D02 amendment 2 of 2026-10-18, final values 2026-10-19): the dHash alone
joins images that look alike only at 9 x 8 pixels (colour, lumen position, the text and the
green box drawn on the image), and misses adjacent frames of one scene. Every pair within
``candidate_bits`` is therefore checked with a finer measure on the centre of the image, where
no overlay is drawn (:class:`CropSettings`, :func:`crop_similarity`), and joined only if that
measure says so: also at 0-4 bits, which the calibration showed can be different images.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from medcompress.data.base import VQASample

HASH_BITS = 64


def image_fingerprint(path: str) -> str:
    """Return the 64-bit dHash of one image as 16 hex characters (Pillow imported lazily)."""
    from PIL import Image

    with Image.open(path) as image:
        return _dhash(image.convert("L"))


def _dhash(grey: Any) -> str:
    from PIL import Image

    pixels = np.asarray(grey.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(a: str, b: str) -> int:
    """Return the number of differing bits between two hex fingerprints."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _to_uint64(fingerprints: Sequence[str]) -> np.ndarray:
    return np.array([int(f, 16) for f in fingerprints], dtype=np.uint64)


def _popcount(x: np.ndarray) -> np.ndarray:
    as_bytes = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(as_bytes, axis=-1).sum(axis=-1)


def near_duplicate_pairs(
    reference: Mapping[str, str], candidates: Mapping[str, str], *, threshold: int
) -> list[tuple[str, str, int]]:
    """Return (candidate key, reference key, distance) for every pair within ``threshold``.

    Inputs map an identifier (image path or sample ID) to its fingerprint.
    """
    if not reference or not candidates:
        return []
    ref_keys = list(reference)
    ref = _to_uint64([reference[k] for k in ref_keys])
    pairs: list[tuple[str, str, int]] = []
    for key, fingerprint in candidates.items():
        distances = _popcount(ref ^ np.uint64(int(fingerprint, 16)))
        for idx in np.nonzero(distances <= threshold)[0]:
            pairs.append((key, ref_keys[int(idx)], int(distances[idx])))
    return pairs


def _fingerprints(samples: Sequence[VQASample], root: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for sample in samples:
        if sample.image_path not in out:
            path = Path(root) / sample.image_path if root else Path(sample.image_path)
            out[sample.image_path] = image_fingerprint(str(path))
    return out


def find_near_duplicates(
    reference: Sequence[VQASample],
    candidates: Sequence[VQASample],
    *,
    threshold: int,
    root: Path | None = None,
) -> list[tuple[str, str]]:
    """Return (candidate sample_id, reference sample_id) pairs within ``threshold`` distance.

    Images are read from ``root / image_path`` (or ``image_path`` if no root).
    """
    ref_fp = _fingerprints(reference, root)
    cand_fp = _fingerprints(candidates, root)
    image_pairs = near_duplicate_pairs(ref_fp, cand_fp, threshold=threshold)
    by_image_ref: dict[str, list[str]] = {}
    for s in reference:
        by_image_ref.setdefault(s.image_path, []).append(s.sample_id)
    by_image_cand: dict[str, list[str]] = {}
    for s in candidates:
        by_image_cand.setdefault(s.image_path, []).append(s.sample_id)
    return sorted(
        (c, r)
        for cand_img, ref_img, _ in image_pairs
        for c in by_image_cand[cand_img]
        for r in by_image_ref[ref_img]
    )


def deduplicate(
    reference: Sequence[VQASample],
    candidates: Sequence[VQASample],
    *,
    threshold: int,
    root: Path | None = None,
) -> list[VQASample]:
    """Return ``candidates`` without near-duplicates of ``reference`` (e.g. external vs train)."""
    duplicated = {
        c for c, _ in find_near_duplicates(reference, candidates, threshold=threshold, root=root)
    }
    return [s for s in candidates if s.sample_id not in duplicated]


IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def folder_fingerprints(folder: Path) -> dict[str, str]:
    """dHash of every image file under ``folder``; keys are paths relative to it.

    Works on any external set before its question format is known (RC-X1 part A).
    """
    folder = Path(folder)
    return {
        path.relative_to(folder).as_posix(): image_fingerprint(str(path))
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def overlap_rows(
    name: str,
    candidates: Mapping[str, str],
    splits: Mapping[str, Mapping[str, str]],
    *,
    threshold: int,
) -> list[dict[str, object]]:
    """Counts of external images that are near-duplicates of each of our splits.

    ``candidates`` maps external images to fingerprints, ``splits`` maps a split name to the
    fingerprints of its images. Only counts are returned (results schema, ``experiment`` =
    ``name``): no image name leaves the machine.
    """
    rows: list[dict[str, object]] = [
        {"experiment": name, "budget": "", "seed": 0, "metric": "images", "value": len(candidates)},
        {"experiment": name, "budget": "", "seed": 0, "metric": "dedup_threshold",
         "value": threshold},
    ]  # fmt: skip
    hit_any: set[str] = set()
    for split, reference in splits.items():
        hits = {c for c, _, _ in near_duplicate_pairs(reference, candidates, threshold=threshold)}
        hit_any |= hits
        rows.append(
            {"experiment": name, "budget": "", "seed": 0,
             "metric": f"near_duplicates_of:{split}", "value": len(hits)}
        )  # fmt: skip
    rows.append(
        {"experiment": name, "budget": "", "seed": 0, "metric": "images_after_dedup",
         "value": len(candidates) - len(hit_any)}
    )  # fmt: skip
    return rows


def candidate_pairs(
    fingerprints: Mapping[str, str], *, threshold: int
) -> list[tuple[str, str, int]]:
    """Every pair of images within ``threshold`` bits, as (smaller key, larger key, distance).

    Sorted by key, so the result does not depend on input order.
    """
    keys = sorted(fingerprints)
    values = _to_uint64([fingerprints[k] for k in keys])
    pairs: list[tuple[str, str, int]] = []
    for i in range(len(keys) - 1):
        distances = _popcount(values[i + 1 :] ^ values[i])
        for offset in np.nonzero(distances <= threshold)[0]:
            pairs.append((keys[i], keys[i + 1 + int(offset)], int(distances[offset])))
    return pairs


def union_groups(keys: Iterable[str], pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Union-find: ``{image: group id}``, the group id being the smallest key of its group.

    Joined transitively: if a~b and b~c, then a, b and c share one group even when a and c
    are far apart. Does not depend on the order of ``keys`` or ``pairs``.
    """
    ordered = sorted(set(keys))
    index = {key: i for i, key in enumerate(ordered)}
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(index[a]), find(index[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # the root is the smallest index: the smallest key
    return {key: ordered[find(i)] for i, key in enumerate(ordered)}


def near_duplicate_groups(fingerprints: Mapping[str, str], *, threshold: int) -> dict[str, str]:
    """Group images whose fingerprints are within ``threshold`` bits, transitively.

    Returns ``{image: group id}`` (:func:`union_groups`). The dHash-only grouping of RC-D1
    run 2; since D02 amendment 2 the run uses :func:`two_stage_groups`.
    """
    pairs = candidate_pairs(fingerprints, threshold=threshold)
    return union_groups(fingerprints, ((a, b) for a, b, _ in pairs))


def distances_to(reference: Sequence[str], fingerprint: str) -> np.ndarray:
    """The Hamming distance from ``fingerprint`` to each of ``reference``, in that order."""
    if not reference:
        return np.zeros(0, dtype=np.int64)
    return _popcount(_to_uint64(reference) ^ np.uint64(int(fingerprint, 16))).astype(np.int64)


def min_distances(reference: Mapping[str, str], candidates: Mapping[str, str]) -> dict[str, int]:
    """For each candidate image, the smallest Hamming distance to any reference image."""
    if not reference:
        return {key: HASH_BITS for key in candidates}
    ref = _to_uint64(list(reference.values()))
    return {
        key: int(_popcount(ref ^ np.uint64(int(fingerprint, 16))).min())
        for key, fingerprint in candidates.items()
    }


# ------------------------------------------------------------------ D02 amendment 2: the crop


@dataclass(frozen=True)
class CropSettings:
    """The finer measure of the second stage (pre-registered, D02 amendment 2 of 2026-10-18).

    - ``crop_fraction``: the central part kept, as a fraction of the width and of the height
      (0.60: 20 % is cut from each side, where the text, the green box and the black border
      are drawn);
    - ``side``: the crop is made grey-scale and resized (area average) to ``side`` x ``side``;
    - ``highpass_sigma``: a Gaussian blur of this many pixels (at ``side``) is subtracted, so
      the measure follows the mucosa's detail rather than the lighting and the lumen position;
    - ``max_shift``: the crop of one image is compared with the other image shifted by up to
      this many pixels (at ``side``) in each direction (adjacent frames move a little);
    - the measure (:func:`crop_similarity`) is the highest Pearson correlation over those
      shifts, from -1 to 1.
    """

    crop_fraction: float = 0.60
    side: int = 32
    highpass_sigma: float = 4.0
    max_shift: int = 3

    @property
    def grid(self) -> int:
        """The side of the stored signature: the crop plus the shift margin on each side."""
        return self.side + 2 * self.max_shift

    @property
    def region_fraction(self) -> float:
        """The central fraction of the image the stored signature covers."""
        return self.crop_fraction * self.grid / self.side


@dataclass(frozen=True)
class NearDuplicateSettings:
    """Both stages: dHash candidates, the crop confirmation, and the chaining guard.

    Every pair within ``candidate_bits`` is joined when, and only when, its crop correlation is
    at least ``min_correlation`` (no pair is joined on the dHash alone since 2026-10-19).
    ``close_bits`` only names the band of very close pairs that is reported separately (the
    pairs within it that are not joined). The run stops if the largest group holds more than
    ``max_group_fraction`` of all images.
    """

    min_correlation: float
    candidate_bits: int = 12
    max_group_fraction: float = 0.10
    close_bits: int = 4
    crop: CropSettings = CropSettings()

    def rows(self) -> dict[str, float]:
        """The settings, as the aggregate rows of ``summary.csv`` record them."""
        return {
            "close_bits": self.close_bits,
            "candidate_bits": self.candidate_bits,
            "min_correlation": self.min_correlation,
            "max_group_fraction": self.max_group_fraction,
            "crop_fraction": self.crop.crop_fraction,
            "crop_side": self.crop.side,
            "crop_highpass_sigma": self.crop.highpass_sigma,
            "crop_max_shift": self.crop.max_shift,
        }


class ChainingError(RuntimeError):
    """The grouping joined too many images into one group: the run stops (chaining guard)."""


def _blur_matrix(n: int, sigma: float) -> np.ndarray:
    """``B`` such that ``B @ x`` is ``x`` blurred by a Gaussian of ``sigma``, edges mirrored."""
    if sigma <= 0:
        return np.eye(n)
    radius = int(np.ceil(3 * sigma))
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    taps /= taps.sum()
    matrix = np.zeros((n, n))
    for row in range(n):
        for offset, weight in zip(range(-radius, radius + 1), taps, strict=True):
            col = row + offset
            while col < 0 or col >= n:  # mirror at the edges (reflect, without repeating)
                col = -col if col < 0 else 2 * (n - 1) - col
            matrix[row, col] += weight
    return matrix


def crop_signature(grey: Any, crop: CropSettings) -> np.ndarray:
    """The stored signature of one grey-scale PIL image: ``crop.grid`` x ``crop.grid`` floats.

    The central ``crop.region_fraction`` of the image (the crop plus the shift margin), resized
    by area average, with the Gaussian-blurred copy subtracted. Its central ``side`` x ``side``
    window is the crop itself.
    """
    from PIL import Image

    width, height = grey.size
    frac = min(1.0, crop.region_fraction)
    box = (
        width * (1 - frac) / 2, height * (1 - frac) / 2,
        width * (1 + frac) / 2, height * (1 + frac) / 2,
    )  # fmt: skip
    small = grey.resize((crop.grid, crop.grid), Image.Resampling.BOX, box=box)
    pixels = np.asarray(small, dtype=np.float64)
    blur = _blur_matrix(crop.grid, crop.highpass_sigma)
    return pixels - blur @ pixels @ blur.T


def image_features(path: str, crop: CropSettings) -> tuple[str, np.ndarray]:
    """The dHash fingerprint and the crop signature of one image (the file is read once)."""
    from PIL import Image

    with Image.open(path) as image:
        grey = image.convert("L")
        return _dhash(grey), crop_signature(grey, crop)


@dataclass(frozen=True)
class _Prepared:
    centres: np.ndarray  # (n, side * side): the crop, zero mean, unit length
    windows: np.ndarray  # (n, grid, grid): the raw signature, for the shifted windows
    norms: np.ndarray  # (n, shifts, shifts): the length of each shifted window, mean removed


def _prepare_signatures(signatures: Sequence[np.ndarray], crop: CropSettings) -> _Prepared:
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in signatures])
    m, side = crop.max_shift, crop.side
    centres = stack[:, m : m + side, m : m + side].reshape(len(stack), -1)
    centres = centres - centres.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(centres, axis=1, keepdims=True)
    centres = np.divide(centres, lengths, out=np.zeros_like(centres), where=lengths > 1e-9)
    sums, squares = _window_sums(stack, side), _window_sums(stack**2, side)
    norms = np.sqrt(np.maximum(squares - sums**2 / (side * side), 0.0))
    return _Prepared(centres, stack, norms)


def _window_sums(stack: np.ndarray, side: int) -> np.ndarray:
    """The sum of every ``side`` x ``side`` window of each image (summed-area table)."""
    table = np.zeros((stack.shape[0], stack.shape[1] + 1, stack.shape[2] + 1))
    table[:, 1:, 1:] = stack.cumsum(axis=1).cumsum(axis=2)
    return (
        table[:, side:, side:] - table[:, :-side, side:]
        - table[:, side:, :-side] + table[:, :-side, :-side]
    )  # fmt: skip


def _directed(prepared: _Prepared, a: np.ndarray, b: np.ndarray, crop: CropSettings) -> np.ndarray:
    """For pairs (a[k], b[k]): the highest correlation of a's crop with b's shifted windows."""

    from numpy.lib.stride_tricks import sliding_window_view

    side = crop.side
    centres = prepared.centres[a].reshape(len(a), side, side)
    views = sliding_window_view(prepared.windows[b], (side, side), axis=(1, 2))
    dots = np.einsum("kij,kxyij->kxy", centres, views)  # the centre has zero mean
    norms = prepared.norms[b]
    corr = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 1e-9)
    return corr.reshape(len(a), -1).max(axis=1)


def crop_similarities(
    signatures: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[str, str]],
    crop: CropSettings,
    *,
    batch: int = 128,
) -> np.ndarray:
    """The crop correlation of each pair (:func:`crop_similarity`), in the order given."""
    if not pairs:
        return np.zeros(0)
    keys = sorted({k for pair in pairs for k in pair})
    index = {k: i for i, k in enumerate(keys)}
    prepared = _prepare_signatures([signatures[k] for k in keys], crop)
    a_all = np.array([index[a] for a, _ in pairs])
    b_all = np.array([index[b] for _, b in pairs])
    out = np.empty(len(pairs))
    for start in range(0, len(pairs), batch):
        a, b = a_all[start : start + batch], b_all[start : start + batch]
        forward = _directed(prepared, a, b, crop)
        backward = _directed(prepared, b, a, crop)
        out[start : start + batch] = np.maximum(forward, backward)
    return out


def crop_similarity(signature_a: np.ndarray, signature_b: np.ndarray, crop: CropSettings) -> float:
    """The finer measure of one pair: the highest Pearson correlation between the crop of one
    image and the other image shifted by up to ``crop.max_shift`` pixels, both ways round.

    1.0 for identical crops; near 0 for unrelated detail. Overlays outside the crop do not count.
    """
    return float(crop_similarities({"a": signature_a, "b": signature_b}, [("a", "b")], crop)[0])


def cross_similarity_matrix(
    rows: Sequence[np.ndarray], cols: Sequence[np.ndarray], crop: CropSettings
) -> np.ndarray:
    """The crop correlation of every (row image, column image) pair, by matrix products.

    The same measure as :func:`crop_similarities`, in float32 (for the checks after the split
    and the final contact sheet, not for the grouping).
    """
    if not len(rows) or not len(cols):
        return np.zeros((len(rows), len(cols)), dtype=np.float32)
    first, second = _prepare_signatures(rows, crop), _prepare_signatures(cols, crop)
    directions = (  # (the crops, the other side, whether the result is transposed)
        (first.centres.astype(np.float32), second, False),
        (second.centres.astype(np.float32), first, True),
    )
    best = np.full((len(rows), len(cols)), -1.0, dtype=np.float32)
    side = crop.side
    for x in range(2 * crop.max_shift + 1):
        for y in range(2 * crop.max_shift + 1):
            for centres, other, transpose in directions:
                window = other.windows[:, x : x + side, y : y + side]
                flat = window.reshape(len(window), -1).astype(np.float32)
                norms = other.norms[:, x, y].astype(np.float32)[None, :]
                dots = centres @ flat.T
                corr = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 1e-9)
                np.maximum(best, corr.T if transpose else corr, out=best)
    return best


@dataclass(frozen=True)
class Grouping:
    """What :func:`two_stage_groups` decided, for the split and the aggregate rows."""

    groups: Mapping[str, str]
    candidates: tuple[tuple[str, str, int], ...]
    scores: tuple[float, ...]
    joined: tuple[bool, ...]

    @property
    def sizes(self) -> dict[str, int]:
        """``{group id: number of images}``."""
        out: dict[str, int] = {}
        for group in self.groups.values():
            out[group] = out.get(group, 0) + 1
        return out


def joined_pairs(
    candidates: Sequence[tuple[str, str, int]],
    scores: Sequence[float],
    settings: NearDuplicateSettings,
) -> list[bool]:
    """Which candidate pairs are joined: those whose crop correlation reaches the threshold,
    at any dHash distance within the candidates (``candidates`` gives the order)."""
    if len(candidates) != len(scores):
        raise ValueError("one score per candidate pair")
    return [float(s) >= settings.min_correlation for s in scores]


CHAIN_MIN_IMAGES = 3
"""Chaining (a~b and b~c joining a and c) needs at least three images, so a group of one or two
never stops the run. This matters only for tiny sets: at 6,449 images, 10 % is 644 images."""


def check_chaining(groups: Mapping[str, str], max_fraction: float) -> None:
    """Raise :class:`ChainingError` if the largest group holds more than ``max_fraction`` of
    the images (and at least :data:`CHAIN_MIN_IMAGES`)."""
    sizes: dict[str, int] = {}
    for group in groups.values():
        sizes[group] = sizes.get(group, 0) + 1
    if not sizes:
        return
    largest = max(sizes.values())
    if largest >= CHAIN_MIN_IMAGES and largest > max_fraction * len(groups):
        top = sorted(sizes.values(), reverse=True)[:5]
        raise ChainingError(
            f"the largest near-duplicate group holds {largest} of {len(groups)} images "
            f"({largest / len(groups):.1%}), more than {max_fraction:.0%}: the grouping chains "
            f"unrelated images. STOP and report (the five largest groups: {top})"
        )


def two_stage_groups(
    fingerprints: Mapping[str, str],
    signatures: Mapping[str, np.ndarray],
    settings: NearDuplicateSettings,
) -> Grouping:
    """Candidates by dHash, confirmation by the crop, union-find, then the chaining guard.

    1. candidates: pairs within ``settings.candidate_bits``;
    2. joined: the candidates whose crop correlation is at least ``min_correlation``;
    3. union-find over the joined pairs (:func:`union_groups`);
    4. :class:`ChainingError` if the largest group holds more than ``max_group_fraction``.
    """
    candidates = candidate_pairs(fingerprints, threshold=settings.candidate_bits)
    scores = crop_similarities(signatures, [(a, b) for a, b, _ in candidates], settings.crop)
    joined = joined_pairs(candidates, scores, settings)
    groups = union_groups(
        fingerprints, ((a, b) for (a, b, _), keep in zip(candidates, joined, strict=True) if keep)
    )
    check_chaining(groups, settings.max_group_fraction)
    return Grouping(groups, tuple(candidates), tuple(float(s) for s in scores), tuple(joined))
