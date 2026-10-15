"""Near-duplicate detection between training images and evaluation sets.

Public endoscopy and fundus datasets often reuse the same source images. Before
any external-test number is reported, external images that are near-duplicates
of training images are removed. The same fingerprints link Kvasir-SEG masks to
Kvasir-VQA-x1 images (D02). The fingerprint method and threshold are fixed in the
preregistration; removed counts go into the data card.

Fingerprint: difference hash (dHash) of a grey-scale image resized to 9 x 8, giving
64 bits as 16 hex characters; distance = number of differing bits (Hamming).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from medcompress.data.base import VQASample

HASH_BITS = 64


def image_fingerprint(path: str) -> str:
    """Return the 64-bit dHash of one image as 16 hex characters (Pillow imported lazily)."""
    from PIL import Image

    with Image.open(path) as image:
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(small, dtype=np.int16)
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
