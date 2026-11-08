"""PolypGen, the external co-primary test (D02 amendment 7; RC-X2): loader and overlap check.

Licence (Ali et al., Sci Data 2023, as read by the team on 2026-11-08): CC BY 4.0, research use
allowed, citation and acknowledgement required. Confirming the terms on the release page is the
downloader's step; nothing here downloads anything.

**Layout.** The loader does not assume file names beyond these rules, which the ``describe``
step prints counts for before anything is used (aggregate counts only leave the machine):

- an image is any image file under the root whose path has no ``mask`` part;
- its **centre** is the first ``C<k>`` token in its path (``data_C1``, ``images_C4``, ...);
- its **sequence** is the first path part like ``seq<k>`` (a video sequence); a frame outside a
  sequence is its own cluster;
- its **mask** is the file with the same stem (optionally with a ``_mask`` suffix) in the
  folder whose path replaces ``images`` by ``masks``; a frame without a mask file counts as a
  negative only when it lies under a folder named ``negative*`` (negative-only sequences),
  otherwise it is reported as unlabelled and left out.

**Probes** (part B plan of RC-X2): one per image, polyp present (a mask with a polyp pixel) or
absent; the lesion area is the mask's polyp pixels over all pixels, stratified with **our**
training quartiles (unchanged). **The bootstrap clusters by video sequence, not by frame**
(:attr:`PolypGenItem.cluster`).

**Overlap check** (:func:`heldout_flags`): the held-out filter of D02 amendment 6 against our
training split, with the same settings as for our test (full-view cosine >= F = 0.86 with the
pinned embeddings, or crop correlation >= 0.70, or within 4 dHash bits with crop >= 0.50).
Flagged images are removed and counted per centre. Centre C4 is Oslo (a Norwegian centre, as
our Kvasir sources are), so its counts are reported apart and the external test without C4 is
a pre-registered sensitivity analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
HELDOUT_F = 0.86
"""F of the held-out filter (D02 amendment 7), the same as for our filtered test."""
SENSITIVITY_CENTRE = "C4"
"""Oslo: reported apart; the external test without it is a sensitivity analysis."""
_CENTRE = re.compile(r"(?:^|[_\-/])C(\d+)(?:$|[_\-/])", re.IGNORECASE)
_SEQUENCE = re.compile(r"^seq[_\-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class PolypGenItem:
    """One PolypGen image: where it is, its mask (if any), centre, sequence and label."""

    image: str
    mask: str | None
    centre: str
    sequence: str | None
    negative_folder: bool

    @property
    def cluster(self) -> str:
        """The bootstrap cluster: the video sequence, or the frame itself outside one."""
        if self.sequence is not None:
            return f"{self.centre}:{self.sequence}"
        return f"{self.centre}:{self.image}"

    @property
    def labelled(self) -> bool:
        """A mask, or a frame of a negative-only folder."""
        return self.mask is not None or self.negative_folder


def centre_of(relative: str) -> str:
    """``C<k>`` of the first centre token in the path ("" if none)."""
    match = _CENTRE.search("/" + relative.replace("\\", "/"))
    return f"C{int(match.group(1))}" if match else ""


def sequence_of(relative: str) -> str | None:
    """The first ``seq<k>`` part of the path, or None."""
    for part in Path(relative).parts[:-1]:
        match = _SEQUENCE.match(part)
        if match:
            return f"seq{int(match.group(1))}"
    return None


def _mask_for(root: Path, image: Path) -> Path | None:
    rel = image.relative_to(root)
    parts = [re.sub("images", "masks", p, flags=re.IGNORECASE) for p in rel.parts[:-1]]
    folder = root.joinpath(*parts) if parts else root
    if folder == image.parent:
        return None
    for stem in (image.stem, f"{image.stem}_mask"):
        for suffix in IMAGE_SUFFIXES:
            candidate = folder / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def scan(root: Path | str) -> list[PolypGenItem]:
    """Every image under ``root`` with its mask, centre and sequence (see the module rules)."""
    root = Path(root)
    items = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        if any("mask" in part.lower() for part in Path(rel).parts):
            continue
        mask = _mask_for(root, path)
        items.append(
            PolypGenItem(
                image=rel,
                mask=None if mask is None else mask.relative_to(root).as_posix(),
                centre=centre_of(rel),
                sequence=sequence_of(rel),
                negative_folder=any(p.lower().startswith("negative") for p in Path(rel).parts),
            )
        )
    return items


def polyp_area(root: Path | str, item: PolypGenItem) -> float:
    """The polyp pixels over all pixels of the item's mask (0 for a negative frame)."""
    from medcompress.data.masks import lesion_area_fraction, load_mask

    if item.mask is None:
        return 0.0
    return lesion_area_fraction(load_mask(str(Path(root) / item.mask)))


def describe_rows(items: Sequence[PolypGenItem], areas: Mapping[str, float] | None = None) -> list:
    """Aggregate counts (results schema, experiment ``DATA``): per centre the images, masks,
    sequences, clusters, unlabelled frames and, with ``areas``, positives and negatives."""
    rows: list[dict[str, Any]] = []

    def row(metric: str, value: float) -> None:
        rows.append({"experiment": "DATA", "budget": "", "seed": 0, "metric": metric,
                     "value": float(value)})  # fmt: skip

    by_centre: dict[str, list[PolypGenItem]] = {}
    for item in items:
        by_centre.setdefault(item.centre or "unknown", []).append(item)
    row("polypgen:images", len(items))
    row("polypgen:clusters", len({i.cluster for i in items}))
    for centre, group in sorted(by_centre.items()):
        row(f"polypgen:images:{centre}", len(group))
        row(f"polypgen:with_mask:{centre}", sum(i.mask is not None for i in group))
        row(f"polypgen:sequences:{centre}", len({i.sequence for i in group if i.sequence}))
        row(f"polypgen:clusters:{centre}", len({i.cluster for i in group}))
        row(f"polypgen:unlabelled:{centre}", sum(not i.labelled for i in group))
        if areas is not None:
            labelled = [i for i in group if i.labelled]
            row(f"polypgen:positives:{centre}", sum(areas.get(i.image, 0.0) > 0 for i in labelled))
            row(f"polypgen:negatives:{centre}", sum(areas.get(i.image, 0.0) == 0 for i in labelled))
    return rows


def heldout_flags(
    external: Sequence[str],
    train: Sequence[str],
    fingerprints: Mapping[str, str],
    signatures: Mapping[str, Any],
    embeddings: Mapping[str, Any],
    images: Sequence[str],
    crop: Any,
) -> dict[str, dict[str, bool]]:
    """``{external image: {reason: removed}}`` by the held-out filter against ``train``.

    Keys of the mappings are the image keys of both sets (external and ours must not collide);
    ``embeddings["full"]`` has one row per key of ``images``.
    """
    from medcompress.data.heldout_filter import HeldoutFilter, heldout_neighbours

    neighbours = heldout_neighbours(
        {"train": list(train), "test": list(external)}, fingerprints, signatures, crop,
        embeddings, images,
    )  # fmt: skip
    rule = HeldoutFilter(HELDOUT_F)
    return {path: rule.reasons(neighbours, path) for path in external}


def overlap_rows(items: Sequence[PolypGenItem], flags: Mapping[str, Mapping[str, bool]]) -> list:
    """Counts of the overlap check per centre (removed in all, and by reason)."""
    rows = []
    removed = Counter()
    reasons = Counter()
    for item in items:
        why = flags.get(item.image) or {}
        if any(why.values()):
            removed[item.centre] += 1
            for reason, hit in why.items():
                reasons[(item.centre, reason)] += int(hit)
    centres = sorted({i.centre for i in items})
    for centre in centres:
        total = sum(i.centre == centre for i in items)
        rows.append(("polypgen:heldout_removed:" + centre, removed[centre]))
        rows.append(("polypgen:heldout_checked:" + centre, total))
        for reason in ("cosine", "crop", "dhash"):
            rows.append(
                (f"polypgen:heldout_removed_by:{reason}:{centre}", reasons[(centre, reason)])
            )
    rows.append(("polypgen:heldout_removed", sum(removed.values())))
    rows.append(("polypgen:heldout_f", HELDOUT_F))
    return [{"experiment": "DATA", "budget": "", "seed": 0, "metric": m, "value": float(v)}
            for m, v in rows]  # fmt: skip
