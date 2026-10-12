"""Near-duplicate detection between training images and evaluation sets.

Public endoscopy and fundus datasets often reuse the same source images. Before
any external-test number is reported, external images that are near-duplicates
of training images are removed. The fingerprint method and threshold are fixed
in the preregistration; removed counts go into the data card.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medcompress.data.base import VQASample


def image_fingerprint(path: str) -> str:
    """Return a perceptual hash of one image (imaging libraries are imported lazily)."""
    raise NotImplementedError("TODO: perceptual hash (method fixed in the preregistration).")


def find_near_duplicates(
    reference: Sequence[VQASample], candidates: Sequence[VQASample], *, threshold: int
) -> list[tuple[str, str]]:
    """Return (candidate sample_id, reference sample_id) pairs within ``threshold`` distance."""
    raise NotImplementedError("TODO: pairwise fingerprint distance with a threshold.")


def deduplicate(
    reference: Sequence[VQASample], candidates: Sequence[VQASample], *, threshold: int
) -> list[VQASample]:
    """Return ``candidates`` without near-duplicates of ``reference`` (e.g. external vs train)."""
    raise NotImplementedError("TODO: drop near-duplicates and report the count.")
