"""Read the splits written by the data-preparation step (run card RC-D1).

``prepare`` writes ``splits.json`` (sample id to split), ``excluded_images.json`` and
``mask_links.json`` to a private directory outside the repository. Every later stage
rebuilds the same ``VQASample`` lists from the dataset on its own machine and that
private directory, so no per-image information ever has to be committed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from medcompress.data.base import VQASample
from medcompress.data.kvasir import KvasirVQAx1
from medcompress.data.masks import lesion_area_fraction, load_mask
from medcompress.data.splits import split_fingerprint


def load_prepared(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    private_dir: Path,
    *,
    expected_split_sha256: str | None = None,
) -> dict[str, list[VQASample]]:
    """Return ``{split: samples}`` for the splits recorded in ``private_dir``.

    Samples that are not in ``splits.json`` (near-duplicates that were dropped) are
    omitted. Mask paths and lesion areas come from ``mask_links.json``. If
    ``expected_split_sha256`` is given (from the pre-registration), the rebuilt splits
    must hash to it.

    Raises:
        FileNotFoundError: the preparation outputs are missing.
        ValueError: the rebuilt splits do not match ``expected_split_sha256``, or the
            dataset on this machine lacks samples that the splits list.
    """
    data_root, private_dir = Path(data_root), Path(private_dir)
    assignment: dict[str, str] = json.loads((private_dir / "splits.json").read_text("utf-8"))
    links: dict[str, str] = json.loads((private_dir / "mask_links.json").read_text("utf-8"))
    areas: dict[str, float] = {}
    names = set(assignment.values()) | {"train", "val", "test"}  # a split may be empty
    out: dict[str, list[VQASample]] = {name: [] for name in sorted(names)}
    seen: set[str] = set()
    for sample in KvasirVQAx1(dataset_config, data_root=data_root).load():
        name = assignment.get(sample.sample_id)
        if name is None:
            continue
        seen.add(sample.sample_id)
        mask = links.get(sample.image_path)
        if mask is not None and mask not in areas:
            areas[mask] = lesion_area_fraction(load_mask(str(data_root / mask)))
        out[name].append(
            replace(
                sample,
                split=name,
                mask_path=mask,
                lesion_area=areas.get(mask) if mask is not None else None,
            )
        )
    missing = sorted(set(assignment) - seen)
    if missing:
        raise ValueError(
            f"{len(missing)} samples listed in splits.json are not in the dataset here "
            f"(first: {missing[0]}); is this the same dataset version?"
        )
    if expected_split_sha256 is not None:
        actual = split_fingerprint(out)
        if actual != expected_split_sha256:
            raise ValueError(f"split fingerprint {actual} != expected {expected_split_sha256}")
    return out
