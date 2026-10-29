"""Read the splits written by the data-preparation step (run card RC-D1).

``prepare`` writes ``splits.json`` (sample id to split), ``excluded_images.json`` and
``mask_links.json`` to a private directory outside the repository. Every later stage
rebuilds the same ``VQASample`` lists from the dataset on its own machine and that
private directory, so no per-image information ever has to be committed.

With the held-out filter of D02 amendment 6, ``prepare`` also writes ``test_unfiltered.json``:
the sample ids of the test split before the filter (the filtered test with the removed test
images added back). ``load_prepared(..., unfiltered_test=True)`` returns it as the split
``test_unfiltered``, for the unfiltered-test sensitivity analysis.
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

UNFILTERED_TEST = "test_unfiltered"
UNFILTERED_TEST_FILE = "test_unfiltered.json"
"""The sample ids of the unfiltered test (D02 amendment 6), in the private folder."""


def load_prepared(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    private_dir: Path,
    *,
    expected_split_sha256: str | None = None,
    unfiltered_test: bool = False,
) -> dict[str, list[VQASample]]:
    """Return ``{split: samples}`` for the splits recorded in ``private_dir``.

    Samples that are not in ``splits.json`` (near-duplicates that were dropped) are
    omitted. Mask paths and lesion areas come from ``mask_links.json``. If
    ``expected_split_sha256`` is given (from the pre-registration), the rebuilt splits
    must hash to it. ``unfiltered_test``: add the split ``test_unfiltered`` from
    ``test_unfiltered.json`` (D02 amendment 6); it is not part of the split hash.

    Raises:
        FileNotFoundError: the preparation outputs are missing.
        ValueError: the rebuilt splits do not match ``expected_split_sha256``, or the
            dataset on this machine lacks samples that the splits list.
    """
    data_root, private_dir = Path(data_root), Path(private_dir)
    if not (private_dir / "splits.json").is_file():
        from medcompress.config import ConfigError

        raise ConfigError(
            f"the RC-D1 files are not at {private_dir} (splits.json): attach the shared data "
            "dataset and pass --data-root <its path>"
        )
    assignment: dict[str, str] = json.loads((private_dir / "splits.json").read_text("utf-8"))
    unfiltered: set[str] = set()
    if unfiltered_test:
        path = private_dir / UNFILTERED_TEST_FILE
        if not path.is_file():
            from medcompress.config import ConfigError

            raise ConfigError(
                f"{path} is missing: the unfiltered test exists only when RC-D1 run 4b applied "
                "the held-out filter (D02 amendment 6)"
            )
        unfiltered = set(json.loads(path.read_text("utf-8")))
    links: dict[str, str] = json.loads((private_dir / "mask_links.json").read_text("utf-8"))
    areas: dict[str, float] = {}
    names = set(assignment.values()) | {"train", "val", "test"}  # a split may be empty
    if unfiltered_test:
        names.add(UNFILTERED_TEST)
    out: dict[str, list[VQASample]] = {name: [] for name in sorted(names)}
    seen: set[str] = set()
    for sample in KvasirVQAx1(dataset_config, data_root=data_root).load():
        name = assignment.get(sample.sample_id)
        wanted = [name] if name is not None else []
        if sample.sample_id in unfiltered:  # removed test images are in this list only
            wanted.append(UNFILTERED_TEST)
        if not wanted:
            continue
        seen.add(sample.sample_id)
        mask = links.get(sample.image_path)
        if mask is not None and mask not in areas:
            areas[mask] = lesion_area_fraction(load_mask(str(data_root / mask)))
        for split in wanted:
            out[split].append(
                replace(
                    sample,
                    split=split,
                    mask_path=mask,
                    lesion_area=areas.get(mask) if mask is not None else None,
                )
            )
    missing = sorted((set(assignment) | unfiltered) - seen)
    if missing:
        raise ValueError(
            f"{len(missing)} samples listed in splits.json are not in the dataset here "
            f"(first: {missing[0]}); is this the same dataset version?"
        )
    if expected_split_sha256 is not None:
        actual = split_fingerprint({k: v for k, v in out.items() if k != UNFILTERED_TEST})
        if actual != expected_split_sha256:
            raise ValueError(f"split fingerprint {actual} != expected {expected_split_sha256}")
    return out
