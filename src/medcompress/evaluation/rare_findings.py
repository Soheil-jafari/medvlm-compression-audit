"""Rare-finding recall by prevalence group (common / medium / rare, fixed in advance)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

POOLING_RULES: tuple[str, ...] = ("pooled", "mean_per_finding")


def recall_by_prevalence_group(
    detected: Mapping[str, Sequence[bool]],
    groups: Mapping[str, str],
    *,
    pooling: str = "pooled",
) -> dict[str, float]:
    """Return recall per prevalence group.

    ``detected`` maps each finding to one entry per positive case (detected or
    not). ``groups`` maps finding -> "common" | "medium" | "rare" from the domain
    config. ``pooling="pooled"`` pools positives of all findings in a group;
    ``"mean_per_finding"`` averages per-finding recall. Which one is used is
    pre-registered. A group without positives gets NaN.

    Used by: all experiments.
    """
    if pooling not in POOLING_RULES:
        raise ValueError(f"pooling must be one of {POOLING_RULES}")
    missing = set(detected) - set(groups)
    if missing:
        raise ValueError(f"Findings without a prevalence group: {sorted(missing)}")
    out: dict[str, float] = {}
    for group in sorted(set(groups.values())):
        members = [f for f in detected if groups[f] == group]
        if pooling == "pooled":
            hits = sum(sum(bool(x) for x in detected[f]) for f in members)
            total = sum(len(detected[f]) for f in members)
            out[group] = hits / total if total else math.nan
        else:
            recalls = [
                sum(bool(x) for x in detected[f]) / len(detected[f]) for f in members if detected[f]
            ]
            out[group] = sum(recalls) / len(recalls) if recalls else math.nan
    return out
