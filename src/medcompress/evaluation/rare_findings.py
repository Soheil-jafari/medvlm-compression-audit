"""Rare-finding recall by prevalence group (common / medium / rare, fixed in advance)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def recall_by_prevalence_group(
    detected: Mapping[str, Sequence[bool]], groups: Mapping[str, str]
) -> dict[str, float]:
    """Return recall per prevalence group, pooling positives of findings in each group.

    ``groups`` maps finding -> "common" | "medium" | "rare" from the domain config.
    Whether to pool positives or average per-finding recall is pre-registered.

    Used by: all experiments.
    """
    raise NotImplementedError("TODO: recall per prevalence group.")
