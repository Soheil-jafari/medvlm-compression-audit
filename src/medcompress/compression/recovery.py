"""Shared recovery training after pruning, IDENTICAL for every pruning arm.

Fairness rule: E5 (all criteria), E6, E7 and A1-A6 use exactly the same recovery
recipe from ``configs/recovery.yaml``: LoRA fine-tuning plus offline distillation
from the cached E0 teacher outputs, same data subset, same number of steps, same
hyper-parameters. Only the pruned model differs. :func:`recovery_fingerprint` is
logged with every run so reviewers can verify this.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from medcompress.config import RunContext
    from medcompress.models.base import VLMWrapper


def recovery_fingerprint(recovery_config: Mapping[str, Any]) -> str:
    """Return a short stable hash of the recovery config (logged with every pruning run)."""
    payload = json.dumps(recovery_config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def run_recovery(
    pruned: VLMWrapper,
    teacher_cache_dir: Path,
    recovery_config: Mapping[str, Any],
    *,
    seed: int,
) -> VLMWrapper:
    """Run the shared LoRA + offline-KD recovery on a pruned model.

    Used by: E5, E6, E7, A1-A6 (3 seeds each).
    """
    raise NotImplementedError("TODO: shared LoRA + offline KD recovery (E5-E7, A1-A6).")


def run_prune_and_recover(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/05_prune_and_recover.py``.

    Reads ``unit_scores.csv`` from step 04, searches the removal set for the
    budget (:func:`medcompress.budgets.search_matched_budget`), physically
    removes units, runs :func:`run_recovery`, then the mask audit
    (:func:`medcompress.evaluation.image_reliance.mask_audit`).

    Used by: E5, E6, E7, A1-A6.
    """
    raise NotImplementedError("TODO: scores -> budget search -> remove -> recover -> audit.")
