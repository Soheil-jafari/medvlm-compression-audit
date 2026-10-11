"""Common interface and registry for pruning criteria.

A criterion assigns an importance score to every candidate unit; units with the
LOWEST scores are removed first until the measured budget is met.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

CRITERIA: dict[str, type[PruningCriterion]] = {}


def register_criterion(name: str) -> Callable[[type[PruningCriterion]], type[PruningCriterion]]:
    """Class decorator that registers a criterion under ``name``."""

    def decorator(cls: type[PruningCriterion]) -> type[PruningCriterion]:
        if name in CRITERIA:
            raise ValueError(f"Criterion {name!r} is already registered")
        cls.name = name
        CRITERIA[name] = cls
        return cls

    return decorator


def get_criterion(name: str) -> type[PruningCriterion]:
    """Return the criterion class registered under ``name``."""
    try:
        return CRITERIA[name]
    except KeyError:
        raise KeyError(f"Unknown criterion {name!r}. Known: {sorted(CRITERIA)}") from None


class PruningCriterion(ABC):
    """Base class: score units on calibration data; higher score = more important."""

    name: str = "base"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return ``{unit_id: importance}`` for every unit."""
        raise NotImplementedError("TODO: implemented by each criterion.")

    def rank(self, scores: Mapping[str, float]) -> list[str]:
        """Return unit IDs from least to most important (removal order).

        Tie-breaking must be deterministic (e.g. by unit ID) and documented.
        """
        raise NotImplementedError("TODO: deterministic ascending sort with tie-breaking.")


def score_units_for_experiment(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/04_score_pruning_units.py``.

    Loads the parent model (E2), enumerates units, runs the configured criterion
    (``pruning.criterion``, or the variant for E5) and writes
    ``unit_scores.csv`` to the run directory.

    Used by: E5, E6, E7, A1-A6.
    """
    raise NotImplementedError("TODO: wire config -> model -> units -> criterion -> scores CSV.")
