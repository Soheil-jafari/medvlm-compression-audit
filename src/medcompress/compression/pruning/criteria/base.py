"""Common interface and registry for pruning criteria.

A criterion assigns an importance score to every candidate unit; units with the
LOWEST scores are removed first until the measured budget is met.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
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
    needs_calibration: bool = True
    """False for a criterion that never looks at calibration data (no ``calibration.size``)."""

    def __init__(self, config: Mapping[str, Any], *, seed: int = 0) -> None:
        self.config = config
        self.seed = seed

    @classmethod
    def check_settings(cls, config: Mapping[str, Any]) -> None:
        """Raise :class:`ConfigError` for a setting of this criterion that is open or invalid.

        Called by stage 04 before any model is loaded; the default has nothing to check.
        """
        return None

    @classmethod
    def settings_fingerprint(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        """Everything besides the units, the data and the model that changes the scores.

        Stage 04 stores it next to ``unit_scores.csv`` and reuses the file only when it is
        unchanged. The default is the criterion's own section ``pruning.<name>``.
        """
        return {"section": (config.get("pruning") or {}).get(cls.name)}

    @property
    def data_root(self) -> Path:
        """``paths.data_root`` of the config (where the images live)."""
        return Path(self.config["paths"]["data_root"])

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

        Ties are broken by unit ID, so the order is deterministic. NaN scores are
        rejected.
        """
        bad = [unit for unit, value in scores.items() if value != value]
        if bad:
            raise ValueError(f"NaN importance for units: {sorted(bad)}")
        return [unit for unit, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))]
