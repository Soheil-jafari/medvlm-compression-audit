"""Stage 04: score the prunable units of the E2 student with the experiment's criterion.

The scores do not depend on the budget, so they are written once per (experiment, variant,
seed) to the run directory without a budget; stage 05 reads them back for every budget.
A second call with the same settings reuses the file instead of scoring again, because
exact scoring of a full unit set takes hours on a T4.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.compression.pruning.units import UnitType
from medcompress.config import (
    ConfigError,
    RunContext,
    apply_variant,
    collect_open_settings,
    find_todos,
)

if TYPE_CHECKING:
    from medcompress.compression.pruning.criteria.base import PruningCriterion
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

SCORES_FILE = "unit_scores.csv"
SETTINGS_FILE = "unit_scores.settings.json"


def _open(value: Any) -> bool:
    return value is None or bool(find_todos({"v": value}))


@dataclass(frozen=True)
class ScoringSettings:
    """What stage 04 needs from the config, resolved for one variant.

    ``calibration_size`` is None for a criterion that never looks at data.
    """

    criterion: str
    unit_types: tuple[str, ...]
    mlp_channel_group_size: int | None
    calibration_size: int | None

    @classmethod
    def from_config(cls, config: Mapping[str, Any], variant: str | None = None) -> ScoringSettings:
        """Read ``pruning.*`` after resolving the ``*_by_variant`` tables.

        Raises:
            ConfigError: a needed value is still ``TODO`` (all of them are listed) or invalid.
        """
        from medcompress.compression.pruning.criteria import CRITERIA

        pruning = apply_variant(config, variant).get("pruning") or {}
        units = pruning.get("units") or {}
        name = pruning.get("criterion")
        wanted: dict[str, Any] = {"pruning.criterion": name}
        if name in CRITERIA:
            needs_data = CRITERIA[str(name)].needs_calibration
        else:
            needs_data = True
        types = units.get("types")
        wanted["pruning.units.types"] = types
        group = units.get("mlp_channel_group_size")
        if isinstance(types, list) and UnitType.MLP_CHANNEL_GROUP.value in types:
            wanted["pruning.units.mlp_channel_group_size"] = group
        size = (pruning.get("calibration") or {}).get("size")
        if needs_data:
            wanted["pruning.calibration.size"] = size
        open_keys = [key for key, value in wanted.items() if _open(value)]
        if open_keys:
            raise ConfigError(f"settings still open: {', '.join(open_keys)}")
        if name not in CRITERIA:
            raise ConfigError(f"pruning.criterion {name!r} is not one of {sorted(CRITERIA)}")
        known = {t.value for t in UnitType}
        if not isinstance(types, list) or not types or not set(types) <= known:
            raise ConfigError(f"pruning.units.types must be a non-empty list of {sorted(known)}")
        if units.get("include_vision_blocks"):
            raise ConfigError("pruning.units.include_vision_blocks: vision blocks are not wired")
        channel_groups = UnitType.MLP_CHANNEL_GROUP.value in types
        if channel_groups and (not isinstance(group, int) or group <= 0):
            raise ConfigError("pruning.units.mlp_channel_group_size must be a positive integer")
        if needs_data and (not isinstance(size, int) or size <= 0):
            raise ConfigError("pruning.calibration.size must be a positive integer")
        return cls(
            criterion=str(name),
            unit_types=tuple(types),
            mlp_channel_group_size=int(group) if channel_groups else None,
            calibration_size=int(size) if needs_data else None,
        )


def scores_run(run: RunContext) -> RunContext:
    """The run whose directory holds the scores of ``run`` (the same without a budget)."""
    return RunContext(run.experiment_id, seed=run.seed, budget=None, variant=run.variant)


def scores_path(config: Mapping[str, Any], run: RunContext) -> Path:
    """Where ``unit_scores.csv`` of ``run`` lives (shared by all budgets)."""
    return scores_run(run).run_dir(config["paths"]["results_root"]) / SCORES_FILE


PARENT = "E2"
"""The training run whose merged adapter every pruning arm starts from."""


def parent_adapter_digest(config: Mapping[str, Any], seed: int) -> str | None:
    """SHA-256 over the files of the E2 adapter of ``seed`` (None when it is not there).

    Part of the scores fingerprint, so scores of an older E2 are not reused after E2 was
    trained again.
    """
    import hashlib

    folder = Path(config["paths"]["checkpoint_root"]) / PARENT / f"seed{seed}" / "adapter"
    if not folder.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _fingerprint(
    effective: Mapping[str, Any], settings: ScoringSettings, run: RunContext
) -> dict[str, Any]:
    """What the scores depend on, as JSON would read it back (tuples as lists).

    ``effective`` is the config with the run's variant applied: the criterion adds its own
    settings (:meth:`PruningCriterion.settings_fingerprint`) and the E2 adapter is hashed.
    """
    from medcompress.compression.pruning.criteria import CRITERIA

    values = {
        **asdict(settings),
        "seed": run.seed,
        "variant": run.variant,
        "criterion_settings": CRITERIA[settings.criterion].settings_fingerprint(effective),
        "parent_adapter": parent_adapter_digest(effective, run.seed),
    }
    return json.loads(json.dumps(values, sort_keys=True, default=str))


def _reusable(target: Path, fingerprint: Mapping[str, Any]) -> bool:
    """True when ``target`` was written with exactly these settings."""
    from medcompress.utils.io import read_json

    sidecar = target.with_name(SETTINGS_FILE)
    return target.exists() and sidecar.exists() and read_json(sidecar) == dict(fingerprint)


def run_scoring(
    config: Mapping[str, Any],
    run: RunContext,
    model: VLMWrapper,
    train: Sequence[VQASample],
    *,
    criterion: PruningCriterion | None = None,
    force: bool = False,
) -> Path:
    """Score the units of ``model`` and write ``unit_scores.csv`` (see the module docstring).

    ``criterion`` replaces the one the config names (tests and special encoders).
    """
    import math

    from medcompress.compression.pruning.criteria import get_criterion
    from medcompress.compression.pruning.criteria.common import write_unit_scores
    from medcompress.compression.pruning.units import enumerate_units
    from medcompress.data.splits import make_calibration_subset
    from medcompress.utils.io import upsert_results, write_json

    effective = apply_variant(config, run.variant)
    settings = ScoringSettings.from_config(config, run.variant)
    target = scores_path(config, run)
    fingerprint = _fingerprint(effective, settings, run)
    if not force and _reusable(target, fingerprint):
        print(f"reusing {target} (same settings)")
        return target

    units: list[PrunableUnit] = enumerate_units(
        model,
        [UnitType(t) for t in settings.unit_types],
        mlp_channel_group_size=settings.mlp_channel_group_size,
    )
    if not units:
        raise ConfigError("the unit types give no prunable units")
    instance = criterion or get_criterion(settings.criterion)(effective, seed=run.seed)
    calibration: list[VQASample] = []
    if settings.calibration_size is not None:
        calibration = make_calibration_subset(train, settings.calibration_size, seed=run.seed)

    peak = _reset_peak_memory()
    started = time.perf_counter()
    scores = instance.score(model, units, calibration)
    seconds = time.perf_counter() - started

    missing = sorted({u.unit_id for u in units} - set(scores))
    if missing:
        raise ValueError(f"{settings.criterion}: no score for {len(missing)} units: {missing[:3]}")
    bad = sorted(u for u, v in scores.items() if not math.isfinite(v))
    if bad:
        raise ValueError(f"{settings.criterion}: non-finite scores for {len(bad)}: {bad[:3]}")

    write_unit_scores(target, units, scores)
    write_json(fingerprint, target.with_name(SETTINGS_FILE))
    rows = [
        {"experiment": run.experiment_tag, "budget": "", "seed": run.seed, "metric": m, "value": v}
        for m, v in (
            ("scoring_units", len(units)),
            ("scoring_calibration_images", len(calibration)),
            ("scoring_seconds", seconds),
        )
    ]
    if (peak_bytes := peak()) is not None:
        rows.append(
            {
                "experiment": run.experiment_tag,
                "budget": "",
                "seed": run.seed,
                "metric": "scoring_peak_gpu_bytes",
                "value": float(peak_bytes),
            }
        )
    upsert_results(rows, target.with_name("metrics.csv"))
    print(f"scored {len(units)} units with {settings.criterion} in {seconds:.0f} s; wrote {target}")
    return target


def _reset_peak_memory() -> Any:
    """Reset the CUDA peak counter; return a function that reads it (None without CUDA)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement of the stage
        return lambda: None
    if not torch.cuda.is_available():
        return lambda: None
    torch.cuda.reset_peak_memory_stats()
    return lambda: int(torch.cuda.max_memory_allocated())


def _check_criterion(config: Mapping[str, Any], run: RunContext) -> None:
    """Let the criterion check its own settings (needs the criterion name to be valid)."""
    from medcompress.compression.pruning.criteria import CRITERIA

    effective = apply_variant(config, run.variant)
    name = (effective.get("pruning") or {}).get("criterion")
    if name in CRITERIA:
        CRITERIA[str(name)].check_settings(effective)


def _check_side(config: Mapping[str, Any], model: Any) -> None:
    """The image side is needed to load the student (not when a model is passed in)."""
    if model is None and _open(config.get("inputs", {}).get("image_longest_side")):
        raise ConfigError("settings still open: inputs.image_longest_side")


def load_scoring_model(config: Mapping[str, Any], run: RunContext) -> VLMWrapper:
    """The E2 student of the run's seed, adapter merged (the parent of every pruning arm)."""
    from medcompress.evaluation.runner import load_trained_student

    side = config.get("inputs", {}).get("image_longest_side")
    if _open(side):
        raise ConfigError("settings still open: inputs.image_longest_side")
    return load_trained_student(config, "E2", run.seed, int(side))


def load_train_samples(config: Mapping[str, Any]) -> list[VQASample]:
    """Training split of the prepared data (the calibration images are drawn from it)."""
    from medcompress.config import load_config
    from medcompress.data.prepared import load_prepared

    paths = config["paths"]
    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    prepared = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))
    return prepared["train"]


def score_units_for_experiment(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    model: VLMWrapper | None = None,
    train: Sequence[VQASample] | None = None,
    criterion: PruningCriterion | None = None,
    force: bool = False,
) -> None:
    """Stage entry point for ``scripts/04_score_pruning_units.py``.

    Resolves the variant, loads the E2 student and the training split, runs the criterion
    and writes ``unit_scores.csv`` under the run directory. Settings that are still open
    stop the stage before anything is loaded (exit code 4).

    Used by: E5, E6, E7, A1-A6.
    """
    if run is None:
        raise ConfigError("--exp is required")
    holder: dict[str, ScoringSettings] = {}

    def reusable() -> bool:
        """True when the scores of these settings exist and no model has to be loaded."""
        if model is not None or force or "settings" not in holder:
            return False
        effective = apply_variant(config, run.variant)
        return _reusable(scores_path(config, run), _fingerprint(effective, holder["settings"], run))

    collect_open_settings(  # fail early, before any model is loaded, with everything that is open
        lambda: holder.update(settings=ScoringSettings.from_config(config, run.variant)),
        lambda: _check_criterion(config, run),
        lambda: None if reusable() else _check_side(config, model),
    )
    settings = holder["settings"]
    target = scores_path(config, run)
    if reusable():
        print(f"reusing {target} (same settings)")
        return  # nothing to load: the scores of these settings exist already
    if model is None:
        model = load_scoring_model(config, run)
    if train is None:
        train = [] if settings.calibration_size is None else load_train_samples(config)
    run_scoring(config, run, model, train, criterion=criterion, force=force)
