"""Stage 04: score the prunable units of the E2 student with the experiment's criterion.

The scores do not depend on the budget, so they are written once per (experiment, variant,
seed) to the run directory without a budget; stage 05 reads them back for every budget.
A second call with the same settings reuses the file instead of scoring again, because
exact scoring of a full unit set takes hours on a T4.

The clinical criterion also writes its per-finding effects (``unit_effects.csv``: for every
unit and finding, the mean Delta over the finding's probes, or the first-order gate gradient
of the finding's mean log-probability). The score is ``sum_c w_c * effect_c``, so A1 (uniform
weights) and A2 (rarity only), whose other settings equal E7's, re-weight E7's effects of the
same seed instead of scoring again (D07 section 1; decided 2026-11-08): no model is loaded.
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
EFFECTS_FILE = "unit_effects.csv"
EFFECTS_SETTINGS_FILE = "unit_effects.settings.json"
EXACT_RESUME_FILE = "exact_effects.resume.json"
"""The exact effects of the clinical criterion, saved when the exact pass ends and removed once
the scores are written; a rerun with the same settings resumes from it (2026-11-14)."""
EFFECTS_SOURCE = "E7"
"""The run whose per-finding effects A1 and A2 re-weight (same seed)."""
REWEIGHTED_SCHEMES: tuple[str, ...] = ("uniform", "rarity_only")
CHECK_FILE = "first_order_check.csv"
CHECK_GROUPS = 30
"""RC-G2's check of the first-order estimate: this many random channel groups (D07)."""
CHECK_MIN_RANK_CORRELATION = 0.7
"""Below this Spearman correlation the agent stops and asks the team (a report, not a gate;
pre-registration section 11 item 13, correction 1, 2026-11-08)."""


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


def scores_source(config: Mapping[str, Any]) -> str | None:
    """The experiment whose scores this one reuses (``pruning.scores_from``; E7U reads E7's,
    amendment of 2026-11-17), or None."""
    value = (config.get("pruning") or {}).get("scores_from")
    return None if value is None else str(value)


def scores_path(config: Mapping[str, Any], run: RunContext) -> Path:
    """Where ``unit_scores.csv`` of ``run`` lives (shared by all budgets; another experiment's
    file when ``pruning.scores_from`` names one)."""
    source = scores_source(config)
    holder = scores_run(run) if source is None else RunContext(source, seed=run.seed)
    return holder.run_dir(config["paths"]["results_root"]) / SCORES_FILE


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


def effects_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """The scores fingerprint without what only the weights change (the weighting scheme, the
    weight table) and without the variant: equal for E7 and for A1 / A2 of the same seed."""
    out = json.loads(json.dumps(dict(fingerprint)))
    out.pop("variant", None)
    criterion = out.get("criterion_settings") or {}
    criterion.pop("weights", None)
    if isinstance(criterion.get("section"), dict):
        criterion["section"].pop("weighting", None)
    return out


def write_effects(target: Path, effects: Mapping[str, Mapping[str, float]],
                  fingerprint: Mapping[str, Any]) -> Path:  # fmt: skip
    """Write ``unit_effects.csv`` (unit_id, finding, effect) and its settings file."""
    from medcompress.utils.io import write_csv, write_json

    path = target.with_name(EFFECTS_FILE)
    rows = [
        (unit_id, finding, float(effects[unit_id][finding]))
        for unit_id in sorted(effects)
        for finding in sorted(effects[unit_id])
    ]
    write_csv(path, ("unit_id", "finding", "effect"), rows)
    write_json(effects_fingerprint(fingerprint), path.with_name(EFFECTS_SETTINGS_FILE))
    return path


def read_effects(path: Path) -> dict[str, dict[str, float]]:
    """``{unit_id: {finding: effect}}`` of a ``unit_effects.csv``."""
    import csv

    out: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["unit_id"], {})[row["finding"]] = float(row["effect"])
    return out


def reweight_source(
    config: Mapping[str, Any], run: RunContext, settings: ScoringSettings
) -> Path | None:
    """E7's run folder of the same seed when its per-finding effects can stand in for scoring
    ``run`` (A1 / A2 with otherwise equal settings), else None."""
    from medcompress.compression.pruning.criteria.clinical import ClinicalSettings
    from medcompress.utils.io import read_json

    if settings.criterion != "clinical" or run.experiment_id == EFFECTS_SOURCE:
        return None
    effective = apply_variant(config, run.variant)
    if ClinicalSettings.from_config(effective).weighting not in REWEIGHTED_SCHEMES:
        return None
    source = RunContext(EFFECTS_SOURCE, seed=run.seed).run_dir(config["paths"]["results_root"])
    files = (source / EFFECTS_FILE, source / EFFECTS_SETTINGS_FILE, source / SCORES_FILE)
    if not all(f.is_file() for f in files):
        return None
    fingerprint = _fingerprint(effective, settings, run)
    return source if read_json(files[1]) == effects_fingerprint(fingerprint) else None


def reweight_from_source(
    config: Mapping[str, Any], run: RunContext, settings: ScoringSettings
) -> Path | None:
    """Write the scores of A1 / A2 from E7's per-finding effects of the same seed, if they
    were made with the same settings apart from the weights; None otherwise (score as usual).
    """
    import csv

    from medcompress.compression.pruning.criteria.clinical import (
        ClinicalSettings,
        finding_weights_for,
        weighted_sum,
    )
    from medcompress.utils.io import upsert_results, write_csv, write_json

    source = reweight_source(config, run, settings)
    if source is None:
        return None
    effective = apply_variant(config, run.variant)
    clinical = ClinicalSettings.from_config(effective)
    fingerprint = _fingerprint(effective, settings, run)
    effects_path, scores_source = source / EFFECTS_FILE, source / SCORES_FILE
    effects = read_effects(effects_path)
    findings = sorted({f for per_unit in effects.values() for f in per_unit})
    weights = finding_weights_for(effective, findings, clinical.weighting,
                                  allow_unfrozen=clinical.debug_unfrozen_weights)  # fmt: skip
    target = scores_path(config, run)
    target.parent.mkdir(parents=True, exist_ok=True)
    with scores_source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {r["unit_id"] for r in rows} != set(effects):
        raise ValueError(f"{effects_path} and {scores_source} hold different units")
    columns = list(rows[0])
    scored = ({**row, "score": weighted_sum(effects[row["unit_id"]], weights)} for row in rows)
    write_csv(target, columns, ([row[c] for c in columns] for row in scored))
    write_json(fingerprint, target.with_name(SETTINGS_FILE))
    upsert_results(
        [
            {"experiment": run.experiment_tag, "budget": "", "seed": run.seed, "metric": m,
             "value": v}
            for m, v in (("scoring_units", len(rows)), ("scoring_reweighted_from_e7", 1.0),
                         ("scoring_seconds", 0.0))
        ],
        target.with_name("metrics.csv"),
    )  # fmt: skip
    print(f"re-weighted {EFFECTS_SOURCE} seed {run.seed}'s per-finding effects with "
          f"{clinical.weighting} weights ({len(rows)} units); wrote {target}")  # fmt: skip
    return target


def subset_source(
    config: Mapping[str, Any], run: RunContext, settings: ScoringSettings
) -> Path | None:
    """E7's run folder of the same seed when E7's scores of this run's unit types stand in for
    scoring it: every other setting equal, this run's unit types among E7's, and each of them
    scored **exactly** by both (A5 ``logprob_drop`` on layers and KV groups, 2026-11-13). An
    exact score of a unit does not depend on which other units are scored, so the values are
    the same as scoring again."""
    from medcompress.compression.pruning.criteria.clinical import ClinicalSettings
    from medcompress.utils.io import read_json

    if settings.criterion != "clinical" or run.experiment_id == EFFECTS_SOURCE:
        return None
    effective = apply_variant(config, run.variant)
    methods = ClinicalSettings.from_config(effective).methods
    if any(methods.get(unit_type) != "exact" for unit_type in settings.unit_types):
        return None
    source = RunContext(EFFECTS_SOURCE, seed=run.seed).run_dir(config["paths"]["results_root"])
    if not ((source / SCORES_FILE).is_file() and (source / SETTINGS_FILE).is_file()):
        return None
    theirs = dict(read_json(source / SETTINGS_FILE))
    ours = dict(_fingerprint(effective, settings, run))
    if not set(ours.get("unit_types") or ()) <= set(theirs.get("unit_types") or ()):
        return None
    no_groups = "mlp_channel_group" not in settings.unit_types
    for fingerprint in (theirs, ours):
        fingerprint.pop("variant", None)
        fingerprint.pop("unit_types", None)
        if no_groups:  # the group size does not touch the layer and KV-group scores
            fingerprint.pop("mlp_channel_group_size", None)
    return source if theirs == ours else None


def subset_from_source(
    config: Mapping[str, Any], run: RunContext, settings: ScoringSettings
) -> Path | None:
    """Write ``run``'s scores as E7's scores of its unit types (see :func:`subset_source`)."""
    import csv

    from medcompress.utils.io import upsert_results, write_csv, write_json

    source = subset_source(config, run, settings)
    if source is None:
        return None
    effective = apply_variant(config, run.variant)
    with (source / SCORES_FILE).open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["unit_type"] in settings.unit_types]
    if not rows:
        return None
    target = scores_path(config, run)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    write_csv(target, columns, ([row[c] for c in columns] for row in rows))
    write_json(_fingerprint(effective, settings, run), target.with_name(SETTINGS_FILE))
    upsert_results(
        [{"experiment": run.experiment_tag, "budget": "", "seed": run.seed, "metric": m,
          "value": v}
         for m, v in (("scoring_units", len(rows)), ("scoring_exact_subset_of_e7", 1.0),
                      ("scoring_seconds", 0.0))],
        target.with_name("metrics.csv"),
    )  # fmt: skip
    print(f"took {EFFECTS_SOURCE} seed {run.seed}'s exact scores of {list(settings.unit_types)} "
          f"({len(rows)} units); wrote {target}")  # fmt: skip
    return target


def _reusable(target: Path, fingerprint: Mapping[str, Any]) -> bool:
    """True when ``target`` was written with exactly these settings."""
    from medcompress.utils.io import read_json

    sidecar = target.with_name(SETTINGS_FILE)
    return target.exists() and sidecar.exists() and read_json(sidecar) == dict(fingerprint)


def whole_layers_within(units: Sequence[PrunableUnit], cap: int) -> list[PrunableUnit]:
    """RC-SMOKE: the units of the first decoder layers, whole layers only, at most ``cap`` units
    (at least one layer). A cut through a layer's channel groups would leave the MLP width
    partly covered, which the first-order gates refuse (2026-11-13: A4, A6 in RC-SMOKE)."""
    by_layer: dict[int, list[PrunableUnit]] = {}
    for unit in units:
        by_layer.setdefault(unit.layer_index, []).append(unit)
    kept: list[PrunableUnit] = []
    for index in sorted(by_layer):
        if kept and len(kept) + len(by_layer[index]) > cap:
            break
        kept += by_layer[index]
    order = {id(u): i for i, u in enumerate(units)}
    return sorted(kept, key=lambda u: order[id(u)])


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
    from medcompress.data.calibration import calibration_set
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
    cap = (config.get("smoke") or {}).get("max_scored_units")  # RC-SMOKE only
    if cap is not None:
        units = whole_layers_within(units, int(cap))
    instance = criterion or get_criterion(settings.criterion)(effective, seed=run.seed)
    calibration: list[VQASample] = []
    if settings.calibration_size is not None:  # without the images the fp16 check replaced
        calibration = calibration_set(config, train, settings.calibration_size, run.seed)

    resume = target.with_name(EXACT_RESUME_FILE)
    if hasattr(instance, "exact_resume"):  # the clinical criterion (2026-11-14)
        instance.exact_resume = (resume, {**fingerprint, "smoke_cap": cap})
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
    effects = getattr(instance, "finding_effects", None)
    if effects:  # the clinical criterion: kept so A1 and A2 can re-weight them
        write_effects(target, effects, fingerprint)
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
    resume.unlink(missing_ok=True)  # the scores are written: the resume file has done its job
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

    def reweighted() -> bool:
        """True when E7's per-finding effects stand in for scoring (A1, A2): no model."""
        if model is not None or criterion is not None or force or "settings" not in holder:
            return False
        return reweight_source(config, run, holder["settings"]) is not None

    def subset() -> bool:
        """True when E7's exact scores of this run's unit types stand in (A5 logprob_drop)."""
        if model is not None or criterion is not None or force or "settings" not in holder:
            return False
        return subset_source(config, run, holder["settings"]) is not None

    collect_open_settings(  # fail early, before any model is loaded, with everything that is open
        lambda: holder.update(settings=ScoringSettings.from_config(config, run.variant)),
        lambda: _check_criterion(config, run),
        lambda: None if reusable() or reweighted() or subset() else _check_side(config, model),
    )
    settings = holder["settings"]
    target = scores_path(config, run)
    if reusable():
        print(f"reusing {target} (same settings)")
        return  # nothing to load: the scores of these settings exist already
    source = scores_source(config)
    if source is not None:  # E7U (2026-11-17): never scores, never writes another arm's file
        raise ConfigError(
            f"{run.experiment_tag} reuses the scores of {source}, but {target} is missing or was "
            f"made with other settings (or the E2 adapter of seed {run.seed} is not attached); "
            f"this stage never writes {source}'s scores"
        )
    if reweighted() and reweight_from_source(config, run, settings) is not None:
        return  # A1 / A2: E7's per-finding effects re-weighted, no model loaded
    if subset() and subset_from_source(config, run, settings) is not None:
        return  # A5 logprob_drop: E7's exact scores of its unit types, no model loaded
    if settings.calibration_size is not None and model is None:
        from medcompress.data.calibration import require_check

        require_check(config, run.seed, settings.calibration_size)  # before the model loads
    if model is None:
        model = load_scoring_model(config, run)
    if train is None:
        train = [] if settings.calibration_size is None else load_train_samples(config)
    run_scoring(config, run, model, train, criterion=criterion, force=force)


# ------------------------------------------------------------ the first-order check (RC-G2)


def check_sample(unit_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """``n`` unit ids drawn at random (seeded, independent of the input order)."""
    import random

    ordered = sorted(unit_ids)
    return sorted(random.Random(f"first_order_check:{seed}").sample(ordered, min(n, len(ordered))))


def rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation (NaN when either side is constant or too short)."""
    import math

    from scipy import stats

    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    return float(stats.spearmanr(x, y)[0])


def run_first_order_check(
    config: Mapping[str, Any],
    run: RunContext,
    model: VLMWrapper,
    train: Sequence[VQASample],
    *,
    criterion: PruningCriterion | None = None,
    n_groups: int = CHECK_GROUPS,
) -> dict[str, float]:
    """Score ``n_groups`` random first-order units exactly and compare with their estimates.

    The estimates are the run's committed ``unit_scores.csv`` (stage 04 first). The exact
    scores use the same probes and weights (:meth:`ClinicalCriterion.exact_scores`). Writes
    ``first_order_check.csv`` (unit id, first-order score, exact score) next to the scores and
    the ``first_order_check:*`` rows to the run's ``metrics.csv``.
    """
    from medcompress.compression.pruning.criteria import get_criterion
    from medcompress.compression.pruning.criteria.clinical import ClinicalSettings
    from medcompress.compression.pruning.criteria.common import read_unit_scores
    from medcompress.compression.pruning.units import enumerate_units
    from medcompress.data.calibration import calibration_set
    from medcompress.utils.io import upsert_results, write_csv

    effective = apply_variant(config, run.variant)
    settings = ScoringSettings.from_config(config, run.variant)
    if settings.criterion != "clinical":
        raise ConfigError(f"the first-order check is for the clinical criterion, not "
                          f"{settings.criterion!r}")  # fmt: skip
    methods = ClinicalSettings.from_config(effective).methods
    target = scores_path(config, run)
    if not target.is_file():
        raise FileNotFoundError(f"no {target}: score the units first (stage 04)")
    estimates = read_unit_scores(target)
    units = enumerate_units(
        model,
        [UnitType(t) for t in settings.unit_types],
        mlp_channel_group_size=settings.mlp_channel_group_size,
    )
    candidates = {u.unit_id: u for u in units if methods.get(u.unit_type.value) == "first_order"}
    if not candidates:
        raise ConfigError("pruning.clinical.ablation_method has no first_order unit type")
    chosen = check_sample(list(candidates), n_groups, run.seed)
    instance = criterion or get_criterion("clinical")(effective, seed=run.seed)
    calibration: list[VQASample] = []
    if settings.calibration_size is not None:
        calibration = calibration_set(config, train, settings.calibration_size, run.seed)
    started = time.perf_counter()
    exact = instance.exact_scores(model, [candidates[c] for c in chosen], calibration)
    seconds = time.perf_counter() - started
    pairs = [(c, float(estimates[c]), float(exact[c])) for c in chosen]
    write_csv(target.with_name(CHECK_FILE), ("unit_id", "first_order", "exact"), pairs)
    rho = rank_correlation([p[1] for p in pairs], [p[2] for p in pairs])
    values = {
        "first_order_check:groups": float(len(pairs)),
        "first_order_check:spearman": rho,
        "first_order_check:min_rank_correlation": CHECK_MIN_RANK_CORRELATION,
        "first_order_check:passed": float(rho >= CHECK_MIN_RANK_CORRELATION),
        "first_order_check:seconds": seconds,
    }
    rows = [{"experiment": run.experiment_tag, "budget": "", "seed": run.seed, "metric": m,
             "value": v} for m, v in values.items()]  # fmt: skip
    upsert_results(rows, target.with_name("metrics.csv"))
    verdict = "at or above" if rho >= CHECK_MIN_RANK_CORRELATION else "BELOW (or undefined)"
    print(f"first-order check: Spearman {rho:.3f} over {len(pairs)} groups, {verdict} "
          f"{CHECK_MIN_RANK_CORRELATION}; wrote {target.with_name(CHECK_FILE)}")  # fmt: skip
    if not rho >= CHECK_MIN_RANK_CORRELATION:
        print("first-order check: below the rule of D07: the agent stops and asks the team "
              "before E7's other runs")  # fmt: skip
    reference = Path(config["paths"]["results_root"]) / "E2" / "nobudget" / "seed0"
    if not (reference / "efficiency.csv").is_file():  # committed in the repository (RC-B0)
        print(f"first-order check: follow-up NOT run: no E2 reference footprint at {reference}; "
              "run it later with --followup-only")  # fmt: skip
        return values
    return {**values, **first_order_followup(config, run)}


CHECK_B2_FILE = "first_order_check_b2.csv"
"""The 30 checked groups at B2: removed with the first-order scores, removed with the exact ones
(amendment of 2026-11-17)."""


def exact_on_estimate_fit(pairs: Sequence[tuple[str, float, float]]) -> dict[str, float]:
    """Least squares of the exact effect on the first-order estimate over the checked groups:
    slope and intercept with 95 % confidence intervals (t with n - 2 degrees of freedom).
    A slope of 1 and an intercept of 0 mean the estimate is on the exact scale; the rank
    correlation alone does not show a scale or an offset (amendment of 2026-11-17)."""
    from scipy import stats

    x = [p[1] for p in pairs]
    y = [p[2] for p in pairs]
    nan = float("nan")
    if len(pairs) < 3 or len(set(x)) < 2:
        return {k: nan for k in ("slope", "slope_ci_low", "slope_ci_high", "intercept",
                                 "intercept_ci_low", "intercept_ci_high")}  # fmt: skip
    fit = stats.linregress(x, y)
    t = float(stats.t.ppf(0.975, len(pairs) - 2))
    return {
        "slope": float(fit.slope),
        "slope_ci_low": float(fit.slope - t * fit.stderr),
        "slope_ci_high": float(fit.slope + t * fit.stderr),
        "intercept": float(fit.intercept),
        "intercept_ci_low": float(fit.intercept - t * fit.intercept_stderr),
        "intercept_ci_high": float(fit.intercept + t * fit.intercept_stderr),
    }


def substituted_selection(
    units: Sequence[PrunableUnit],
    scores: Mapping[str, float],
    exact: Mapping[str, float],
    removal: Any,
    spec: Any,
    target: float,
    measure: Any,
) -> tuple[dict[str, float], list[tuple[str, int, int]]]:
    """The B2 selection with the first-order scores and again with the checked groups' exact
    effects put in their place (same search, same measurer; amendment of 2026-11-17).

    Returns the counts (checked groups that switch, each way; KV groups and layers whose
    decision changes; units in the two sets) and one row per checked group: ``(unit id,
    removed with the estimate, removed with the exact effect)``."""
    from medcompress.compression.recovery import search_budget

    before, _ = search_budget(units, scores, removal, spec, target, measure)
    after, _ = search_budget(units, {**scores, **exact}, removal, spec, target, measure)
    old = {u.unit_id for u in before.removal}
    new = {u.unit_id for u in after.removal}
    rows = [(c, int(c in old), int(c in new)) for c in sorted(exact)]
    coarse = {u.unit_id for u in units
              if u.unit_type in (UnitType.ATTENTION_HEAD, UnitType.DECODER_LAYER)}  # fmt: skip
    counts = {
        "b2_switched": float(sum(1 for _, a, b in rows if a != b)),
        "b2_switched_to_removed": float(sum(1 for _, a, b in rows if not a and b)),
        "b2_switched_to_kept": float(sum(1 for _, a, b in rows if a and not b)),
        "b2_checked_removed_with_estimate": float(sum(a for _, a, _ in rows)),
        "b2_kv_or_layer_changes": float(len((old ^ new) & coarse)),
        "b2_units_with_estimate": float(len(old)),
        "b2_units_with_exact": float(len(new)),
        "b2_units_changed": float(len(old ^ new)),
    }
    return counts, rows


def first_order_followup(config: Mapping[str, Any], run: RunContext) -> dict[str, float]:
    """From the committed files alone (CPU): the fit of the exact effect on the estimate and the
    B2 selection with the exact effects substituted (amendment of 2026-11-17). Reads
    ``unit_scores.csv`` and ``first_order_check.csv``; the B2 search uses the offline estimate
    of :mod:`medcompress.compression.pruning.replay` (E2's measured peak minus 2 bytes per
    removed parameter) for both selections. Writes ``first_order_check_b2.csv`` and the rows."""
    import csv

    from medcompress.budgets import (
        budget_spec_for_run,
        metric_value,
        reference_decoder_bytes,
        reference_report,
        target_value,
    )
    from medcompress.compression.pruning.replay import parameter_measurer, units_from_scores
    from medcompress.compression.recovery import RemovalSettings
    from medcompress.utils.io import upsert_results, write_csv

    target_file = scores_path(config, run)
    check = target_file.with_name(CHECK_FILE)
    if not check.is_file():
        raise FileNotFoundError(f"no {check}: run the first-order check first")
    with check.open(encoding="utf-8", newline="") as handle:
        pairs = [(r["unit_id"], float(r["first_order"]), float(r["exact"]))
                 for r in csv.DictReader(handle)]  # fmt: skip
    fit = exact_on_estimate_fit(pairs)
    units, scores = units_from_scores(target_file, ScoringSettings.from_config(
        config, run.variant).mlp_channel_group_size or 128)  # fmt: skip
    results = config["paths"]["results_root"]
    reference = reference_report(results, "E2")
    spec = budget_spec_for_run(config, "B2")
    target = target_value(spec, reference, decoder_bytes=reference_decoder_bytes(results, "E2"))
    counts, rows = substituted_selection(
        units, scores, {c: e for c, _, e in pairs}, RemovalSettings.from_config(config), spec,
        target, parameter_measurer(metric_value(reference, "peak_memory")),
    )  # fmt: skip
    write_csv(target_file.with_name(CHECK_B2_FILE),
              ("unit_id", "removed_with_estimate", "removed_with_exact"), rows)  # fmt: skip
    values = {f"first_order_check:{k}": v for k, v in {**fit, **counts}.items()}
    upsert_results([{"experiment": run.experiment_tag, "budget": "", "seed": run.seed,
                     "metric": m, "value": v} for m, v in values.items()],
                   target_file.with_name("metrics.csv"))  # fmt: skip
    print(
        f"first-order check, exact on estimate: slope {fit['slope']:.4g} "
        f"[{fit['slope_ci_low']:.4g}, {fit['slope_ci_high']:.4g}], intercept "
        f"{fit['intercept']:.4g} [{fit['intercept_ci_low']:.4g}, {fit['intercept_ci_high']:.4g}]"
    )
    print(f"first-order check, B2 with the exact effects (estimated footprints): "
          f"{counts['b2_switched']:.0f} of {len(rows)} checked groups switch "
          f"({counts['b2_switched_to_removed']:.0f} to removed, "
          f"{counts['b2_switched_to_kept']:.0f} to kept); KV-group or layer decisions changed: "
          f"{counts['b2_kv_or_layer_changes']:.0f}")  # fmt: skip
    return values


def first_order_check_for_experiment(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    model: VLMWrapper | None = None,
    train: Sequence[VQASample] | None = None,
    criterion: PruningCriterion | None = None,
) -> None:
    """Stage entry point for ``scripts/check_first_order.py`` (RC-G2, E7 seed 0)."""
    if run is None:
        raise ConfigError("--exp is required")
    settings = ScoringSettings.from_config(config, run.variant)
    _check_criterion(config, run)
    _check_side(config, model)
    if settings.calibration_size is not None and model is None:
        from medcompress.data.calibration import require_check

        require_check(config, run.seed, settings.calibration_size)
    if model is None:
        model = load_scoring_model(config, run)
    if train is None:
        train = [] if settings.calibration_size is None else load_train_samples(config)
    run_first_order_check(config, run, model, train, criterion=criterion)
