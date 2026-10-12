"""YAML configuration loading with ``_base_`` inheritance (no Hydra).

Rules
-----
* A config file may contain ``_base_: <path>`` or ``_base_: [<path>, ...]``.
  Paths are relative to the file that declares them. Bases are merged left to
  right, then the file itself is merged on top.
* Merging is recursive for mappings; any other value (including lists) in the
  child replaces the value from the base.
* String values may reference environment variables as ``${env:NAME}`` or
  ``${env:NAME,default}``. This is how Kaggle/Colab paths are injected without
  editing configs.
* Values that are still undecided are written as strings starting with ``TODO``.
  :func:`find_todos` lists them so a run can refuse to start with open TODOs.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BASE_KEY = "_base_"
TODO_MARKER = "TODO"
CANDIDATE_STATUS = "CANDIDATE - NOT DECIDED"

_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")

REQUIRED_EXPERIMENT_KEYS: tuple[str, ...] = (
    "id",
    "name",
    "description",
    "kind",
    "parent",
    "variant_of",
    "requires",
    "budgets",
    "seeds",
    "variants",
    "scripts",
    "outputs",
)


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


# --------------------------------------------------------------------------- paths


def repo_root() -> Path:
    """Return the repository root.

    Uses ``$MEDCOMPRESS_REPO_ROOT`` if set, otherwise assumes the src layout
    (``<root>/src/medcompress/config.py``), which holds for editable installs and
    for running directly from a clone.
    """
    env = os.environ.get("MEDCOMPRESS_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def config_root() -> Path:
    """Return the ``configs/`` directory (``$MEDCOMPRESS_CONFIG_ROOT`` overrides it)."""
    env = os.environ.get("MEDCOMPRESS_CONFIG_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "configs"


def experiment_config_path(experiment_id: str) -> Path:
    """Return the config path for an experiment ID (``E*`` or ablation ``A*``)."""
    if experiment_id.startswith("A"):
        return config_root() / "experiments" / "ablations" / f"{experiment_id}.yaml"
    return config_root() / "experiments" / f"{experiment_id}.yaml"


# ------------------------------------------------------------------------ loading


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dict without resolving ``_base_``.

    An empty file yields ``{}``. A file whose top level is not a mapping raises
    :class:`ConfigError`.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` merged recursively onto ``base``.

    Nested mappings are merged; every other value in ``override`` (including
    lists and ``None``) replaces the corresponding value in ``base``. Inputs are
    not modified.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def expand_env(value: Any) -> Any:
    """Recursively replace ``${env:NAME}`` / ``${env:NAME,default}`` in strings.

    A reference to an unset variable without a default raises :class:`ConfigError`.
    """
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable {name!r} is not set and has no default")

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, Mapping):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute() and not path.exists():
        candidate = config_root() / path
        if candidate.exists():
            path = candidate
    return path.resolve()


def _load_with_bases(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ConfigError(f"Cyclic _base_ inheritance: {chain}")
    raw = load_yaml(path)
    bases = raw.pop(BASE_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(b, str) for b in bases):
        raise ConfigError(f"{BASE_KEY} in {path} must be a string or a list of strings")
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        merged = deep_merge(merged, _load_with_bases(base_path, (*stack, path)))
    return deep_merge(merged, raw)


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    *,
    resolve_env: bool = True,
) -> dict[str, Any]:
    """Load a YAML config, resolving ``_base_`` inheritance, overrides and env vars.

    Args:
        path: Config file. Relative paths that do not exist from the current
            directory are looked up under :func:`config_root`.
        overrides: Optional ``"dotted.key=value"`` strings applied after merging
            (values are parsed as YAML, so ``seed=1`` gives an int).
        resolve_env: Expand ``${env:...}`` references.

    Returns:
        The fully merged config as a plain dict.
    """
    config = _load_with_bases(_resolve_path(path), ())
    if overrides:
        config = apply_overrides(config, overrides)
    if resolve_env:
        config = expand_env(config)
    return config


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``"a.b.c=value"`` overrides to a copy of ``config``."""
    result = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override {item!r} must look like key.path=value")
        dotted, raw_value = item.split("=", 1)
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ConfigError(f"Override {item!r} has an empty key")
        node = result
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[keys[-1]] = yaml.safe_load(raw_value)
    return result


VARIANT_SUFFIX = "_by_variant"


def apply_variant(config: Mapping[str, Any], variant: str | None) -> dict[str, Any]:
    """Resolve every ``<key>_by_variant`` table of a config for one variant.

    A table such as ``size_by_variant: {n32: 32, n64: 64}`` sets ``size`` to the entry of the
    variant and is then dropped; an experiment that lists ``criterion: by_variant`` next to
    ``criterion_by_variant`` gets the entry in the same way. A config with such a table needs
    a variant that the table names (:class:`ConfigError` otherwise), and a config without
    tables is returned unchanged (a copy), whatever the variant.
    """
    result = copy.deepcopy(dict(config))

    def walk(node: dict[str, Any], path: str) -> None:
        for key in [k for k in node if isinstance(k, str) and k.endswith(VARIANT_SUFFIX)]:
            table, target = node.pop(key), key[: -len(VARIANT_SUFFIX)]
            where = f"{path}{key}"
            if not isinstance(table, Mapping):
                raise ConfigError(f"{where} must map variants to values")
            if variant is None:
                raise ConfigError(f"{where} needs a variant (--variant), one of {sorted(table)}")
            if variant not in table:
                raise ConfigError(f"{where} has no entry for variant {variant!r}: {sorted(table)}")
            node[target] = copy.deepcopy(table[variant])
        for key, child in node.items():
            if isinstance(child, dict):
                walk(child, f"{path}{key}.")

    walk(result, "")
    return result


def collect_open_settings(*readers: Callable[[], object]) -> None:
    """Call each settings reader and raise ONE :class:`ConfigError` that lists every open value.

    A reader raises ``ConfigError("settings still open: a, b")`` (or another message for an
    invalid value). The open names of all readers are merged, so one run of a stage shows
    everything that still has to be decided.
    """
    marker = "settings still open: "
    open_keys: list[str] = []
    other: list[str] = []
    for read in readers:
        try:
            read()
        except ConfigError as exc:
            text = str(exc)
            if marker in text:
                open_keys += [k.strip() for k in text.split(marker, 1)[1].split(",")]
            else:
                other.append(text)
    if open_keys or other:
        parts = [marker + ", ".join(dict.fromkeys(open_keys))] if open_keys else []
        raise ConfigError("; ".join([*parts, *other]))


def get_by_path(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Return ``config["a"]["b"]`` for ``dotted="a.b"``, or ``default`` if missing."""
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def find_todos(config: Any, prefix: str = "") -> list[str]:
    """List dotted paths whose value is a string starting with ``TODO``."""
    found: list[str] = []
    if isinstance(config, Mapping):
        for key, value in config.items():
            found.extend(find_todos(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(config, list):
        for index, value in enumerate(config):
            found.extend(find_todos(value, f"{prefix}[{index}]"))
    elif isinstance(config, str) and config.strip().startswith(TODO_MARKER):
        found.append(prefix)
    return found


# ---------------------------------------------------------------------- schemas


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations from the ``paths`` section of ``configs/base.yaml``."""

    data_root: Path
    cache_root: Path
    checkpoint_root: Path
    output_root: Path
    results_root: Path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PathsConfig:
        missing = [name for name in cls.__dataclass_fields__ if name not in data]
        if missing:
            raise ConfigError(f"paths section is missing keys: {missing}")
        return cls(**{name: Path(str(data[name])) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ExperimentConfig:
    """The ``experiment`` section of an experiment config (E0-E9, A1-A6)."""

    id: str
    name: str
    description: str
    kind: str
    parent: str | None
    variant_of: str | None
    requires: tuple[str, ...]
    budgets: tuple[str, ...]
    seeds: tuple[int, ...]
    variants: tuple[str, ...]
    scripts: tuple[str, ...]
    outputs: tuple[str, ...]
    status: str = "planned"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        missing = [key for key in REQUIRED_EXPERIMENT_KEYS if key not in data]
        if missing:
            raise ConfigError(f"experiment section is missing keys: {missing}")
        known = set(REQUIRED_EXPERIMENT_KEYS) | {"status"}
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            description=str(data["description"]).strip(),
            kind=str(data["kind"]),
            parent=data["parent"],
            variant_of=data["variant_of"],
            requires=tuple(data["requires"] or ()),
            budgets=tuple(data["budgets"] or ()),
            seeds=tuple(int(s) for s in (data["seeds"] or ())),
            variants=tuple(str(v) for v in (data["variants"] or ())),
            scripts=tuple(data["scripts"] or ()),
            outputs=tuple(data["outputs"] or ()),
            status=str(data.get("status", "planned")),
            extra={k: v for k, v in data.items() if k not in known},
        )


@dataclass(frozen=True)
class RunContext:
    """Identifies one run: experiment ID, optional variant, budget and seed."""

    experiment_id: str
    seed: int
    budget: str | None = None
    variant: str | None = None

    @property
    def experiment_tag(self) -> str:
        """``"E3:w4"`` style tag used in the ``experiment`` column of results CSVs."""
        return f"{self.experiment_id}:{self.variant}" if self.variant else self.experiment_id

    def run_dir(self, root: str | Path) -> Path:
        """Return ``<root>/<ID>[/<variant>]/<budget or nobudget>/seed<seed>``."""
        path = Path(root) / self.experiment_id
        if self.variant:
            path = path / self.variant
        return path / (self.budget or "nobudget") / f"seed{self.seed}"


def load_experiment(
    experiment_id: str, overrides: Iterable[str] | None = None
) -> tuple[dict[str, Any], ExperimentConfig]:
    """Load ``configs/experiments/<ID>.yaml`` and parse its ``experiment`` section."""
    config = load_config(experiment_config_path(experiment_id), overrides)
    if "experiment" not in config:
        raise ConfigError(f"Config for {experiment_id} has no 'experiment' section")
    parsed = ExperimentConfig.from_dict(config["experiment"])
    if parsed.id != experiment_id:
        raise ConfigError(f"Config for {experiment_id} declares experiment.id={parsed.id!r}")
    return config, parsed
