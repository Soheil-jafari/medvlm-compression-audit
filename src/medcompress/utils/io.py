"""File I/O helpers, including the shared results CSV schema."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

RESULTS_COLUMNS: tuple[str, ...] = (
    "experiment",
    "budget",
    "seed",
    "metric",
    "value",
    "ci_low",
    "ci_high",
)
"""Long-format results schema; see results/README.md."""


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: str | Path) -> Any:
    """Read a YAML file (no ``_base_`` handling; use ``medcompress.config`` for configs)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(data: Any, path: str | Path) -> None:
    """Write ``data`` as YAML, creating parent directories."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    """Write ``data`` as JSON, creating parent directories."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, sort_keys=True, default=str)
        handle.write("\n")


def append_results(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """Append rows in the :data:`RESULTS_COLUMNS` schema to a CSV file.

    Writes the header if the file is new. Missing optional fields (``budget``,
    ``ci_low``, ``ci_high``) are written as empty strings.

    Returns:
        Number of rows written.

    Raises:
        ValueError: a row has a column outside the schema or lacks a required one.
    """
    path = Path(path)
    ensure_dir(path.parent)
    required = {"experiment", "seed", "metric", "value"}
    new_file = not path.exists()
    count = 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS_COLUMNS))
        if new_file:
            writer.writeheader()
        for row in rows:
            unknown = set(row) - set(RESULTS_COLUMNS)
            if unknown:
                raise ValueError(f"Unknown results columns: {sorted(unknown)}")
            missing = required - set(row)
            if missing:
                raise ValueError(f"Missing results columns: {sorted(missing)}")
            writer.writerow({col: row.get(col, "") for col in RESULTS_COLUMNS})
            count += 1
    return count


def upsert_results(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """Write rows to a results CSV, replacing existing rows with the same key.

    The key is (experiment, budget, seed, metric), so running a check again updates
    its rows instead of duplicating them, while rows of other metrics in the file
    are kept. Returns the number of rows written for ``rows``.
    """
    path = Path(path)
    new = [dict(row) for row in rows]
    fresh = {(r["experiment"], str(r.get("budget", "")), str(r["seed"]), r["metric"]) for r in new}
    kept: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for old in csv.DictReader(handle):
                key = (old["experiment"], old.get("budget", ""), str(old["seed"]), old["metric"])
                if key not in fresh:
                    kept.append(old)
        path.unlink()
    append_results(kept + new, path)
    return len(new)
