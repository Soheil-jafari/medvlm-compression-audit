"""File I/O helpers, including the shared results CSV schema."""

from __future__ import annotations

import csv
import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
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


def format_cell(value: Any) -> str:
    """The text of one CSV cell, the same with every pandas and numpy version.

    Floats are written with Python's shortest round-trip ``repr`` (reading the text back gives
    exactly the same float); a numpy float32 keeps numpy's shortest text for its own type, as
    the ``csv`` module always wrote it. Integers as integers, booleans as ``True`` / ``False``,
    ``None`` and NaN as an empty cell, everything else ``str``. This is the one float format of
    every results file and every generated CSV (``write_csv``, ``write_frame``,
    ``append_results``).
    """
    if value is None:
        return ""
    if isinstance(value, bool | str):
        return str(value)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard requirement
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.bool_):
        return str(bool(value))
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if np is not None and isinstance(value, np.floating) and value.dtype != np.float64:
        return "" if np.isnan(value) else str(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return "" if math.isnan(number) else repr(number)
    return str(value)


def parse_float(text: Any) -> float:
    """A results cell as a float, correctly rounded (Python's ``float``); NaN when empty or not
    a number. Used instead of the pandas C parser, whose default is not exact in every version."""
    if text is None:
        return math.nan
    if isinstance(text, numbers.Real):
        return float(text)
    try:
        return float(str(text).strip()) if str(text).strip() else math.nan
    except ValueError:
        return math.nan


def write_csv(
    path: str | Path,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    lineterminator: str = "\r\n",
) -> Path:
    """Write a CSV (replaced) with :func:`format_cell` for every cell; returns the path.

    The line ending defaults to the ``csv`` module's (as every run file was written);
    :func:`write_frame` passes ``"\n"``, as ``DataFrame.to_csv`` wrote the summaries."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator=lineterminator)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([format_cell(v) for v in row])
    return path


def write_frame(frame: Any, path: str | Path) -> Path:
    """Write a pandas DataFrame like ``to_csv(index=False)``, but with :func:`format_cell`, so
    the bytes do not depend on the pandas version."""
    columns = [str(c) for c in frame.columns]
    rows = (list(r) for r in frame.itertuples(index=False, name=None))
    return write_csv(path, columns, rows, lineterminator="\n")


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
            writer.writerow({col: format_cell(row.get(col, "")) for col in RESULTS_COLUMNS})
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
