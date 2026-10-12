"""Reading, validating and summarising the committed results tree.

Layout (``results/README.md``)::

    results/<ID>[/<variant>]/<budget|nobudget>/seed<k>/metrics.csv | efficiency.csv
    results/comparisons/<name>/<budget|nobudget>/comparison.csv
    results/selection/<round>/<candidate>/metrics.csv          (student selection, D05)
    results/data/summary.csv                                    (aggregate data statistics)
    results/smoke/<card>.csv                                    (smoke tests)
    results/summary/                                            (written by aggregation)

Every CSV uses the long schema ``medcompress.utils.io.RESULTS_COLUMNS``. This
module never estimates numbers: it only reads committed CSVs and summarises them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from medcompress.registry import BUDGET_IDS, REGISTRY
from medcompress.utils.io import RESULTS_COLUMNS

RUN_FILES: tuple[str, ...] = ("metrics.csv", "efficiency.csv")
SPECIAL_DIRS: tuple[str, ...] = ("comparisons", "selection", "data", "smoke", "summary")
SELECTION_ROUNDS: tuple[str, ...] = ("R0", "R1", "R2", "R3")
_SEED_DIR = re.compile(r"^seed(\d+)$")


@dataclass(frozen=True)
class ResultFile:
    """One results CSV and what its path says about its rows."""

    path: Path
    kind: str  # "run" | "comparison" | "selection" | "data" | "smoke"
    experiment: str | None = None
    budget: str | None = None
    seed: int | None = None


def split_experiment(experiment: str) -> tuple[str, str | None]:
    """Split ``"E5:taylor"`` into ``("E5", "taylor")`` and ``"E7"`` into ``("E7", None)``."""
    base, _, variant = experiment.partition(":")
    return base, (variant or None)


def classify_path(path: Path, root: Path) -> ResultFile | None:
    """Return what a CSV path under ``root`` encodes, or None if it is not a results file."""
    rel = path.relative_to(root)
    parts = rel.parts
    if not parts or parts[0] == "summary" or path.suffix != ".csv":
        return None
    if parts[0] == "comparisons":
        budget = parts[2] if len(parts) == 4 else None
        return ResultFile(path, "comparison", budget=None if budget == "nobudget" else budget)
    if parts[0] == "selection":
        return ResultFile(path, "selection")
    if parts[0] in ("data", "smoke"):
        return ResultFile(path, parts[0])
    if path.name not in RUN_FILES:
        return None
    # <ID>[/<variant>]/<budget|nobudget>/seed<k>/<file>
    match = _SEED_DIR.match(parts[-2]) if len(parts) >= 4 else None
    if match is None:
        return ResultFile(path, "run")
    budget_dir = parts[-3]
    id_parts = parts[:-3]
    experiment = ":".join(id_parts)
    return ResultFile(
        path,
        "run",
        experiment=experiment,
        budget="" if budget_dir == "nobudget" else budget_dir,
        seed=int(match.group(1)),
    )


def iter_result_files(root: Path) -> list[ResultFile]:
    """Return every results CSV under ``root`` (summary outputs excluded), sorted by path."""
    root = Path(root)
    if not root.is_dir():
        return []
    files = (classify_path(p, root) for p in sorted(root.rglob("*.csv")))
    return [f for f in files if f is not None]


def _known_experiment(experiment: str, kind: str) -> bool:
    base, variant = split_experiment(experiment)
    if kind == "selection":
        return base in SELECTION_ROUNDS and variant is not None
    if kind == "comparison":
        return "_vs_" in base
    if kind in ("data", "smoke"):
        return True
    spec = REGISTRY.get(base)
    if spec is None:
        return False
    return variant is None if not spec.variants else variant in spec.variants


def read_result_file(result: ResultFile) -> tuple[pd.DataFrame, list[str]]:
    """Read one CSV and return (rows, problems). Rows get a ``source`` column."""
    problems: list[str] = []
    name = str(result.path)
    try:
        frame = pd.read_csv(result.path, dtype={"experiment": str, "budget": str, "metric": str})
    except Exception as exc:  # noqa: BLE001 - report any parse failure as a problem
        return pd.DataFrame(columns=[*RESULTS_COLUMNS, "source"]), [f"{name}: unreadable ({exc})"]
    if tuple(frame.columns) != RESULTS_COLUMNS:
        problems.append(f"{name}: columns {tuple(frame.columns)} != {RESULTS_COLUMNS}")
        return pd.DataFrame(columns=[*RESULTS_COLUMNS, "source"]), problems
    frame["budget"] = frame["budget"].fillna("")
    for column in ("value", "ci_low", "ci_high"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for i, row in frame.iterrows():
        where = f"{name} row {i + 2}"
        if not _known_experiment(str(row["experiment"]), result.kind):
            problems.append(f"{where}: unknown experiment {row['experiment']!r}")
        if row["budget"] not in ("", *BUDGET_IDS):
            problems.append(f"{where}: unknown budget {row['budget']!r}")
        if not str(row["seed"]).isdigit():
            problems.append(f"{where}: seed must be a non-negative integer")
        if not isinstance(row["metric"], str) or not row["metric"]:
            problems.append(f"{where}: empty metric")
        if math.isnan(row["value"]):
            problems.append(f"{where}: value is not a number")
        low, high = row["ci_low"], row["ci_high"]
        if math.isnan(low) != math.isnan(high):
            problems.append(f"{where}: give both CI bounds or neither")
        elif not math.isnan(low) and not low <= row["value"] <= high:
            problems.append(f"{where}: value outside its CI")
        if result.kind == "run" and result.experiment is not None:
            if row["experiment"] != result.experiment:
                problems.append(
                    f"{where}: experiment {row['experiment']!r} != path {result.experiment!r}"
                )
            if row["budget"] != result.budget:
                problems.append(f"{where}: budget {row['budget']!r} != path {result.budget!r}")
            if str(row["seed"]) != str(result.seed):
                problems.append(f"{where}: seed {row['seed']} != path seed{result.seed}")
    if result.kind == "run" and result.experiment is None:
        problems.append(f"{name}: path does not follow <ID>[/<variant>]/<budget>/seed<k>/<file>")
    frame["source"] = str(result.path)
    return frame, problems


def validate_results_tree(root: Path) -> list[str]:
    """Return every schema or layout problem under ``root`` (empty list = valid)."""
    problems: list[str] = []
    for result in iter_result_files(root):
        problems.extend(read_result_file(result)[1])
    return problems


def load_results(root: Path) -> pd.DataFrame:
    """Return all rows under ``root``; raises ``ValueError`` listing problems if invalid."""
    frames: list[pd.DataFrame] = []
    problems: list[str] = []
    for result in iter_result_files(root):
        frame, found = read_result_file(result)
        frames.append(frame)
        problems.extend(found)
    if problems:
        raise ValueError("Invalid results:\n" + "\n".join(problems))
    if not frames:
        return pd.DataFrame(columns=[*RESULTS_COLUMNS, "source"])
    data = pd.concat(frames, ignore_index=True)
    data["seed"] = data["seed"].astype(int)
    return data


def summarise_over_seeds(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (experiment, budget, metric) with n_seeds, mean, sd, min, max.

    SD is the sample SD (ddof=1) and NaN for a single seed; ``ci_low`` / ``ci_high``
    are carried over only when there is a single seed (multi-seed CIs come from the
    comparison step, as pre-registered).
    """
    columns = ["experiment", "budget", "metric", "n_seeds", "mean", "sd", "min", "max"]
    if rows.empty:
        return pd.DataFrame(columns=[*columns, "ci_low", "ci_high"])
    grouped = rows.groupby(["experiment", "budget", "metric"], sort=True)
    summary = grouped["value"].agg(
        n_seeds="count", mean="mean", sd=lambda v: v.std(ddof=1), min="min", max="max"
    )
    summary = summary.reset_index()
    single = grouped.agg(ci_low=("ci_low", "first"), ci_high=("ci_high", "first")).reset_index()
    summary = summary.merge(single, on=["experiment", "budget", "metric"])
    multi = summary["n_seeds"] > 1
    summary.loc[multi, ["ci_low", "ci_high"]] = math.nan
    return summary


def write_summary(rows: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write ``all_rows.csv`` and ``by_arm.csv`` to ``out_dir`` and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "all_rows.csv", out_dir / "by_arm.csv"]
    rows.to_csv(paths[0], index=False)
    summarise_over_seeds(rows).to_csv(paths[1], index=False)
    return paths


def metric_names(rows: pd.DataFrame) -> list[str]:
    """Return the sorted metric names present in the rows."""
    return sorted(set(rows["metric"])) if not rows.empty else []


def experiments_present(rows: pd.DataFrame) -> Iterable[str]:
    """Return the sorted experiment labels present in the rows."""
    return sorted(set(rows["experiment"])) if not rows.empty else []
