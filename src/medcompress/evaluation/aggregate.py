"""Reading, validating and summarising the committed results tree.

Layout (``results/README.md``)::

    results/<ID>[/<variant>]/<budget|nobudget>/seed<k>/metrics.csv | metrics_test.csv
                                                     | efficiency.csv
    results/comparisons/<name>/<budget|nobudget>/comparison.csv
    results/selection/<round>/<candidate>/metrics.csv          (student selection, D05)
    results/data/summary.csv                                    (aggregate data statistics)
    results/smoke/<card>.csv                                    (smoke tests)
    results/summary/                                            (written by aggregation)

Every CSV uses the long schema ``medcompress.utils.io.RESULTS_COLUMNS``. This
module never estimates numbers: it only reads committed CSVs and summarises them.

The split a row was scored on is not a CSV column; it follows from the file name
(``metrics.csv`` = validation, ``metrics_test.csv`` = test, see :data:`SPLIT_OF_RUN_FILE`)
and is added as a ``split`` column when the rows are loaded, so the summary never averages
validation and test numbers of the same metric together.
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

RUN_FILES: tuple[str, ...] = (
    "metrics.csv",
    "metrics_test.csv",
    "efficiency.csv",
    "efficiency_final.csv",
)
SPLIT_OF_RUN_FILE: dict[str, str] = {
    "metrics.csv": "val",
    "metrics_test.csv": "test",
    "efficiency.csv": "",
    "efficiency_final.csv": "",  # the final timing (RC-T1), metric names prefixed "final:"
}
"""Split of the rows in each run file; ``""`` = not tied to a split (footprints, training)."""
SUMMARY_KEYS: tuple[str, ...] = ("experiment", "budget", "split", "metric")
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
    split: str = ""
    relative: str = ""
    """The path relative to the results folder (``smoke/RC-M1.csv``), the same on every machine."""


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
        return ResultFile(
            path,
            kind="comparison",
            relative=rel.as_posix(),
            budget=None if budget == "nobudget" else budget,
        )
    if parts[0] == "selection":
        return ResultFile(path, relative=rel.as_posix(), kind="selection", split="val")
    if parts[0] in ("data", "smoke"):
        return ResultFile(path, relative=rel.as_posix(), kind=parts[0])
    if path.name not in RUN_FILES:
        return None
    # <ID>[/<variant>]/<budget|nobudget>/seed<k>/<file>
    split = SPLIT_OF_RUN_FILE[path.name]
    match = _SEED_DIR.match(parts[-2]) if len(parts) >= 4 else None
    if match is None:
        return ResultFile(path, relative=rel.as_posix(), kind="run", split=split)
    budget_dir = parts[-3]
    id_parts = parts[:-3]
    experiment = ":".join(id_parts)
    return ResultFile(
        path,
        "run",
        relative=rel.as_posix(),
        experiment=experiment,
        budget="" if budget_dir == "nobudget" else budget_dir,
        seed=int(match.group(1)),
        split=split,
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
        return "_vs_" in experiment  # e.g. "E5:taylor_vs_E2", "best_pruning_vs_E3:w8"
    if kind in ("data", "smoke"):
        return True
    spec = REGISTRY.get(base)
    if spec is None:
        return False
    return variant is None if not spec.variants else variant in spec.variants


def read_result_file(result: ResultFile) -> tuple[pd.DataFrame, list[str]]:
    """Read one CSV and return (rows, problems). Rows get ``source`` and ``split`` columns."""
    problems: list[str] = []
    name = str(result.path)
    try:
        frame = pd.read_csv(result.path, dtype={"experiment": str, "budget": str, "metric": str})
    except Exception as exc:  # noqa: BLE001 - report any parse failure as a problem
        return _empty(), [f"{name}: unreadable ({exc})"]
    if tuple(frame.columns) != RESULTS_COLUMNS:
        problems.append(f"{name}: columns {tuple(frame.columns)} != {RESULTS_COLUMNS}")
        return _empty(), problems
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
    frame["source"] = result.relative or str(result.path)
    frame["split"] = result.split
    return frame, problems


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[*RESULTS_COLUMNS, "source", "split"])


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
        return _empty()
    data = pd.concat(frames, ignore_index=True)
    data["seed"] = data["seed"].astype(int)
    return data


def summarise_over_seeds(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (experiment, budget, split, metric) with n_seeds, mean, sd, min, max.

    SD is the sample SD (ddof=1) and NaN for a single seed; ``ci_low`` / ``ci_high``
    are carried over only when there is a single seed (multi-seed CIs come from the
    comparison step, as pre-registered). Rows without a ``split`` column count as ``""``.
    """
    keys = list(SUMMARY_KEYS)
    columns = [*keys, "n_seeds", "mean", "sd", "min", "max"]
    if rows.empty:
        return pd.DataFrame(columns=[*columns, "ci_low", "ci_high"])
    rows = rows.assign(split=rows["split"].fillna("") if "split" in rows else "")
    grouped = rows.groupby(keys, sort=True)
    summary = grouped["value"].agg(
        n_seeds="count", mean="mean", sd=lambda v: v.std(ddof=1), min="min", max="max"
    )
    summary = summary.reset_index()
    single = grouped.agg(ci_low=("ci_low", "first"), ci_high=("ci_high", "first")).reset_index()
    summary = summary.merge(single, on=keys)
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


def paper_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows the paper reports: the test split plus rows not tied to a split.

    Validation rows (``split == "val"``) serve Gate 2, the choice of the best of E5/E6 and the
    student checks; they never reach the paper tables, figures or verdicts.
    """
    if frame.empty or "split" not in frame:
        return frame
    return frame[frame["split"].fillna("") != "val"]


def metric_names(rows: pd.DataFrame) -> list[str]:
    """Return the sorted metric names present in the rows."""
    return sorted(set(rows["metric"])) if not rows.empty else []


def experiments_present(rows: pd.DataFrame) -> Iterable[str]:
    """Return the sorted experiment labels present in the rows."""
    return sorted(set(rows["experiment"])) if not rows.empty else []
