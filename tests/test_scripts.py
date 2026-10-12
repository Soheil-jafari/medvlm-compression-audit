"""Entry-point scripts parse arguments, validate IDs and exit cleanly while stubs remain."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from medcompress.cli import NOT_IMPLEMENTED_EXIT_CODE

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py"))


def _run(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_help(script: Path, tmp_path: Path) -> None:
    result = _run(script.name, "--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout


@pytest.mark.parametrize(
    ("script", "args"),
    [
        ("00_check_teacher_image_reliance.py", []),
        ("01_cache_teacher_outputs.py", []),
        ("02_train_student.py", ["--exp", "E2", "--seed", "1"]),
        ("03_quantize.py", ["--exp", "E3", "--variant", "w4"]),
        ("04_score_pruning_units.py", ["--exp", "E7"]),
        ("05_prune_and_recover.py", ["--exp", "E7", "--budget", "B2"]),
        ("06_evaluate.py", ["--exp", "A4", "--budget", "B2", "--variant", "n64"]),
        ("07_benchmark_efficiency.py", ["--exp", "E9"]),
        ("run_experiment.py", ["--exp", "E7", "--budget", "B2", "--seed", "0"]),
    ],
)
def test_stub_stage_exits_cleanly(script: str, args: list[str], tmp_path: Path) -> None:
    result = _run(script, *args, cwd=tmp_path)
    assert result.returncode == NOT_IMPLEMENTED_EXIT_CODE, result.stderr
    assert "not implemented yet" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["--exp", "E99"],
        ["--exp", "E7"],  # missing budget
        ["--exp", "E0", "--budget", "B2"],  # E0 is not budget-matched
        ["--exp", "E7", "--budget", "B2", "--seed", "7"],  # seed not in config
        ["--exp", "E5", "--budget", "B2"],  # E5 needs a variant
    ],
)
def test_run_experiment_rejects_invalid_runs(args: list[str], tmp_path: Path) -> None:
    result = _run("run_experiment.py", *args, cwd=tmp_path)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_stage_script_rejects_wrong_experiment(tmp_path: Path) -> None:
    result = _run("03_quantize.py", "--exp", "E7", "--budget", "B2", cwd=tmp_path)
    assert result.returncode == 2
    assert "not part of E7" in result.stderr


def test_run_experiment_list_and_plan(tmp_path: Path) -> None:
    listing = _run("run_experiment.py", "--list", cwd=tmp_path)
    assert listing.returncode == 0
    for exp_id in ("E0", "E7", "E9", "A6"):
        assert exp_id in listing.stdout
    plan = _run("run_experiment.py", "--exp", "E3", "--variant", "w4", "--plan-only", cwd=tmp_path)
    assert plan.returncode == 0
    assert "E3:w4" in plan.stdout


def test_aggregate_script_writes_summary_and_tables(tmp_path: Path) -> None:
    result = _run("08_aggregate_results.py", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results" / "summary" / "by_arm.csv").is_file()
    for name in ("main.tex", "primary.tex", "numbers.tex", "setup_data.tex"):
        assert (tmp_path / "paper" / "tables" / name).is_file(), name
    assert (tmp_path / "paper" / "figures" / "headline_vs_hidden.pdf").is_file()
    assert "pending" in (tmp_path / "docs" / "results" / "VERDICTS.md").read_text()
