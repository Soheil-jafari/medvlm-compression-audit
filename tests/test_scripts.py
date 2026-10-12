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


def test_a_stage_that_is_still_a_stub_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    from medcompress.cli import run_stage

    def stub() -> None:
        raise NotImplementedError("TODO: a later stage")

    assert run_stage("XX", stub) == NOT_IMPLEMENTED_EXIT_CODE
    assert "not implemented yet" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("script", "args", "open_setting"),
    [
        ("01_cache_teacher_outputs.py", [], "max_new_tokens"),
        ("00_check_teacher_image_reliance.py", [], "domain.findings"),
        ("02_train_student.py", ["--exp", "E2", "--seed", "1"], "training.steps"),
        (
            "06_evaluate.py",
            ["--exp", "A4", "--budget", "B2", "--variant", "n64"],
            "domain.findings",
        ),
        ("select_student.py", ["--round", "R1", "--candidate", "qwen3_vl_2b"], "compute_unit_b"),
        (
            "select_student.py",
            ["--round", "R0", "--candidate", "qwen3_vl_2b"],
            "image_longest_side",
        ),
        ("select_student.py", ["--rank", "R1"], "domain.small_findings.thresholds"),
        ("04_score_pruning_units.py", ["--exp", "E7"], "pruning.clinical.ablation_method"),
        ("05_prune_and_recover.py", ["--exp", "E7", "--budget", "B2"], "training.steps"),
        (
            "05_prune_and_recover.py",
            ["--exp", "E7", "--budget", "B2"],
            "budgets.levels.B2.target_fraction",
        ),
        ("07_benchmark_efficiency.py", ["--exp", "E9"], "budgets.measurement.hardware"),
        ("03_quantize.py", ["--exp", "E3", "--variant", "w4"], "inputs.image_longest_side"),
        ("03_quantize.py", ["--exp", "E8", "--budget", "B2"], "inputs.image_longest_side"),
        (
            "run_experiment.py",
            ["--exp", "E3", "--variant", "w4", "--seed", "0"],
            "inputs.image_longest_side",
        ),
        (
            "04_score_pruning_units.py",
            ["--exp", "E5", "--variant", "taylor"],
            "inputs.image_longest_side",
        ),
        ("run_experiment.py", ["--exp", "E7", "--budget", "B2", "--seed", "0"], "pruning"),
    ],
)
def test_implemented_stages_stop_cleanly_while_settings_are_open(
    script: str, args: list[str], open_setting: str, tmp_path: Path
) -> None:
    """Implemented stages whose settings wait for a team decision exit with code 4."""
    from medcompress.cli import CONFIG_INCOMPLETE_EXIT_CODE

    result = _run(script, *args, cwd=tmp_path)
    assert result.returncode == CONFIG_INCOMPLETE_EXIT_CODE, result.stderr
    assert "configuration incomplete" in result.stdout and open_setting in result.stdout
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


def test_select_student_lists_the_pool_and_checks_its_arguments(tmp_path: Path) -> None:
    listed = _run("select_student.py", "--list", cwd=tmp_path)
    assert listed.returncode == 0, listed.stderr
    assert "qwen3_vl_2b" in listed.stdout.split() and len(listed.stdout.split()) == 10
    for args in (["--round", "R1"], ["--round", "R9", "--candidate", "x"], []):
        result = _run("select_student.py", *args, cwd=tmp_path)
        assert result.returncode == 2 and "Traceback" not in result.stderr
    unknown = _run("select_student.py", "--round", "R1", "--candidate", "nope", cwd=tmp_path)
    assert unknown.returncode != 0  # an unknown candidate is a mistake, not an open setting


def test_select_student_writes_the_report_from_committed_results(tmp_path: Path) -> None:
    target = tmp_path / "SELECTION.md"
    result = _run("select_student.py", "--report", "--report-path", str(target), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    text = target.read_text()
    assert text.startswith("# Student selection (D05)") and "Not run yet." in text
