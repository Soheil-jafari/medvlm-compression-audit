"""Attached Kaggle inputs are found by their manifests, not by fixed paths."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from medcompress.inputs import (
    InputNotFound,
    find_handoff,
    find_handoffs,
    find_shared_data,
)
from medcompress.utils.bundles import MANIFEST_NAME

REPO = Path(__file__).resolve().parents[1]


def _bundle(folder: Path, *, private: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MANIFEST_NAME).write_text("{}")
    if private:
        (folder / "private").mkdir()
    return folder


def _tree(root: Path) -> dict[str, Path]:
    user = root / "datasets" / "soheil"
    return {
        "data": _bundle(user / "medcompress-data" / "bundle", private=True),
        "adapter": _bundle(user / "e0-teacher-adapter" / "E0_teacher_adapter"),
        "cache": _bundle(user / "e0-teacher-cache"),  # the slug only: - and case do not matter
    }


def test_the_shared_data_and_the_hand_offs_are_found_by_their_manifests(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    assert find_shared_data(tmp_path) == paths["data"]
    assert find_handoff("E0_teacher_adapter", tmp_path) == paths["adapter"]
    assert find_handoff("E0_teacher_cache", tmp_path) == paths["cache"]
    assert find_handoffs(tmp_path) == sorted([paths["adapter"], paths["cache"]])


def test_a_missing_or_doubled_input_stops_with_a_clear_message(tmp_path: Path) -> None:
    with pytest.raises(InputNotFound, match="the shared data is not attached.*medcompress-data"):
        find_shared_data(tmp_path)
    _tree(tmp_path)
    with pytest.raises(InputNotFound, match="E2_seed1_adapter is not attached"):
        find_handoff("E2_seed1_adapter", tmp_path)
    _bundle(tmp_path / "notebooks" / "soheil" / "old-run" / "handoff" / "E0_teacher_adapter")
    with pytest.raises(InputNotFound, match="more than once"):
        find_handoff("E0_teacher_adapter", tmp_path)


def test_the_script_prints_the_path_or_stops(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    script = str(REPO / "scripts" / "find_inputs.py")

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, script, *args, "--root", str(tmp_path)],
                              capture_output=True, text=True, check=False)  # fmt: skip

    ok = run("data")
    assert ok.returncode == 0 and ok.stdout.strip() == str(paths["data"])
    assert run("handoff", "E0_teacher_cache").stdout.strip() == str(paths["cache"])
    missing = run("handoff", "E1_seed0_adapter")
    assert missing.returncode == 5 and missing.stdout == ""
    assert "STOP: the hand-off E1_seed0_adapter is not attached" in missing.stderr


def _runner():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("run_card", REPO / "scripts" / "run_card.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_finds_its_inputs_when_none_are_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _tree(tmp_path)
    monkeypatch.setenv("MEDCOMPRESS_INPUT_ROOT", str(tmp_path))
    runner = _runner()
    args = runner.build_parser().parse_args(["--card", "RC-E2", "--seed", "0"])
    assert runner.find_inputs(args) is None
    assert args.data_root == paths["data"]
    assert sorted(args.handoffs) == sorted([paths["adapter"], paths["cache"]])
    code = runner.main(["--card", "RC-E2", "--seed", "0", "--dry-run", "--allow-blocked"])
    out = capsys.readouterr().out
    assert code == 0 and f"--data-root {paths['data']}" in out
    given = runner.build_parser().parse_args(["--card", "RC-E2", "--seed", "0",
                                              "--data-root", "/elsewhere"])  # fmt: skip
    runner.find_inputs(given)
    assert given.data_root == Path("/elsewhere")  # what is given is kept


def test_the_runner_stops_when_the_shared_data_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MEDCOMPRESS_INPUT_ROOT", str(tmp_path))
    runner = _runner()
    code = runner.main(["--card", "RC-E1", "--seed", "0", "--allow-blocked"])
    assert code == 5
    assert "STOP: the shared data is not attached" in capsys.readouterr().err
    rc_d1 = runner.build_parser().parse_args(["--card", "RC-D1"])
    assert runner.find_inputs(rc_d1) is None and rc_d1.data_root is None  # RC-D1 makes them
