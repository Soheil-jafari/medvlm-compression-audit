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


def _real_bundle(folder: Path, kind: str, items: list[tuple[str, str]], **meta: object) -> Path:
    """A hand-off folder with a real manifest (files hashed) holding ``items``."""
    from medcompress.utils.bundles import write_bundle_manifest

    for root, rel in items:
        target = folder / root / rel
        target.mkdir(parents=True, exist_ok=True)
        (target / "weights.bin").write_bytes(b"w" * 16)
    listed = [{"root": r, "path": p} for r, p in items]
    write_bundle_manifest(folder, kind=kind, meta=meta, items=listed, commit="abc")
    return folder


def test_bundles_inside_the_shared_dataset_or_of_other_kinds_are_not_hand_offs(
    tmp_path: Path,
) -> None:
    from medcompress.inputs import classify_bundles

    paths = _tree(tmp_path)
    data = paths["data"].parent  # the medcompress-data dataset
    embed = data / "rc_d1_embed"  # RC-D1-embed's bundle, next to the data (the Kaggle failure)
    embed.mkdir()
    (embed / MANIFEST_NAME).write_text('{"kind": "near_duplicates"}')
    nested = paths["data"] / "extra"  # anything inside the shared-data folder itself
    nested.mkdir()
    (nested / MANIFEST_NAME).write_text("{}")
    handoffs, notes = classify_bundles(tmp_path)
    assert handoffs == sorted([paths["adapter"], paths["cache"]])
    assert any("rc_d1_embed" in n and "'near_duplicates' is not a hand-off" in n for n in notes)
    assert any("extra" in n and "part of the shared dataset" in n for n in notes)
    assert find_handoffs(tmp_path) == handoffs


def test_only_the_hand_offs_the_plan_reads_are_attached(tmp_path: Path) -> None:
    from medcompress.card_runner import select_handoffs
    from medcompress.cards import build_plan
    from medcompress.config import load_config

    cache = _real_bundle(
        tmp_path / "e0-teacher-cache", "teacher_cache", [("cache_root", "teacher")]
    )
    e1 = [("checkpoint_root", "E1/seed2/adapter")]
    extra = _real_bundle(tmp_path / "e1-seed2-adapter", "adapter", e1, experiment="E1", seed=2)
    e2 = [("checkpoint_root", "E2/seed0/adapter")]
    own = _real_bundle(tmp_path / "e2-seed0-adapter", "adapter", e2, experiment="E2", seed=0)
    config = load_config("base.yaml")
    needed, ignored = select_handoffs(build_plan("RC-E2", 0), config, [cache, extra, own])
    assert needed == [cache, own]  # the cache it reads, and its own adapter (skips training)
    assert len(ignored) == 1 and "E1 seed 2 adapter" in ignored[0] and "ignored" in ignored[0]
    b0, ignored = select_handoffs(build_plan("RC-B0"), config, [cache, extra, own])
    assert b0 == [own] and len(ignored) == 2  # RC-B0 reads only E2 seed 0's adapter


def test_an_input_or_environment_failure_is_reported_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from medcompress.card_runner import Session, StepOutcome, report_text
    from medcompress.cards import build_plan

    empty = Session(build_plan("RC-C1"), tmp_path, "now")
    assert not empty.ok and empty.outcome == "FAILED: no step ran"  # never "all steps ok"
    empty.failure = "input check failed: x"
    assert "- Outcome: FAILED: input check failed: x" in report_text(empty)
    done = Session(build_plan("RC-C1"), tmp_path, "now", steps=[StepOutcome("a", "ok", 0)])
    assert done.ok and "- Outcome: all steps ok" in report_text(done)

    monkeypatch.setenv("MEDCOMPRESS_INPUT_ROOT", str(tmp_path / "inputs"))
    runner = _runner()
    out = tmp_path / "work"
    code = runner.main(["--card", "RC-C1", "--allow-blocked", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == runner.INPUT_CHECK_FAILED  # no shared data attached: stops before a session
    assert "Outcome: FAILED: input check failed: the shared data is not attached" in text
    fake = tmp_path / "not-the-data"  # given, but no manifest: the input check fails
    fake.mkdir()
    args = ["--card", "RC-C1", "--allow-blocked", "--out", str(out), "--data-root", str(fake)]
    monkeypatch.setattr(runner, "_environment", lambda session: ["no GPU visible"])
    code = runner.main(args)
    text = capsys.readouterr().out
    assert code == runner.ENVIRONMENT_FAILED and "Outcome: FAILED: environment check" in text
    monkeypatch.setattr(runner, "_environment", lambda session: [])
    code = runner.main(args)
    text = capsys.readouterr().out
    assert code == runner.INPUT_CHECK_FAILED and "Outcome: FAILED: input check failed" in text
    assert "all steps ok" not in text
