"""Disk space on Kaggle (2026-11-13): checks before saves, hard-linked hand-offs, exit code 7."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from medcompress.utils import disk


def test_a_save_that_does_not_fit_stops_with_a_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(disk, "free_bytes", lambda _path: 3 * disk.GIB)
    disk.require_free_disk(tmp_path, 2 * disk.GIB, "the pruned model")  # 2 + 0.5 <= 3: fits
    with pytest.raises(
        disk.DiskSpaceError, match=r"the pruned model: needs 5\.85 GiB .* 3\.00 GiB free"
    ):
        disk.require_free_disk(tmp_path / "new" / "dir", int(5.85 * disk.GIB), "the pruned model")


def test_hand_off_files_are_hard_links_not_copies(tmp_path: Path) -> None:
    from medcompress.utils.bundles import export_handoff

    model = tmp_path / "ckpt" / "E5" / "B2" / "seed0" / "pruned"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"w" * 4096)
    (model / "manifest.json").write_text("{}")
    config = {"paths": {"checkpoint_root": str(tmp_path / "ckpt")}}
    bundle = export_handoff(config, [("checkpoint_root", "E5/B2/seed0/pruned")],
                            tmp_path / "handoff" / "x", kind="pruned", meta={})  # fmt: skip
    linked = bundle.root / "checkpoint_root" / "E5/B2/seed0/pruned" / "model.safetensors"
    assert os.stat(linked).st_ino == os.stat(model / "model.safetensors").st_ino  # one copy
    assert disk.tree_bytes(model) == 4096 + 2


def test_a_full_disk_ends_the_stage_with_its_own_exit_code() -> None:
    from medcompress.cli import DISK_SPACE_EXIT_CODE, run_stage

    def stage(**_kwargs: object) -> None:
        raise disk.DiskSpaceError("the pruned model: needs 5.85 GiB")

    assert run_stage("05_prune_and_recover", stage) == DISK_SPACE_EXIT_CODE == 7


def test_the_pruned_model_checks_the_disk_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from medcompress.compression.pruning.checkpoint import save_pruned

    monkeypatch.setattr(disk, "free_bytes", lambda _path: 0)
    with pytest.raises(disk.DiskSpaceError, match="the pruned model"):
        save_pruned(torch.nn.Linear(64, 64), tmp_path / "pruned", [], mlp_channel_group_size=128)
    assert not (tmp_path / "pruned").exists()  # nothing written


def test_the_smoke_deletes_a_jobs_checkpoints_once_nothing_reads_them() -> None:
    from medcompress import smoke

    jobs = smoke.smoke_jobs()
    by_name = {j.name: j for j in jobs}
    assert smoke.job_experiments(by_name["E7"]) == {("E7", None)}
    assert smoke.job_experiments(by_name["E3 w4"]) == {("E3", "w4")}
    assert smoke.job_experiments(by_name["E9 evaluation path"]) == set()  # attached links kept
    finished = {"E7"}
    assert by_name["E7"] not in smoke.ready_for_cleanup(jobs, finished)  # E8 and A1 still read it
    finished |= {"E7 30-group check", "E8 B2", "A1"}
    assert by_name["E7"] in smoke.ready_for_cleanup(jobs, finished)
    reduced = smoke.with_dependencies(jobs, ["RC-A1", "RC-E3"])
    names = [j.name for j in reduced]
    assert "E7" in names and names[0].startswith("calibration check")  # what they read, first
    assert "E5 magnitude" not in names
