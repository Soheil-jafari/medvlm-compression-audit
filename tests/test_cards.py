"""Unattended runs: card plans, the runner (fake steps), the environment check (CPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from medcompress import card_runner as runner_mod
from medcompress.cards import (
    RUNNABLE_CARDS,
    CardError,
    Run,
    build_plan,
    card_file,
    card_status,
    g2_runs,
)
from medcompress.config import experiment_config_path, load_config
from medcompress.registry import REGISTRY
from medcompress.utils import kaggle

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "docs" / "runs"
EXTRA = {
    "kvasir_seg": Path("/kaggle/input/kvasir-seg"),
    "data_root": Path("/kaggle/input/data"),
    "external_images": Path("/kaggle/input/endobench"),
}
SPLIT = {"RC-E1", "RC-E2", "RC-E3", "RC-E4", "RC-E5", "RC-E6", "RC-E7", "RC-E8"} | {
    f"RC-A{n}" for n in range(1, 7)
}


def _plans() -> list:
    plans = []
    for card in RUNNABLE_CARDS:
        seeds = (0, 1, 2) if card in SPLIT else (None,)
        plans += [build_plan(card, s, **EXTRA) for s in seeds]
    return plans


def test_every_runnable_card_has_a_plan_whose_scripts_exist_and_are_registered() -> None:
    for plan in _plans():
        assert card_status(card_file(plan.card, RUNS))  # the card file and its status line
        for step in plan.steps:
            if step.argv[0] == "-c":
                continue
            script = step.argv[0]
            assert (REPO / script).is_file(), (plan.tag, script)
            if "--exp" in step.argv and script != "scripts/run_experiment.py":
                exp = step.argv[step.argv.index("--exp") + 1]
                assert script in REGISTRY[exp].scripts, (plan.tag, script)
        for handoff in plan.handoffs:
            assert handoff.runs and handoff.name


def test_seeds_are_split_one_per_session() -> None:
    for card in SPLIT:
        with pytest.raises(CardError, match="give --seed"):
            build_plan(card)
        with pytest.raises(CardError, match="not 3"):
            build_plan(card, 3)
    for card in ("RC-E0a", "RC-G2", "RC-R1"):
        with pytest.raises(CardError, match="no seed"):
            build_plan(card, 1)
    with pytest.raises(CardError, match="rented GPU"):
        build_plan("RC-T1")
    with pytest.raises(CardError, match="no runnable commands"):
        build_plan("RC-L1")


def test_each_run_reads_the_e2_of_its_own_seed_and_the_hand_offs_it_needs() -> None:
    labels = {p.tag: [n.label for n in p.needs] for p in _plans()}
    assert labels["RC-E2_seed1"] == ["teacher cache"]
    assert labels["RC-E1_seed2"] == []
    assert labels["RC-E3_seed2"] == ["E2 seed2 adapter"]
    assert labels["RC-E4_seed1"] == ["teacher cache"]  # E4 starts from the pretrained student
    assert labels["RC-B0"] == ["E2 seed0 adapter"]
    assert labels["RC-E8_seed1"] == [
        "E2 seed1 adapter", "E7 B1 seed1 pruned", "E7 B2 seed1 pruned", "E7 B3 seed1 pruned",
    ]  # fmt: skip
    for s in (0, 1, 2):
        assert labels[f"RC-E7_seed{s}"][:2] == [f"E2 seed{s} adapter", "teacher cache"]
    seed0 = build_plan("RC-E5", 0)
    optional = [n.run for n in seed0.needs if n.optional]
    assert optional == [r for r in g2_runs() if r.exp == "E5"]  # RC-G2's runs are reused
    assert not [n for n in build_plan("RC-E5", 1).needs if n.optional]
    skipped = [s.skip_if_pruned for s in seed0.steps if s.skip_if_pruned]
    assert Run("E5", 0, "B2", "taylor") in skipped
    names = {h.name: h.kind for p in _plans() for h in p.handoffs}
    assert names["E0_teacher_cache"] == "teacher_cache"
    assert names["E2_seed2_adapter"] == "adapter" and names["E1_seed0_adapter"] == "adapter"
    assert names["E7_seed1_pruned"] == "pruned" and names["E8_seed0_outcomes"] == "outcomes"
    a4 = build_plan("RC-A4", 0)
    assert [
        s.argv for s in a4.steps if "--variant" in s.argv and "run_experiment" in s.argv[0]
    ] == [
        (
            "scripts/run_experiment.py",
            "--exp",
            "A4",
            "--seed",
            "0",
            "--variant",
            v,
            "--budget",
            "B2",
        )
        for v in ("n32", "n64", "n256")
    ]  # n128 is E7 itself


def test_the_runner_refuses_a_card_that_is_not_ready() -> None:
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "run_card.py"), *args],
            capture_output=True, text=True, cwd=REPO, check=False, timeout=120,
        )  # fmt: skip

    status = card_status(card_file("RC-E2", RUNS))
    refused = run("--card", "RC-E2", "--seed", "0", "--dry-run")
    if status != "ready":
        assert refused.returncode == 2 and "not `ready`" in refused.stderr
    shown = run("--card", "RC-E2", "--seed", "0", "--dry-run", "--allow-blocked",
                "--data-root", "/kaggle/input/data", "--from", "/kaggle/input/cache")  # fmt: skip
    assert shown.returncode == 0, shown.stderr
    assert (
        "scripts/02_train_student.py --exp E2 --seed 0 --data-root /kaggle/input/data"
        in shown.stdout
    )
    assert "--from /kaggle/input/cache" in shown.stdout and "reads: teacher cache" in shown.stdout
    assert run("--card", "RC-E2", "--dry-run").returncode == 2  # no seed


def _config(root: Path) -> dict:
    config = load_config(experiment_config_path("E2"))
    config["paths"] = {
        "data_root": str(root / "data"), "cache_root": str(root / "cache"),
        "checkpoint_root": str(root / "checkpoints"), "output_root": str(root / "outputs"),
        "results_root": "results",
    }  # fmt: skip
    return config


def _git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def test_an_unattended_e2_run_trains_hands_off_and_reports(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    config = _config(tmp_path / "work")
    plan = build_plan("RC-E2", 1)
    session = runner_mod.Session(plan, tmp_path / "work", "2026-10-17 09:00", commit="abc")
    (Path(config["paths"]["cache_root"]) / "teacher").mkdir(parents=True)
    assert runner_mod.missing_needs(plan, config) == []
    seen: list[list[str]] = []

    def fake(command: Sequence[str], env: Mapping[str, str], log: Path) -> int:
        seen.append(list(command))
        if "02_train_student.py" in command[1]:
            adapter = Path(config["paths"]["checkpoint_root"]) / "E2" / "seed1" / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"w" * 64)
            metrics = repo / "results" / "E2" / "nobudget" / "seed1" / "metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                "experiment,budget,seed,metric,value,ci_low,ci_high\n"
                "E2,,1,train_peak_gpu_bytes,11372899328.0,,\n"
            )
            return 0
        return 4 if "07_benchmark" in command[1] else 0  # the hardware setting is still open

    runner_mod.run_steps(
        session, config, {}, repo=repo, data_root=Path("/in/data"), handoffs=[Path("/in/cache")],
        runner=fake,
    )  # fmt: skip
    assert [s.status for s in session.steps] == ["ok", "ok", "waiting on a setting"]
    assert all(c[-4:] == ["--data-root", "/in/data", "--from", "/in/cache"] for c in seen)
    assert session.ok
    runner_mod.export_handoffs(session, config)
    manifest = json.loads(
        (tmp_path / "work" / "handoff" / "E2_seed1_adapter" / "bundle_manifest.json").read_text()
    )
    assert (manifest["kind"], manifest["experiment"], manifest["seed"]) == ("adapter", "E2", 1)
    assert manifest["card"] == "RC-E2" and manifest["runs"] == ["E2 seed1"]
    runner_mod.collect_results(session, repo)
    copied = tmp_path / "work" / "results_to_commit" / "results" / "E2" / "nobudget" / "seed1"
    assert (copied / "metrics.csv").is_file()
    text = runner_mod.report_text(session)
    for part in (
        "### RC-E2 run report (seed 1)", "Commit actually used: abc", "train E2 seed1:",
        "train_peak_gpu_bytes = 11,372,899,328", "E2_seed1_adapter:", "make it a private dataset",
        "results/E2/nobudget/seed1/metrics.csv", "waits on an open setting", "all steps ok",
    ):  # fmt: skip
        assert part in text, part


def test_a_failed_step_stops_the_card_and_missing_inputs_are_named(tmp_path: Path) -> None:
    config = _config(tmp_path / "work")
    plan = build_plan("RC-E2", 0)
    assert runner_mod.missing_needs(plan, config) == [
        f"teacher cache (expected at {tmp_path / 'work' / 'cache' / 'teacher'})"
    ]
    session = runner_mod.Session(plan, tmp_path / "work", "now")
    codes = iter([0, 1, 0])
    runner_mod.run_steps(
        session, config, {}, repo=tmp_path, data_root=None, handoffs=[],
        runner=lambda c, e, log: next(codes),
    )  # fmt: skip
    assert [s.status for s in session.steps] == ["ok", "failed"] and not session.ok
    assert "stopped early" in runner_mod.report_text(session)
    open_setting = runner_mod.Session(plan, tmp_path / "work", "now")
    runner_mod.run_steps(
        open_setting, config, {}, repo=tmp_path, data_root=None, handoffs=[],
        runner=lambda c, e, log: 4,
    )  # fmt: skip
    assert [s.status for s in open_setting.steps] == ["stopped: setting open"]
    assert not open_setting.ok


def test_a_grid_run_made_by_rc_g2_is_skipped_when_its_model_is_attached(tmp_path: Path) -> None:
    from medcompress.compression.recovery import pruned_dir

    config = _config(tmp_path / "work")
    run = Run("E7", 0, "B2")
    metrics = tmp_path / "results" / "E7" / "B2" / "seed0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("experiment,budget,seed,metric,value\nE7,B2,0,budget_within_tolerance,1\n")
    assert not runner_mod.made_by_g2(run, config, tmp_path)  # no pruned model attached yet
    pruned_dir(config, run.context).mkdir(parents=True)
    assert runner_mod.made_by_g2(run, config, tmp_path)
    plan = build_plan("RC-E7", 0, budgets=["B2"])
    session = runner_mod.Session(plan, tmp_path / "work", "now")
    calls: list[str] = []
    runner_mod.run_steps(
        session, config, {}, repo=tmp_path, data_root=None, handoffs=[],
        runner=lambda c, e, log: calls.append(c[1]) or 0,
    )  # fmt: skip
    assert [s.status for s in session.steps] == ["skipped", "ok"]
    assert calls == ["scripts/06_evaluate.py"]  # only its test evaluation runs


def test_the_environment_check_knows_the_kaggle_pitfalls() -> None:
    pins = kaggle.parse_requirements(REPO / "requirements" / "train.txt")
    by_name = {p.name: p for p in pins}
    assert (by_name["torchao"].op, by_name["torchao"].version) == (">=", "0.16.0")
    assert (by_name["optimum"].op, by_name["optimum"].version) == (">=", "1.24.0")
    assert by_name["peft"].op == "==" and "pyyaml" in by_name  # base.txt is followed
    kaggle_default = {
        "torchao": "0.10.0", "optimum": "1.23.0", "peft": "0.21.2", "gptqmodel": None,
    }  # fmt: skip

    def version_of(name: str) -> str | None:
        return kaggle_default.get(name, "99.0")

    problems, lines = kaggle.check_versions(pins, quantization=False, version_of=version_of)
    assert any("torchao 0.10.0" in p and "pip uninstall -y torchao" in p for p in problems)
    assert any(p.startswith("optimum 1.23.0") for p in problems)
    assert not any("gptqmodel" in p for p in problems)  # only quantization cards need it
    quant, _ = kaggle.check_versions(pins, quantization=True, version_of=version_of)
    assert any("gptqmodel is not installed" in p for p in quant)
    assert kaggle.satisfied("2.11.0+cu128", kaggle.Pin("torch", ">=", "2.8"))
    assert kaggle.check_gpus(2, 1) and not kaggle.check_gpus(1, 2) and not kaggle.check_gpus(0, 0)


def test_kvasir_seg_is_found_below_the_attached_dataset(tmp_path: Path) -> None:
    seg = tmp_path / "input" / "kvasir-seg" / "Kvasir-SEG"
    for name in ("images", "masks"):
        (seg / name).mkdir(parents=True)
        (seg / name / "a.jpg").write_bytes(b"x")
    counts = kaggle.copy_kvasir_seg(tmp_path / "input", tmp_path / "data")
    assert counts == {"images": 1, "masks": 1}
    assert (tmp_path / "data" / "kvasir_seg" / "masks" / "a.jpg").is_file()
    with pytest.raises(FileExistsError):
        kaggle.copy_kvasir_seg(tmp_path / "input", tmp_path / "data")


def test_the_generic_notebook_takes_the_card_and_seed_in_its_first_cell() -> None:
    notebook = json.loads((REPO / "notebooks" / "run_card.ipynb").read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code[0].lstrip().startswith("# ---- fill in")
    assert 'CARD = "RC-E2"' in code[0] and "SEED = 0" in code[0] and "FROM = []" in code[0]
    assert "scripts/run_card.py" in code[-1]
    assert notebook["nbformat"] == 4


def test_a_later_session_runs_only_what_is_left_when_the_adapter_is_attached(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "work")
    adapter = Path(config["paths"]["checkpoint_root"]) / "E1" / "seed2" / "adapter"
    adapter.mkdir(parents=True)  # attached from the first session's hand-off
    session = runner_mod.Session(build_plan("RC-E1", 2), tmp_path / "work", "now")
    calls: list[str] = []
    runner_mod.run_steps(
        session, config, {}, repo=tmp_path, data_root=None, handoffs=[],
        runner=lambda c, e, log: calls.append(c[1]) or 0,
    )  # fmt: skip
    assert [s.status for s in session.steps] == ["skipped", "ok", "ok"]
    assert calls == ["scripts/06_evaluate.py", "scripts/07_benchmark_efficiency.py"]
