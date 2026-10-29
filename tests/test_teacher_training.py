"""Our teacher (D01 amendment 2026-10-18): rule 3, the plan reader, the examples, resume (CPU)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from medcompress.config import load_config
from medcompress.models import teacher_training as tt
from medcompress.utils.io import append_results

REPO = Path(__file__).resolve().parents[1]
SETTINGS = load_config("teacher_training.yaml")["teacher_training"]
SWIFT_STYLE = (
    r"^(model)(?!.*\.(visual|merger)).*\."
    r"(down_proj|gate_proj|k_proj|o_proj|q_proj|up_proj|v_proj)$"
)
"""A regular expression of the form ms-swift writes as ``target_modules`` (made up for the test;
the published file's own value is printed by RC-TT0 and RC-TT1)."""


def _measured(
    fp16: float | None, fp16_gc: float | None, nf4: float | None, nf4_gc: float | None
) -> dict:
    values = {}
    for setup, gc, seconds in (
        ("fp16_2gpu", False, fp16), ("fp16_2gpu", True, fp16_gc),
        ("nf4_1gpu", False, nf4), ("nf4_1gpu", True, nf4_gc),
    ):  # fmt: skip
        label = "lora_gc_step" if gc else "lora_step"
        values[f"{setup}_{label}_ok_mb1"] = 0.0 if seconds is None else 1.0
        if seconds is not None:
            values[f"{setup}_{label}_seconds_mb1"] = seconds
    return values


def test_rule_3_prefers_fp16_over_two_t4s_unless_it_fails_or_is_twice_as_slow() -> None:
    pick = lambda *s: tt.choose_setup(tt.setup_results(_measured(*s)))  # noqa: E731
    assert pick(3.0, 4.0, 2.0, 2.5).name == "fp16_2gpu"  # 1.5x slower: still (a)
    assert pick(None, 4.0, 2.5, 3.0).name == "fp16_2gpu_gc"  # (a) only with checkpointing
    assert pick(5.1, None, 2.5, 3.0).name == "nf4_1gpu"  # more than 2x slower: (b)
    assert pick(5.0, None, 2.5, 3.0).name == "fp16_2gpu"  # exactly 2x is not "more than"
    assert pick(None, None, None, 3.0).name == "nf4_1gpu_gc"  # (a) out of memory
    with pytest.raises(tt.RuleError, match="no setup ran"):
        pick(None, None, None, None)


def _twice() -> dict:
    """The settings as RC-TT0 ran them (a T4 x2 hour counted twice until it was checked)."""
    return {**SETTINGS, "rule": {**SETTINGS["rule"], "t4x2_counts_twice": True}}


def test_rule_3_sizes_n_by_15_gpu_hours_in_steps_of_1000() -> None:
    assert tt.rule_examples(2.0, 2, budget_gpu_hours=15, step=1000) == 13000  # 13,500 fit
    assert tt.rule_examples(2.0, 1, budget_gpu_hours=15, step=1000) == 27000
    rows, chosen, n, n_once = tt.rule_rows(_measured(2.0, 3.0, 2.0, 2.5), _twice())
    values = {r["metric"]: r["value"] for r in rows}
    assert chosen.name == "fp16_2gpu" and (n, n_once) == (13000, 27000)
    assert values["rule_gpus_counted"] == 2.0  # counted twice, as RC-TT0 ran
    assert values["rule_projected_gpu_hours"] <= 15.0
    assert values["rule_projected_wall_hours"] == pytest.approx(13000 * 2.0 / 3600)
    # measured 2026-10-19: a T4 x2 hour counts once; the committed config says so
    assert tt.t4x2_counts_once(SETTINGS) and not tt.t4x2_counts_once(_twice())
    rows, _, n, n_once = tt.rule_rows(_measured(2.0, 3.0, 2.0, 2.5), SETTINGS)
    assert (
        n == n_once == 27000 and {r["metric"]: r["value"] for r in rows}["rule_gpus_counted"] == 1.0
    )
    with pytest.raises(tt.RuleError, match="not even 1000"):
        tt.rule_rows(_measured(60.0, None, None, None), SETTINGS)


def _feasibility_csv(path: Path, values: dict) -> Path:
    rows = [{"experiment": tt.EXPERIMENT, "budget": "", "seed": 0, "metric": k, "value": v}
            for k, v in values.items()]  # fmt: skip
    append_results(rows, path)
    return path


def _full_values() -> dict:
    values = _measured(2.0, 3.0, 2.0, 2.5)
    lora = tt.lora_settings(
        {"r": 16, "lora_alpha": 64, "lora_dropout": 0.05,
         "target_modules": sorted(tt.LM_TARGETS)},
        SETTINGS["lora_default"],
    )  # fmt: skip
    values.update({f"lora_{k}": v for k, v in lora.items()})
    rows, *_ = tt.rule_rows(values, _twice())  # as RC-TT0 ran (both readings recorded)
    values.update({r["metric"]: r["value"] for r in rows})
    return values


def test_the_training_run_reads_its_plan_from_rc_tt0_only(tmp_path: Path) -> None:
    path = _feasibility_csv(tmp_path / "rc_tt0.csv", _full_values())
    plan = tt.read_plan(path)
    assert (plan.setup, plan.gradient_checkpointing, plan.n_examples) == ("fp16_2gpu", False, 13000)
    assert (plan.lora_rank, plan.lora_alpha, plan.lora_from_published) == (16, 64.0, True)
    assert tt.read_plan(path, t4x2_counts_once=True).n_examples == 27000
    partial = _feasibility_csv(tmp_path / "partial.csv", _measured(2.0, 3.0, 2.0, 2.5))
    with pytest.raises(tt.RuleError, match="did not finish its rule step"):
        tt.read_plan(partial)
    with pytest.raises(tt.RuleError, match="run RC-TT0 first"):
        tt.read_plan(tmp_path / "missing.csv")


def test_lora_settings_come_from_the_published_adapter_when_readable() -> None:
    default = SETTINGS["lora_default"]
    assert tt.lora_settings(None, default)["from_published"] == 0.0
    assert tt.lora_settings(None, default)["rank"] == 16
    published = tt.lora_settings(
        {"r": 16, "lora_alpha": 64, "target_modules": "all-linear"}, default
    )
    assert (published["alpha"], published["targets_match"], published["dropout"]) == (
        64.0,
        1.0,
        0.05,
    )
    other = tt.lora_settings({"r": 8, "target_modules": ["qkv"]}, default)
    assert other["rank"] == 8 and other["targets_match"] == 0.0
    assert tt.lora_settings({"target_modules": SWIFT_STYLE}, default)["targets_match"] == 1.0


def test_the_published_targets_are_read_and_applied_inside_the_decoder_only() -> None:
    from medcompress.compression.lora import layer_regex

    regex = SWIFT_STYLE
    assert tt.published_target_names(regex) == tuple(sorted(tt.LM_TARGETS))
    assert tt.published_target_names(["model.layers.0.self_attn.q_proj", "v_proj"]) == (
        "q_proj", "v_proj",
    )  # fmt: skip
    assert tt.published_target_names("all-linear") == ("all-linear",)
    for bad in ("^model\\..*$", [], None):
        with pytest.raises(tt.RuleError):
            tt.published_target_names(bad)
    # the layout of Qwen2.5-VL in recent transformers: the vision blocks have gate/up/down too
    linear = [f"model.language_model.layers.{i}.self_attn.{p}" for i in range(2)
              for p in ("q_proj", "k_proj", "v_proj", "o_proj")]  # fmt: skip
    linear += [f"model.language_model.layers.{i}.mlp.{p}" for i in range(2)
               for p in ("gate_proj", "up_proj", "down_proj")]  # fmt: skip
    linear += [f"model.visual.blocks.{i}.mlp.{p}" for i in range(2)
               for p in ("gate_proj", "up_proj", "down_proj")]  # fmt: skip
    linear += ["model.visual.blocks.0.attn.qkv", "model.visual.merger.mlp.0", "lm_head"]
    names = tt.published_target_names(regex)
    counts = tt.adapted_counts(linear, layer_regex("model.language_model.layers", names), names)
    assert counts == {name: 2 for name in names}  # 2 decoder layers; no vision block, no head
    stack = "model.language_model.layers"
    only_q = tt.adapted_counts(linear, layer_regex(stack, ["q_proj", "qkv"]), ["q_proj", "qkv"])
    assert only_q == {"q_proj": 2, "qkv": 0}  # qkv is only in the vision tower: the run stops


def _sample(i: int, kind: str) -> SimpleNamespace:
    return SimpleNamespace(sample_id=f"s{i:04d}", metadata={"question_class": (kind,)})


def test_the_examples_are_one_nested_order_stratified_by_question_type() -> None:
    train = [_sample(i, "common") for i in range(900)] + [
        _sample(900 + i, "mid") for i in range(95)
    ]
    train += [_sample(2000 + i, f"rare{i}") for i in range(5)]
    order = tt.teacher_order(train, 20261012)
    assert len(order) == 1000 and len({s.sample_id for s in order}) == 1000
    first = {tt.question_type(s) for s in order[:7]}
    assert first == {"common", "mid"} | {f"rare{i}" for i in range(5)}  # every type at once
    head = [tt.question_type(s) for s in order[:107]]
    assert 85 <= head.count("common") <= 100 and 8 <= head.count("mid") <= 13  # in proportion
    assert tt.teacher_order(list(reversed(train)), 20261012) == order  # input order is irrelevant
    assert tt.teacher_order(train, 1) != order
    plan = SimpleNamespace(n_examples=13000)
    assert tt.training_size(plan, 6000) == 6000 and tt.training_size(plan, 200000) == 13000
    assert tt.extension_size(13000) == 6500 and tt.extension_size(1001) == 501


def test_training_settings_follow_the_recipe() -> None:
    fake = tt.TeacherPlan("fp16_2gpu", False, 2.0, 13000, 27000, 16, 64.0, 0.05, True)
    fields = tt.train_settings(fake, SETTINGS, 13000)
    assert fields["steps"] == 813 and fields["warmup_steps"] == 25  # ceil(13000/16), ceil(3 %)
    assert (fields["learning_rate"], fields["lr_schedule"], fields["weight_decay"]) == (
        2e-5,
        "cosine",
        0.1,
    )
    assert (fields["batch_size"], fields["grad_accumulation"], fields["kd_weight"]) == (1, 16, 0.0)
    assert fields["checkpoint_every"] == round(30 * 60 / (2.0 * 16))  # about every 30 minutes
    assert fields["lora"]["target_modules"] == "published"  # D01, 2026-10-19
    d06 = tt.train_settings(fake, {**SETTINGS, "target_modules": ["decoder_mlp"]}, 13000)
    assert d06["lora"]["target_modules"] == ["decoder_mlp"]


def test_the_cosine_schedule_warms_up_then_decays_to_zero() -> None:
    from medcompress.compression.distillation.offline_kd import TrainConfig, _learning_rate
    from medcompress.compression.lora import LoRAConfig

    cfg = TrainConfig("sft", LoRAConfig(16, 32, 0.05, ("decoder_attention",)), 100, 2e-5, 1, 16,
                      0.0, 1.0, 1600,
                      extra={"warmup_steps": 10, "lr_schedule": "cosine"})  # fmt: skip
    rates = [_learning_rate(cfg, s) for s in range(100)]
    assert rates[0] == pytest.approx(2e-6) and rates[9] == pytest.approx(2e-5)
    assert rates[10] == pytest.approx(2e-5) and rates[55] == pytest.approx(1e-5, rel=1e-2)
    assert rates[99] < 1e-7 and all(a >= b for a, b in zip(rates[10:], rates[11:], strict=False))


def _state(folder: Path, step: int) -> Path:
    (folder / "current").mkdir(parents=True)
    (folder / "current" / "progress.json").write_text(
        json.dumps({"fingerprint": "f", "step": step})
    )
    return folder


def test_a_rerun_finds_the_furthest_saved_state_also_in_an_attached_output(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import train_teacher
    finally:
        sys.path.remove(str(REPO / "scripts"))
    old = _state(tmp_path / "input" / "v1" / "checkpoints" / "E0" / "teacher_resume", 3)
    newer = _state(tmp_path / "input" / "v2" / "checkpoints" / "E0" / "teacher_resume", 7)
    assert tt.find_resume_states([tmp_path / "input"]) == [newer, old]
    local = tmp_path / "working" / "checkpoints" / "E0" / "teacher_resume"
    assert "from step 7" in train_teacher._restore_state(local, [tmp_path / "input"])
    assert tt.saved_step(local) == 7
    _state(tmp_path / "input" / "v0" / "checkpoints" / "E0" / "teacher_resume", 5)
    assert "saved locally" in train_teacher._restore_state(local, [tmp_path / "input"])  # 7 > 5


def test_the_feasibility_script_records_failures_and_applies_rule_3(tmp_path: Path) -> None:
    script = REPO / "scripts" / "teacher_feasibility.py"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(
        [sys.executable, str(script), "run", "--out-dir", str(tmp_path),
         "--steps", "lora_config", "fp16_2gpu"],
        capture_output=True, text=True, cwd=REPO, env=env, check=False, timeout=300,
    )  # fmt: skip
    assert done.returncode == 3 and "no setup ran" in done.stderr, done.stdout + done.stderr
    values = tt.read_values(tmp_path / "teacher_training.csv")
    assert values["lora_rank"] == 16 and values["lora_from_published"] == 0.0  # D06 defaults
    assert values["fp16_2gpu_lora_step_ok_mb1"] == 0.0
    good = _feasibility_csv(tmp_path / "good.csv", _full_values())
    shown = subprocess.run(
        [sys.executable, str(script), "rule", "--csv", str(good)],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=120,
    )  # fmt: skip
    assert shown.returncode == 0 and "estimates" in shown.stdout
    assert "N = 27000" in shown.stdout  # the rule again, with a T4 x2 hour counted once


def test_the_training_dry_run_reads_the_plan_and_the_shared_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_bundles import _shared_data

    shared = tmp_path / "working" / "medcompress_data"
    _shared_data(shared)
    made = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_data.py"), "bundle",
         "--bundle-dir", str(shared)],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=300,
    )  # fmt: skip
    assert made.returncode == 0, made.stderr
    plan = _feasibility_csv(tmp_path / "rc_tt0.csv", _full_values())
    refused = subprocess.run(  # the config pins RC-D1 run 4b's split, not this fixture's
        [sys.executable, str(REPO / "scripts" / "train_teacher.py"), "--data-root", str(shared),
         "--feasibility", str(plan), "--out", str(tmp_path / "out"), "--dry-run"],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=300,
    )  # fmt: skip
    assert refused.returncode == 5 and "the config pins" in refused.stderr
    held = json.loads((shared / "bundle_manifest.json").read_text())["split_sha256"]
    monkeypatch.setenv("MEDCOMPRESS_SPLIT_SHA256", held)  # the fixture's own split (tests only)
    run = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train_teacher.py"), "--data-root", str(shared),
         "--feasibility", str(plan), "--out", str(tmp_path / "out"), "--dry-run"],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=300,
    )  # fmt: skip
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"split: split_sha256 {held} (pinned, checked)" in run.stdout
    assert "setup fp16_2gpu" in run.stdout and "N = " in run.stdout and "estimate" in run.stdout
    assert "counted once" in run.stdout  # the config, measured 2026-10-19
    assert "read from the hub when the run starts" in run.stdout
    published = tmp_path / "adapter_config.json"
    published.write_text(json.dumps({
        "r": 16, "lora_alpha": 64, "lora_dropout": 0.05,
        "target_modules": SWIFT_STYLE,
    }))  # fmt: skip
    local = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train_teacher.py"), "--data-root", str(shared),
         "--feasibility", str(plan), "--out", str(tmp_path / "out"), "--dry-run",
         "--published-adapter-config", str(published)],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=300,
    )  # fmt: skip
    assert local.returncode == 0, local.stdout + local.stderr
    assert "down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj" in local.stdout
    published.write_text(json.dumps({"r": 8, "lora_alpha": 64, "target_modules": "all-linear"}))
    differ = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train_teacher.py"), "--data-root", str(shared),
         "--feasibility", str(plan), "--out", str(tmp_path / "out"), "--dry-run",
         "--published-adapter-config", str(published)],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=300,
    )  # fmt: skip
    assert differ.returncode == 5 and "differ from the published file" in differ.stderr
    missing = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train_teacher.py"), "--data-root", str(shared),
         "--feasibility", str(tmp_path / "none.csv"), "--dry-run"],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=120,
    )  # fmt: skip
    assert missing.returncode == 4 and "run RC-TT0 first" in missing.stderr


def test_our_teacher_adapter_must_be_attached_before_anything_loads(tmp_path: Path) -> None:
    from medcompress.models.loading import ModelConfigError, ModelSpec, check_loadable

    ours = ModelSpec.from_config(load_config("models/qwen2_5_vl_7b_teacher.yaml"))
    assert ours.adapter.endswith("E0/teacher_adapter") and ours.adapter_revision is None
    with pytest.raises(ModelConfigError, match="RC-TT1"):
        check_loadable(ours)
    published = ModelSpec.from_config(load_config("models/qwen2_5_vl_7b_teacher_simulamet.yaml"))
    assert (
        published.adapter == "SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft" and published.adapter_revision
    )
    check_loadable(published)  # the appendix item can still load the published adapter
    folder = tmp_path / "E0" / "teacher_adapter"
    folder.mkdir(parents=True)
    os.environ["MEDCOMPRESS_CHECKPOINT_ROOT"] = str(tmp_path)
    try:
        attached = ModelSpec.from_config(load_config("models/qwen2_5_vl_7b_teacher.yaml"))
        check_loadable(attached)  # attached: loadable
    finally:
        del os.environ["MEDCOMPRESS_CHECKPOINT_ROOT"]


def test_rc_tt1_reads_n_38000_from_the_committed_rc_tt0_file() -> None:
    committed = REPO / tt.DEFAULT_CSV  # RC-TT0's run (commit 3afb035)
    plan = tt.read_plan(committed, t4x2_counts_once=tt.t4x2_counts_once(SETTINGS))
    assert plan.n_examples == 38000 and plan.setup == "fp16_2gpu"  # a T4 x2 hour counted once
    assert (plan.lora_rank, plan.lora_alpha, plan.lora_dropout) == (16, 64.0, 0.05)
    assert plan.lora_from_published and not plan.gradient_checkpointing
    assert tt.train_settings(plan, SETTINGS, plan.n_examples)["steps"] == 2375
