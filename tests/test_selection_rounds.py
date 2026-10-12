"""Selection rounds R0 to R3, the ranking files and the report on the tiny offline model.

The tiny model has random weights, so only mechanics are asserted: what each round
writes, that the candidates are scored on the same probes, and that the driver ranks,
gates and reports from those files.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

import numpy as np  # noqa: E402
import tiny_vlm  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eval_fixtures import eval_config, write_dataset  # noqa: E402

from medcompress.compression.distillation import batching, offline_kd  # noqa: E402
from medcompress.config import ConfigError, load_config  # noqa: E402
from medcompress.data.base import VQASample  # noqa: E402
from medcompress.evaluation.aggregate import validate_results_tree  # noqa: E402
from medcompress.models.teacher_cache import CacheRecord  # noqa: E402
from medcompress.selection import driver, rounds  # noqa: E402
from medcompress.selection.ranking import CandidateOutcomes  # noqa: E402
from medcompress.utils.io import upsert_results  # noqa: E402

TINY_TEMPLATE = "Is there a {finding} in this image ? Answer yes or no ."
POOL = ["qwen2_5_vl_3b", "qwen3_vl_2b", "internvl3_5_2b"]


def _config(root: Path, **selection: Any) -> dict[str, Any]:
    config = eval_config(
        root,
        domain={"probe": {"template": TINY_TEMPLATE, "display_names": {}}},
        max_new_tokens=4,
    )
    config["paths"]["cache_root"] = str(root / "cache")
    loaded = load_config("student_selection.yaml")
    chosen = copy.deepcopy(loaded["student_selection"])
    chosen["candidates"] = [f"models/{name}_student.yaml" for name in POOL]
    chosen["compute_unit_b"] = 4
    chosen["rounds"]["R1"]["validation_subset"]["n_images"] = 4
    chosen["rounds"]["R1"]["keep"] = 2
    chosen["gates"]["G2_uses_image"]["threshold"] = -1.0  # mechanics only: always passes
    chosen.update(selection)
    config["student_selection"] = chosen
    config["training"] = {
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["decoder_attention", "decoder_mlp"]},
        "steps": 3, "learning_rate": 1e-2, "batch_size": 2, "grad_accumulation": 1,
        "kd_weight": 0.5, "kd_temperature": 1.0, "data_subset": "all",
    }  # fmt: skip
    return config


def _splits(root: Path) -> dict[str, list[VQASample]]:
    """Images 0 to 5 are the validation split (they hold the small lesions), 6 to 11 train."""
    samples = write_dataset(root)
    val = [s for s in samples if int(s.image_path.split("/")[1].split(".")[0]) < 6]
    train = [s for s in samples if s not in val]
    return {"train": train, "val": val}


def _records(root: Path, samples: list[VQASample], *, shift: int = 0) -> dict[str, CacheRecord]:
    """Teacher-forced top-k records from a second random model, like the real cache."""
    teacher = tiny_vlm.tiny_wrapper(seed=1)
    pad = teacher.tokenizer.pad_token_id
    out = {}
    for s in samples:
        ids = batching.ground_truth_ids(teacher, s.answer)
        prompt = batching.prompt_inputs(teacher, root / s.image_path, s.question)
        batch = batching.assemble([prompt], [ids], pad)
        with torch.no_grad():
            logits = offline_kd.forward_answer_logits(teacher.module, batch)[0]
        top = F.log_softmax(logits.float(), dim=-1).topk(5, dim=-1)
        out[s.sample_id] = CacheRecord(
            s.sample_id,
            s.answer,
            np.array(ids[:-1] + [ids[-1] + shift], dtype=np.int64),
            top.indices.numpy().astype(np.int64),
            top.values.numpy().astype(np.float32),
        )
    return out


def _metrics(config: dict[str, Any], round_name: str, candidate: str) -> dict[str, float]:
    path = Path(config["paths"]["results_root"]) / "selection" / round_name / candidate
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        return {r["metric"]: float(r["value"]) for r in csv.DictReader(handle)}


def _outcomes(config: dict[str, Any], round_name: str, candidate: str, seed: int = 0) -> Path:
    base = Path(config["paths"]["output_root"]) / "selection" / round_name / candidate
    return base / f"seed{seed}" / "outcomes.npz"


# ------------------------------------------------------------------------ plans


def test_plan_names_every_open_setting(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["student_selection"]["compute_unit_b"] = "TODO"
    config["student_selection"]["rounds"]["R1"]["validation_subset"]["n_images"] = "TODO"
    config["student_selection"]["gates"]["G2_uses_image"]["threshold"] = "TODO"
    with pytest.raises(ConfigError) as caught:
        rounds.plan_round(config, "R1", "qwen3_vl_2b", need_side=False)
    message = str(caught.value)
    for key in ("compute_unit_b", "validation_subset.n_images", "G2_uses_image.threshold"):
        assert key in message
    with pytest.raises(ConfigError, match="image_longest_side"):
        rounds.plan_round(_config(tmp_path), "R0", "qwen3_vl_2b")


def test_plan_rejects_unknown_rounds_candidates_and_seeds(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="round must be one of"):
        rounds.plan_round(config, "R9", "qwen3_vl_2b")
    with pytest.raises(ValueError, match="unknown candidate"):
        rounds.plan_round(config, "R1", "not_a_model", need_side=False)
    with pytest.raises(ValueError, match="1 seed"):
        rounds.plan_round(config, "R1", "qwen3_vl_2b", seed=1, need_side=False)
    assert rounds.plan_round(config, "R3", "qwen3_vl_2b", seed=1, need_side=False).seed == 1
    with pytest.raises(ValueError, match="2 seed"):
        rounds.plan_round(config, "R3", "qwen3_vl_2b", seed=2, need_side=False)


def test_round_lengths_follow_the_compute_unit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    r1 = rounds.plan_round(config, "R1", "qwen3_vl_2b", need_side=False)
    r2 = rounds.plan_round(config, "R2", "qwen3_vl_2b", need_side=False)
    r3 = rounds.plan_round(config, "R3", "qwen3_vl_2b", need_side=False)
    assert (r1.train.mode, r1.train.kd_weight, r1.train.data_subset) == ("sft", 0.0, 4)
    assert (r2.train.mode, r2.train.data_subset) == ("offline_kd", 8)
    assert r2.train.steps == 4 and r1.train.steps == 2  # ceil(examples / batch)
    assert (r3.train.mode, r3.train.steps) == ("offline_kd", 3)  # the E2 recipe unchanged
    assert r1.g2_threshold == -1.0 and r2.g2_threshold is None
    assert "blank_image" in r1.eval_config["evaluation"]["suites"]
    assert "mask_audit" in r2.eval_config["evaluation"]["suites"]
    assert r1.eval_config["evaluation"]["split"] == "val"


def test_validation_subset_is_the_same_for_every_caller(tmp_path: Path) -> None:
    val = _splits(tmp_path)["val"]
    first = rounds.select_validation_subset(val, 4, seed=0)
    again = rounds.select_validation_subset(list(reversed(val)), 4, seed=0)
    assert {s.sample_id for s in first} == {s.sample_id for s in again}
    assert len({s.image_path for s in first}) == 4
    assert all(len([x for x in first if x.image_path == s.image_path]) == 2 for s in first)
    other = rounds.select_validation_subset(val, 4, seed=1)
    assert {s.sample_id for s in other} != {s.sample_id for s in first}
    assert len(rounds.select_validation_subset(val, None, seed=0)) == len(val)
    assert len(rounds.select_validation_subset(val, 99, seed=0)) == len(val)


def test_candidate_names_and_paths() -> None:
    assert rounds.candidate_name("models/qwen3_vl_2b_student.yaml") == "qwen3_vl_2b"
    selection = {"candidates": ["models/a_student.yaml", "models/b_student.yaml"]}
    assert rounds.candidate_config_path(selection, "b") == "models/b_student.yaml"
    with pytest.raises(ValueError, match=r"pool is \['a', 'b'\]"):
        rounds.candidate_config_path(selection, "c")


# ----------------------------------------------------------------------- tokenizer


def test_tokenizer_match_is_full_for_the_same_tokenizer(tmp_path: Path) -> None:
    splits = _splits(tmp_path)
    records = _records(tmp_path, splits["train"])
    student = tiny_vlm.tiny_wrapper(seed=2)
    assert rounds.tokenizer_match_share(student, list(records.values())) == 1.0
    assert rounds.check_cache_compatible(student, list(records.values())) == 1.0
    broken = _records(tmp_path, splits["train"], shift=1)  # the stop token differs
    assert rounds.tokenizer_match_share(student, list(broken.values())) == 0.0
    with pytest.raises(ValueError, match="not valid for this candidate"):
        rounds.check_cache_compatible(student, list(broken.values()))
    with pytest.raises(ValueError, match="no cached teacher records"):
        rounds.tokenizer_match_share(student, [])


# ---------------------------------------------------------------------------- R0


def test_round_0_records_numbers_and_cannot_pass_the_gpu_gate_on_a_cpu(tmp_path: Path) -> None:
    config = _config(tmp_path)
    student = tiny_vlm.tiny_wrapper(seed=2)
    rounds.run_round(config, "R0", "qwen3_vl_2b", model=student)
    metrics = _metrics(config, "R0", "qwen3_vl_2b")
    assert metrics["r0_scores_finite"] == 1.0 and metrics["r0_lora_step_ok"] == 1.0
    assert metrics["r0_on_cuda"] == 0.0 and metrics["gate_g1_hardware"] == 0.0
    assert metrics["r0_visual_tokens"] > 0 and metrics["r0_params"] > 0
    assert metrics["r0_load_ok"] == 1.0 and metrics["r0_lora_step_seconds"] > 0
    assert not any(math.isnan(v) for v in metrics.values())
    assert validate_results_tree(tmp_path / "results") == []
    rounds.run_round(config, "R0", "qwen3_vl_2b", model=tiny_vlm.tiny_wrapper(seed=3))
    assert _metrics(config, "R0", "qwen3_vl_2b")["r0_lora_step_ok"] == 1.0  # rows are replaced


def test_round_0_records_a_candidate_that_cannot_load_as_a_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import medcompress.models.loading as loading

    config = _config(tmp_path)
    config["inputs"] = {"image_longest_side": 56}

    def refuse(spec: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(loading, "load_model", refuse)
    rounds.run_round(config, "R0", "qwen3_vl_2b")
    metrics = _metrics(config, "R0", "qwen3_vl_2b")
    assert metrics == {"r0_load_ok": 0.0, "gate_g1_hardware": 0.0}
    assert validate_results_tree(tmp_path / "results") == []
    # a missing opt-in for model code is a setting to fix, not a failure of the candidate
    config["student_selection"]["candidates"] = ["models/ovis2_5_2b_student.yaml"]
    monkeypatch.undo()
    with pytest.raises(ConfigError, match="allow_remote_code"):
        rounds.run_round(config, "R0", "ovis2_5_2b")
    assert not (Path(config["paths"]["results_root"]) / "selection/R0/ovis2_5_2b").exists()


def test_visual_tokens_are_the_tokens_the_image_adds() -> None:
    student = tiny_vlm.tiny_wrapper()
    count = rounds.visual_token_count(student, tiny_vlm.tiny_image(0), tiny_vlm.PROBE)
    assert count is not None and 4 <= count <= 16 + 2  # up to 16 image pads plus two markers
    assert rounds.visual_token_count(object(), tiny_vlm.tiny_image(0), "q") is None


# ---------------------------------------------------------------------------- R1


def test_round_1_scores_before_and_after_and_applies_the_gate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits(tmp_path)
    rounds.run_round(
        config, "R1", "qwen3_vl_2b", model=tiny_vlm.tiny_wrapper(seed=2), samples=splits
    )
    metrics = _metrics(config, "R1", "qwen3_vl_2b")
    assert metrics["train_examples"] == 4 and metrics["train_steps"] == 2
    assert "zero_shot:macro_f1" in metrics and "macro_f1" in metrics
    assert "blank_accuracy_drop" in metrics and metrics["gate_g2_uses_image"] == 1.0
    assert validate_results_tree(tmp_path / "results") == []
    outcomes = CandidateOutcomes.from_npz(_outcomes(config, "R1", "qwen3_vl_2b"))
    assert len(set(outcomes.image_ids.tolist())) == 4  # the fixed validation subset
    assert len(outcomes.audit_image_ids) == 0  # the mask audit belongs to R2 and R3
    # a threshold nobody reaches fails the gate
    config["student_selection"]["gates"]["G2_uses_image"]["threshold"] = 2.0
    rounds.run_round(
        config, "R1", "qwen3_vl_2b", model=tiny_vlm.tiny_wrapper(seed=2), samples=splits
    )
    assert _metrics(config, "R1", "qwen3_vl_2b")["gate_g2_uses_image"] == 0.0


def test_round_1_trains_every_candidate_on_the_same_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    splits = _splits(tmp_path)
    seen: list[list[str]] = []
    real = offline_kd.train_lora

    def spy(model: Any, examples: list[Any], cfg: Any, seed: int = 0) -> Any:
        seen.append([e.sample_id for e in examples])
        return real(model, examples, cfg, seed=seed)

    monkeypatch.setattr(offline_kd, "train_lora", spy)
    for name in POOL[:2]:
        rounds.run_round(config, "R1", name, model=tiny_vlm.tiny_wrapper(seed=2), samples=splits)
    assert len(seen) == 2 and seen[0] == seen[1] and len(seen[0]) == 4


# ------------------------------------------------------------------------ R2, R3


def test_round_2_distils_from_the_cache_and_saves_audit_answers(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits(tmp_path)
    records = _records(tmp_path, splits["train"])
    rounds.run_round(
        config, "R2", "qwen3_vl_2b", model=tiny_vlm.tiny_wrapper(seed=2),
        samples=splits, records=records,
    )  # fmt: skip
    metrics = _metrics(config, "R2", "qwen3_vl_2b")
    assert metrics["tokenizer_match"] == 1.0 and metrics["gate_tokenizer_match"] == 1.0
    assert metrics["train_examples"] == 8 and metrics["train_steps"] == 4
    assert "lesion_mask_effect" in metrics and "ece" in metrics
    assert "gate_g2_uses_image" not in metrics
    assert validate_results_tree(tmp_path / "results") == []
    outcomes = CandidateOutcomes.from_npz(_outcomes(config, "R2", "qwen3_vl_2b"))
    assert len(set(outcomes.image_ids.tolist())) == 6  # the whole validation split
    assert len(outcomes.audit_image_ids) > 0 and outcomes.in_small.any()
    adapter = Path(config["paths"]["checkpoint_root"]) / "selection" / "R2" / "qwen3_vl_2b"
    assert (adapter / "seed0" / "adapter").exists()


def test_round_2_stops_when_the_tokenizer_cannot_reproduce_the_cache(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits(tmp_path)
    records = _records(tmp_path, splits["train"], shift=1)
    rounds.run_round(
        config, "R2", "qwen3_vl_2b", model=tiny_vlm.tiny_wrapper(seed=2),
        samples=splits, records=records,
    )  # fmt: skip
    metrics = _metrics(config, "R2", "qwen3_vl_2b")
    assert metrics["gate_tokenizer_match"] == 0.0 and metrics["tokenizer_match"] == 0.0
    assert "train_examples" not in metrics
    assert not _outcomes(config, "R2", "qwen3_vl_2b").exists()


def test_round_3_writes_one_run_per_seed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits(tmp_path)
    records = _records(tmp_path, splits["train"])
    for seed in (0, 1):
        rounds.run_round(
            config, "R3", "qwen3_vl_2b", seed=seed, model=tiny_vlm.tiny_wrapper(seed=2),
            samples=splits, records=records,
        )  # fmt: skip
    path = Path(config["paths"]["results_root"]) / "selection" / "R3" / "qwen3_vl_2b"
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        seeds = {int(r["seed"]) for r in csv.DictReader(handle)}
    assert seeds == {0, 1}
    assert _outcomes(config, "R3", "qwen3_vl_2b", 0).exists()
    assert _outcomes(config, "R3", "qwen3_vl_2b", 1).exists()
    a = CandidateOutcomes.from_npz(_outcomes(config, "R3", "qwen3_vl_2b", 0))
    b = CandidateOutcomes.from_npz(_outcomes(config, "R3", "qwen3_vl_2b", 1))
    assert np.array_equal(a.image_ids, b.image_ids) and np.array_equal(a.truth, b.truth)


def test_outcomes_keep_the_probe_order_across_candidates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits(tmp_path)
    for i, name in enumerate(POOL[:2]):
        rounds.run_round(
            config, "R1", name, model=tiny_vlm.tiny_wrapper(seed=2 + i), samples=splits
        )
    a, b = (CandidateOutcomes.from_npz(_outcomes(config, "R1", n)) for n in POOL[:2])
    assert np.array_equal(a.image_ids, b.image_ids)
    assert np.array_equal(a.finding_idx, b.finding_idx) and np.array_equal(a.truth, b.truth)


# ----------------------------------------------------------------------- the driver


def _run_pool(config: dict[str, Any], round_name: str, names: list[str], splits: Any,
              records: Any = None) -> None:  # fmt: skip
    for i, name in enumerate(names):
        rounds.run_round(
            config, round_name, name, model=tiny_vlm.tiny_wrapper(seed=2 + i),
            samples=splits, records=records,
        )  # fmt: skip


def test_the_chain_from_round_1_to_the_report(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits(tmp_path)
    _run_pool(config, "R1", POOL, splits)
    result = driver.rank_selection(config, "R1")
    assert len(result.candidates) == 3 and len(result.advance) == 2
    ranking = Path(config["paths"]["results_root"]) / "selection" / "R1" / "ranking.csv"
    assert validate_results_tree(tmp_path / "results") == []
    advanced = driver.expected_candidates(config, "R2")
    assert sorted(advanced) == sorted(result.advance)
    records = _records(tmp_path, splits["train"])
    _run_pool(config, "R2", advanced, splits, records)
    second = driver.rank_selection(config, "R2")
    assert sorted(second.candidates) == sorted(advanced)
    assert ranking.exists()
    report = driver.render_report(config)
    for name in POOL:
        assert name in report
    assert "Round 2" in report and "S [95% CI]" in report
    assert "Not run yet." in report  # R0 and R3 have no files
    assert driver.write_report(config, tmp_path / "SELECTION.md").read_text() == report


def test_ranking_waits_for_every_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits(tmp_path)
    _run_pool(config, "R1", POOL[:2], splits)
    with pytest.raises(ValueError, match="no 'gate_g2_uses_image' row"):
        driver.rank_selection(config, "R1")  # the third candidate has not been run
    base = Path(config["paths"]["results_root"]) / "selection" / "R1" / POOL[2]
    row = {"experiment": f"R1:{POOL[2]}", "budget": "", "seed": 0, "value": 1.0}
    upsert_results([{**row, "metric": "gate_g2_uses_image"}], base / "metrics.csv")
    with pytest.raises(ValueError, match="per-probe answers are missing"):
        driver.rank_selection(config, "R1")  # its metrics came in but its answers did not
    with pytest.raises(ValueError, match="rank R1 first"):
        driver.expected_candidates(config, "R2")


def test_gate_failures_leave_the_ranking(tmp_path: Path) -> None:
    config = _config(tmp_path, manual_gate_failures={"qwen3_vl_2b": ["G5_no_test_exposure"]})
    splits = _splits(tmp_path)
    _run_pool(config, "R1", POOL, splits)
    result = driver.rank_selection(config, "R1")
    assert "qwen3_vl_2b" not in result.candidates
    assert result.failed_gates == {"qwen3_vl_2b": ("G5_no_test_exposure",)}
    assert "qwen3_vl_2b" not in result.advance
    report = driver.render_report(config)
    assert "Left the ranking by a gate: qwen3_vl_2b (G5_no_test_exposure)" in report


def test_a_stopped_tokenizer_run_is_a_gate_failure_not_a_missing_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["student_selection"]["rounds"]["R1"]["keep"] = 3
    splits = _splits(tmp_path)
    _run_pool(config, "R1", POOL, splits)
    driver.rank_selection(config, "R1")
    good = _records(tmp_path, splits["train"])
    bad = _records(tmp_path, splits["train"], shift=1)
    _run_pool(config, "R2", POOL[:2], splits, good)
    rounds.run_round(
        config, "R2", POOL[2], model=tiny_vlm.tiny_wrapper(seed=9), samples=splits, records=bad
    )
    result = driver.rank_selection(config, "R2")
    assert result.failed_gates == {POOL[2]: ("tokenizer_match",)}
    assert sorted(result.candidates) == sorted(POOL[:2])


def test_everyone_failing_a_gate_is_an_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["student_selection"]["gates"]["G2_uses_image"]["threshold"] = 2.0
    splits = _splits(tmp_path)
    _run_pool(config, "R1", POOL, splits)
    with pytest.raises(ValueError, match="every candidate failed a gate"):
        driver.rank_selection(config, "R1")


def test_open_ranking_settings_are_reported(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["evaluation"]["bootstrap_resamples"] = "TODO"
    with pytest.raises(ConfigError, match="bootstrap_resamples"):
        driver.ranking_settings(config, "R1")
    config = _config(tmp_path)
    config["student_selection"]["score"]["weights"]["image_reliance"] = "TODO"
    with pytest.raises(ConfigError, match="score.weights.image_reliance"):
        driver.ranking_settings(config, "R1")
    settings = driver.ranking_settings(_config(tmp_path), "R2")
    assert settings["reliance_metric"] == "lesion_mask_effect" and settings["keep"] == 3
    assert settings["weights"] == (0.4, 0.3, 0.3) and settings["n_findings"] == 2


def _r0_rows(name: str, *, g1: float, peak: float | None = None) -> list[dict[str, Any]]:
    rows = [("gate_g1_hardware", g1), ("r0_params", 2.0e9), ("r0_visual_tokens", 256.0),
            ("r0_scoring_seconds_per_probe", 0.5), ("r0_lora_step_ok", g1)]  # fmt: skip
    if peak is not None:
        rows.append((driver.PEAK_MEMORY_METRIC, peak))
    return [
        {"experiment": f"R0:{name}", "budget": "", "seed": 0, "metric": m, "value": v}
        for m, v in rows
    ]


def test_round_0_verdicts_the_pool_and_the_tie_break(tmp_path: Path) -> None:
    config = _config(tmp_path)
    base = Path(config["paths"]["results_root"]) / "selection" / "R0"
    peaks = {"qwen2_5_vl_3b": 9.0e9, "qwen3_vl_2b": 6.0e9, "internvl3_5_2b": None}
    for name in POOL:
        g1 = 0.0 if name == "internvl3_5_2b" else 1.0
        upsert_results(_r0_rows(name, g1=g1, peak=peaks[name]), base / name / "metrics.csv")
    verdicts = driver.screen_feasibility(config)
    assert verdicts == {"qwen2_5_vl_3b": (), "qwen3_vl_2b": (), "internvl3_5_2b": ("G1_hardware",)}
    assert driver.expected_candidates(config, "R1") == ["qwen2_5_vl_3b", "qwen3_vl_2b"]
    ties = driver.tie_break_values(config, POOL)
    assert ties["qwen3_vl_2b"] < ties["qwen2_5_vl_3b"] < ties["internvl3_5_2b"] == (math.inf,)
    assert validate_results_tree(tmp_path / "results") == []
    report = driver.render_report(config)
    assert "FAIL: G1_hardware" in report and "pass" in report and "| 2.00 |" in report
    with pytest.raises(ValueError, match="Round 0 has gates only"):
        driver.rank_selection(config, "R0")


def test_round_0_needs_every_candidate_to_have_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    base = Path(config["paths"]["results_root"]) / "selection" / "R0"
    upsert_results(_r0_rows(POOL[0], g1=1.0), base / POOL[0] / "metrics.csv")
    with pytest.raises(ValueError, match="run the round before ranking it"):
        driver.screen_feasibility(config)


def test_reading_missing_results_gives_nothing(tmp_path: Path) -> None:
    assert driver.read_rows(tmp_path / "nothing.csv") == []
    assert driver.candidate_metrics(_config(tmp_path), "R1", "qwen3_vl_2b") == {}
