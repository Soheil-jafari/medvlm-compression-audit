"""The student-selection protocol (D05) is well formed and points at real candidate configs."""

from __future__ import annotations

import math

import pytest

from medcompress.config import CANDIDATE_STATUS, config_root, load_config

CONFIG_ROOT = config_root()
SELECTION = load_config(CONFIG_ROOT / "student_selection.yaml")["student_selection"]
FULL_POOL = SELECTION["candidates_full_protocol"]  # the record of the 10-model protocol
ARCHITECTURE_KEYS = {
    "vision_encoder",
    "connector",
    "language_model",
    "tokenizer_family",
    "vocab_size",
}


def test_protocol_uses_validation_split_only() -> None:
    assert SELECTION["split"] == "val"
    assert SELECTION["tokenizer_rule"] in {"TODO", "A", "B", "C"}


def test_rounds_halve_10_5_3_1() -> None:
    rounds = SELECTION["rounds"]
    assert list(rounds) == ["R0", "R1", "R2", "R3"]
    assert [rounds[r]["keep"] for r in rounds] == [10, 5, 3, 1]
    assert len(FULL_POOL) == rounds["R0"]["keep"]


def test_rounds_cost_about_the_same() -> None:
    rounds = SELECTION["rounds"]
    pool = [10, 5, 3]
    cost = [
        n * rounds[r]["compute_units"] * rounds[r]["seeds"]
        for n, r in zip(pool, ["R1", "R2", "R3"], strict=True)
    ]
    assert max(cost) / min(cost) < 1.25, cost


def test_score_weights_sum_to_one() -> None:
    weights = SELECTION["score"]["weights"]
    assert math.isclose(sum(weights.values()), 1.0)
    assert set(SELECTION["score"]["image_reliance_metric"]) == {"R1", "R2", "R3"}


def test_reserves_present_and_distinct() -> None:
    reserves = SELECTION["reserves"]
    assert len(reserves) >= 3 and len(set(reserves)) == len(reserves)


@pytest.mark.parametrize("path", FULL_POOL)
def test_candidates_share_the_teacher_tokenizer_family(path: str) -> None:
    teacher = load_config(CONFIG_ROOT / "models" / "qwen2_5_vl_7b_teacher.yaml")["model"]
    student = load_config(CONFIG_ROOT / path)["model"]
    assert SELECTION["tokenizer_rule"] == "C"
    assert (
        student["architecture"]["tokenizer_family"] == teacher["architecture"]["tokenizer_family"]
    )


@pytest.mark.parametrize("path", FULL_POOL)
def test_candidate_config(path: str) -> None:
    model = load_config(CONFIG_ROOT / path)["model"]
    assert model["role"] == "student"
    assert model["status"] == CANDIDATE_STATUS
    assert ARCHITECTURE_KEYS <= set(model["architecture"])


def test_candidates_are_distinct_families() -> None:
    families = [load_config(CONFIG_ROOT / p)["model"]["family"] for p in FULL_POOL]
    assert len(set(families)) == len(families), families


def test_the_d05_amendment_ranks_only_the_student_and_its_fallback() -> None:
    from medcompress.selection.driver import pool

    assert SELECTION["candidates"] == [
        "models/qwen2_5_vl_3b_student.yaml",
        "models/qwen3_vl_2b_student.yaml",
    ]
    assert set(SELECTION["candidates"]) <= set(FULL_POOL)  # the record keeps all ten
    assert pool({"student_selection": SELECTION}) == ["qwen2_5_vl_3b", "qwen3_vl_2b"]


def test_round_0_ranks_the_committed_results(tmp_path) -> None:  # noqa: ANN001
    """The R0 ranking reads only the committed metrics (no torch, no GPU)."""
    import shutil

    from medcompress.selection.driver import read_rows, screen_feasibility

    committed = CONFIG_ROOT.parent / "results" / "selection" / "R0"
    for name in ("qwen2_5_vl_3b", "qwen3_vl_2b"):
        target = tmp_path / "selection" / "R0" / name
        target.mkdir(parents=True)
        shutil.copy(committed / name / "metrics.csv", target / "metrics.csv")
    config = {"student_selection": SELECTION, "paths": {"results_root": str(tmp_path)}}
    assert screen_feasibility(config) == {"qwen2_5_vl_3b": (), "qwen3_vl_2b": ()}
    rows = read_rows(tmp_path / "selection" / "R0" / "ranking.csv")
    assert {(r["experiment"], r["metric"], r["value"]) for r in rows} == {
        ("R0:qwen2_5_vl_3b", "advance", 1.0),
        ("R0:qwen3_vl_2b", "advance", 1.0),
    }
