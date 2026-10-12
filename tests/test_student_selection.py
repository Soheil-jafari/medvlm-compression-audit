"""The student-selection protocol (D05) is well formed and points at real candidate configs."""

from __future__ import annotations

import math

import pytest

from medcompress.config import CANDIDATE_STATUS, config_root, load_config

CONFIG_ROOT = config_root()
SELECTION = load_config(CONFIG_ROOT / "student_selection.yaml")["student_selection"]
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
    assert len(SELECTION["candidates"]) == rounds["R0"]["keep"]


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


@pytest.mark.parametrize("path", SELECTION["candidates"])
def test_candidates_share_the_teacher_tokenizer_family(path: str) -> None:
    teacher = load_config(CONFIG_ROOT / "models" / "qwen2_5_vl_7b_teacher.yaml")["model"]
    student = load_config(CONFIG_ROOT / path)["model"]
    assert SELECTION["tokenizer_rule"] == "C"
    assert (
        student["architecture"]["tokenizer_family"] == teacher["architecture"]["tokenizer_family"]
    )


@pytest.mark.parametrize("path", SELECTION["candidates"])
def test_candidate_config(path: str) -> None:
    model = load_config(CONFIG_ROOT / path)["model"]
    assert model["role"] == "student"
    assert model["status"] == CANDIDATE_STATUS
    assert ARCHITECTURE_KEYS <= set(model["architecture"])


def test_candidates_are_distinct_families() -> None:
    families = [load_config(CONFIG_ROOT / p)["model"]["family"] for p in SELECTION["candidates"]]
    assert len(set(families)) == len(families), families
