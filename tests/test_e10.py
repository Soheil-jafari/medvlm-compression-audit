"""E10, the native 2B student (pre-freeze addition of 2026-11-08): registry, config, the card
plan and the tokenizer check that picks its recipe (CPU, fake tokenizers)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from medcompress.cards import build_plan
from medcompress.config import load_experiment, student_config_path
from medcompress.registry import REGISTRY

REPO = Path(__file__).resolve().parents[1]


def _check_module(monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location(
        "tokcheck", REPO / "scripts" / "check_tokenizer_match.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "tokcheck", module)
    spec.loader.exec_module(module)
    return module


@dataclass
class Record:
    answer_text: str
    answer_token_ids: np.ndarray
    topk_token_ids: np.ndarray


VOCAB = {1: "a", 2: "b", 3: "c", 9: "<|im_end|>"}


def _records() -> list[Record]:
    return [
        Record("a b", np.array([1, 2, 9]), np.array([[1, 2], [2, 3], [9, 1]])),
        Record("c", np.array([3, 9]), np.array([[3, 1], [9, 2]])),
    ]


def _encode(text: str) -> list[int]:
    back = {v: k for k, v in VOCAB.items()}
    return [back[w] for w in text.split()]


def test_the_registry_and_the_config_name_the_2b_student() -> None:
    assert "E10" in REGISTRY and REGISTRY["E10"].trains and REGISTRY["E10"].requires == ("E0",)
    config, experiment = load_experiment("E10")
    assert experiment.seeds == (0,)
    assert student_config_path(config) == "models/qwen3_vl_2b_student.yaml"
    assert config["training"]["mode"] == "offline_kd"  # set from the tokenizer check, 2026-11-12


def test_the_card_checks_the_tokenizer_before_training() -> None:
    plan = build_plan("RC-E10")
    labels = [s.label for s in plan.steps]
    assert labels[0].startswith("tokenizer check") and "train E10 seed0" in labels
    assert [h.name for h in plan.handoffs] == ["E10_seed0_adapter"]


def test_the_tokenizer_check_picks_the_e2_recipe_only_when_every_cached_id_agrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _check_module(monkeypatch)

    def tokens(vocab: dict[int, str]):
        return lambda ids: [vocab.get(i, "?") for i in ids]

    same = module.compare(tokens(VOCAB), tokens(VOCAB), _encode, _records(), min_share=0.99)
    assert same.passed and same.recipe == "offline_kd"
    assert (same.ids_checked, same.ids_differing, same.answers_matching) == (4, 0, 2)
    other = {**VOCAB, 9: "<|endoftext|>"}  # a special token with another meaning
    differ = module.compare(tokens(VOCAB), tokens(other), _encode, _records(), min_share=0.99)
    assert not differ.passed and differ.recipe == "sft" and differ.ids_differing == 1
    split = module.compare(tokens(VOCAB), tokens(VOCAB), lambda t: [1, 2, 3][: len(t)],
                           _records(), min_share=0.99)  # fmt: skip
    assert split.share == 0.0 and split.recipe == "sft"  # answers re-encoded differently
    rows = {r["metric"]: r["value"] for r in module.rows(same, "E10")}
    assert rows["tokenizer_check_passed"] == 1.0 and rows["tokenizer_answer_match_share"] == 1.0
