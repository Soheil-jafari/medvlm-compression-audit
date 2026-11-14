"""E10 at 4 bit (E10Q, 2026-11-14): its parent, its calibration check, its card and smoke job."""

from __future__ import annotations

from typing import Any

import pytest

from medcompress.cards import build_plan
from medcompress.compression.quantization import ptq
from medcompress.config import RunContext, load_experiment
from medcompress.data.calibration import MODELS
from medcompress.smoke import GATED_CARDS, smoke_jobs


def test_e10q_quantizes_the_merged_e10_with_the_w4_recipe() -> None:
    run = RunContext("E10Q", seed=0)
    assert ptq.parent_of(run) == {"experiment": "E10", "budget": None, "seed": 0}
    config, spec = load_experiment("E10Q")
    assert spec.parent == "E10" and spec.seeds == (0,)
    assert config["quantization"]["recipe"] == "w4"
    assert config["model"]["config"] == load_experiment("E10")[0]["model"]["config"]  # the 2B
    e8 = load_experiment("E8")[0]
    assert ptq.PTQRecipe.from_config(config) == ptq.PTQRecipe.from_config(e8)  # the same recipe


def test_each_quantized_arm_is_calibrated_on_the_check_of_its_own_model() -> None:
    assert "e10" in MODELS
    assert ptq.calibration_model(RunContext("E10Q", seed=0)) == "e10"
    assert ptq.calibration_model(RunContext("E9", seed=0)) == "teacher"
    assert ptq.calibration_model(RunContext("E3", seed=0, variant="w4")) == "student"
    assert ptq.calibration_model(RunContext("E8", seed=0, budget="B2")) == "student"


def test_the_parent_of_e10q_is_e10s_trained_2b(monkeypatch: pytest.MonkeyPatch) -> None:
    from medcompress.evaluation import runner

    calls: list[tuple[str, int, int]] = []

    def trained(config: Any, tag: str, seed: int, side: int, **kwargs: Any) -> str:
        calls.append((tag, seed, side))
        return "model"

    monkeypatch.setattr(runner, "load_trained_student", trained)
    assert ptq.load_parent({}, RunContext("E10Q", seed=0), 672) == "model"
    assert calls == [("E10", 0, 672)]


def test_the_card_checks_the_2bs_calibration_first_and_hands_off_the_codes() -> None:
    plan = build_plan("RC-E10Q")
    labels = [s.label for s in plan.steps]
    assert labels[0].startswith("fp16 calibration check of the 2B")
    assert plan.steps[0].argv[:5] == ("scripts/check_calibration.py", "--exp", "E10Q", "--seed",
                                      "0")  # fmt: skip
    assert [s.argv[0] for s in plan.steps[1:]] == [
        "scripts/03_quantize.py", "scripts/06_evaluate.py", "scripts/07_benchmark_efficiency.py",
    ]  # fmt: skip
    assert [(n.kind, n.run.label if n.run else None) for n in plan.needs] == [
        ("adapter", "E10 seed0")
    ]
    assert [h.name for h in plan.handoffs] == ["E10Q_seed0_quantized"] and plan.quantization
    assert GATED_CARDS["RC-E10Q"] == "RC-E10Q"
    (job,) = [j for j in smoke_jobs() if j.card == "RC-E10Q"]
    assert job.after == "E10" and job.stages[0].args[-2:] == ("--size", "2")
