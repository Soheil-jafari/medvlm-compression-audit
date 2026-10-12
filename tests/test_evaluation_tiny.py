"""The evaluation stage on the tiny offline model (heavy: needs torch).

Checks the glue the fake-model tests cannot: real chat prompts with images, the yes/no
scoring path, greedy generation, and loading the adapter that a training run saved.
The tiny model has random weights, so only mechanics are asserted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

import tiny_vlm  # noqa: E402
from eval_fixtures import eval_config, write_dataset  # noqa: E402

from medcompress.compression import lora  # noqa: E402
from medcompress.compression.distillation import batching, offline_kd  # noqa: E402
from medcompress.compression.distillation.offline_kd import TrainConfig  # noqa: E402
from medcompress.compression.lora import LoRAConfig  # noqa: E402
from medcompress.config import RunContext  # noqa: E402
from medcompress.evaluation.aggregate import validate_results_tree  # noqa: E402
from medcompress.evaluation.runner import load_run_model, run_evaluation  # noqa: E402

TINY_TEMPLATE = "Is there a {finding} in this image ? Answer yes or no ."


def _config(root: Path, **evaluation: object) -> dict:
    return eval_config(
        root, domain={"probe": {"template": TINY_TEMPLATE, "display_names": {}}}, **evaluation
    )


def test_stage_runs_every_suite_on_the_tiny_model(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)
    run = RunContext("E2", seed=0)
    run_evaluation(
        _config(tmp_path, max_new_tokens=4), run, model=tiny_vlm.tiny_wrapper(), samples=samples
    )
    path = tmp_path / "results" / "E2" / "nobudget" / "seed0" / "metrics.csv"
    values = dict(zip(*pd.read_csv(path)[["metric", "value"]].T.values, strict=True))
    assert validate_results_tree(tmp_path / "results") == []
    assert values["n_probes_scored"] == 24 and values["n_answers"] == 24
    assert 0.0 <= values["overall_accuracy"] <= 1.0 and 0.0 <= values["ece"] <= 1.0
    assert values["answer_words_mean"] > 0
    # the scores must not depend on how the probes were batched
    again = tiny_vlm.tiny_wrapper()
    again.batch_size = 5
    run_evaluation(_config(tmp_path, max_new_tokens=4), run, model=again, samples=samples)
    after = dict(zip(*pd.read_csv(path)[["metric", "value"]].T.values, strict=True))
    for name in ("overall_accuracy", "macro_f1", "brier", "small_finding_recall"):
        assert after[name] == pytest.approx(values[name], abs=1e-6)


def test_e2_evaluation_loads_the_adapter_that_training_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import peft

    import medcompress.config as config_module
    import medcompress.models.loading as loading_module

    student = tiny_vlm.tiny_wrapper()
    examples = [
        batching.TrainExample(f"s{i}", tiny_vlm.tiny_image(i), tiny_vlm.PROBE, ["Yes", "No"][i % 2])
        for i in range(4)
    ]
    cfg = TrainConfig(
        "sft", LoRAConfig(4, 8, 0.0, ("decoder_attention", "decoder_mlp")),
        steps=10, learning_rate=1e-2, batch_size=2, grad_accumulation=1,
        kd_weight=0.0, kd_temperature=1.0, data_subset="all",
    )  # fmt: skip
    offline_kd.train_student_sft(student, examples, cfg, seed=0)
    config = _config(tmp_path)
    config["model"] = {"config": "fake/student.yaml"}
    adapter = tmp_path / "checkpoints" / "E2" / "seed0" / "adapter"
    lora.save_adapter(student, adapter)

    real_load_config = config_module.load_config
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda path, *a, **k: {}
        if str(path).startswith("fake/")
        else real_load_config(path, *a, **k),
    )
    monkeypatch.setattr(
        loading_module, "ModelSpec", SimpleNamespace(from_config=lambda cfg: "spec")
    )
    monkeypatch.setattr(
        loading_module, "load_model", lambda spec, **kwargs: tiny_vlm.tiny_wrapper()
    )
    loaded = load_run_model(config, RunContext("E2", seed=0), 56)
    assert not isinstance(loaded.module, peft.PeftModel)  # merged into the weights
    images, questions = [tiny_vlm.tiny_image(7), tiny_vlm.tiny_image(8)], [tiny_vlm.PROBE] * 2
    trained = student.score_yes_no(images, questions)
    reloaded = loaded.score_yes_no(images, questions)
    for a, b in zip(trained, reloaded, strict=True):
        assert (a.logp_yes, a.logp_no) == pytest.approx((b.logp_yes, b.logp_no), abs=1e-4)
    untouched = tiny_vlm.tiny_wrapper().score_yes_no(images, questions)
    assert any(abs(a.logp_yes - u.logp_yes) > 1e-3 for a, u in zip(trained, untouched, strict=True))
