"""Finding-balanced recovery (amendment of 2026-11-22), on the tiny model: the exact paired
margin gradient, the sampling-only control, one training run, the training pass's margin and the
rebuild of the pruned model from ``removal.json``."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from medcompress.compression import fbr

# ------------------------------------------------------------------- torch: the gradients

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

import tiny_vlm  # noqa: E402
from eval_fixtures import write_dataset  # noqa: E402
from kd_fixtures import teacher_records  # noqa: E402

TINY = ("polyp", "bleeding")
TEMPLATE = "Is there a {finding} in this image ? Answer yes or no ."


def _setup(root: Path, coefficient: float) -> tuple[Any, ...]:
    from medcompress.compression.distillation.offline_kd import TrainConfig, examples_from_samples
    from medcompress.compression.lora import attach_lora, trainable_parameters

    samples = write_dataset(root)
    records = teacher_records(samples[:4], root)
    examples = examples_from_samples(samples[:4], root, records=records)
    cfg = TrainConfig.from_config({
        "mode": "offline_kd", "steps": 1, "learning_rate": 1e-2, "batch_size": 1,
        "grad_accumulation": 6, "kd_weight": 0.5, "kd_temperature": 2.0, "data_subset": 4,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["decoder_mlp"]},
    })  # fmt: skip
    settings = dataclasses.replace(fbr.FBRSettings.load(), general_per_update=2,
                                   margin_coefficient=coefficient)  # fmt: skip
    student = tiny_vlm.tiny_wrapper(seed=0)
    attach_lora(student, cfg.lora)
    student.module.train()
    probes = fbr.training_probes(samples, TINY, TEMPLATE)
    schedule = fbr.probe_schedule(probes, TINY, steps=1, seed=0)
    yes, no = int(student.yes_ids[0]), int(student.no_ids[0])
    ctx = fbr.UpdateContext(
        student, student.module, trainable_parameters(student.module), cfg, settings, TINY,
        {True: fbr.target_ids(student, yes), False: fbr.target_ids(student, no)},
        list(student.yes_ids), list(student.no_ids), root, torch.device("cpu"),
        lambda t: t, contextlib.nullcontext,
    )  # fmt: skip
    return ctx, examples[:2], schedule.updates[0], (yes, no)


def _reference_grads(ctx: Any, general: list, probes: list, e2: dict) -> list[Any]:
    """The update's loss L in one graph (fp32), differentiated directly."""
    from medcompress.compression.distillation.losses import answer_ce_loss
    from medcompress.compression.distillation.offline_kd import (
        _micro_batches,
        _step_loss,
        forward_answer_logits,
    )

    accumulation = ctx.cfg.grad_accumulation
    total = 0.0
    for example in general:
        gt, teacher = _micro_batches(ctx.student, [example], ctx.cfg)
        total = total + _step_loss(ctx.module, gt, teacher, ctx.cfg)[0] / accumulation
    share = len(probes) / accumulation
    for f_index in range(len(ctx.findings)):
        pair = probes[2 * f_index : 2 * f_index + 2]
        margins = []
        for probe in pair:
            batch = fbr.probe_batch(ctx.student, probe, ctx.data_root, ctx.targets[probe.present])
            logits = forward_answer_logits(ctx.module, batch)
            total = total + answer_ce_loss(logits, batch.targets) / accumulation
            margins.append(fbr.answer_margin(logits[:, 0, :], ctx.yes_ids, ctx.no_ids)[0])
        r = (margins[0] - margins[1]) - (e2[pair[0].key] - e2[pair[1].key])
        delta = ctx.settings.huber_delta
        h = torch.where(r.abs() <= delta, 0.5 * r * r, delta * (r.abs() - 0.5 * delta))
        total = total + ctx.settings.margin_coefficient * share * h / len(ctx.findings)
    return list(torch.autograd.grad(total, ctx.params, allow_unused=True))


@pytest.mark.parametrize("regime", ["quadratic", "linear"])
def test_the_paired_margin_gradient_equals_the_joint_graph(tmp_path: Path, regime: str) -> None:
    ctx, general, probes, _ids = _setup(tmp_path, 0.25)
    student_margins, _first = fbr.score_margins(ctx.student, probes, tmp_path)
    ctx.module.train()
    offset = 0.3 if regime == "quadratic" else 6.0  # |r| below or above delta = 1
    e2 = {p.key: student_margins[p.key] + (offset if p.present else 0.0) for p in probes}
    reference = _reference_grads(ctx, general, probes, e2)
    for p in ctx.params:
        p.grad = None
    parts, broken = fbr.accumulate_update(ctx, general, probes, e2)
    assert not broken and parts["huber"] > 0
    expected = 0.5 * offset**2 if regime == "quadratic" else offset - 0.5
    assert parts["huber"] == pytest.approx(expected, rel=1e-3)
    for p, ref in zip(ctx.params, reference, strict=True):
        if ref is None:
            continue
        assert torch.allclose(p.grad, ref, rtol=1e-4, atol=1e-7)


def test_the_margin_term_is_invariant_to_a_common_offset_and_zero_is_plain_probe_ce(
    tmp_path: Path,
) -> None:
    ctx, general, probes, _ids = _setup(tmp_path, 0.25)
    e2 = {p.key: (1.5 if p.present else -0.5) for p in probes}

    def grads(context: Any, margins: dict) -> list[Any]:
        for p in context.params:
            p.grad = None
        context.module.train()
        fbr.accumulate_update(context, general, probes, margins)
        return [None if p.grad is None else p.grad.clone() for p in context.params]

    base = grads(ctx, e2)
    shifted = grads(ctx, {k: v + 7.0 for k, v in e2.items()})  # both members move together
    assert all(a is None or torch.allclose(a, b) for a, b in zip(base, shifted, strict=True))
    # coefficient 0 (sampling-only): the gradient is L0 plus plain probe CE, whatever E2 says
    control = dataclasses.replace(
        ctx, settings=dataclasses.replace(ctx.settings, margin_coefficient=0.0)
    )
    zero = grads(control, e2)
    other = grads(control, {k: -v * 3 for k, v in e2.items()})
    assert all(a is None or torch.allclose(a, b) for a, b in zip(zero, other, strict=True))
    plain = dataclasses.replace(control.settings, margin_coefficient=0.25)
    reference = _reference_grads(dataclasses.replace(control, settings=dataclasses.replace(
        plain, margin_coefficient=0.0)), general, probes, e2)  # fmt: skip
    assert all(r is None or torch.allclose(z, r, rtol=1e-4, atol=1e-7)
               for z, r in zip(zero, reference, strict=True))  # fmt: skip
    assert any(a is not None and not torch.allclose(a, z) for a, z in zip(base, zero, strict=True))


def test_one_fbr_update_trains_and_logs(tmp_path: Path) -> None:
    ctx, general, probes, ids = _setup(tmp_path, 0.25)
    samples = write_dataset(tmp_path)
    schedule = fbr.probe_schedule(fbr.training_probes(samples, TINY, TEMPLATE),
                                  TINY, steps=2, seed=0)  # fmt: skip
    from medcompress.compression.distillation.offline_kd import examples_from_samples

    examples = examples_from_samples(samples[:4], tmp_path,
                                     records=teacher_records(samples[:4], tmp_path))  # fmt: skip
    cfg = dataclasses.replace(ctx.cfg, steps=2)
    e2 = {k: 0.0 for k in schedule.unique_keys()}
    student = tiny_vlm.tiny_wrapper(seed=0)
    result = fbr.train_fbr(student, examples, schedule, e2, cfg, ctx.settings, ids,
                           data_root=tmp_path, seed=0)  # fmt: skip
    assert result.steps == 2 and len(result.losses) == 2 and not result.nonfinite_steps
    assert all(v > 0 for v in result.probe_ce) and all(v > 0 for v in result.margin_huber)
    lora_b = [p for n, p in student.module.named_parameters() if "lora_B" in n]
    assert lora_b and any(float(p.detach().abs().sum()) > 0 for p in lora_b)  # zero at the start
    with pytest.raises(ValueError, match="margin cache lacks"):
        fbr.train_fbr(tiny_vlm.tiny_wrapper(seed=0), examples, schedule, {}, cfg, ctx.settings,
                      ids, data_root=tmp_path, seed=0)  # fmt: skip


def test_the_training_pass_margin_is_the_evaluations(tmp_path: Path) -> None:
    """The teacher-forced first answer slot gives the scoring path's margin on the same probe."""
    samples = write_dataset(tmp_path)
    wrapper = tiny_vlm.tiny_wrapper(seed=0)
    probes = fbr.training_probes(samples, TINY, TEMPLATE)[:4]
    margins, first = fbr.score_margins(wrapper, probes, tmp_path)
    scores = wrapper.score_yes_no([tmp_path / p.image_path for p in probes],
                                  [p.question for p in probes])  # fmt: skip
    for probe, score in zip(probes, scores, strict=True):
        assert margins[probe.key] == pytest.approx(score.logp_yes - score.logp_no, abs=1e-4)
    assert set(first) == {p.key for p in probes}


def test_the_rebuild_from_removal_json_reproduces_the_pruned_model(tmp_path: Path) -> None:
    from test_recovery import _config, _factory  # the tiny pruning config of the recovery tests

    from medcompress.compression.pruning.apply import remove_units
    from medcompress.compression.pruning.units import UnitType, enumerate_units
    from medcompress.compression.recovery import removal_by_layer
    from medcompress.config import RunContext
    from medcompress.evaluation.efficiency import count_parameters

    write_dataset(tmp_path)
    config = _config(tmp_path)
    run = RunContext("E5", seed=0, budget="B2", variant="magnitude")
    model = _factory()
    units = enumerate_units(model, [UnitType(t) for t in ("decoder_layer", "attention_head",
                                                           "mlp_channel_group")],
                            mlp_channel_group_size=16)  # fmt: skip
    chosen = [u for u in units if u.unit_type.value == "mlp_channel_group"][:2] + [
        u for u in units if u.unit_type.value == "attention_head" and u.layer_index == 1
    ][:1]
    sizes = {i: sum(p.numel() for p in layer.parameters())
             for i, layer in enumerate(model.module.model.language_model.layers)}  # fmt: skip
    remove_units(model, chosen)
    run_dir = run.run_dir(config["paths"]["results_root"])
    run_dir.mkdir(parents=True)
    (run_dir / "removal.json").write_text(json.dumps({"units": [u.unit_id for u in chosen]}))
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "budget", "seed", "metric", "value", "ci_low", "ci_high"])
        writer.writerow(["E5:magnitude", "B2", 0, "parameters_after", count_parameters(model),
                         "", ""])  # fmt: skip
    with (run_dir / "removal_by_layer.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer_index", "layer_params", "removed_params", "share_of_removed",
                         "share_of_layer"])  # fmt: skip
        writer.writerows(removal_by_layer(sizes, chosen))
    rebuilt, info = fbr.rebuild_pruned(config, run, student_factory=_factory)
    assert info["parameters_after"] == count_parameters(model) == info["committed_parameters_after"]
    assert info["units"] == 3 and info["layers_checked"] == len(sizes)
    shapes = {k: tuple(v.shape) for k, v in model.module.state_dict().items()}
    assert shapes == {k: tuple(v.shape) for k, v in rebuilt.module.state_dict().items()}
    (run_dir / "removal.json").write_text(json.dumps({"units": [u.unit_id for u in chosen[:2]]}))
    with pytest.raises(ValueError, match="committed run has"):
        fbr.rebuild_pruned(config, run, student_factory=_factory)


def test_the_probe_answer_pass_matches_the_evaluations_probes(tmp_path: Path) -> None:
    from eval_fixtures import eval_config

    from medcompress.evaluation import fbr_analysis as fa
    from medcompress.evaluation.runner import resolve_settings, score_probes

    samples = write_dataset(tmp_path)
    settings = resolve_settings(eval_config(tmp_path))
    wrapper = tiny_vlm.tiny_wrapper(seed=0)
    arrays = fa.probe_answers(wrapper, samples, settings, tmp_path, max_new_tokens=4)
    scored = score_probes(wrapper, samples, settings, tmp_path)
    assert list(arrays["probe_sample_ids"]) == list(scored.sample_ids)
    assert list(arrays["probe_findings"]) == list(scored.findings)
    assert arrays["probe_margin"] == pytest.approx(scored.margin, abs=1e-4)
    assert (arrays["probe_yes_rank"] >= 1).all() and (arrays["probe_no_rank"] >= 1).all()
    assert set(arrays["probe_decoded"].tolist()) <= {-1, 0, 1}
    assert len(arrays["probe_decoded_text"]) == len(scored.sample_ids)
    assert len(arrays["probe_first_token"]) == len(scored.sample_ids)  # note of 2026-11-25
    assert list(arrays["yes_ids"]) == list(wrapper.yes_ids)
    test_split = fa.probe_answers(wrapper, samples, dataclasses.replace(settings, split="test"),
                                  tmp_path, max_new_tokens=4, any_split=True)  # fmt: skip
    assert len(test_split["probe_first_token"]) == len(test_split["probe_sample_ids"])
    with pytest.raises(ValueError, match="validation only"):
        fa.probe_answers(wrapper, samples, dataclasses.replace(settings, split="test"),
                         tmp_path, max_new_tokens=4)  # fmt: skip


def _with_answers(config: dict) -> None:
    """The tiny domain file with the template's canonical answers (as the real one has)."""
    import yaml

    path = Path(config["components"]["domain"])
    domain = yaml.safe_load(path.read_text())
    domain["domain"]["probe"]["answers"] = {"positive": "yes", "negative": "no"}
    path.write_text(yaml.safe_dump(domain))


def test_the_arm_stage_runs_the_trial_and_the_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_fbr_arm`` end to end on the tiny model, with the data, the teacher cache and the
    rebuild replaced by the tiny fixtures: the trial writes its timing; the full run trains,
    merges, scores validation, saves the arrays and the rows."""
    from test_recovery import _config

    from medcompress.compression import fbr_stage
    from medcompress.compression.pruning import stage as pruning_stage
    from medcompress.config import RunContext
    from medcompress.evaluation import runner
    from medcompress.models import teacher_cache

    samples = write_dataset(tmp_path)
    config = _config(tmp_path)
    config["recovery"] = {**config["recovery"], "batch_size": 1, "grad_accumulation": 14,
                          "steps": 2, "data_subset": 6}  # 10 ordinary + 2 x 2 probes  # fmt: skip
    config["paths"]["output_root"] = str(tmp_path / "outputs")
    records = teacher_records(samples, tmp_path)
    monkeypatch.setattr(pruning_stage, "load_train_samples", lambda cfg: samples)
    monkeypatch.setattr(teacher_cache, "load_pinned_teacher_cache",
                        lambda cfg, folder: iter(records.values()))  # fmt: skip
    monkeypatch.setattr(fbr, "rebuild_pruned",
                        lambda cfg, arm: (tiny_vlm.tiny_wrapper(seed=0), {"units": 0}))  # fmt: skip
    monkeypatch.setattr(runner, "split_samples", lambda cfg, settings: (samples, {}))
    schedule = fbr_stage._schedule(config, samples, steps=2)[0]
    keys = set(schedule.unique_keys())
    probes = [p for u in schedule.updates for p in u if p.key in keys]
    margins, first = fbr.score_margins(tiny_vlm.tiny_wrapper(seed=0), probes, tmp_path)
    _with_answers(config)
    words = tiny_vlm.tiny_processor().tokenizer.convert_tokens_to_ids
    cache = {
        "schedule_fingerprint": schedule.fingerprint(),
        "steps": 2,
        "margins": margins,
        "yes_target_id": int(words("yes")),
        "no_target_id": int(words("no")),
    }
    margins_file = tmp_path / "fbr_e2_margins.json"
    margins_file.write_text(json.dumps(cache))
    run = RunContext("E5", seed=0, budget="B2", variant="magnitude")
    out = tmp_path / "fbr_out"
    trial = fbr_stage.run_fbr_arm(config, run, margins_file=margins_file, out=out, trial_steps=1)
    record = json.loads(trial.read_text())
    assert record["trial"]["updates"] == 1 and record["trial"]["seconds_per_update"] > 0
    folder = fbr_stage.run_fbr_arm(config, run, margins_file=margins_file, out=out, control=True)
    assert folder.name == "E5_magnitude_sampling_only"
    training = json.loads((folder / "training.json").read_text())
    assert training["margin_coefficient"] == 0.0 and training["training"]["steps"] == 2
    with (folder / "fbr_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {r["experiment"] for r in rows} == {"FBR:E5_magnitude_sampling_only"}
    arrays = tmp_path / "outputs" / "fbr" / "E5_magnitude_sampling_only"
    assert {p.name for p in arrays.iterdir()} == {"outcomes.npz", "probe_answers_val.npz",
                                                  "fbr_run.json"}  # fmt: skip
    cache["schedule_fingerprint"] = "other"
    margins_file.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="not the one E2's margins"):
        fbr_stage.run_fbr_arm(config, run, margins_file=margins_file, out=out, trial_steps=1)


def test_the_e2_margin_cache_matches_the_schedule_every_arm_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cell (ii) runs with a pruning arm's config (E2's has no recovery section) and writes the
    fingerprint the arm run checks; E5's and E7's configs give the same schedule."""
    from test_recovery import _config

    from medcompress.compression import fbr_stage
    from medcompress.compression.pruning import stage as pruning_stage
    from medcompress.config import load_config
    from medcompress.evaluation import runner

    assert load_config("experiments/E2.yaml").get("recovery") is None
    e5, e7 = load_config("experiments/E5.yaml"), load_config("experiments/E7.yaml")
    assert (
        e5["recovery"] == e7["recovery"]
        and e5["components"]["domain"] == (e7["components"]["domain"])
    )
    samples = write_dataset(tmp_path)
    config = _config(tmp_path)
    config["recovery"] = {**config["recovery"], "batch_size": 1, "grad_accumulation": 14,
                          "steps": 3}  # fmt: skip
    monkeypatch.setattr(pruning_stage, "load_train_samples", lambda cfg: samples)
    monkeypatch.setattr(runner, "load_trained_student",
                        lambda cfg, exp, seed, side: tiny_vlm.tiny_wrapper(seed=0))  # fmt: skip
    config["inputs"] = {"image_longest_side": 56}
    _with_answers(config)
    target = fbr_stage.run_e2_margin_cache(config, tmp_path / "margins", examples=2)
    cache = json.loads(target.read_text())
    schedule = fbr_stage._schedule(config, samples, steps=int(cache["steps"]))[0]
    assert cache["schedule_fingerprint"] == schedule.fingerprint() and cache["steps"] == 3
    assert set(cache["margins"]) == set(schedule.unique_keys())
    diagnosis = cache["diagnosis"]
    yes_word = tiny_vlm.tiny_processor().tokenizer.convert_tokens_to_ids("yes")
    assert cache["yes_target_id"] == yes_word == diagnosis["targets"]["positive"]["token"]
    positives = {p.key for u in schedule.updates for p in u if p.present}
    assert sum(diagnosis["first_tokens"]["by_label"]["positive"].values()) == len(positives)
    parsed = diagnosis["parsed_by_label"]
    assert sum(sum(c.values()) for c in parsed.values()) == len(cache["margins"])
    saved = json.loads((tmp_path / "margins" / fbr_stage.DIAGNOSIS_FILE).read_text())
    assert "margins" not in saved and saved["probes"] == len(cache["margins"])


def test_a_non_finite_e2_margin_skips_its_update_instead_of_stopping(tmp_path: Path) -> None:
    """Note of 2026-11-25: a non-finite margin of E2 on a training image (an fp16 overflow)
    makes that update non-finite; it is skipped and logged, as the registered recovery does."""
    from medcompress.compression.distillation.offline_kd import examples_from_samples

    ctx, _general, _probes, ids = _setup(tmp_path, 0.25)
    samples = write_dataset(tmp_path)
    schedule = fbr.probe_schedule(fbr.training_probes(samples, TINY, TEMPLATE), TINY, steps=2,
                                  seed=0)  # fmt: skip
    examples = examples_from_samples(samples[:4], tmp_path,
                                     records=teacher_records(samples[:4], tmp_path))  # fmt: skip
    e2 = {k: 0.0 for k in schedule.unique_keys()}
    e2[schedule.updates[1][0].key] = float("nan")  # the second update's first probe
    result = fbr.train_fbr(tiny_vlm.tiny_wrapper(seed=0), examples, schedule, e2,
                           dataclasses.replace(ctx.cfg, steps=2), ctx.settings, ids,
                           data_root=tmp_path, seed=0)  # fmt: skip
    assert result.steps == 2 and result.nonfinite_steps == [2] and result.skipped_steps == [2]


def test_the_rescaled_paired_term_gives_the_same_gradient_as_the_first_formula(
    tmp_path: Path,
) -> None:
    """Note of 2026-11-26: the margin gradient taken of ``scale * w * m`` and multiplied by the
    slope equals the first implementation's (``scale * m``, then ``w * slope``) in fp32."""
    ctx, general, probes, _ids = _setup(tmp_path, 0.25)
    e2 = {p.key: (1.5 if p.present else -0.5) for p in probes}

    def grads(context: Any) -> list[Any]:
        for p in context.params:
            p.grad = None
        context.module.train()
        _parts, causes = fbr.accumulate_update(context, general, probes, e2)
        assert causes == []
        return [None if p.grad is None else p.grad.clone() for p in context.params]

    new = grads(ctx)
    legacy = grads(dataclasses.replace(ctx, legacy_pair_scaling=True))
    assert all(a is None or torch.allclose(a, b, rtol=1e-5, atol=1e-8)
               for a, b in zip(new, legacy, strict=True))  # fmt: skip


def test_the_margin_cache_replaces_probes_with_a_non_finite_e2_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_recovery import _config

    from medcompress.compression import fbr_stage
    from medcompress.compression.pruning import stage as pruning_stage
    from medcompress.evaluation import runner

    samples = write_dataset(tmp_path)
    config = _config(tmp_path)
    config["recovery"] = {**config["recovery"], "batch_size": 1, "grad_accumulation": 14,
                          "steps": 3}  # fmt: skip
    config["inputs"] = {"image_longest_side": 56}
    _with_answers(config)
    monkeypatch.setattr(pruning_stage, "load_train_samples", lambda cfg: samples)
    monkeypatch.setattr(runner, "load_trained_student",
                        lambda cfg, exp, seed, side: tiny_vlm.tiny_wrapper(seed=0))  # fmt: skip
    first_schedule = fbr_stage._schedule(config, samples, steps=3)[0]
    bad_key = first_schedule.updates[1][0].key  # update 2's first probe overflows for E2
    real = fbr.score_margins

    def scored(wrapper: Any, probes: list, data_root: Path, ids: Any = None) -> tuple:
        margins, first = real(wrapper, probes, data_root, ids)
        return {k: (float("nan") if k == bad_key else v) for k, v in margins.items()}, first

    monkeypatch.setattr(fbr, "score_margins", scored)
    target = fbr_stage.run_e2_margin_cache(config, tmp_path / "m", examples=1,
                                           diagnosis_decode=False)  # fmt: skip
    cache = json.loads(target.read_text())
    assert cache["excluded"] == [bad_key] and bad_key not in cache["margins"]
    assert all(math.isfinite(v) for v in cache["margins"].values())
    rounds = cache["diagnosis"]["e2_nonfinite_rounds"]
    assert [r["nonfinite"] for r in rounds] == [1, 0] and rounds[0]["updates"] == [2]
    final = fbr_stage._schedule(config, samples, steps=3, exclude={bad_key})[0]
    assert cache["schedule_fingerprint"] == final.fingerprint() != first_schedule.fingerprint()
