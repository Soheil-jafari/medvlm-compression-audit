"""One candidate through one selection round (decision D05).

* R0: feasibility, no dataset. Load in fp16, score synthetic probes, time them, count the
  visual tokens of an image and take one LoRA step. Gate G1.
* R1: simplified screen on a fixed subset of validation images. Zero-shot scores, then the
  same short LoRA fine-tune on ground-truth answers for every candidate, then scores
  again. Gate G2 (does the model use the image).
* R2: the full student recipe (distillation from the cached teacher outputs) with about
  twice the R1 examples, scored on the whole validation split with the lesion-mask audit.
* R3: the E2 recipe itself, one run per seed.

Every round writes ``results/selection/<round>/<candidate>/metrics.csv`` (experiment
``<round>:<candidate>``) and, for R1 to R3, the per-probe answers the ranking needs to
``<output_root>/selection/<round>/<candidate>/seed<k>/outcomes.npz`` (never committed).
Selection uses the validation split only.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from medcompress.selection.ranking import CandidateOutcomes

if TYPE_CHECKING:
    from medcompress.compression.distillation.offline_kd import TrainConfig
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

ROUNDS: tuple[str, ...] = ("R0", "R1", "R2", "R3")
ROUND_SUITES: Mapping[str, tuple[str, ...]] = {
    "R1": ("clinical", "small_findings", "blank_image"),
    "R2": ("clinical", "small_findings", "rare_findings", "calibration", "mask_audit"),
    "R3": ("clinical", "small_findings", "rare_findings", "calibration", "mask_audit"),
}
MIN_TOKENIZER_MATCH = 0.99
"""Share of cached teacher answers that the student's tokenizer must reproduce (R2, R3)."""
R0_PROBES = 20


def candidate_name(config_path: str | Path) -> str:
    """Short name of a candidate: ``models/qwen3_vl_2b_student.yaml`` gives ``qwen3_vl_2b``."""
    stem = Path(config_path).stem
    return stem.removesuffix("_student")


def candidate_config_path(selection: Mapping[str, Any], name: str) -> str:
    """The config path of the candidate called ``name`` (or a clear error listing the pool)."""
    by_name = {candidate_name(p): p for p in selection["candidates"]}
    if name not in by_name:
        raise ValueError(f"unknown candidate {name!r}; the pool is {sorted(by_name)}")
    return by_name[name]


def select_validation_subset(
    samples: Sequence[VQASample], n_images: int | None, seed: int
) -> list[VQASample]:
    """Rows of a seeded random subset of ``n_images`` distinct images (all rows if None).

    The subset depends only on the images and the seed, so every candidate is scored on
    the same ones without any list of image names being stored.
    """
    if n_images is None:
        return list(samples)
    images = sorted({s.image_path for s in samples})
    if n_images >= len(images):
        return list(samples)
    chosen = set(random.Random(f"val-subset:{seed}:{len(images)}").sample(images, n_images))
    return [s for s in samples if s.image_path in chosen]


# ------------------------------------------------------------------------ plan


@dataclass(frozen=True)
class RoundPlan:
    """Everything one candidate run needs, resolved from the configs."""

    round: str
    candidate: str
    config_path: str
    seed: int
    side: int | None
    eval_config: Mapping[str, Any] | None
    train: TrainConfig | None
    n_val_images: int | None
    val_seed: int
    data_seed: int
    g2_threshold: float | None
    r0: Mapping[str, Any] | None


def is_open(value: Any) -> bool:
    """A setting that is missing, null or still a ``TODO`` placeholder."""
    from medcompress.config import find_todos

    return value is None or bool(find_todos({"v": value}))


def round_eval_config(config: Mapping[str, Any], round_name: str) -> dict[str, Any]:
    """The config with the evaluation suites of ``round_name`` switched on (validation split)."""
    out = copy.deepcopy(dict(config))
    out["evaluation"] = {
        **out.get("evaluation", {}),
        "suites": list(ROUND_SUITES[round_name]),
        "split": "val",
    }
    return out


def plan_round(
    config: Mapping[str, Any],
    round_name: str,
    candidate: str,
    seed: int = 0,
    *,
    need_side: bool = True,
) -> RoundPlan:
    """Resolve the settings of one round for one candidate.

    Raises:
        ConfigError: a setting the round needs is missing or still ``TODO``.
        ValueError: unknown round or candidate, or a seed the round does not allow.
    """
    from dataclasses import replace

    from medcompress.compression.distillation.offline_kd import TrainConfig
    from medcompress.config import ConfigError

    if round_name not in ROUNDS:
        raise ValueError(f"round must be one of {ROUNDS}")
    selection = config["student_selection"]
    path = candidate_config_path(selection, candidate)
    spec = selection["rounds"][round_name]
    if not 0 <= seed < int(spec.get("seeds", 1)):
        raise ValueError(f"{round_name} runs with {spec.get('seeds', 1)} seed(s); got seed {seed}")
    open_keys: set[str] = set()
    side = config.get("inputs", {}).get("image_longest_side")
    if need_side and is_open(side):
        open_keys.add("inputs.image_longest_side")
    if round_name == "R0":
        step = spec.get("lora_step", {})
        for key in ("rank", "alpha", "dropout", "learning_rate"):
            if is_open(step.get(key)):
                open_keys.add(f"student_selection.rounds.R0.lora_step.{key}")
        if open_keys:
            raise ConfigError(f"selection settings still open: {', '.join(sorted(open_keys))}")
        return RoundPlan(
            "R0", candidate, path, seed, int(side) if not is_open(side) else None, None, None,
            None, 0, 0, None, step,
        )  # fmt: skip

    from medcompress.evaluation.runner import resolve_settings

    eval_config = round_eval_config(config, round_name)
    try:
        resolve_settings(eval_config)
    except ConfigError as exc:
        open_keys.add(str(exc))
    unit = selection.get("compute_unit_b")
    n_val = None
    if round_name == "R1":
        subset = spec.get("validation_subset", {})
        n_val = subset.get("n_images")
        if is_open(n_val):
            open_keys.add("student_selection.rounds.R1.validation_subset.n_images")
        gate = selection["gates"]["G2_uses_image"].get("threshold")
        if is_open(gate):
            open_keys.add("student_selection.gates.G2_uses_image.threshold")
    if round_name in ("R1", "R2") and is_open(unit):
        open_keys.add("student_selection.compute_unit_b")
    section = {
        **config.get("training", {}),
        "mode": "sft" if round_name == "R1" else "offline_kd",
    }
    if round_name == "R1":
        section["kd_weight"] = 0.0
    if round_name in ("R1", "R2"):  # the length of these rounds comes from compute_unit_b below
        section["steps"], section["data_subset"] = 1, "all"
    train = None
    try:
        train = TrainConfig.from_config(section)
    except ConfigError as exc:
        open_keys.add(str(exc))
    if open_keys:
        raise ConfigError(f"selection settings still open: {'; '.join(sorted(open_keys))}")
    assert train is not None
    per_step = train.batch_size * train.grad_accumulation
    if round_name in ("R1", "R2"):
        n_train = int(round(float(unit) * float(spec["compute_units"])))
        train = replace(train, steps=math.ceil(n_train / per_step), data_subset=n_train)
    return RoundPlan(
        round_name,
        candidate,
        path,
        seed,
        int(side) if not is_open(side) else None,
        eval_config,
        train,
        int(n_val) if n_val is not None else None,
        int(spec.get("validation_subset", {}).get("seed", 0)),
        int(selection.get("data_seed", 0)),
        float(selection["gates"]["G2_uses_image"]["threshold"]) if round_name == "R1" else None,
        None,
    )


# ------------------------------------------------------------------------ R0


def visual_token_count(model: VLMWrapper, image: Any, question: str) -> int | None:
    """Tokens the image adds to a prompt (prompt with the image minus prompt without it).

    None when the processor of a family cannot build the text-only prompt; the run report
    then records the number by hand.
    """
    try:
        from medcompress.compression.distillation.batching import prompt_inputs

        with_image = int(prompt_inputs(model, image, question)["input_ids"].shape[1])  # type: ignore[attr-defined]
        text = model.processor.apply_chat_template(  # type: ignore[attr-defined]
            [{"role": "user", "content": [{"type": "text", "text": question}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        without = len(model.tokenizer(text)["input_ids"])  # type: ignore[attr-defined]
        return with_image - without
    except Exception:  # noqa: BLE001 - a family quirk must not stop the feasibility run
        return None


def feasibility_rows(model: VLMWrapper, plan: RoundPlan, *, seed: int = 0) -> list[dict[str, Any]]:
    """Round 0 measurements for a loaded candidate; ``gate_g1_hardware`` is the verdict.

    The gate needs CUDA: a run on a CPU records the numbers but cannot pass G1.
    """
    from medcompress.compression.distillation.batching import TrainExample
    from medcompress.compression.distillation.offline_kd import TrainConfig, train_lora
    from medcompress.compression.lora import LoRAConfig
    from medcompress.models.smoke import (
        cuda_available,
        measure_model,
        peak_gpu_bytes,
        probe_questions,
        reset_peak,
        synthetic_images,
    )

    assert plan.r0 is not None
    questions = (probe_questions() * R0_PROBES)[:R0_PROBES]
    images = synthetic_images(R0_PROBES)
    rows, texts = measure_model(model, images, questions, "r0")
    values = {r["metric"]: float(r["value"]) for r in rows}
    tokens = visual_token_count(model, images[0], questions[0])
    step = plan.r0
    examples = [
        TrainExample(f"r0-{i}", images[i], questions[i], ["Yes", "No"][i % 2]) for i in range(2)
    ]
    cfg = TrainConfig(
        mode="sft",
        lora=LoRAConfig(
            int(step["rank"]), float(step["alpha"]), float(step["dropout"]),
            ("decoder_attention", "decoder_mlp"),
        ),
        steps=1, learning_rate=float(step["learning_rate"]), batch_size=1, grad_accumulation=1,
        kd_weight=0.0, kd_temperature=1.0, data_subset="all",
    )  # fmt: skip
    reset_peak()
    lora_ok, lora_loss, lora_seconds = 0.0, math.nan, math.nan
    try:
        result = train_lora(model, examples, cfg, seed=seed)
        lora_loss, lora_seconds = result.losses[0], result.seconds
        lora_ok = float(math.isfinite(lora_loss))
    except Exception as exc:  # noqa: BLE001 - out of memory and family quirks are the finding
        print(f"LoRA step failed: {type(exc).__name__}: {exc}")
    peak = peak_gpu_bytes()
    on_cuda = cuda_available()
    finite = values.get("r0_scores_finite", 0.0) == 1.0
    extra = {
        "r0_load_ok": 1.0,
        "r0_lora_step_ok": lora_ok,
        "r0_lora_step_loss": lora_loss,
        "r0_lora_step_seconds": lora_seconds,
        "r0_on_cuda": float(on_cuda),
        "gate_g1_hardware": float(finite and lora_ok == 1.0 and on_cuda),
    }
    if tokens is not None:
        extra["r0_visual_tokens"] = float(tokens)
    if peak is not None:
        extra["r0_peak_gpu_bytes_lora_step"] = float(peak)
    print("generation sample:", texts)
    return [dict(r) for r in rows] + [
        {"experiment": "", "budget": "", "seed": 0, "metric": k, "value": v}
        for k, v in extra.items()
        if not math.isnan(v)  # an undefined value (a failed LoRA step has no loss) is left out
    ]


# ------------------------------------------------------------------ R1 to R3


def tokenizer_match_share(
    student: VLMWrapper, records: Sequence[Any], *, sample: int = 200, seed: int = 0
) -> float:
    """Share of cached teacher answers that the student's tokenizer reproduces exactly.

    Re-encodes the cached answer text with the student tokenizer and compares it with the
    cached token ids (the stop token must match too). Logit distillation on the cached
    top-k needs the two vocabularies to agree on the tokens that occur.
    """
    if not records:
        raise ValueError("no cached teacher records")
    rng = random.Random(f"tok-check:{seed}")
    picks = rng.sample(list(records), min(sample, len(records)))
    stop = student.stop_token_id()  # type: ignore[attr-defined]
    matches = 0
    for record in picks:
        ids = [int(i) for i in record.answer_token_ids]
        encoded = list(
            student.tokenizer.encode(record.answer_text.strip(), add_special_tokens=False)
        )  # type: ignore[attr-defined]
        matches += int(ids[-1:] == [stop] and ids[:-1] == [int(i) for i in encoded])
    return matches / len(picks)


def check_cache_compatible(
    student: VLMWrapper, records: Sequence[Any], *, sample: int = 200, seed: int = 0
) -> float:
    """:func:`tokenizer_match_share`, or an error when it is below :data:`MIN_TOKENIZER_MATCH`.

    Raises:
        ValueError: too few cached answers are reproduced for logit distillation to be valid.
    """
    share = tokenizer_match_share(student, records, sample=sample, seed=seed)
    if share < MIN_TOKENIZER_MATCH:
        raise ValueError(
            f"the student tokenizer reproduces {share:.1%} of the checked cached teacher answers "
            f"(need {MIN_TOKENIZER_MATCH:.0%}); logit distillation is not valid for this candidate"
        )
    return share


def outcomes_from_result(result: Any, settings: Any) -> CandidateOutcomes:
    """Per-probe answers in the form the ranking needs, from an ``EvaluationResult``.

    There is one probe per image and finding, so ``image_ids`` (the id of the image's
    representative row) identifies the image for the paired bootstrap of the ranking.
    """
    from medcompress.evaluation import probe_suites

    out = result.probes
    if out is None:
        raise ValueError("the round scored no probes")
    in_small = np.zeros(len(out.sample_ids), dtype=bool)
    if settings.thresholds is not None and settings.mask_findings is not None:
        keep, strata = probe_suites.small_positive_strata(
            out, settings.thresholds, settings.mask_findings
        )
        in_small[np.flatnonzero(keep)[strata == 0]] = True
    audit = result.audit
    return CandidateOutcomes(
        image_ids=np.array(out.sample_ids),
        finding_idx=out.finding_index(settings.findings),
        truth=out.truth,
        yes=out.yes,
        yes_blank=out.yes_blank if out.yes_blank is not None else np.zeros_like(out.yes),
        in_small=in_small,
        audit_image_ids=np.array(audit.sample_ids if audit is not None else [], dtype=str),
        audit_lesion_flip=audit.lesion_flip if audit is not None else np.zeros(0, dtype=bool),
        audit_random_flip=audit.random_flip if audit is not None else np.zeros(0, dtype=bool),
    )


def _prefixed(metrics: Sequence[Any], prefix: str, keep: Sequence[str]) -> list[Any]:
    return [dataclasses.replace(m, name=f"{prefix}{m.name}") for m in metrics if m.name in keep]


def run_round(
    config: Mapping[str, Any],
    round_name: str,
    candidate: str,
    *,
    seed: int = 0,
    model: VLMWrapper | None = None,
    samples: Mapping[str, Sequence[VQASample]] | None = None,
    records: Mapping[str, Any] | None = None,
    allow_remote_code: bool = False,
) -> None:
    """Run one candidate through one round and write its rows.

    ``model``, ``samples`` (``{"train": ..., "val": ...}``) and ``records`` (cached teacher
    outputs by sample id) can be passed in; tests do. Otherwise they are built from the
    config.
    """
    from medcompress.config import load_config
    from medcompress.utils.io import upsert_results

    plan = plan_round(config, round_name, candidate, seed, need_side=model is None)
    paths = config["paths"]
    tag = f"{round_name}:{candidate}"
    target = Path(paths["results_root"]) / "selection" / round_name / candidate / "metrics.csv"
    if model is None:
        from medcompress.config import ConfigError
        from medcompress.models.loading import ModelSpec, load_model
        from medcompress.models.smoke import cuda_available

        try:
            spec = ModelSpec.from_config(load_config(plan.config_path))
            model = load_model(
                spec,
                device_map={"": 0} if cuda_available() else "cpu",
                allow_candidate=True,
                allow_remote_code=allow_remote_code,
                image_longest_side=plan.side,
            )
        except ConfigError:
            raise  # a setting or a missing opt-in, not a property of the candidate
        except Exception as exc:  # noqa: BLE001 - a candidate that cannot load fails G1
            if round_name != "R0":
                raise
            print(f"{candidate} did not load: {type(exc).__name__}: {exc}")
            failed = {"r0_load_ok": 0.0, "gate_g1_hardware": 0.0}
            rows = [
                {"experiment": tag, "budget": "", "seed": seed, "metric": k, "value": v}
                for k, v in failed.items()
            ]
            upsert_results(rows, target)
            print(f"wrote {len(rows)} rows to {target}")
            return
    if round_name == "R0":
        rows = [{**r, "experiment": tag, "seed": seed} for r in feasibility_rows(model, plan)]
        upsert_results(rows, target)
        print(f"wrote {len(rows)} rows to {target}")
        return

    from medcompress.compression.distillation.offline_kd import (
        examples_from_samples,
        select_subset,
        train_lora,
    )
    from medcompress.evaluation.runner import evaluate_model, resolve_settings, to_rows

    assert plan.eval_config is not None and plan.train is not None
    settings = resolve_settings(plan.eval_config)
    data_root = Path(paths["data_root"])
    if samples is None:
        from medcompress.data.prepared import load_prepared

        dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
        samples = load_prepared(data_root, dataset, Path(paths["private_root"]))
    val = select_validation_subset(samples["val"], plan.n_val_images, plan.val_seed)
    metrics: list[Any] = []
    if round_name == "R1":
        zero = evaluate_model(model, val, settings, data_root, seed)
        keep = ("macro_f1", "small_finding_recall", "blank_accuracy_drop")
        metrics += _prefixed(zero.metrics, "zero_shot:", keep)
    train_cfg = plan.train
    if train_cfg.mode == "offline_kd":
        if records is None:
            from medcompress.models.teacher_cache import load_teacher_cache

            records = {
                r.sample_id: r for r in load_teacher_cache(Path(paths["cache_root"]) / "teacher")
            }
        share = tokenizer_match_share(model, list(records.values()))
        metrics.append(_metric("tokenizer_match", share))
        if share < MIN_TOKENIZER_MATCH:  # a finding about the candidate, not an error
            metrics.append(_metric("gate_tokenizer_match", 0.0))
            upsert_results(to_rows(metrics, tag, "", seed), target)
            print(
                f"{candidate} reproduces {share:.1%} of the cached teacher answers "
                f"(need {MIN_TOKENIZER_MATCH:.0%}): logit distillation is not valid, round stopped"
            )
            return
        metrics.append(_metric("gate_tokenizer_match", 1.0))
    examples = examples_from_samples(
        samples["train"], data_root, records=records if train_cfg.mode == "offline_kd" else None
    )
    n = None if train_cfg.data_subset == "all" else int(train_cfg.data_subset)
    chosen = select_subset([e.sample_id for e in examples], n, plan.data_seed)
    examples = [examples[i] for i in chosen]
    result = train_lora(model, examples, train_cfg, seed=seed)
    metrics += [
        _metric("train_examples", len(examples)),
        _metric("train_steps", result.steps),
        _metric("train_loss_final", result.final_loss()),
        _metric("train_seconds", result.seconds),
    ]
    if result.peak_gpu_bytes is not None:
        metrics.append(_metric("train_peak_gpu_bytes", result.peak_gpu_bytes))
    evaluated = evaluate_model(model, val, settings, data_root, seed)
    metrics += evaluated.metrics
    if round_name == "R1":
        drop = next(m.value for m in evaluated.metrics if m.name == "blank_accuracy_drop")
        metrics.append(_metric("gate_g2_uses_image", float(drop >= float(plan.g2_threshold))))
    if round_name != "R1":
        from medcompress.compression.lora import save_adapter

        save_adapter(
            model,
            Path(paths["checkpoint_root"])
            / "selection"
            / round_name
            / candidate
            / f"seed{seed}"
            / "adapter",
        )
    rows = to_rows(metrics, tag, "", seed)
    upsert_results(rows, target)
    outcomes = outcomes_from_result(evaluated, settings)
    out_path = (
        Path(paths["output_root"])
        / "selection"
        / round_name
        / candidate
        / f"seed{seed}"
        / "outcomes.npz"
    )
    outcomes.to_npz(out_path)
    print(f"wrote {len(rows)} rows to {target} and the probe answers to {out_path}")


def _metric(name: str, value: float) -> Any:
    from medcompress.evaluation.probe_suites import Metric

    return Metric(name, float(value))
