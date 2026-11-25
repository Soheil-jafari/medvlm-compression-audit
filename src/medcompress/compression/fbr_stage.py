"""The GPU steps of the FBR study (amendment of 2026-11-22; cells (i) to (iv) of
``docs/runs/RC-FBR_finding_balanced_recovery.md``).

* :func:`run_probe_answers` (cell (i)): the decoded free answer and the yes / no token ranks of
  every validation probe of a run, beside its outcomes;
* :func:`run_e2_margin_cache` (cell (ii)): E2 seed 0's margin of every training probe of the
  schedule, the answer tokens and the spelling report;
* :func:`run_fbr_arm` (cells (iii) and (iv)): an arm's pruned model rebuilt before recovery, FBR
  (or the sampling-only control), then the validation pass and the probe answers; with
  ``trial_steps`` only that many updates, timed, for the budget estimate.

Nothing here writes into ``results/<run>``: the FBR rows go to ``--out`` (by default
``results/exploratory/fbr/<label>/fbr_metrics.csv``), the arrays to ``<output_root>/fbr/``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from medcompress.config import ConfigError, RunContext

MARGINS_FILE = "fbr_e2_margins.json"
RUN_FILE = "fbr_run.json"
DIAGNOSIS_FILE = "fbr_e2_diagnosis.json"
"""E2's answer-format diagnosis: counts and token strings only (no sample id; committable)."""
ROWS_FILE = "fbr_metrics.csv"
"""The FBR model's validation rows (not ``metrics.csv``: the aggregator reads that name as a
registered run's file)."""
OUTPUT_SUBDIR = "fbr"


def arm_label(run: RunContext, control: bool) -> str:
    """``E5_magnitude``, ``E7``, ``E5_activation_sampling_only`` ..."""
    label = run.experiment_tag.replace(":", "_")
    return f"{label}_sampling_only" if control else label


def _side(config: Mapping[str, Any]) -> int:
    side = (config.get("inputs") or {}).get("image_longest_side")
    if side is None:
        raise ConfigError("settings still open: inputs.image_longest_side")
    return int(side)


def _validation_settings(config: Mapping[str, Any]) -> Any:
    from medcompress.evaluation.fbr_analysis import require_validation
    from medcompress.evaluation.runner import resolve_settings

    settings = resolve_settings(config)
    require_validation(settings.split)
    return settings


# ----------------------------------------------------------------------------- cell (i)


def run_probe_answers(config: Mapping[str, Any], run: RunContext | None) -> Path:
    """Decoded answers and token ranks of the run's validation probes, written to
    ``<output_root>/fbr/probe_answers/<run dir>/probe_answers_val.npz``."""
    from medcompress.compression.fbr import FBRSettings
    from medcompress.evaluation.fbr_analysis import ANSWERS_FILE, probe_answers
    from medcompress.evaluation.runner import load_run_model, split_samples

    if run is None:
        raise ConfigError("--exp is required")
    settings = _validation_settings(config)
    samples, _drops = split_samples(config, settings)
    model = load_run_model(config, run, _side(config))
    arrays = probe_answers(model, samples, settings, Path(config["paths"]["data_root"]),
                           FBRSettings.load().decode_max_new_tokens)  # fmt: skip
    folder = Path(config["paths"]["output_root"]) / OUTPUT_SUBDIR / "probe_answers"
    target = folder / run.run_dir("") / ANSWERS_FILE.format(split=settings.split)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    print(f"wrote {target}: {len(arrays['probe_decoded'])} probes, "
          f"{int((arrays['probe_decoded'] < 0).sum())} parse failures", flush=True)  # fmt: skip
    return target


# ----------------------------------------------------------------------------- cell (ii)


def _schedule(config: Mapping[str, Any], train: Any, steps: int | None = None) -> Any:
    from medcompress.compression.distillation.offline_kd import TrainConfig
    from medcompress.compression.fbr import (
        FBRSettings,
        probe_schedule,
        recovery_config,
        training_probes,
    )
    from medcompress.evaluation.runner import resolve_settings

    settings = resolve_settings(config)
    fbr = FBRSettings.load()
    cfg = TrainConfig.from_config({**recovery_config(config), "mode": "offline_kd"})
    probes = training_probes(train, settings.findings, settings.template, settings.display_names)
    schedule = probe_schedule(probes, settings.findings, steps or cfg.steps, fbr.seed)
    return schedule, settings, fbr, cfg


def probe_answers_config(config: Mapping[str, Any]) -> dict[str, str]:
    """The probe template's canonical answers (``domain.probe.answers``)."""
    from medcompress.config import load_config

    domain = load_config(config["components"]["domain"])["domain"]
    answers = (domain.get("probe") or {}).get("answers") or {}
    if not {"positive", "negative"} <= set(answers):
        raise ConfigError("domain.probe.answers needs positive and negative")
    return {"positive": str(answers["positive"]), "negative": str(answers["negative"])}


def run_e2_margin_cache(
    config: Mapping[str, Any], out: Path, *, examples: int = 10, max_new_tokens: int = 32
) -> Path:
    """E2 seed 0's pooled margin of every probe of the FBR schedule (the training pass's
    function, no gradient), the (yes, no) cross-entropy targets (the template's canonical
    answers, checked to be single tokens) and the diagnosis of E2's answer format
    (clarification of 2026-11-24): its greedy first token by label with the 20 most frequent
    other tokens decoded; every schedule probe decoded (up to 8 tokens, as the FBR rule) and
    read by the pre-registered parser, with the unparsed count; ``examples`` full answers each
    to positive and negative probes (up to ``max_new_tokens``) printed. Writes
    ``fbr_e2_margins.json`` (sample ids inside: never committed) and ``fbr_e2_diagnosis.json``
    (counts and token strings only)."""
    from collections import Counter

    from medcompress.compression.fbr import (
        FBRSettings,
        canonical_answer_ids,
        first_token_report,
        score_margins,
    )
    from medcompress.compression.pruning.stage import load_train_samples
    from medcompress.evaluation.clinical import parse_yes_no
    from medcompress.evaluation.runner import load_trained_student
    from medcompress.models.tokens import NO_SPELLINGS, YES_SPELLINGS

    train = load_train_samples(config)
    schedule, _settings, _fbr, _cfg = _schedule(config, train)
    keys = set(schedule.unique_keys())
    probes = {p.key: p for u in schedule.updates for p in u if p.key in keys}
    e2 = load_trained_student(config, "E2", 0, _side(config))
    yes_id, no_id, targets = canonical_answer_ids(e2, probe_answers_config(config))
    data_root = Path(config["paths"]["data_root"])
    started = time.perf_counter()
    margins, first = score_margins(e2, list(probes.values()), data_root, (yes_id, no_id))
    tokenizer = e2.tokenizer
    present = {k: p.present for k, p in probes.items()}
    tokens = first_token_report(first, present, tokenizer.decode, e2.yes_ids, e2.no_ids)
    listed = list(probes.values())
    decoded = e2.generate([data_root / p.image_path for p in listed], [p.question for p in listed],
                          max_new_tokens=FBRSettings.load().decode_max_new_tokens)  # fmt: skip
    parsed: dict[str, Counter] = {"positive": Counter(), "negative": Counter()}
    for probe, answer in zip(listed, decoded, strict=True):
        reading = parse_yes_no(answer.text) or "unparsed"
        parsed["positive" if probe.present else "negative"][reading] += 1
    positives = [p for p in listed if p.present][:examples]
    shown = positives + [p for p in listed if not p.present][:examples]
    full = e2.generate([data_root / p.image_path for p in shown], [p.question for p in shown],
                       max_new_tokens=max_new_tokens)  # fmt: skip
    diagnosis = {
        "spelling_tokens": {s: list(tokenizer.encode(s, add_special_tokens=False))
                            for s in (*YES_SPELLINGS, *NO_SPELLINGS)},
        "single_token_yes_ids": list(e2.yes_ids), "single_token_no_ids": list(e2.no_ids),
        "targets": targets,
        "first_tokens": tokens,
        "parser": "clinical.parse_yes_no (pre-registered): lower case, leading non-letters "
                  "dropped, the answer must start with the word yes or no",
        "decoded_up_to_tokens": FBRSettings.load().decode_max_new_tokens,
        "parsed_by_label": {k: {r: int(n) for r, n in c.items()} for k, c in parsed.items()},
        "probes": len(listed),
        # a non-finite E2 margin (fp16 overflow of an image): that update is skipped (2026-11-25)
        "e2_nonfinite_margins": int(sum(1 for v in margins.values() if not math.isfinite(v))),
    }  # fmt: skip
    examples_shown = [{"finding": p.finding, "present": p.present, "answer": a.text,
                       "parsed": parse_yes_no(a.text) or "unparsed"}
                      for p, a in zip(shown, full, strict=True)]  # fmt: skip
    document = {
        "schedule_fingerprint": schedule.fingerprint(), "schedule_log": schedule.log,
        "findings": list(schedule.findings), "steps": len(schedule.updates),
        "margins": margins, "yes_target_id": yes_id, "no_target_id": no_id,
        "seconds": time.perf_counter() - started, "diagnosis": diagnosis,
    }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    target = out / MARGINS_FILE
    target.write_text(json.dumps(document, indent=1), encoding="utf-8")
    (out / DIAGNOSIS_FILE).write_text(json.dumps(diagnosis, indent=1), encoding="utf-8")
    print(f"wrote {target}: {len(margins)} probes of {len(schedule.updates)} updates "
          f"(schedule {schedule.fingerprint()}); targets {targets}", flush=True)  # fmt: skip
    print(json.dumps(diagnosis, indent=1))
    print("decoded answers (not committed):")
    for row in examples_shown:
        print(
            f"  [{'positive' if row['present'] else 'negative'} {row['finding']}] "
            f"{row['answer']!r} -> {row['parsed']}"
        )
    return target


# ----------------------------------------------------------------------------- cells (iii), (iv)


def run_fbr_arm(
    config: Mapping[str, Any],
    arm: RunContext,
    *,
    margins_file: Path,
    out: Path,
    control: bool = False,
    trial_steps: int | None = None,
) -> Path:
    """FBR (or, with ``control``, the sampling-only recipe) on one arm, from its pruned model
    before recovery; then the validation pass and the probe answers. With ``trial_steps`` it
    stops after that many updates and writes the timing and the estimate instead. (``arm``, not
    ``run``: the stage records its environment in its own ``training.json``, never in the
    registered run's ``environment.json``.)"""
    import dataclasses

    from medcompress.compression.distillation.offline_kd import (
        data_seed,
        examples_from_samples,
        require_teacher_records,
        training_pool,
    )
    from medcompress.compression.fbr import rebuild_pruned, recovery_config, train_fbr
    from medcompress.compression.lora import merge_adapter
    from medcompress.compression.pruning.stage import load_train_samples
    from medcompress.evaluation.fbr_analysis import ANSWERS_FILE, probe_answers
    from medcompress.evaluation.runner import evaluate_model, split_samples, to_rows
    from medcompress.models.teacher_cache import load_pinned_teacher_cache
    from medcompress.utils.environment import environment_record
    from medcompress.utils.io import upsert_results

    if arm.budget != "B2" or arm.seed != 0:
        raise ConfigError("the FBR pilot runs at B2, seed 0")
    cache = json.loads(Path(margins_file).read_text(encoding="utf-8"))
    paths = config["paths"]
    data_root = Path(paths["data_root"])
    label = arm_label(arm, control)
    t0 = time.perf_counter()
    train = load_train_samples(config)
    schedule, settings, fbr, cfg = _schedule(config, train)
    full = _schedule(config, train, steps=int(cache["steps"]))[0]
    if full.fingerprint() != cache["schedule_fingerprint"]:
        raise ValueError(f"the probe schedule ({full.fingerprint()}) is not the one E2's margins "
                         f"were cached for ({cache['schedule_fingerprint']})")  # fmt: skip
    if control:
        fbr = dataclasses.replace(fbr, margin_coefficient=0.0)
    model, rebuild = rebuild_pruned(config, arm)
    from medcompress.compression.fbr import canonical_answer_ids

    yes_id, no_id, _targets = canonical_answer_ids(model, probe_answers_config(config))
    if (yes_id, no_id) != (int(cache["yes_target_id"]), int(cache["no_target_id"])):
        raise ValueError(
            f"the canonical answer tokens ({yes_id}, {no_id}) are not the ones E2's "
            f"cache was made with ({cache['yes_target_id']}, {cache['no_target_id']})"
        )
    print(f"rebuilt {label} before recovery: {rebuild}", flush=True)
    records = {r.sample_id: r for r in load_pinned_teacher_cache(
        config, Path(paths["cache_root"]) / "teacher")}  # fmt: skip
    pool = training_pool(train, recovery_config(config).get("data_subset"), data_seed(config))
    examples = examples_from_samples(pool, data_root, records=records)
    require_teacher_records(pool, examples)
    setup_seconds = time.perf_counter() - t0
    result = train_fbr(
        model, examples, schedule, cache["margins"], cfg, fbr,
        (yes_id, no_id),
        data_root=data_root, seed=arm.seed, stop_after_steps=trial_steps,
    )  # fmt: skip
    folder = Path(out) / label
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "arm": arm.experiment_tag, "control": control, "label": label,
        "margin_coefficient": fbr.margin_coefficient, "schedule": schedule.fingerprint(),
        "schedule_log": schedule.log, "rebuild": rebuild, "setup_seconds": setup_seconds,
        "training": result.summary(), "environment": environment_record(),
    }  # fmt: skip
    if trial_steps is not None:
        timed = result.seconds_per_update[1:] or result.seconds_per_update
        per_update = float(np.mean(timed))
        record["trial"] = {
            "updates": result.steps, "seconds_per_update": per_update,
            "estimated_training_hours": per_update * cfg.steps / 3600,
            "note": "an estimate from the timed updates (the first one left out)",
        }  # fmt: skip
        (folder / "trial.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"trial: {per_update:.1f} s per update, about "
              f"{record['trial']['estimated_training_hours']:.2f} h for {cfg.steps} updates "
              "(estimate)", flush=True)  # fmt: skip
        return folder / "trial.json"
    merge_adapter(model)
    model.module.eval()
    eval_settings = _validation_settings(config)
    samples, _drops = split_samples(config, eval_settings)
    t1 = time.perf_counter()
    evaluation = evaluate_model(model, samples, eval_settings, data_root, arm.seed)
    arrays_dir = Path(paths["output_root"]) / OUTPUT_SUBDIR / label
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_dir / "outcomes.npz", **evaluation.arrays)
    answers = probe_answers(model, samples, eval_settings, data_root, fbr.decode_max_new_tokens)
    np.savez_compressed(arrays_dir / ANSWERS_FILE.format(split="val"), **answers)
    record["evaluation_seconds"] = time.perf_counter() - t1
    kept = ("arm", "control", "label", "margin_coefficient", "schedule")
    (arrays_dir / RUN_FILE).write_text(json.dumps({k: record[k] for k in kept}), encoding="utf-8")
    upsert_results(to_rows(evaluation.metrics, f"FBR:{label}", "B2", arm.seed),
                   folder / ROWS_FILE)  # fmt: skip
    (folder / "training.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {folder} and {arrays_dir}", flush=True)
    return folder
