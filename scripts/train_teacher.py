#!/usr/bin/env python
"""Train our teacher adapter (run card RC-TT1): rule 3's setup and N from RC-TT0, resumable.

Reads the setup (fp16 over two T4s, or QLoRA on one), the gradient checkpointing, N and the
LoRA rank, alpha and dropout from RC-TT0's committed result file; no number is typed by hand.
N is the reading for a T4 x2 hour counted once (measured 2026-10-19; `rule.t4x2_counts_twice`
in configs/teacher_training.yaml). The LoRA targets are the published adapter's: its
adapter_config.json is read at the start (pinned revision), printed, applied inside the
language model's decoder layers (the vision tower stays frozen) and recorded; the run stops if
a published target matches no linear layer there. Trains
Qwen2.5-VL-7B-Instruct (pinned) with a new LoRA on the first N examples of our training split
(a seeded order stratified by question type), saving its state about every 30 minutes. Run
again, it continues from the furthest saved state it finds: the local one, or one in a
previous notebook version's output attached as input (searched under /kaggle/input and
--resume-from). At the end it writes the adapter as a hand-off (one private Kaggle dataset)
and runs the validation evaluation and Gate 1, which wait cleanly while the finding list is
still open (RC-E0a and RC-E0b then run them).

The shared data must hold the split the dataset config pins (``split_sha256``, RC-D1 run 4b):
the manifest is checked when the data are attached, and the training split is rebuilt and
hashed again. With ``--stop-after-hours H`` the training stops on a saved step once H hours
have passed since the script started, and the script exits with code 3: the notebook version
then ends normally, its output keeps the saved state, and the next session resumes from it.

Examples (Kaggle, from the repository root):
    python scripts/train_teacher.py --data-root /kaggle/input/<shared-data> --dry-run
    python scripts/train_teacher.py --data-root /kaggle/input/<shared-data> --stop-after-hours 11
    python scripts/train_teacher.py --data-root ... --extend --from /kaggle/input/<teacher-adapter>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from medcompress.models.teacher_training import (  # noqa: E402
    ADAPTER_DIR,
    DEFAULT_CSV,
    EXTENSION_RESUME_DIR,
    NF4,
    RESUME_DIR,
    RuleError,
    adapted_counts,
    extension_size,
    find_resume_states,
    published_target_names,
    read_plan,
    saved_step,
    t4x2_counts_once,
    teacher_order,
    train_settings,
    training_size,
)

RESUME_NEEDED = 3
"""The training stopped on a saved step for ``--stop-after-hours``: run it again to resume."""
CONFIG_INCOMPLETE = 4
INPUT_CHECK_FAILED = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True, help="the shared-data dataset")
    parser.add_argument("--from", dest="handoffs", type=Path, action="append", default=[])
    parser.add_argument("--feasibility", type=Path, default=REPO / DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    parser.add_argument(
        "--resume-from", type=Path, action="append", default=[],
        help="extra folders to search for a saved state (/kaggle/input is always searched)",
    )  # fmt: skip
    parser.add_argument(
        "--t4x2-counts-once", action="store_true",
        help="kept for old commands; the config says so since 2026-10-19",
    )  # fmt: skip
    parser.add_argument(
        "--published-adapter-config", type=Path, default=None,
        help="a local copy of the published adapter_config.json (default: read from the hub)",
    )  # fmt: skip
    parser.add_argument(
        "--extend", action="store_true",
        help="the one extension of the quality rule: +50 %% of N, on the next examples",
    )  # fmt: skip
    parser.add_argument(
        "--stop-after-hours", type=float, default=None, metavar="H",
        help="stop on a saved step once H hours have passed since the start, and exit with "
             "code 3 so the session ends cleanly (set it below the session time limit)",
    )  # fmt: skip
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; load no model")
    return parser


def _session_env(out: Path) -> None:
    for name, sub in (("CHECKPOINT", "checkpoints"), ("CACHE", "cache"), ("OUTPUT", "outputs")):
        os.environ.setdefault(f"MEDCOMPRESS_{name}_ROOT", str(out / sub))
    if Path("/kaggle").is_dir():
        os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _restore_state(local: Path, roots: list[Path]) -> str:
    """Copy the furthest saved state found under ``roots`` to ``local`` if it is further."""
    here = saved_step(local)
    for folder in find_resume_states(roots, folder_name=local.name):
        if folder.resolve() == local.resolve():
            continue
        if saved_step(folder) > here:
            shutil.rmtree(local, ignore_errors=True)
            shutil.copytree(folder, local)
            return f"continuing from step {saved_step(local)} saved in {folder}"
        break
    return f"continuing from step {here} saved locally" if here else "starting from step 0"


def _published_config(settings: dict, local: Path | None) -> dict:
    """The published adapter_config.json: a local copy, or the pinned revision on the hub."""
    if local is not None:
        return json.loads(Path(local).read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    from medcompress.config import load_config

    published = load_config(settings["published_adapter"])["model"]
    path = hf_hub_download(
        published["adapter"], "adapter_config.json", revision=published["adapter_revision"]
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _evaluate(script: str, data_root: Path, *extra: str) -> int:
    command = [sys.executable, str(REPO / "scripts" / script), "--exp", "E0", "--seed", "0",
               "--data-root", str(data_root), *extra]  # fmt: skip
    print(f"== {' '.join(command[1:])}", flush=True)
    return subprocess.run(command, cwd=REPO, check=False).returncode


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - one linear procedure
    started = time.monotonic()
    args = build_parser().parse_args(argv)
    if args.stop_after_hours is not None and args.stop_after_hours <= 0:
        print("--stop-after-hours must be above 0", file=sys.stderr)
        return INPUT_CHECK_FAILED
    from medcompress.config import load_config

    settings = load_config("teacher_training.yaml")["teacher_training"]
    once = args.t4x2_counts_once or t4x2_counts_once(settings)
    try:
        plan = read_plan(args.feasibility, t4x2_counts_once=once)
    except RuleError as exc:
        print(exc, file=sys.stderr)
        return CONFIG_INCOMPLETE
    if plan.setup == "nf4_1gpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # before torch is imported
    out = args.out.resolve()
    _session_env(out)

    from medcompress.utils.bundles import BundleError, attach_data, attach_handoff

    config = load_config("experiments/E0.yaml")
    try:
        attach_data(config, args.data_root)
        for path in args.handoffs:
            attach_handoff(config, path)
    except BundleError as exc:
        print(f"[inputs] check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    paths = config["paths"]
    targets, published_spec = None, None
    if settings["target_modules"] == "published":
        if args.dry_run and args.published_adapter_config is None:
            print("LoRA targets: the published adapter's, read from the hub when the run starts")
        else:
            try:
                published = _published_config(settings, args.published_adapter_config)
                published_spec = published.get("target_modules")
                targets = published_target_names(published_spec)
            except (OSError, ValueError, RuleError) as exc:
                print(f"the published adapter_config.json could not be used: {exc}",
                      file=sys.stderr)  # fmt: skip
                return INPUT_CHECK_FAILED
            print(f"published target_modules (as written): {published_spec}")
            print(f"LoRA targets inside the decoder layers: {', '.join(targets)}")
            mismatch = [
                (key, published.get(key), mine)
                for key, mine in (("r", plan.lora_rank), ("lora_alpha", plan.lora_alpha),
                                  ("lora_dropout", plan.lora_dropout))
                if key in published and float(published[key]) != float(mine)
            ]  # fmt: skip
            if mismatch:
                print(f"RC-TT0's LoRA values differ from the published file: {mismatch}",
                      file=sys.stderr)  # fmt: skip
                return INPUT_CHECK_FAILED

    from medcompress.compression.distillation.offline_kd import TrainConfig, data_seed
    from medcompress.compression.lora import LoRAConfig
    from medcompress.data.prepared import load_prepared, pinned_split_sha256

    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    pinned = pinned_split_sha256(dataset)
    try:  # the split rebuilt here must hash to the pinned one (RC-D1 run 4b)
        train = load_prepared(
            Path(paths["data_root"]), dataset, Path(paths["private_root"]),
            expected_split_sha256=pinned,
        )["train"]  # fmt: skip
    except ValueError as exc:
        print(f"[inputs] check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    print(f"split: {'split_sha256 ' + pinned + ' (pinned, checked)' if pinned else 'not pinned'}"
          f"; {len(train)} training rows")  # fmt: skip
    order = teacher_order(train, data_seed(config))
    n = training_size(plan, len(train))
    if args.extend:
        extra = extension_size(n, float(settings["quality"]["extension_fraction"]))
        pool, size = order[n : n + extra], extra
    else:
        pool, size = order[:n], n
    fields = train_settings(plan, settings, size)
    lora = fields.pop("lora")
    core = {k: fields.pop(k) for k in ("mode", "steps", "learning_rate", "batch_size",
            "grad_accumulation", "kd_weight", "kd_temperature", "data_subset")}  # fmt: skip
    groups = () if lora["target_modules"] == "published" else tuple(lora["target_modules"])
    cfg = TrainConfig(
        lora=LoRAConfig(int(lora["rank"]), float(lora["alpha"]), float(lora["dropout"]),
                        groups, layer_modules=tuple(targets or ())),
        extra=fields, **core,
    )  # fmt: skip
    hours = size * plan.seconds_per_example / 3600
    print(
        f"teacher plan (RC-TT0, rule 3): setup {plan.setup}, gradient checkpointing "
        f"{'on' if plan.gradient_checkpointing else 'off'}, N = {n} of {len(train)} training rows"
        f" (a T4 x2 hour counted {'once' if once else 'twice'})"
        + (f"; extension: {size} more examples" if args.extend else "")
    )
    print(
        f"{cfg.steps} updates of 16, warm-up {cfg.extra['warmup_steps']}, lr {cfg.learning_rate} "
        f"cosine, weight decay {cfg.extra['weight_decay']}; LoRA rank {lora['rank']} alpha "
        f"{lora['alpha']} dropout {lora['dropout']}; a save every {cfg.extra['checkpoint_every']} "
        f"updates; projected about {hours:.1f} h wall (an estimate from RC-TT0)"
    )
    if args.dry_run:
        return 0
    if len(pool) < size:
        print(f"only {len(pool)} examples are left for the extension; stop", file=sys.stderr)
        return CONFIG_INCOMPLETE

    checkpoint_root = Path(paths["checkpoint_root"])
    resume = checkpoint_root / (EXTENSION_RESUME_DIR if args.extend else RESUME_DIR)
    resume.parent.mkdir(parents=True, exist_ok=True)
    print(_restore_state(resume, [*args.resume_from, Path("/kaggle/input")]))

    from medcompress.compression.distillation.offline_kd import examples_from_samples, train_lora
    from medcompress.compression.distillation.resume import ResumeStore
    from medcompress.compression.lora import save_adapter
    from medcompress.models.loading import ModelSpec, load_model

    spec = ModelSpec.from_config(load_config(settings["model"]))
    base = dataclasses.replace(spec, adapter=None, adapter_revision=None)
    fp16 = plan.setup == "fp16_2gpu"
    side = int(config["inputs"]["image_longest_side"])
    model = load_model(
        base, device_map="auto" if fp16 else {"": 0}, quantization=None if fp16 else NF4,
        image_longest_side=side,
    )  # fmt: skip
    counts: dict[str, int] = {}
    if targets:  # check the published targets on the loaded model before any training
        import torch

        from medcompress.compression.lora import lora_target_regex

        regex = lora_target_regex(model.module, cfg.lora)
        linear = [name for name, sub in model.module.named_modules()
                  if isinstance(sub, torch.nn.Linear)]  # fmt: skip
        names = targets if targets != ("all-linear",) else ()
        counts = adapted_counts(linear, regex, names)
        print("adapted linear layers per target: "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))  # fmt: skip
        empty = sorted(k for k, v in counts.items() if v == 0)
        if empty or not counts:
            print(f"published targets that match no linear layer of the language model: "
                  f"{empty or 'all'}; a decision is needed (D01), nothing was trained",
                  file=sys.stderr)  # fmt: skip
            return CONFIG_INCOMPLETE
    adapter_dir = checkpoint_root / ADAPTER_DIR
    if args.extend:
        import peft

        if not (adapter_dir / "adapter_config.json").is_file():
            print(f"--extend needs the trained adapter at {adapter_dir} (attach it with --from)")
            return INPUT_CHECK_FAILED
        model.module = peft.PeftModel.from_pretrained(
            model.module, str(adapter_dir), is_trainable=True
        )
    examples = examples_from_samples(pool, Path(paths["data_root"]))
    budget = None
    if args.stop_after_hours is not None:  # what is left of H after the model was loaded
        budget = max(0.0, args.stop_after_hours * 3600 - (time.monotonic() - started))
    result = train_lora(model, examples, cfg, seed=0, resume_dir=resume, stop_after_seconds=budget)
    if result.steps < cfg.steps:
        print(f"STOPPED for a resume after {args.stop_after_hours} h: step {result.steps} of "
              f"{cfg.steps} is saved in {resume} (this version's output). Start the resume "
              "session: attach this version's output and run the same command (RC-TT1, "
              "section 3)")  # fmt: skip
        return RESUME_NEEDED

    target = checkpoint_root / ADAPTER_DIR
    if target.is_symlink():  # an attached adapter (the extension): write a new folder
        target.unlink()
    save_adapter(model, target)
    record = {
        "card": "RC-TT1", "setup": plan.setup,
        "gradient_checkpointing": plan.gradient_checkpointing,
        "n_examples": n, "extension_examples": size if args.extend else 0,
        "train_rows": len(train), "steps": result.steps, "examples_seen": result.examples_seen,
        "seconds": result.seconds, "final_loss": result.final_loss(),
        "peak_gpu_bytes": result.peak_gpu_bytes, "lora": lora, "settings": cfg.extra,
        "published_target_modules": published_spec, "target_names": list(targets or ()),
        "adapted_per_target": counts, "t4x2_counts_once": once,
        "feasibility_sha256": _sha256(args.feasibility),
    }  # fmt: skip
    (target / "teacher_training.json").write_text(json.dumps(record, indent=1, default=str))
    ResumeStore(resume, "").clear()
    del model

    from medcompress.config import RunContext
    from medcompress.utils.bundles import export_handoff, handoff_meta
    from medcompress.utils.io import upsert_results

    run = RunContext("E0", seed=0)
    rows = [
        {"experiment": "E0", "budget": "", "seed": 0, "metric": f"teacher_{name}", "value": value}
        for name, value in (
            ("n_examples", n), ("extension_examples", record["extension_examples"]),
            ("train_steps", result.steps), ("train_examples_seen", result.examples_seen),
            ("train_seconds", result.seconds), ("train_loss_final", result.final_loss()),
            ("setup_fp16_2gpu", float(fp16)),
            ("gradient_checkpointing", float(plan.gradient_checkpointing)),
            ("train_peak_gpu_bytes", float(result.peak_gpu_bytes or 0)),
            ("lora_rank", float(plan.lora_rank)), ("lora_alpha", float(plan.lora_alpha)),
            ("lora_dropout", float(plan.lora_dropout)),
            ("lora_adapted_layers", float(sum(counts.values()))),
            ("t4x2_counts_once", float(once)),
        )
    ]  # fmt: skip
    rows += [
        {"experiment": "E0", "budget": "", "seed": 0, "metric": f"teacher_lora_target:{name}",
         "value": float(value)}
        for name, value in sorted(counts.items())
    ]  # fmt: skip
    upsert_results(rows, run.run_dir(REPO / "results") / "metrics.csv")
    name = "E0_teacher_adapter_extended" if args.extend else "E0_teacher_adapter"
    meta = handoff_meta(
        config, experiment="E0", seed=0,
        extra={k: record[k] for k in ("card", "setup", "gradient_checkpointing", "n_examples",
               "extension_examples", "steps", "feasibility_sha256")},
    )  # fmt: skip
    shutil.rmtree(out / "handoff" / name, ignore_errors=True)
    bundle = export_handoff(
        config, [("checkpoint_root", ADAPTER_DIR)], out / "handoff" / name,
        kind="teacher_adapter", meta=meta,
    )  # fmt: skip
    print(f"teacher adapter: {len(bundle.files)} files in {bundle.root}")
    print("make that folder ONE private Kaggle dataset (docs/team_setup.md)")

    if args.skip_eval:
        return 0
    if not fp16:  # the teacher is evaluated in fp16 over both T4s, whatever trained it
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    code = _evaluate("06_evaluate.py", args.data_root)
    if code == 0 and not fp16:  # QLoRA was used: record the nf4 teacher too (fp16 is the teacher)
        nf4 = json.dumps(NF4)
        code_nf4 = _evaluate("06_evaluate.py", args.data_root, "-o",
                             f"evaluation.teacher_quantization={nf4}", "-o",
                             f"paths.results_root={out / 'results_nf4'}")  # fmt: skip
        print(f"nf4 record: exit code {code_nf4}; rows in {out / 'results_nf4'}")
    gate = _evaluate("00_check_teacher_image_reliance.py", args.data_root)
    for label, value in (("validation", code), ("Gate 1", gate)):
        if value == CONFIG_INCOMPLETE:
            print(f"{label}: waits on an open setting (finding list); RC-E0a / RC-E0b run it later")
        elif value != 0:
            print(f"{label}: exit code {value}; see the log above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
