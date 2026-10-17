#!/usr/bin/env python
"""Train our teacher adapter (run card RC-TT1): rule 3's setup and N from RC-TT0, resumable.

Reads the setup (fp16 over two T4s, or QLoRA on one), the gradient checkpointing, N and the
LoRA settings from RC-TT0's committed result file; no number is typed by hand. Trains
Qwen2.5-VL-7B-Instruct (pinned) with a new LoRA on the first N examples of our training split
(a seeded order stratified by question type), saving its state about every 30 minutes. Run
again, it continues from the furthest saved state it finds: the local one, or one in a
previous notebook version's output attached as input (searched under /kaggle/input and
--resume-from). At the end it writes the adapter as a hand-off (one private Kaggle dataset)
and runs the validation evaluation and Gate 1, which wait cleanly while the finding list is
still open (RC-E0a and RC-E0b then run them).

Examples (Kaggle, from the repository root):
    python scripts/train_teacher.py --data-root /kaggle/input/<shared-data> --dry-run
    python scripts/train_teacher.py --data-root /kaggle/input/<shared-data>
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
    extension_size,
    find_resume_states,
    read_plan,
    saved_step,
    teacher_order,
    train_settings,
    training_size,
)

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
        help="only once the team has checked that Kaggle counts a T4 x2 hour once",
    )  # fmt: skip
    parser.add_argument(
        "--extend", action="store_true",
        help="the one extension of the quality rule: +50 %% of N, on the next examples",
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


def _evaluate(script: str, data_root: Path, *extra: str) -> int:
    command = [sys.executable, str(REPO / "scripts" / script), "--exp", "E0", "--seed", "0",
               "--data-root", str(data_root), *extra]  # fmt: skip
    print(f"== {' '.join(command[1:])}", flush=True)
    return subprocess.run(command, cwd=REPO, check=False).returncode


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - one linear procedure
    args = build_parser().parse_args(argv)
    try:
        plan = read_plan(args.feasibility, t4x2_counts_once=args.t4x2_counts_once)
    except RuleError as exc:
        print(exc, file=sys.stderr)
        return CONFIG_INCOMPLETE
    if plan.setup == "nf4_1gpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # before torch is imported
    out = args.out.resolve()
    _session_env(out)

    from medcompress.config import load_config
    from medcompress.utils.bundles import BundleError, attach_data, attach_handoff

    config = load_config("experiments/E0.yaml")
    try:
        attach_data(config, args.data_root)
        for path in args.handoffs:
            attach_handoff(config, path)
    except BundleError as exc:
        print(f"[inputs] check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    settings = load_config("teacher_training.yaml")["teacher_training"]
    paths = config["paths"]

    from medcompress.compression.distillation.offline_kd import TrainConfig, data_seed
    from medcompress.compression.lora import LoRAConfig
    from medcompress.data.prepared import load_prepared

    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    train = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))["train"]
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
    cfg = TrainConfig(
        lora=LoRAConfig(int(lora["rank"]), float(lora["alpha"]), float(lora["dropout"]),
                        tuple(lora["target_modules"])),
        extra=fields, **core,
    )  # fmt: skip
    hours = size * plan.seconds_per_example / 3600
    print(
        f"teacher plan (RC-TT0, rule 3): setup {plan.setup}, gradient checkpointing "
        f"{'on' if plan.gradient_checkpointing else 'off'}, N = {n} of {len(train)} training rows"
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
    result = train_lora(model, examples, cfg, seed=0, resume_dir=resume)

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
        )
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
