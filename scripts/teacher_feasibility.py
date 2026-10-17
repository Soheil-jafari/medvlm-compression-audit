#!/usr/bin/env python
"""Teacher feasibility (run card RC-TT0): one LoRA step of the 7B per setup, then rule 3.

  run    each step in a child process of its own:
           lora_config   print the published adapter_config.json; rank, alpha, dropout
           fp16_2gpu     the fp16 base over two T4s, one LoRA step at micro-batch 1
           fp16_2gpu_gc  the same with gradient checkpointing
           nf4_1gpu      QLoRA: the base in 4-bit NF4 on one T4
           nf4_1gpu_gc   the same with gradient checkpointing
         rows go to <out-dir>/teacher_training.csv as each child ends; an out-of-memory error
         is a result. At the end rule 3 (configs/teacher_training.yaml) is applied and the
         chosen setup, N and the projected hours (estimates) are printed and written.
  rule   apply rule 3 again to an existing result file and print it (no GPU)
  child  one step; started by run, not by hand

Synthetic images at side 672 (no dataset). Example (Kaggle, GPU T4 x2):
    python scripts/teacher_feasibility.py run 2>&1 | tee rc_tt0.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from medcompress.config import load_config  # noqa: E402
from medcompress.models.image_sides import (  # noqa: E402
    DEFAULT_CHILD_TIMEOUT,
    lora_step_rows,
    read_child_result,
    safe_peak,
    source_size,
    write_child_result,
)
from medcompress.models.teacher_training import (  # noqa: E402
    DEFAULT_CSV,
    EXPERIMENT,
    FEASIBILITY_STEPS,
    NF4,
    RuleError,
    lora_settings,
    read_values,
    rule_rows,
)
from medcompress.utils.io import upsert_results  # noqa: E402

SIDE = 672
QUESTION = (
    "Describe every abnormality visible in this endoscopic image, its location, its colour and "
    "its size, and state whether an instrument or a text overlay is present."
)
ANSWER = " ".join(
    ["The image shows one small polyp on the left wall with a reddish colour and no instrument."]
    * 3
).split()[:36]  # 36 words: the longest training answer of RC-D1 run 1
ANSWER_TEXT = " ".join(ANSWER)


def settings() -> dict:
    return load_config("teacher_training.yaml")["teacher_training"]


def _row(metric: str, value: float) -> dict:
    return {"experiment": EXPERIMENT, "budget": "", "seed": 0, "metric": metric, "value": value}


def _failed(step: str) -> list[dict]:
    if step == "lora_config":
        default = settings()["lora_default"]
        s = lora_settings(None, default)
        return [_row(f"lora_{k}", v) for k, v in s.items()]
    setup, gc = step.removesuffix("_gc"), step.endswith("_gc")
    label = "lora_gc_step" if gc else "lora_step"
    return [_row(f"{setup}_{label}_ok_mb1", 0.0)]


# ------------------------------------------------------------------ child


def _lora_config_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download

    cfg = settings()
    published_cfg = load_config(cfg["published_adapter"])["model"]
    repo, revision = published_cfg["adapter"], published_cfg["adapter_revision"]
    try:
        path = hf_hub_download(repo, "adapter_config.json", revision=revision)
        published = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"== {repo} at {revision}: adapter_config.json")
        for key in sorted(published):
            print(f"  {key}: {published[key]}")
    except Exception as exc:  # noqa: BLE001 - not readable: the D06 values are used
        print(f"adapter_config.json could not be read ({type(exc).__name__}: {exc}); D06 values")
        published = None
    chosen = lora_settings(published, cfg["lora_default"])
    print(
        f"LoRA for our teacher: rank {chosen['rank']}, alpha {chosen['alpha']}, dropout "
        f"{chosen['dropout']} ({'published' if chosen['from_published'] else 'D06 default'}); "
        f"targets: the language model's linear layers (match {chosen['targets_match']})"
    )
    return [_row(f"lora_{k}", v) for k, v in chosen.items()]


def _setup_rows(step: str, rank: int, alpha: float, dropout: float) -> list[dict]:
    from medcompress.compression.distillation.batching import TrainExample
    from medcompress.models.loading import ModelSpec, load_model
    from medcompress.models.smoke import synthetic_images

    cfg = settings()
    setup, gc = step.removesuffix("_gc"), step.endswith("_gc")
    spec = ModelSpec.from_config(load_config(cfg["model"]))
    base = dataclasses.replace(spec, adapter=None, adapter_revision=None)  # the base only
    image = synthetic_images(1, size=source_size((SIDE,)))[0]
    fp16 = setup == "fp16_2gpu"

    def build() -> object:
        return load_model(
            base,
            device_map="auto" if fp16 else {"": 0},
            quantization=None if fp16 else NF4,
            image_longest_side=SIDE,
        )

    lora = {"rank": rank, "alpha": alpha, "dropout": dropout}
    lora["target_modules"] = cfg["target_modules"]
    return lora_step_rows(
        build, [TrainExample("tt0-0", image, QUESTION, ANSWER_TEXT)], 1, setup, lora=lora,
        learning_rate=float(cfg["learning_rate"]), gradient_checkpointing=gc,
    )  # fmt: skip


def child(args: argparse.Namespace) -> int:
    try:
        if args.step == "lora_config":
            rows = _lora_config_rows()
        else:
            rows = _setup_rows(args.step, args.rank, args.alpha, args.dropout)
    except Exception as exc:  # noqa: BLE001 - every failure is a recorded result
        print(f"{args.step} failed: {type(exc).__name__}: {exc}")
        rows = _failed(args.step)
        peak = safe_peak()
        if peak is not None and args.step != "lora_config":
            setup, gc = args.step.removesuffix("_gc"), args.step.endswith("_gc")
            label = "lora_gc_step" if gc else "lora_step"
            rows.append(_row(f"{setup}_peak_gpu_bytes_{label}_mb1", peak))
    tagged = [{**r, "experiment": EXPERIMENT, "budget": "", "seed": 0} for r in rows]
    write_child_result(args.result_file, tagged)
    return 0


# ------------------------------------------------------------------ parent


def step_metrics(step: str) -> set[str]:
    """The metrics a step writes (its earlier rows are replaced when it runs again)."""
    if step == "lora_config":
        return {
            f"lora_{k}" for k in ("rank", "alpha", "dropout", "from_published", "targets_match")
        }
    setup = step.removesuffix("_gc")
    label = "lora_gc_step" if step.endswith("_gc") else "lora_step"
    return {
        f"{setup}_{label}_ok_mb1",
        f"{setup}_{label}_first_seconds_mb1",
        f"{setup}_{label}_seconds_mb1",
        f"{setup}_peak_gpu_bytes_{label}_mb1",
    }


def _replace_step_rows(rows: list[dict], target: Path, step: str) -> None:
    """Write a step's rows instead of its earlier ones (a failed rerun leaves no stale time)."""
    import csv

    from medcompress.utils.io import append_results

    mine = step_metrics(step)
    kept = []
    if target.exists():
        with target.open(encoding="utf-8", newline="") as handle:
            kept = [old for old in csv.DictReader(handle) if old["metric"] not in mine]
        target.unlink()
    append_results(kept + rows, target)


def print_rule(target: Path) -> int:
    try:
        rows, chosen, n, n_once = rule_rows(read_values(target), settings())
    except RuleError as exc:
        print(f"rule 3 could not be applied: {exc}", file=sys.stderr)
        return 3
    upsert_results(rows, target)
    values = {r["metric"]: r["value"] for r in rows}
    checkpointing = "on" if chosen.gradient_checkpointing else "off"
    counted = int(values["rule_gpus_counted"])
    print("== rule 3 (configs/teacher_training.yaml; all hours are estimates)")
    print(f"setup: {chosen.setup}, gradient checkpointing {checkpointing}")
    print(f"warm step at micro-batch 1: {values['rule_seconds_per_example']:.3f} s per example")
    print(f"N = {n} training examples (a T4 x2 hour counted as {counted} GPU hour(s))")
    print(f"if Kaggle counts a T4 x2 hour once: N = {n_once}")
    print(
        f"projected training: {values['rule_projected_wall_hours']:.1f} h wall, "
        f"{values['rule_projected_gpu_hours']:.1f} GPU hours "
        f"(budget {values['rule_budget_gpu_hours']:.0f}); the training run caps N at one pass "
        "over the training split"
    )
    print(f"written to {target}; RC-TT1 reads these rows")
    return 0


def run(args: argparse.Namespace) -> int:
    target = args.out_dir / "teacher_training.csv"
    rank, alpha, dropout = None, None, None
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="rc_tt0_") as workdir:
        for step in args.steps:
            result = Path(workdir) / f"{step}.json"
            command = [sys.executable, "-u", str(script), "child", "--step", step]
            command += ["--result-file", str(result)]
            if step != "lora_config":
                if rank is None:
                    values = read_values(target) if target.exists() else {}
                    default = settings()["lora_default"]
                    rank = int(values.get("lora_rank", default["rank"]))
                    alpha = float(values.get("lora_alpha", default["alpha"]))
                    dropout = float(values.get("lora_dropout", default["dropout"]))
                command += ["--rank", str(rank), "--alpha", str(alpha), "--dropout", str(dropout)]
            env = dict(os.environ)
            if step.startswith("nf4"):
                first = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                env["CUDA_VISIBLE_DEVICES"] = first or "0"  # QLoRA uses one T4
            print(f"== {step}", flush=True)
            try:
                subprocess.run(command, env=env, timeout=args.child_timeout, check=False)
            except subprocess.TimeoutExpired:
                print(f"{step}: stopped after {args.child_timeout} s")
            rows = read_child_result(result) or _failed(step)
            _replace_step_rows(rows, target, step)
            ok = next((r["value"] for r in rows if r["metric"].endswith("_ok_mb1")), None)
            print(f"{step}: {len(rows)} rows" + ("" if ok is None else f", ok={int(ok)}"))
            if step == "lora_config":
                got = {r["metric"]: r["value"] for r in rows}
                rank, alpha = int(got["lora_rank"]), float(got["lora_alpha"])
                dropout = float(got["lora_dropout"])
    return print_rule(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="part", required=True)
    r = sub.add_parser("run", help="every step, then rule 3")
    r.add_argument("--out-dir", type=Path, default=DEFAULT_CSV.parent)
    r.add_argument("--steps", nargs="+", default=list(FEASIBILITY_STEPS), choices=FEASIBILITY_STEPS)
    r.add_argument("--child-timeout", type=float, default=DEFAULT_CHILD_TIMEOUT)
    q = sub.add_parser("rule", help="apply rule 3 to an existing result file")
    q.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    c = sub.add_parser("child", help="one step; started by run")
    c.add_argument("--step", choices=FEASIBILITY_STEPS, required=True)
    c.add_argument("--result-file", type=Path, required=True)
    c.add_argument("--rank", type=int, default=16)
    c.add_argument("--alpha", type=float, default=32.0)
    c.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.part == "child":
        return child(args)
    if args.part == "rule":
        return print_rule(args.csv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
