#!/usr/bin/env python
"""Image-side measurement on synthetic images (run card RC-S0b): student, teacher, settings.

  student   per side, each step in a child process of its own: scoring, visual tokens, peak
            memory, and two LoRA optimizer steps at each micro-batch, with and without
            gradient checkpointing (one T4: CUDA_VISIBLE_DEVICES=0); rows to
            <out-dir>/student.csv as each child finishes
  teacher   per side, in a child process: scoring, visual tokens, peak memory, generation
            speed (two T4s); rows to <out-dir>/teacher.csv
  settings  prints the revisions in the configs and on the hub, and the image settings the
            published teacher adapter and the Qwen processors state; nothing is written
  child     one (side, step); started by student/teacher, not by hand

A step that fails, out of memory included, is recorded as ..._ok = 0 (with the peak memory if
known) and the run goes on. No dataset is used. Rows are tagged RC-S0b:side<N> and never touch
the RC-M1 or RC-S0 files.

Example (Kaggle):
    CUDA_VISIBLE_DEVICES=0 python scripts/measure_image_sides.py student \\
        --config configs/models/qwen2_5_vl_3b_student.yaml --allow-candidate
    python scripts/measure_image_sides.py teacher --config configs/models/qwen2_5_vl_7b_teacher.yaml
    python scripts/measure_image_sides.py settings
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.config import load_config  # noqa: E402
from medcompress.models.image_sides import (  # noqa: E402
    DEFAULT_CHILD_TIMEOUT,
    DEFAULT_SIDES,
    NOT_STATED,
    REVISION_NOT_PINNED,
    STUDENT_STEPS,
    TEACHER_STEPS,
    Step,
    check_sides,
    check_steps,
    failure_rows,
    lora_step_rows,
    matching_lines,
    measure_sides,
    ok_metric,
    parse_step,
    report_lines,
    safe_peak,
    source_size,
    write_child_result,
)
from medcompress.models.loading import ModelConfigError, ModelSpec, check_loadable  # noqa: E402

TEACHER_CONFIG = Path("configs/models/qwen2_5_vl_7b_teacher.yaml")
STUDENT_CONFIG = Path("configs/models/qwen2_5_vl_3b_student.yaml")
FALLBACK_CONFIG = Path("configs/models/qwen3_vl_2b_student.yaml")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
PROCESSOR_FIELDS = ("min_pixels", "max_pixels", "patch_size", "merge_size", "size")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="part", required=True)
    for part, steps in (("student", STUDENT_STEPS), ("teacher", TEACHER_STEPS)):
        p = sub.add_parser(part, help=f"measure the {part} at every side")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--sides", type=int, nargs="+", default=list(DEFAULT_SIDES))
        p.add_argument(
            "--steps", nargs="+", default=list(steps), help=f"default: {' '.join(steps)}"
        )
        p.add_argument("--out-dir", type=Path, default=Path("results/smoke/RC-S0b"))
        p.add_argument("--allow-candidate", action="store_true", help="the model is a candidate")
        p.add_argument(
            "--child-timeout", type=float, default=DEFAULT_CHILD_TIMEOUT, help="seconds per step"
        )
    child = sub.add_parser("child", help="one (side, step); started by student or teacher")
    child.add_argument("--of", choices=("student", "teacher"), required=True)
    child.add_argument("--config", type=Path, required=True)
    child.add_argument("--side", type=int, required=True)
    child.add_argument("--sides", type=int, nargs="+", required=True, help="sets the image size")
    child.add_argument("--step", required=True)
    child.add_argument("--result-file", type=Path, required=True)
    child.add_argument("--allow-candidate", action="store_true")
    settings = sub.add_parser("settings", help="print the stated settings, write nothing")
    settings.add_argument("--teacher-config", type=Path, default=TEACHER_CONFIG)
    settings.add_argument("--student-config", type=Path, default=STUDENT_CONFIG)
    settings.add_argument("--fallback-config", type=Path, default=FALLBACK_CONFIG)
    return parser


# ------------------------------------------------------------------ parent


def measure(args: argparse.Namespace, sides: tuple[int, ...], steps: tuple[str, ...]) -> int:
    try:
        check_loadable(
            ModelSpec.from_config(load_config(args.config)), allow_candidate=args.allow_candidate
        )
    except (ModelConfigError, OSError) as exc:
        print(f"{args.config}: {exc}", file=sys.stderr)
        return 2
    target = args.out_dir / f"{args.part}.csv"
    script = Path(__file__).resolve()

    def command_for(side: int, step: Step, result_file: Path) -> list[str]:
        command = [
            sys.executable, "-u", str(script), "child", "--of", args.part,  # -u: keep a crash log
            "--config", str(args.config), "--side", str(side), "--sides", *map(str, sides),
            "--step", step.name, "--result-file", str(result_file),
        ]  # fmt: skip
        return command + (["--allow-candidate"] if args.allow_candidate else [])

    print(f"synthetic images {source_size(sides)}; sides {list(sides)}; steps {list(steps)}")
    print(f"each step in its own process; writing {target}")
    with tempfile.TemporaryDirectory(prefix="rc_s0b_") as workdir:
        outcomes = measure_sides(
            args.part, sides, steps, target, command_for, Path(workdir),
            timeout=args.child_timeout,
        )  # fmt: skip
    failed = [f"side {side} {name}" for side, name, ok in outcomes if not ok]
    print(f"done: {len(outcomes) - len(failed)} of {len(outcomes)} steps ok")
    if failed:
        print("failed (recorded as ..._ok = 0): " + ", ".join(failed))
    return 0


# ------------------------------------------------------------------ child


def _probes(sides: tuple[int, ...]) -> tuple[list, list]:
    from medcompress.models.smoke import probe_questions, synthetic_images

    questions = probe_questions() * 2  # the 8 probes of RC-M1
    return synthetic_images(len(questions), size=source_size(sides)), questions


def _scoring_rows(model: object, images: list, questions: list, prefix: str) -> list[dict]:
    from medcompress.models.smoke import measure_model
    from medcompress.selection.rounds import visual_token_count

    rows, texts = measure_model(model, images, questions, prefix)  # type: ignore[arg-type]
    print(f"{prefix} generation sample: {texts}")
    tokens = visual_token_count(model, images[0], questions[0])  # type: ignore[arg-type]
    if tokens is not None:
        rows.append({"metric": f"{prefix}_visual_tokens", "value": float(tokens)})
    return rows


def _child_rows(args: argparse.Namespace, step: Step) -> list[dict]:
    from medcompress.models.loading import load_model
    from medcompress.models.smoke import cuda_available

    sides = check_sides(args.sides)
    if args.side not in sides:
        raise ValueError(f"side {args.side} is not one of {list(sides)}")
    spec = ModelSpec.from_config(load_config(args.config))
    images, questions = _probes(sides)
    student = args.of == "student"
    device_map = ({"": 0} if cuda_available() else "cpu") if student else "auto"

    def build() -> object:
        return load_model(
            spec,
            device_map=device_map,
            allow_candidate=args.allow_candidate,
            image_longest_side=args.side,
        )

    if step.kind == "score":
        rows = _scoring_rows(build(), images, questions, args.of)
        return rows + [{"metric": ok_metric(args.of, step), "value": 1.0}]
    from medcompress.compression.distillation.batching import TrainExample

    training = load_config("training.yaml")["training"]
    examples = [
        TrainExample(f"s0b-{i}", images[i], questions[i], ("Yes", "No")[i % 2])
        for i in range(step.micro_batch)
    ]
    return lora_step_rows(
        build,
        examples,
        step.micro_batch,
        args.of,
        lora=training["lora"],
        learning_rate=float(training["learning_rate"]),
        gradient_checkpointing=step.gradient_checkpointing,
    )


def run_one_step(args: argparse.Namespace) -> int:
    """The child: measure one (side, step), always hand back rows, exit 0."""
    step = parse_step(args.step)
    if args.of == "teacher" and step.kind != "score":
        raise SystemExit("the teacher has only the step 'score'")
    try:
        rows = _child_rows(args, step)
    except Exception as exc:  # noqa: BLE001 - every failure is a recorded result
        print(f"{args.of} {step.name} at side {args.side} failed: {type(exc).__name__}: {exc}")
        rows = failure_rows(args.of, step, safe_peak())
    write_child_result(args.result_file, rows)
    return 0


# ------------------------------------------------------------------ settings


def _revision_lines(path: Path, model_info: Callable[[str], Any]) -> tuple[ModelSpec, list[str]]:
    spec = ModelSpec.from_config(load_config(path))
    lines = [
        f"== {spec.source} ({path})",
        f"config revision: {spec.revision or REVISION_NOT_PINNED}",
        f"config sha_seen: {spec.sha_seen or NOT_STATED}",
    ]
    try:
        lines.append(f"hub main now: {model_info(spec.source).sha}")
    except Exception as exc:  # noqa: BLE001 - print and go on
        lines.append(f"hub main now: could not read ({type(exc).__name__}: {exc})")
    return spec, lines


def print_settings(args: argparse.Namespace) -> int:
    from huggingface_hub import hf_hub_download, list_repo_files, model_info
    from transformers import AutoProcessor

    teacher = load_config(args.teacher_config)["model"]
    repo, revision = teacher["adapter"], teacher["adapter_revision"]
    files = [
        name
        for name in list_repo_files(repo, revision=revision)
        if name.lower().endswith((".md", ".json", ".yaml", ".yml", ".txt"))
        and Path(name).name not in TOKENIZER_FILES
    ]
    print(f"== {repo} at {revision} (text files: {files or NOT_STATED})")
    for name in files:
        text = Path(hf_hub_download(repo, name, revision=revision)).read_text(errors="replace")
        for line in report_lines(name, matching_lines(text)):
            print(line)
    for path in (args.teacher_config, args.student_config, args.fallback_config):
        spec, lines = _revision_lines(path, model_info)
        print("\n".join(lines))
        if path == args.fallback_config:
            continue  # its revision only: it is not measured here
        if spec.revision is None:
            print(f"{spec.source}: {REVISION_NOT_PINNED}; processor not read")
            continue
        try:
            processor = AutoProcessor.from_pretrained(spec.source, revision=spec.revision)
        except Exception as exc:  # noqa: BLE001 - print and go on
            print(f"processor could not be read: {type(exc).__name__}: {exc}")
            continue
        image_processor = getattr(processor, "image_processor", None)
        print(f"processor at {spec.revision}: {type(image_processor).__name__}")
        for field in PROCESSOR_FIELDS:
            print(f"image_processor.{field}: {getattr(image_processor, field, NOT_STATED)}")
    print(
        "== this repository: images are shrunk so their longest side is at most "
        "inputs.image_longest_side (models/hf.py); max_pixels is passed to the processor "
        "only when a config sets it (none does)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.part == "settings":
        return print_settings(args)
    if args.part == "child":
        return run_one_step(args)
    try:
        sides = check_sides(args.sides)
        steps = check_steps(args.steps, args.part)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return measure(args, sides, steps)


if __name__ == "__main__":
    sys.exit(main())
