#!/usr/bin/env python
"""Image-side measurement on synthetic images (run card RC-S0b): student, teacher, settings.

  student   per side: scoring, visual tokens, peak memory, one LoRA step at each micro-batch
            (one T4: CUDA_VISIBLE_DEVICES=0); rows to <out-dir>/student.csv
  teacher   per side: scoring, visual tokens, peak memory, generation speed (two T4s);
            rows to <out-dir>/teacher.csv
  settings  prints the image settings the published teacher adapter and the Qwen processors
            state (adapter README and config files, processor defaults); nothing is written

No dataset is used. Rows are tagged RC-S0b:side<N> and never touch the RC-M1 or RC-S0 files.

Example (Kaggle):
    CUDA_VISIBLE_DEVICES=0 python scripts/measure_image_sides.py student \\
        --config configs/models/qwen2_5_vl_3b_student.yaml --allow-candidate
    python scripts/measure_image_sides.py teacher --config configs/models/qwen2_5_vl_7b_teacher.yaml
    python scripts/measure_image_sides.py settings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.config import load_config  # noqa: E402
from medcompress.models.image_sides import (  # noqa: E402
    DEFAULT_MICRO_BATCHES,
    DEFAULT_SIDES,
    NOT_STATED,
    check_sides,
    free_memory,
    lora_step_rows,
    matching_lines,
    report_lines,
    source_size,
    tag_rows,
)
from medcompress.utils.io import upsert_results  # noqa: E402

TEACHER_CONFIG = Path("configs/models/qwen2_5_vl_7b_teacher.yaml")
STUDENT_CONFIG = Path("configs/models/qwen2_5_vl_3b_student.yaml")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
PROCESSOR_FIELDS = ("min_pixels", "max_pixels", "patch_size", "merge_size", "size")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="part", required=True)
    for part in ("student", "teacher"):
        p = sub.add_parser(part, help=f"measure the {part} at every side")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--sides", type=int, nargs="+", default=list(DEFAULT_SIDES))
        p.add_argument("--out-dir", type=Path, default=Path("results/smoke/RC-S0b"))
        p.add_argument("--allow-candidate", action="store_true", help="the model is a candidate")
        if part == "student":
            p.add_argument(
                "--micro-batches", type=int, nargs="+", default=list(DEFAULT_MICRO_BATCHES)
            )
    settings = sub.add_parser("settings", help="print the stated image settings, write nothing")
    settings.add_argument("--teacher-config", type=Path, default=TEACHER_CONFIG)
    settings.add_argument("--student-config", type=Path, default=STUDENT_CONFIG)
    return parser


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


def measure(args: argparse.Namespace) -> int:
    from medcompress.models.loading import ModelSpec, load_model
    from medcompress.models.smoke import cuda_available

    sides = check_sides(args.sides)
    images, questions = _probes(sides)
    spec = ModelSpec.from_config(load_config(args.config))
    student = args.part == "student"
    device_map = ({"": 0} if cuda_available() else "cpu") if student else "auto"
    target = args.out_dir / f"{args.part}.csv"
    print(f"synthetic images {source_size(sides)}; sides {list(sides)}; writing {target}")
    for side in sides:

        def build(side: int = side) -> object:
            return load_model(
                spec,
                device_map=device_map,
                allow_candidate=args.allow_candidate,
                image_longest_side=side,
            )

        model = build()
        rows = _scoring_rows(model, images, questions, args.part)
        del model
        free_memory()
        if student:
            from medcompress.compression.distillation.batching import TrainExample

            training = load_config("training.yaml")["training"]
            examples = [
                TrainExample(f"s0b-{i}", images[i], questions[i], ("Yes", "No")[i % 2])
                for i in range(max(args.micro_batches))
            ]
            for micro_batch in sorted(args.micro_batches):
                rows += lora_step_rows(
                    build,
                    examples,
                    micro_batch,
                    "student",
                    lora=training["lora"],
                    learning_rate=float(training["learning_rate"]),
                )
        upsert_results(tag_rows(rows, side), target)  # each side is saved as soon as it is done
        print(f"side {side}: {len(rows)} rows")
    return 0


def print_settings(args: argparse.Namespace) -> int:
    from huggingface_hub import hf_hub_download, list_repo_files
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
    for path in (args.teacher_config, args.student_config):
        model = load_config(path)["model"]
        revision = model.get("revision") or model.get("sha_seen")
        processor = AutoProcessor.from_pretrained(model["source"], revision=revision)
        image_processor = getattr(processor, "image_processor", None)
        print(f"== {model['source']} at {revision}: {type(image_processor).__name__}")
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
    try:
        check_sides(args.sides)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return measure(args)


if __name__ == "__main__":
    sys.exit(main())
