#!/usr/bin/env python
"""GPU smoke test of the teacher and a student (run card RC-M1).

Loads the teacher and a student in the planned precision, checks that their
tokenizers are identical, scores yes/no probes on synthetic images, times scoring and
generation, removes one decoder layer physically and records the measured memory
drop. No dataset is used. Rows are written to results/smoke/RC-M1.csv.

Example (Kaggle, 2 x T4):
    python scripts/smoke_models.py \\
        --teacher-config configs/models/qwen2_5_vl_7b_teacher.yaml \\
        --student-config configs/models/qwen2_5_vl_3b_student.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.config import load_config  # noqa: E402
from medcompress.utils.io import append_results  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/smoke/RC-M1.csv"))
    parser.add_argument(
        "--teacher-bits",
        type=int,
        choices=(4, 8),
        default=None,
        help="load the teacher in 4 or 8 bit (needs bitsandbytes); default is fp16",
    )
    parser.add_argument("--image-longest-side", type=int, default=448)
    parser.add_argument(
        "--layer", type=int, default=None, help="student layer to remove (default: middle)"
    )
    parser.add_argument(
        "--allow-candidate", action="store_true", help="the student is a selection candidate"
    )
    parser.add_argument(
        "--allow-remote-code", action="store_true", help="only after reading the model code"
    )
    parser.add_argument(
        "--skip-teacher", action="store_true", help="student only (no tokenizer comparison)"
    )
    return parser


def _free() -> None:
    gc.collect()
    from medcompress.models.smoke import cuda_available

    if cuda_available():
        import torch

        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from medcompress.models.loading import ModelSpec, load_model
    from medcompress.models.smoke import (
        cuda_available,
        measure_layer_removal,
        measure_model,
        probe_questions,
        reset_peak,
        synthetic_images,
    )
    from medcompress.utils.hardware import describe_hardware

    print(json.dumps(describe_hardware().to_dict(), indent=2))
    questions = probe_questions() * 2  # 8 probes
    images = synthetic_images(len(questions))
    rows: list[dict] = []
    teacher_fingerprint = None

    if not args.skip_teacher:
        spec = ModelSpec.from_config(load_config(args.teacher_config))
        quant = {"bits": args.teacher_bits} if args.teacher_bits else None
        reset_peak()
        teacher = load_model(
            spec,
            device_map="auto",
            quantization=quant,
            allow_candidate=args.allow_candidate,
            allow_remote_code=args.allow_remote_code,
            image_longest_side=args.image_longest_side,
        )
        teacher_rows, texts = measure_model(teacher, images, questions, "teacher")
        rows += teacher_rows
        print("teacher generation:", texts)
        teacher_fingerprint = teacher.tokenizer_fingerprint()
        del teacher
        _free()

    spec = ModelSpec.from_config(load_config(args.student_config))
    device_map = {"": 0} if cuda_available() else "cpu"
    reset_peak()
    student = load_model(
        spec,
        device_map=device_map,
        allow_candidate=args.allow_candidate,
        allow_remote_code=args.allow_remote_code,
        image_longest_side=args.image_longest_side,
    )
    if teacher_fingerprint is not None:
        same = student.tokenizer_fingerprint() == teacher_fingerprint
        rows.append(
            {
                "experiment": "RC-M1",
                "budget": "",
                "seed": 0,
                "metric": "tokenizers_identical",
                "value": float(same),
            }
        )
        print("tokenizers identical:", same)
    student_rows, texts = measure_model(student, images, questions, "student")
    rows += student_rows
    print("student generation:", texts)
    layer = (
        args.layer
        if args.layer is not None
        else student.module.config.get_text_config().num_hidden_layers // 2
    )
    rows += measure_layer_removal(student, layer, images, questions, "student")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    append_results(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")
    for row in rows:
        print(f"{row['metric']:<48} {row['value']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
