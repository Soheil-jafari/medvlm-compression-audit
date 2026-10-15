#!/usr/bin/env python
"""GPU smoke test of the quantization toolchain (run card RC-Q0).

Tries every GPTQModel kernel class on our 4- and 8-bit codes, then quantizes the student
with GPTQ (w4, w8) and LLM.int8 (w8a8) on synthetic calibration images, runs the
low-precision kernel check, scores probes and measures memory. No dataset is used. Rows are
written to results/smoke/RC-Q0.csv; the versions and the chosen kernel class are printed for
the run card.

Example (Kaggle, one T4):
    python scripts/smoke_quantization.py --allow-candidate
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.config import load_config  # noqa: E402
from medcompress.utils.io import append_results  # noqa: E402

DEFAULT_STUDENT = Path(__file__).resolve().parents[1] / "configs/models/qwen2_5_vl_3b_student.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student-config", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--out", type=Path, default=Path("results/smoke/RC-Q0.csv"))
    parser.add_argument("--image-longest-side", type=int, default=448)
    parser.add_argument("--calibration", type=int, default=16, help="synthetic calibration images")
    parser.add_argument("--kernel-class", default=None, help="default: fastest class that passes")
    parser.add_argument("--allow-candidate", action="store_true", help="the student is a candidate")
    return parser


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in ("torch", "transformers", "peft", "gptqmodel", "bitsandbytes"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _free() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from medcompress.compression.quantization.kernels import DECODER_TARGETS, kernel_report
    from medcompress.compression.quantization.ptq import (
        PTQRecipe,
        apply_llm_int8,
        calibration_batches,
        gptq_quantize_decoder,
        install_quantized,
    )
    from medcompress.compression.quantization.smoke import (
        answers_agree,
        choose_kernel_class,
        discover_kernel_classes,
        kernel_rows,
        row,
        write_synthetic_calibration,
    )
    from medcompress.evaluation.efficiency import state_bytes
    from medcompress.models.loading import ModelSpec, load_model
    from medcompress.models.smoke import (
        peak_gpu_bytes,
        probe_questions,
        reset_peak,
        synthetic_images,
    )
    from medcompress.utils.hardware import describe_hardware

    print(
        json.dumps({"hardware": describe_hardware().to_dict(), "versions": _versions()}, indent=2)
    )
    rows = []
    classes = discover_kernel_classes()
    print(f"{len(classes)} GPTQModel kernel classes found")
    rows += kernel_rows(classes)
    chosen = args.kernel_class
    if chosen is None:
        chosen = choose_kernel_class(rows, classes)  # fastest at w4 among those passing w4 and w8
        if chosen is None:
            print("no kernel class passed both w4 and w8; stopping before the model steps")
            return _write(rows, args.out)
    print(f"chosen kernel class: {chosen}")

    questions = probe_questions() * 4
    images = synthetic_images(len(questions))
    workdir = Path(tempfile.mkdtemp(prefix="rcq0_"))
    calibration = write_synthetic_calibration(
        workdir,
        synthetic_images(args.calibration, seed=1),
        (probe_questions() * 8)[: args.calibration],
    )
    spec = ModelSpec.from_config(load_config(args.student_config))

    def student() -> object:
        return load_model(
            spec,
            device_map={"": 0},
            allow_candidate=args.allow_candidate,
            image_longest_side=args.image_longest_side,
        )

    def measure(model: object, prefix: str) -> tuple[list, list]:
        reset_peak()
        scores = model.score_yes_no(images, questions)  # type: ignore[attr-defined]
        return scores, [
            row(f"{prefix}_peak_bytes", peak_gpu_bytes() or 0),
            row(f"{prefix}_state_bytes", state_bytes(model)),
        ]

    model = student()
    reference, measured = measure(model, "fp16")
    rows += measured
    del model
    _free()

    for name, bits in (("w4", 4), ("w8", 8)):
        recipe = PTQRecipe(name, "gptq", bits, None, 128, DECODER_TARGETS, args.calibration)
        model = student()
        started = time.perf_counter()
        batches = calibration_batches(model, calibration, workdir)
        records = gptq_quantize_decoder(model, batches, recipe)
        rows.append(row(f"{name}_gptq_seconds", time.perf_counter() - started))
        install_quantized(model, records, recipe, kernel_class=chosen)
        report = kernel_report(model)
        print(f"{name}: kernel check {report.ok}, classes {report.classes}, {report.problems[:3]}")
        rows.append(row(f"{name}_kernels_ok", report.ok))
        scores, measured = measure(model, name)
        rows += measured
        rows.append(row(f"{name}_agreement_with_fp16", answers_agree(scores, reference)))
        del model, batches, records
        _free()

    recipe = PTQRecipe("w8a8", "llm_int8", 8, 8, None, DECODER_TARGETS, None)
    model = student()
    apply_llm_int8(model, recipe)
    report = kernel_report(model)
    print(f"w8a8: kernel check {report.ok}, classes {report.classes}, {report.problems[:3]}")
    rows.append(row("w8a8_kernels_ok", report.ok))
    scores, measured = measure(model, "w8a8")
    rows += measured
    rows.append(row("w8a8_agreement_with_fp16", answers_agree(scores, reference)))
    return _write(rows, args.out)


def _write(rows: list, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    append_results(rows, out)
    print(f"wrote {len(rows)} rows to {out}")
    for r in rows:
        print(f"{r['metric']:<56} {r['value']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
