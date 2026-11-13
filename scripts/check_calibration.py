#!/usr/bin/env python
"""The fp16 calibration check of one seed (amendment of 2026-11-13; one model per call).

Walks the seeded, stratified calibration order and runs each leading image's calibration batch
(image, question, ground-truth answer: what GPTQ and the criteria read) through the model in
fp16; an image with any non-finite value (vision tower output, a decoder layer's output, the
logits) is replaced by the next eligible image, until the first ``--size`` images are finite.
Writes ``results/calibration/seed<k>_replaced.csv`` (commit it). The student is E2 of the seed
(``--exp E3``, the parent of every student arm); the teacher is E9's parent (``--exp E9``).

Examples (Kaggle):
    python scripts/check_calibration.py --exp E3 --seed 0 --size 256   # student, A4's largest
    python scripts/check_calibration.py --exp E9 --seed 0 --size 128   # teacher (T4 x2)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_context  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, required=True, help="images to check (A4: 256)")
    args = parser.parse_args(argv)
    if args.exp not in ("E3", "E9"):
        parser.error("use --exp E3 (the student, E2 of the seed) or --exp E9 (the teacher)")
    from medcompress.compression.pruning.stage import load_train_samples
    from medcompress.compression.quantization.ptq import calibration_batches, load_parent
    from medcompress.data.calibration import find_replacements, forward_is_finite, write_check

    args.variant = args.variant or ("w4" if args.exp == "E3" else None)
    run = run_context(args)
    assert run is not None
    config = resolve_config(args)
    model_name = "teacher" if args.exp == "E9" else "student"
    train = load_train_samples(config)
    model = load_parent(config, run, int(config["inputs"]["image_longest_side"]))
    data_root = Path(config["paths"]["data_root"])
    checked = 0

    def is_finite(sample: object) -> bool:
        nonlocal checked
        checked += 1
        (batch,) = calibration_batches(model, [sample], data_root)  # type: ignore[list-item]
        return forward_is_finite(model, batch)

    subset, replaced = find_replacements(train, args.size, run.seed, is_finite)
    path = write_check(config["paths"]["results_root"], run.seed, model_name, len(subset), replaced)
    print(f"calibration check seed {run.seed} ({model_name}): {checked} images checked, "
          f"{len(replaced)} replaced {replaced}; the first {len(subset)} are finite; wrote "
          f"{path} (commit it)")  # fmt: skip
    return 0


if __name__ == "__main__":
    main_guard(main)
