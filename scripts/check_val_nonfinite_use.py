#!/usr/bin/env python
"""Where the validation images that are not finite in fp16 are used (CPU; 2026-11-14).

For the given sample ids (one per image, as RC-CAL0's scan lists them), reports whether the
image is among the probe images (clinical, small findings, calibration, blank image), the
mask-audit probes (the other-image partners are probe images too), and the rows of the
free-text set (the seeded
``evaluation.generation_samples`` rows that E1 and E2 answered). No model is loaded.

Example (CPU notebook, medcompress-data attached):
    python scripts/check_val_nonfinite_use.py clbnxtowu6umq079oar8d56lx:e28d18f5f7e5 \\
        clbnxtoncarbp07554f82d7ij:43bda143abf0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)
    from medcompress.cli import attach_inputs
    from medcompress.config import load_config, load_experiment
    from medcompress.data.prepared import load_prepared
    from medcompress.evaluation import free_text
    from medcompress.evaluation.image_reliance import unique_image_probes
    from medcompress.evaluation.runner import PROBE_SAMPLE_SEED, resolve_settings
    from medcompress.inputs import find_shared_data

    config, _ = load_experiment("E2")
    attach_inputs(config, args.data_root or find_shared_data(), ())
    settings = resolve_settings(config)
    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    paths = config["paths"]
    val = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))["val"]
    by_id = {s.sample_id: s for s in val}
    images = {by_id[i].image_path: i for i in args.ids if i in by_id}
    for i in args.ids:
        if i not in by_id:
            print(f"{i}: not a validation row")
    probes = unique_image_probes(val, settings.findings, settings.template,
                                 display_names=settings.display_names)  # fmt: skip
    probe_images = {p.image_path for p in probes}
    masked = {p.image_path for p in probes if getattr(p, "mask_path", None)}
    rows = free_text.sample_qa_rows(val, settings.generation_samples, PROBE_SAMPLE_SEED)
    print(f"validation: {len({s.image_path for s in val})} images, {len(val)} rows; probe images "
          f"{len(probe_images)}; free-text rows {len(rows)}")  # fmt: skip
    for image, sample_id in images.items():
        in_rows = sum(r.image_path == image for r in rows)
        # the other-image partners are drawn from the probe images (image_reliance)
        print(f"{sample_id}: probe image (and so a possible other-image partner) "
              f"{image in probe_images}; mask-audit {image in masked}; free-text rows {in_rows} "
              f"of {len(rows)} "
              f"(all its rows: {sum(s.image_path == image for s in val)})")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
