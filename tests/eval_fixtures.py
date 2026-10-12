"""A tiny labelled dataset and an evaluation config for the stage tests (no model needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from medcompress.data.base import VQASample

TEMPLATE = "Is there a {finding} in this image? Answer yes or no."
SIZE = (64, 48)
BLOB = 4  # blob heights; widths are set per sample so the lesion areas differ


def write_dataset(root: Path) -> list[VQASample]:
    """12 images: even ones show a bright blob of growing size (and have a mask)."""
    samples = []
    for i in range(12):
        positive = i % 2 == 0
        array = np.random.default_rng(i).integers(20, 120, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
        mask_rel, area = None, None
        if positive:
            width = 3 + 3 * (i // 2)  # 3, 6, 9, 12, 15, 18 pixels wide
            array[10 : 10 + BLOB, 10 : 10 + width] = 255
            mask = np.zeros(SIZE[::-1], dtype=np.uint8)
            mask[10 : 10 + BLOB, 10 : 10 + width] = 255
            mask_rel = f"masks/{i}.png"
            (root / "masks").mkdir(exist_ok=True)
            Image.fromarray(mask).save(root / mask_rel)
            area = BLOB * width / (SIZE[0] * SIZE[1])
        (root / "images").mkdir(exist_ok=True)
        Image.fromarray(array).save(root / f"images/{i}.png")
        answer = "yes there is a polyp" if positive else "no polyp is visible"
        findings = {"polyp": positive, "bleeding": i % 3 == 0}
        for suffix in ("", "b"):
            samples.append(
                VQASample(
                    f"s{i}{suffix}",
                    "d",
                    f"images/{i}.png",
                    "What is seen?",
                    answer,
                    findings=findings,
                    mask_path=mask_rel,
                    lesion_area=area,
                )  # fmt: skip
            )
    return samples


def eval_config(root: Path, *, domain: dict | None = None, **evaluation: object) -> dict:
    domain_section = {
        "findings": ["polyp", "bleeding"],
        "prevalence_groups": {"common": ["polyp"], "medium": [], "rare": ["bleeding"]},
        "probe": {"template": TEMPLATE, "display_names": {}},
        "small_findings": {"thresholds": [0.006, 0.013, 0.018]},
        **(domain or {}),
    }
    (root / "domain.yaml").write_text(yaml.safe_dump({"domain": domain_section}))
    (root / "weights.yaml").write_text(yaml.safe_dump({"preregistration": {"frozen_on": "TODO"}}))
    return {
        "seeds": {"data_split": 0},
        "paths": {
            "data_root": str(root),
            "results_root": str(root / "results"),
            "output_root": str(root / "outputs"),
            "private_root": str(root / "private"),
            "checkpoint_root": str(root / "checkpoints"),
        },
        "components": {
            "domain": str(root / "domain.yaml"),
            "finding_weights": str(root / "weights.yaml"),
            "datasets": {"train_qa": "data/kvasir_vqa_x1.yaml"},
        },
        "evaluation": {
            "suites": [
                "clinical", "small_findings", "rare_findings", "image_reliance", "calibration",
                "hallucination", "language",
            ],
            "split": "val",
            "bootstrap_resamples": 60,
            "calibration_bins": 5,
            "ci_level": 0.95,
            "max_new_tokens": 8,
            "generation_samples": None,
            **evaluation,
        },
        "image_reliance_check": {
            "fill": 128,
            "mask_findings": ["polyp"],
            "max_probes_per_finding": None,
            "max_mask_probes": None,
        },
    }  # fmt: skip
