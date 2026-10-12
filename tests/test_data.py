"""Data-layer helpers that already run, and a guard against ignoring source folders."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from medcompress.data.base import VQASample
from medcompress.data.probes import format_probe_question
from medcompress.data.splits import CALIBRATION_SIZES

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "Is there a {finding} in this image? Answer yes or no."


def test_format_probe_question() -> None:
    assert format_probe_question(TEMPLATE, "polyp") == (
        "Is there a polyp in this image? Answer yes or no."
    )
    assert format_probe_question(TEMPLATE, "  ulcer ") == (
        "Is there a ulcer in this image? Answer yes or no."
    )


@pytest.mark.parametrize(
    ("template", "finding"), [("Is there a lesion?", "polyp"), (TEMPLATE, "  ")]
)
def test_format_probe_question_rejects_bad_input(template: str, finding: str) -> None:
    with pytest.raises(ValueError):
        format_probe_question(template, finding)


def test_vqa_sample_defaults() -> None:
    sample = VQASample("s1", "kvasir_vqa_x1", "img/1.jpg", "Is there a polyp?", "yes")
    assert sample.patient_id is None and sample.findings == {} and sample.mask_path is None


def test_calibration_sizes_match_ablation_a4() -> None:
    assert CALIBRATION_SIZES == (32, 64, 128, 256)


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(), reason="needs a git checkout"
)
@pytest.mark.parametrize(
    "path", ["src/medcompress/data/__init__.py", "configs/data/kvasir_vqa_x1.yaml"]
)
def test_source_folders_named_data_are_not_ignored(path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO_ROOT, capture_output=True, check=False
    )
    assert result.returncode == 1, f"{path} is matched by .gitignore"


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(), reason="needs a git checkout"
)
@pytest.mark.parametrize("path", ["data/x.jpg", "checkpoints/model.safetensors", ".env"])
def test_data_and_secrets_are_ignored(path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO_ROOT, capture_output=True, check=False
    )
    assert result.returncode == 0, f"{path} should be ignored"
