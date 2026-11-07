"""The software and hardware a run used, written next to its results (2026-11-06).

The Kaggle image changed between the teacher's runs and E1's (Torch 2.11.0+cu128 to
2.11.0+cu130, TensorFlow 2.20 to 2.21), so every stage records what it ran on:
``results/<run dir>/environment.json``, one entry per stage. Only versions and device names
are written, never paths, tokens or account names.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata, util
from pathlib import Path
from typing import Any

ENVIRONMENT_FILE = "environment.json"
PACKAGES: tuple[str, ...] = (
    "torch", "transformers", "peft", "accelerate", "optimum", "torchao", "bitsandbytes",
    "safetensors", "numpy", "tensorflow",
)  # fmt: skip


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def nvidia_driver() -> str | None:
    """The NVIDIA driver version from ``nvidia-smi`` (None without a GPU or the tool)."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return lines[0] if done.returncode == 0 and lines else None


def environment_record() -> dict[str, Any]:
    """Python, the package versions (the torch build string included), CUDA, cuDNN, the GPUs
    and the driver."""
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "nvidia_driver": nvidia_driver(),
        "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").is_dir(),
    }
    if util.find_spec("torch") is not None:
        import torch

        record["torch_build"] = torch.__version__
        record["torch_cuda"] = torch.version.cuda
        record["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        record["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return record


def record_environment(
    results_dir: Path, stage: str, record: Mapping[str, Any] | None = None
) -> Path:
    """Add ``stage``'s record to ``<results_dir>/environment.json`` (other stages are kept)."""
    path = Path(results_dir) / ENVIRONMENT_FILE
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    data[stage] = dict(record if record is not None else environment_record())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def summary_line(record: Mapping[str, Any]) -> str:
    """One printed line: torch build, CUDA, driver, transformers, peft."""
    packages = record.get("packages", {})
    return (f"environment: torch {record.get('torch_build') or packages.get('torch')}, CUDA "
            f"{record.get('torch_cuda')}, driver {record.get('nvidia_driver')}, transformers "
            f"{packages.get('transformers')}, peft {packages.get('peft')}")  # fmt: skip
