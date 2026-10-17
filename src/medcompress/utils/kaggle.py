"""Session checks for unattended runs on Kaggle (``scripts/run_card.py``).

Before a card runs, the installed library versions are compared with the pins of
``requirements/train.txt`` (``requirements/base.txt`` for a CPU card), including the two
lessons of RC-M1 and RC-S0: torchao must be 0.16.0 or newer and optimum 1.24.0 or newer, or
peft 0.21.2 fails. The number of visible GPUs must match the card.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

_PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(==|>=)\s*([0-9][0-9A-Za-z.+\-]*)")
QUANTIZATION_PACKAGES = frozenset({"gptqmodel", "bitsandbytes"})


@dataclass(frozen=True)
class Pin:
    name: str
    op: str
    version: str


def parse_requirements(path: Path) -> list[Pin]:
    """The ``name==x`` and ``name>=x`` lines of a requirements file (``-r`` files followed)."""
    pins: list[Pin] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text.startswith("-r "):
            pins += parse_requirements(Path(path).parent / text[3:].strip())
            continue
        match = _PIN.match(text)
        if match:
            pins.append(Pin(match.group(1).lower(), match.group(2), match.group(3)))
    return pins


def version_tuple(text: str) -> tuple[int, ...]:
    """``2.11.0+cu128`` -> ``(2, 11, 0)``: the leading numeric parts."""
    parts = []
    for piece in text.split("+")[0].split("."):
        digits = re.match(r"\d+", piece)
        if digits is None:
            break
        parts.append(int(digits.group()))
    return tuple(parts)


def satisfied(installed: str, pin: Pin) -> bool:
    """True if the installed version meets the pin (``==``: same numbers; ``>=``: at least)."""
    have, want = version_tuple(installed), version_tuple(pin.version)
    if pin.op == "==":
        return have == want
    return have >= want


def installed_version(name: str) -> str | None:
    """The installed version of a package, or ``None``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_versions(
    pins: list[Pin],
    *,
    quantization: bool,
    version_of: Callable[[str], str | None] = installed_version,
) -> tuple[list[str], list[str]]:
    """(problems, report lines) for the pins; quantization packages only if the card needs them."""
    problems, lines = [], []
    for pin in pins:
        if pin.name in QUANTIZATION_PACKAGES and not quantization:
            continue
        have = version_of(pin.name)
        lines.append(f"{pin.name} {have or 'not installed'} (needs {pin.op}{pin.version})")
        if have is None:
            problems.append(f"{pin.name} is not installed (needs {pin.op}{pin.version})")
        elif not satisfied(have, pin):
            hint = " (run `pip uninstall -y torchao` first, RC-M1)" if pin.name == "torchao" else ""
            problems.append(f"{pin.name} {have} does not meet {pin.op}{pin.version}{hint}")
    return problems, lines


def gpu_report() -> tuple[int, list[str]]:
    """(number of visible GPUs, their names), read in a child process (no CUDA here)."""
    code = (
        "import torch\n"
        "n = torch.cuda.device_count() if torch.cuda.is_available() else 0\n"
        "print(n)\n"
        "for i in range(n): print(torch.cuda.get_device_name(i))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300
        ).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        return 0, []
    return int(out[0] or 0), [line for line in out[1:] if line.strip()]


def check_gpus(needed: int, visible: int) -> list[str]:
    """Problems with the GPUs: a GPU card needs at least ``needed`` visible GPUs."""
    if needed and visible < needed:
        return [f"the card needs {needed} GPU(s), {visible} visible (Kaggle: GPU T4 x2)"]
    return []


def find_kvasir_seg(source: Path, *, max_depth: int = 6) -> Path:
    """The folder under ``source`` that holds both ``images/`` and ``masks/`` (Kvasir-SEG)."""
    level = [Path(source)]
    for _ in range(max_depth + 1):
        found = [p for p in level if (p / "images").is_dir() and (p / "masks").is_dir()]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise FileNotFoundError(f"more than one Kvasir-SEG folder under {source}: {found}")
        level = [c for p in level if p.is_dir() for c in sorted(p.iterdir()) if c.is_dir()]
    raise FileNotFoundError(f"no folder with images/ and masks/ under {source}")


def copy_kvasir_seg(source: Path, data_root: Path) -> Mapping[str, int]:
    """Copy the attached Kvasir-SEG images and masks to ``<data_root>/kvasir_seg/``."""
    seg = find_kvasir_seg(source)
    target = Path(data_root) / "kvasir_seg"
    counts = {}
    for name in ("images", "masks"):
        if (target / name).exists():
            raise FileExistsError(f"{target / name} exists already; start from a fresh session")
        shutil.copytree(seg / name, target / name)
        counts[name] = sum(1 for p in (target / name).iterdir() if p.is_file())
    print(f"copied Kvasir-SEG from {seg}: {counts['images']} images, {counts['masks']} masks")
    return counts
