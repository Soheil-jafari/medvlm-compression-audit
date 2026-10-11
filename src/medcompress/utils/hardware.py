"""Describe the hardware a run executes on (recorded next to every efficiency result)."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class HardwareInfo:
    """Static description of the current machine."""

    platform: str
    python: str
    cpu_count: int | None
    gpus: list[dict[str, Any]] = field(default_factory=list)
    torch_version: str | None = None
    cuda_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_hardware() -> HardwareInfo:
    """Collect platform, Python, CPU and (if torch is installed) GPU information.

    torch is imported lazily; without it ``gpus`` is empty.
    """
    gpus: list[dict[str, Any]] = []
    torch_version = cuda_version = None
    try:
        import torch
    except ImportError:
        pass
    else:
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpus.append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_bytes": props.total_memory,
                        "capability": f"{props.major}.{props.minor}",
                    }
                )
    return HardwareInfo(
        platform=platform.platform(),
        python=sys.version.split()[0],
        cpu_count=os.cpu_count(),
        gpus=gpus,
        torch_version=torch_version,
        cuda_version=cuda_version,
    )
