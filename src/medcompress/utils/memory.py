"""Host and GPU memory readings, freeing host memory, and a step-over-step growth check.

Added 2026-11-14 after two kills:

* RC-G2 session 3: the first-order pass of E7's scoring ran out of GPU memory on one T4
  (14.18 GiB allocated); nothing logged the memory step by step.
* RC-SMOKE at a237c5c: E7's stage 05 was killed with exit -9 (the kernel's out-of-memory killer)
  after loading the 3B student many times in one process; nothing logged host RAM.

Every reading prints as one ``MEDCOMPRESS_MEM`` line, so a log shows where memory went.
Host readings come from ``/proc`` (Linux, as on Kaggle) and are ``None`` elsewhere.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path

GIB = float(1 << 30)


def _proc_value(path: str, key: str) -> int | None:
    """A ``kB`` value of a ``/proc`` file in bytes (``None`` when it is not there)."""
    try:
        with Path(path).open(encoding="ascii") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def host_rss_bytes() -> int | None:
    """Resident memory of this process now."""
    return _proc_value("/proc/self/status", "VmRSS")


def host_peak_rss_bytes() -> int | None:
    """The highest resident memory of this process so far."""
    return _proc_value("/proc/self/status", "VmHWM")


def host_available_bytes() -> int | None:
    """Memory the kernel can still hand out without swapping (``MemAvailable``)."""
    return _proc_value("/proc/meminfo", "MemAvailable")


def gpu_reserved_bytes() -> int | None:
    """This process's reserved GPU memory (its allocator's cache included), summed over GPUs."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None
    return sum(int(torch.cuda.memory_reserved(d)) for d in range(torch.cuda.device_count()))


def gpu_bytes() -> tuple[int, int] | None:
    """``(allocated, peak allocated)`` summed over the visible GPUs (``None`` without CUDA)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement of the stages
        return None
    if not torch.cuda.is_available():
        return None
    devices = range(torch.cuda.device_count())
    return (
        sum(int(torch.cuda.memory_allocated(d)) for d in devices),
        sum(int(torch.cuda.max_memory_allocated(d)) for d in devices),
    )


def _gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / GIB:.2f}"


def memory_line(label: str) -> str:
    """``MEDCOMPRESS_MEM <label>: gpu allocated .. GiB, peak .. GiB; host rss .. GiB, peak ..
    GiB, available .. GiB``."""
    gpu = gpu_bytes()
    allocated, peak = (None, None) if gpu is None else gpu
    return (
        f"MEDCOMPRESS_MEM {label}: gpu allocated {_gib(allocated)} GiB, reserved "
        f"{_gib(gpu_reserved_bytes())} GiB, peak {_gib(peak)} GiB; "
        f"host rss {_gib(host_rss_bytes())} GiB, peak {_gib(host_peak_rss_bytes())} GiB, "
        f"available {_gib(host_available_bytes())} GiB"
    )


def print_memory(label: str) -> None:
    """Print :func:`memory_line` (flushed, so a kill does not swallow it)."""
    print(memory_line(label), flush=True)


def trim_host_memory() -> None:
    """Give freed host memory back to the system: collect garbage, empty PyTorch's pinned-memory
    cache and ask glibc to return free heap pages (``malloc_trim``).

    Freed Python and PyTorch objects stay in the process's heap until glibc returns the pages;
    a heap fragmented by many model loads keeps growing although nothing is referenced
    (2026-11-14). None of this changes a tensor.
    """
    gc.collect()
    try:
        import torch

        empty_host = getattr(torch._C, "_host_emptyCache", None)  # pinned host blocks
        if empty_host is not None and torch.cuda.is_available():
            empty_host()
    except ImportError:  # pragma: no cover - torch is a hard requirement of the stages
        pass
    if os.name == "posix":
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):  # not glibc (macOS, musl): nothing to trim
            pass


class MemoryGrowthError(RuntimeError):
    """Memory grew step over step (the scale check fails on it)."""


@dataclass
class GrowthCheck:
    """Readings of one quantity after each step of a loop, and the rule that fails on growth.

    The first ``warmup`` readings are skipped (caches, the allocator's first blocks, lazily
    built kernels). After that, a reading more than ``tolerance_bytes`` above the first kept
    reading counts as growth, and growth in more than half of the later readings fails: a
    one-off spike (a larger image batch) passes, a quantity that keeps climbing does not.
    """

    name: str
    warmup: int = 2
    tolerance_bytes: int = 256 << 20
    readings: list[int] = field(default_factory=list)

    def add(self, value: int | None) -> None:
        """Record one reading (``None``, e.g. no GPU, is skipped)."""
        if value is not None:
            self.readings.append(int(value))

    def kept(self) -> list[int]:
        return self.readings[self.warmup :]

    def growth_bytes(self) -> int:
        """Last kept reading minus the first kept one (0 with fewer than two)."""
        kept = self.kept()
        return 0 if len(kept) < 2 else kept[-1] - kept[0]

    def grows(self) -> bool:
        kept = self.kept()
        if len(kept) < 2:
            return False
        above = sum(1 for v in kept[1:] if v - kept[0] > self.tolerance_bytes)
        return above > (len(kept) - 1) / 2

    def summary(self) -> str:
        kept = self.kept()
        if not kept:
            return f"{self.name}: no readings"
        return (
            f"{self.name}: {len(self.readings)} readings, first kept {kept[0] / GIB:.2f} GiB, "
            f"last {kept[-1] / GIB:.2f} GiB, max {max(kept) / GIB:.2f} GiB, growth "
            f"{self.growth_bytes() / GIB:+.2f} GiB ({'GROWS' if self.grows() else 'flat'})"
        )

    def require_flat(self) -> None:
        """Raise :class:`MemoryGrowthError` when :meth:`grows`."""
        if self.grows():
            raise MemoryGrowthError(self.summary())


@dataclass
class WatchedRun:
    """A finished child process and the host memory seen while it ran."""

    returncode: int
    stdout: str
    stderr: str
    peak_rss: int | None  # the child's highest resident memory (sampled)
    min_available: int | None  # the lowest MemAvailable of the machine while it ran


def _rss_of(pid: int) -> int | None:
    return _proc_value(f"/proc/{pid}/status", "VmRSS")


def run_watched(command: list[str], env: dict[str, str] | None = None,
                interval: float = 1.0) -> WatchedRun:  # fmt: skip
    """Run ``command`` to the end and sample its resident memory and the machine's available
    memory every ``interval`` seconds (RC-SMOKE, 2026-11-14).

    Sampling from outside works even when the kernel kills the child (exit -9 leaves no last
    line of its own). The output goes to temporary files, so a long log cannot block the pipe.
    """
    import subprocess
    import tempfile
    import time

    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        child = subprocess.Popen(command, env=env, stdout=out, stderr=err, text=True)
        peak: int | None = None
        lowest: int | None = None
        while True:
            rss, available = _rss_of(child.pid), host_available_bytes()
            if rss is not None:
                peak = rss if peak is None else max(peak, rss)
            if available is not None:
                lowest = available if lowest is None else min(lowest, available)
            if child.poll() is not None:
                break
            time.sleep(interval)
        out.seek(0)
        err.seek(0)
        return WatchedRun(int(child.returncode), out.read(), err.read(), peak, lowest)


def live_cuda_tensors(limit: int = 5) -> list[tuple[int, tuple[int, ...], str, list[str]]]:
    """The largest CUDA tensors this process still holds: ``(bytes, shape, dtype, the types of
    the objects that refer to them)``, largest first (2026-11-14: to name what keeps GPU memory
    after a release)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement of the stages
        return []
    if not torch.cuda.is_available():
        return []
    gc.collect()
    found = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                found.append(obj)
        except Exception:  # noqa: BLE001 - some objects refuse attribute access
            continue
    found.sort(key=lambda t: t.numel() * t.element_size(), reverse=True)
    out = []
    for tensor in found[:limit]:
        owners = sorted({type(r).__name__ for r in gc.get_referrers(tensor)} - {"list", "frame"})
        out.append((tensor.numel() * tensor.element_size(), tuple(tensor.shape),
                    str(tensor.dtype), owners[:6]))  # fmt: skip
    return out


def report_leftover_gpu(what: str, limit_bytes: int = 1 << 30) -> int:
    """After a release: print this process's allocated and reserved GPU memory, and, above
    ``limit_bytes``, the largest live CUDA tensors (``MEDCOMPRESS_LEFTOVER`` line). Returns the
    allocated bytes (0 without CUDA)."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return 0
    if not torch.cuda.is_available():
        return 0
    allocated = sum(int(torch.cuda.memory_allocated(d)) for d in range(torch.cuda.device_count()))
    reserved = sum(int(torch.cuda.memory_reserved(d)) for d in range(torch.cuda.device_count()))
    line = (f"MEDCOMPRESS_LEFTOVER {what}: allocated {allocated / GIB:.2f} GiB, reserved "
            f"{reserved / GIB:.2f} GiB")  # fmt: skip
    if allocated > limit_bytes:
        line += f"; largest live CUDA tensors {live_cuda_tensors()}"
    print(line, flush=True)
    return allocated


def live_tensor_bytes(device_type: str = "cuda") -> int:
    """Bytes of all live tensors of one device type that ``gc`` can see (the CPU stand-in for the
    GPU's allocated memory in tests; storages shared by several tensors count once)."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return 0
    gc.collect()
    storages: dict[int, int] = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == device_type:
                storage = obj.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        except Exception:  # noqa: BLE001 - some objects refuse attribute access
            continue
    return sum(storages.values())
