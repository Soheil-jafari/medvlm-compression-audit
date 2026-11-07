#!/usr/bin/env python
"""Check an adapter hand-off on CPU: are all its tensors finite, and what loss history is kept.

Reads every ``*.safetensors`` file of the adapter folder (found below the given path) and
prints, per file, the number of tensors and values and every tensor that holds NaN or inf.
Then it reads whatever per-step history the run kept:

* ``train_history.json`` in the adapter folder (written by every training run since
  2026-11-06: the per-step losses, the non-finite steps and the skipped updates);
* a ``progress.json`` of a resume state (``--progress``), if one was kept.

Older runs (E1, 2026-11-04/05) kept neither: their resume state is removed once the adapter
is saved, and they printed no per-step lines.

Exit code 0 if every tensor is finite, 6 if not, 5 if no adapter was found.

Example (Kaggle CPU notebook, e1-seed0-adapter attached):
    python scripts/inspect_adapter.py "$(python scripts/find_inputs.py handoff E1_seed0_adapter)"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

NOT_FINITE = 6
NOT_FOUND = 5


def tensor_report(path: Path) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """``(tensors, values, {name: (nan, inf)})`` of one safetensors file."""
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    values, bad = 0, {}
    for name, value in tensors.items():
        data = value.float()
        values += data.numel()
        nan, inf = int(data.isnan().sum()), int(data.isinf().sum())
        if nan or inf:
            bad[name] = (nan, inf)
    return len(tensors), values, bad


def loss_history(losses: list) -> str:
    """One line about a list of per-step losses (``None`` or NaN = not finite)."""
    flags = [x is None or not math.isfinite(float(x)) for x in losses]
    steps = [i + 1 for i, f in enumerate(flags) if f]
    finite = [float(x) for x, f in zip(losses, flags, strict=True) if not f]
    head = f"{len(losses)} steps; {len(steps)} non-finite"
    if steps:
        head += f" at steps {steps[:20]}{' ...' if len(steps) > 20 else ''}"
    if finite:
        head += f"; finite losses first {finite[0]:.4f}, last {finite[-1]:.4f}"
    return head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adapter", type=Path, help="the adapter hand-off (or its folder)")
    parser.add_argument("--progress", type=Path, default=None, help="a resume progress.json")
    args = parser.parse_args(argv)
    files = sorted(Path(args.adapter).rglob("*.safetensors"))
    if not files:
        print(f"STOP: no .safetensors file below {args.adapter}", file=sys.stderr)
        return NOT_FOUND
    broken = 0
    for path in files:
        count, values, bad = tensor_report(path)
        print(f"{path.name}: {count} tensors, {values:,} values, {len(bad)} non-finite tensor(s)")
        for name, (nan, inf) in list(bad.items())[:20]:
            print(f"  {name}: {nan} NaN, {inf} inf")
        broken += len(bad)
    histories = sorted(Path(args.adapter).rglob("train_history.json"))
    for path in histories:
        data = json.loads(path.read_text(encoding="utf-8"))
        skipped = data.get("skipped_steps", [])
        print(f"{path.name}: {loss_history(data.get('losses', []))}; skipped updates "
              f"{len(skipped)} {skipped[:20]}")  # fmt: skip
    if args.progress is not None and args.progress.is_file():
        data = json.loads(args.progress.read_text(encoding="utf-8"))
        print(f"{args.progress.name}: {loss_history(data.get('losses', []))}")
    if not histories and args.progress is None:
        print("no per-step loss history kept: the resume state was removed when the adapter was "
              "saved, and the run printed no per-step lines (runs before 2026-11-06)")  # fmt: skip
    print(
        "adapter: all tensors finite" if not broken else f"adapter: {broken} tensor(s) NOT finite"
    )
    return NOT_FINITE if broken else 0


if __name__ == "__main__":
    sys.exit(main())
