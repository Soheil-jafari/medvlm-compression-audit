#!/usr/bin/env python
"""Print the path of an attached dataset, found by its manifest (src/medcompress/inputs.py).

  data           the shared data: the folder of a bundle_manifest.json with private/ next to it
  handoff NAME   a hand-off: the folder of a bundle_manifest.json whose path contains NAME
                 (case and - against _ do not matter: e0-teacher-adapter matches
                 E0_teacher_adapter)
  handoffs       every attached hand-off, one per line

It prints the path and nothing else, so a cell can write DATA=$(python scripts/find_inputs.py
data) || exit 1. If nothing (or more than one folder) matches, it says what to attach and
exits with code 5.

Examples (Kaggle):
    DATA=$(python scripts/find_inputs.py data) || exit 1
    CACHE=$(python scripts/find_inputs.py handoff E0_teacher_cache) || exit 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import INPUT_CHECK_EXIT_CODE  # noqa: E402
from medcompress.inputs import (  # noqa: E402
    InputNotFound,
    find_handoff,
    find_handoffs,
    find_shared_data,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("data", "handoff", "handoffs"))
    parser.add_argument("name", nargs="?", help="the hand-off's name, e.g. E0_teacher_adapter")
    parser.add_argument("--root", type=Path, default=None, help="default /kaggle/input")
    args = parser.parse_args(argv)
    try:
        if args.what == "data":
            print(find_shared_data(args.root))
        elif args.what == "handoff":
            if not args.name:
                parser.error("handoff needs the hand-off's name, e.g. E0_teacher_adapter")
            print(find_handoff(args.name, args.root))
        else:
            for path in find_handoffs(args.root):
                print(path)
    except InputNotFound as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return INPUT_CHECK_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
