#!/usr/bin/env python
"""Write ``results/data/nonfinite_drop.csv`` from RC-CAL0's two scan files (second amendment of
2026-11-13): the union of the student's and the teacher's non-finite test images.

Example:
    python scripts/make_drop_list.py vision_scan_student.csv vision_scan_teacher.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("student", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument(
        "--out", type=Path, default=REPO / "results" / "data" / "nonfinite_drop.csv"
    )
    args = parser.parse_args(argv)
    from medcompress.data.nonfinite import drop_rows_from_scans
    from medcompress.utils.io import write_csv

    def read(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    rows = drop_rows_from_scans(read(args.student), read(args.teacher))
    write_csv(args.out, ("split", "sample_id", "encoder", "first_non_finite"), rows)
    for split in ("test", "test_unfiltered"):
        print(f"{split}: {sum(r[0] == split for r in rows)} image(s) dropped for every arm")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
