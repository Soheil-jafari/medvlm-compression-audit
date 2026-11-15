#!/usr/bin/env python
"""Scale check: each heavy step at its real settings for a few steps, failing on memory growth.

Runs every check of ``medcompress.scale_check`` in its own process, one after the other (the GPU
and host memory of one check never meet the next), samples each process's host RSS from outside
(a kill leaves a reading), and prints a table with the GPU peak, the host RSS peak, the lowest
available host RAM and what each check measured; saves it as ``<out>/scale_report.md``. Nothing
is written to ``results/`` (a temporary copy is used). Exit code 0 only if no check failed.

Example (Kaggle T4 x2; the shared data, the E2 seed-0 adapter and the teacher cache attached):
    python scripts/scale_check.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

QUANTIZATION_CHECKS = ("e4_training", "gptq", "e9_eval")  # the env of their cards
PLAN_OF: dict[str, tuple[str, int | None, tuple[str, ...]]] = {
    "e7_first_order": ("RC-G2", None, ()),
    "budget_search": ("RC-G2", None, ()),
    "recovery": ("RC-G2", None, ()),
    "e4_training": ("RC-E4", 0, ()),
    "gptq": ("RC-E3", 0, ("w4",)),
    "e9_eval": ("RC-E9", None, ()),
}
"""The card whose hand-offs each check reads (2026-11-14: E9's check was given an unrelated E1
adapter, whose settings were then compared with the teacher's)."""
TWO_GPU_CHECKS = ("e9_eval",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="default /kaggle/working/scale")
    parser.add_argument("--data-root", type=Path, default=None, help="found by its manifest")
    parser.add_argument("--from", dest="handoffs", type=Path, action="append", default=[])
    parser.add_argument("--check", action="append", default=None, help="only these checks")
    parser.add_argument("--steps", type=int, default=40, help="E7 first-order steps (+3 largest)")
    parser.add_argument(
        "--loads",
        type=int,
        default=15,
        help="model loads of the budget search (the first + 14 measurements)",
    )
    parser.add_argument("--updates", type=int, default=20, help="recovery updates")
    parser.add_argument("--e4-updates", type=int, default=10, help="E4 training updates")
    parser.add_argument("--layers", type=int, default=4, help="GPTQ decoder layers")
    parser.add_argument("--batches", type=int, default=30, help="E9 evaluation batches")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def child(args: argparse.Namespace) -> int:
    """Run one check in this process (the parent starts one process per check)."""
    from medcompress.cli import resolve_config
    from medcompress.scale_check import EXPERIMENT_OF, run_check

    name = args.check[0]
    namespace = argparse.Namespace(config=None, exp=EXPERIMENT_OF[name], override=[],
                                   data_root=args.data_root, handoffs=args.handoffs)  # fmt: skip
    config = resolve_config(namespace)
    names = ("steps", "loads", "updates", "e4_updates", "layers", "batches")
    options = {name: getattr(args, name) for name in names}
    return run_check(name, config, args.out, options)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out = (args.out or Path("/kaggle/working/scale")).resolve()
    if args.child:
        return child(args)

    from medcompress.inputs import InputNotFound, classify_bundles, find_shared_data
    from medcompress.scale_check import CHECKS, report_table
    from medcompress.utils.memory import run_watched

    if args.out.exists():
        shutil.rmtree(args.out)
    roots = {
        name: args.out / name for name in ("results", "checkpoints", "outputs", "cache", "logs")
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "results", roots["results"], dirs_exist_ok=True)  # read, never written
    data_root = args.data_root
    if data_root is None:
        try:
            data_root = find_shared_data()
        except InputNotFound as exc:
            print(f"STOP: {exc}")
            return 2
    found = list(args.handoffs) or classify_bundles()[0]
    env = dict(os.environ)
    env.update(
        MEDCOMPRESS_RESULTS_ROOT=str(roots["results"]),
        MEDCOMPRESS_CHECKPOINT_ROOT=str(roots["checkpoints"]),
        MEDCOMPRESS_OUTPUT_ROOT=str(roots["outputs"]),
        MEDCOMPRESS_CACHE_ROOT=str(roots["cache"]),
        PYTHONUNBUFFERED="1",
    )
    if Path("/kaggle").is_dir():
        env.setdefault("HF_HOME", "/kaggle/temp/hf")
    for key in ("RESULTS", "CHECKPOINT", "OUTPUT", "CACHE"):  # the base config reads them
        os.environ[f"MEDCOMPRESS_{key}_ROOT"] = env[f"MEDCOMPRESS_{key}_ROOT"]
    from medcompress.card_runner import select_handoffs
    from medcompress.cards import build_plan
    from medcompress.config import load_config

    base = load_config("base.yaml")
    checks = [c for c in CHECKS if args.check is None or c in args.check]
    rows = []
    started = time.perf_counter()
    for name in checks:
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--check", name,
                   "--out", str(args.out), "--data-root", str(data_root),
                   "--steps", str(args.steps), "--loads", str(args.loads),
                   "--updates", str(args.updates), "--e4-updates", str(args.e4_updates),
                   "--layers", str(args.layers), "--batches", str(args.batches)]  # fmt: skip
        card, seed, variants = PLAN_OF[name]
        plan = build_plan(card, seed, variants=list(variants) or None)
        handoffs, ignored = select_handoffs(plan, base, found)  # only what this card reads
        for path in handoffs:
            command += ["--from", str(path)]
        check_env = dict(env)
        check_env["CUDA_VISIBLE_DEVICES"] = "0,1" if name in TWO_GPU_CHECKS else "0"
        if name in QUANTIZATION_CHECKS:
            check_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print(f"== scale check {name} (own process)", flush=True)
        begun = time.perf_counter()
        done = run_watched(command, env=check_env)
        minutes = (time.perf_counter() - begun) / 60
        log = roots["logs"] / f"{name}.log"
        log.write_text(done.stdout + "\n" + done.stderr, encoding="utf-8")
        report = args.out / f"{name}.json"
        row = json.loads(report.read_text()) if report.is_file() else {
            "check": name, "status": "FAIL", "numbers": {},
            "lines": [f"FAIL: exit {done.returncode} before a verdict"
                      + (" (killed, exit -9: host RAM)" if done.returncode == -9 else "")],
        }  # fmt: skip
        if done.returncode != 0 and row["status"] != "FAIL":
            row["status"] = "FAIL"
            row["lines"].append(f"FAIL: exit {done.returncode}")
        rss = "n/a" if done.peak_rss is None else f"{done.peak_rss / 2**30:.1f}"
        low = "n/a" if done.min_available is None else f"{done.min_available / 2**30:.1f}"
        row["host"], row["minutes"] = f"{rss} / {low}", minutes
        rows.append(row)
        print(f"== {name}: {row['status']} in {minutes:.1f} min; host RSS peak {rss} GiB, lowest "
              f"available {low} GiB (log {log})", flush=True)  # fmt: skip
        if row["status"] == "FAIL":
            tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-20:])
            print(f"--- last lines of {log.name} ---\n{tail}\n---", flush=True)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()  # fmt: skip
    text = (f"# Scale check report\n\nCommit {commit}; "
            f"{(time.perf_counter() - started) / 60:.0f} min in all.\n\n"
            f"{report_table(rows)}\n")  # fmt: skip
    (args.out / "scale_report.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
