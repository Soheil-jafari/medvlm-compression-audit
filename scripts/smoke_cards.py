#!/usr/bin/env python
"""RC-SMOKE: run every card still to run end to end with tiny settings (one T4 x2 session).

Each stage runs in its own process with the real stage script (``medcompress.smoke`` lists the
jobs and the tiny settings). Two lanes run at once, one per T4; the teacher's jobs follow on
both T4s. Nothing is written to ``results/`` (a temporary copy is used). Prints a PASS / WAIT /
FAIL table per card and stage with the peak GPU memory of every load, and saves it as
``<out>/smoke_report.md``. Exit code 0 only if every card passed.

Example (Kaggle, the shared data and the hand-offs attached):
    python scripts/smoke_cards.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="default /kaggle/working/smoke")
    parser.add_argument("--data-root", type=Path, default=None, help="found by its manifest")
    parser.add_argument("--from", dest="handoffs", type=Path, action="append", default=[])
    parser.add_argument("--card", action="append", default=None, help="only these cards")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    return parser


def main(argv: list[str] | None = None) -> int:
    import yaml

    from medcompress.card_runner import select_handoffs
    from medcompress.cards import build_plan
    from medcompress.config import config_root, load_config
    from medcompress.inputs import InputNotFound, classify_bundles, find_shared_data
    from medcompress.smoke import (
        SMOKE_OVERRIDES,
        StageResult,
        card_verdicts,
        parse_peaks,
        report_table,
        smoke_budgets,
        smoke_jobs,
    )

    args = build_parser().parse_args(argv)
    out = (args.out or Path("/kaggle/working/smoke")).resolve()
    if out.exists():
        shutil.rmtree(out)
    roots = {name: out / name for name in ("results", "checkpoints", "outputs", "cache", "logs")}
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "results", roots["results"], dirs_exist_ok=True)  # read, never written
    budgets = out / "smoke_budgets.yaml"
    budgets.write_text(yaml.safe_dump(smoke_budgets(load_config(config_root() / "budgets.yaml"))))
    overrides = [*SMOKE_OVERRIDES, f"components.budgets={budgets}"]

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
        MEDCOMPRESS_REPORT_PEAKS="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONUNBUFFERED="1",
    )
    if Path("/kaggle").is_dir():
        env.setdefault("HF_HOME", "/kaggle/temp/hf")
    for key in ("RESULTS", "CHECKPOINT", "OUTPUT", "CACHE"):  # the base config reads them
        os.environ[f"MEDCOMPRESS_{key}_ROOT"] = env[f"MEDCOMPRESS_{key}_ROOT"]
    config = load_config("base.yaml")

    jobs = [j for j in smoke_jobs() if args.card is None or j.card in args.card]
    results: list[StageResult] = []
    failed: set[str] = set()
    lock = threading.Lock()

    def run_job(job, devices: str) -> None:  # type: ignore[no-untyped-def]
        card, seed, variants, budget_list = job.plan
        plan = build_plan(card, seed, variants=list(variants) or None,
                          budgets=list(budget_list) or None)  # fmt: skip
        handoffs, _ignored = select_handoffs(plan, config, found)
        stage_env = {**env, "CUDA_VISIBLE_DEVICES": devices}
        blocked = job.after is not None and job.after in failed
        for stage in job.stages:
            command = [sys.executable, str(REPO / stage.script), *stage.args,
                       "--data-root", str(data_root)]  # fmt: skip
            for path in handoffs:
                command += ["--from", str(path)]
            for item in [*overrides, *stage.overrides]:
                command += ["-o", item]
            if args.dry_run:
                print(f"[{devices}] {job.card} {job.name} {stage.label}: {' '.join(command)}")
                continue
            if blocked:
                with lock:
                    results.append(StageResult(job.card, job.name, stage.label, None,
                                               note=f"skipped: {job.after} failed"))  # fmt: skip
                continue
            log = roots["logs"] / f"{job.card}_{job.name}_{stage.label}.log".replace(" ", "_")
            started = time.perf_counter()
            done = subprocess.run(command, env=stage_env, capture_output=True, text=True,
                                  check=False)  # fmt: skip
            seconds = time.perf_counter() - started
            text = done.stdout + "\n" + done.stderr
            log.write_text(text, encoding="utf-8")
            loads, peak = parse_peaks(done.stdout)
            result = StageResult(job.card, job.name, stage.label, done.returncode, seconds,
                                 loads, peak, job.note)  # fmt: skip
            with lock:
                results.append(result)
                minutes = seconds / 60
                print(f"[{devices}] {job.card} {job.name} {stage.label}: exit {done.returncode} "
                      f"in {minutes:.1f} min (log {log})", flush=True)  # fmt: skip
            if done.returncode != 0:
                tail = "\n".join(text.strip().splitlines()[-15:])
                print(f"--- last lines of {log.name} ---\n{tail}\n---", flush=True)
                with lock:
                    failed.add(job.name)
                blocked = True  # the job's later stages read this stage's output

    def lane(name: str, devices: str) -> None:
        for job in jobs:
            if job.lane == name:
                run_job(job, devices)

    started = time.perf_counter()
    lane("pre", "0")  # the calibration check that the lanes' calibration sets read
    lanes = [threading.Thread(target=lane, args=(n, d)) for n, d in (("A", "0"), ("B", "1"))]
    for thread in lanes:
        thread.start()
    for thread in lanes:
        thread.join()
    lane("both", "0,1")
    if args.dry_run:
        return 0
    order = {(j.card, j.name): i for i, j in enumerate(jobs)}
    results.sort(key=lambda r: order.get((r.card, r.job), 0))
    table = report_table(results)
    verdicts = card_verdicts(results)
    summary = "\n".join(f"- {card}: {verdict}" for card, verdict in sorted(verdicts.items()))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()  # fmt: skip
    report = (f"# RC-SMOKE report\n\nCommit {commit}; {(time.perf_counter() - started) / 60:.0f} "
              f"min in all.\n\n{table}\n\nPer card:\n{summary}\n")  # fmt: skip
    (out / "smoke_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0 if all(v == "PASS" for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
