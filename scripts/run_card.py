#!/usr/bin/env python
"""Run one run card unattended (Kaggle "Save & Run All"): check, run, hand off, report.

Steps: refuse a card that is not `ready` (docs/runs/); check the library versions (torchao,
optimum and the pins of requirements/) and the GPUs; verify the shared data (--data-root) and
the hand-offs (--from) and check that everything the card reads is there; run the card's
steps (src/medcompress/cards.py); write the hand-offs for other accounts under
<out>/handoff/; copy the results files to commit to <out>/results_to_commit/; print the
card's section 6 and save it as <out>/run_report_<card>.md.

Examples (Kaggle, from the repository root):
    python scripts/run_card.py --card RC-D1 --kvasir-seg /kaggle/input/<kvasir-seg>
    python scripts/run_card.py --card RC-E2 --seed 1 --data-root /kaggle/input/<shared-data> \\
        --from /kaggle/input/<e0-teacher-cache>
    python scripts/run_card.py --card RC-E5 --seed 2 --data-root ... --from ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from medcompress.card_runner import (  # noqa: E402
    ENVIRONMENT_FAILED,
    INPUT_CHECK_FAILED,
    Session,
    collect_results,
    export_handoffs,
    missing_needs,
    now_utc,
    report_text,
    run_steps,
    session_env,
    step_command,
    write_report,
)
from medcompress.cards import CardError, build_plan, card_file, card_status  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--card", required=True, help="e.g. RC-E2 (RC-D1-counts: second pass)")
    parser.add_argument("--seed", type=int, default=None, help="cards split by seed: 0, 1 or 2")
    parser.add_argument("--data-root", type=Path, default=None, help="the shared-data dataset")
    parser.add_argument("--from", dest="handoffs", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=None, help="default /kaggle/working")
    parser.add_argument("--variant", action="append", default=None, help="only these variants")
    parser.add_argument("--budget", action="append", default=None, help="only these budgets")
    parser.add_argument("--kvasir-seg", type=Path, default=None, help="RC-D1: Kvasir-SEG input")
    parser.add_argument("--external-images", type=Path, default=None, help="RC-X1: EndoBench")
    parser.add_argument("--allow-blocked", action="store_true", help="tests only")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    return parser


def _out_dir(args: argparse.Namespace) -> Path | None:
    if args.out is not None:
        return args.out.resolve()
    return Path("/kaggle/working") if Path("/kaggle/working").is_dir() else None


def _environment(session: Session) -> list[str]:
    from medcompress.utils.kaggle import check_gpus, check_versions, gpu_report, parse_requirements

    plan = session.plan
    requirements = REPO / "requirements" / ("train.txt" if plan.gpus else "base.txt")
    problems, session.environment = check_versions(
        parse_requirements(requirements), quantization=plan.quantization
    )
    session.environment.insert(0, f"python {sys.version.split()[0]}")
    if any("export" in step.argv for step in plan.steps):  # it downloads with `datasets`
        from medcompress.utils.kaggle import installed_version

        if installed_version("datasets") is None:
            problems.append("datasets is not installed (pip install datasets)")
    if plan.gpus:
        visible, session.gpus = gpu_report()
        problems += check_gpus(plan.gpus, visible)
    return problems


def _finish(session: Session, code: int) -> int:
    path = write_report(session)
    committed = session.out / "results_to_commit" / "docs" / "runs" / "reports" / path.name
    committed.parent.mkdir(parents=True, exist_ok=True)  # each run's report is its own file
    committed.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print("\n" + report_text(session))
    print(f"report saved to {path}")
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = _out_dir(args)
    try:
        plan = build_plan(
            args.card, args.seed, variants=args.variant, budgets=args.budget,
            out=out or Path("/kaggle/working"), kvasir_seg=args.kvasir_seg,
            data_root=args.data_root, external_images=args.external_images,
        )  # fmt: skip
        status = card_status(card_file(plan.card, REPO / "docs" / "runs"))
    except CardError as exc:
        print(exc, file=sys.stderr)
        return 2
    allowed = {"ready", "running", "done"} if plan.card == "RC-D1-COUNTS" else {"ready"}
    if status not in allowed and not args.allow_blocked:  # pass 2 follows a finished RC-D1
        print(f"{plan.card} is `{status}`, not `ready`: see its card, section 2", file=sys.stderr)
        return 2
    print(f"{plan.card}: {plan.title} ({plan.gpus} GPU(s); card status `{status}`)")
    if args.dry_run:
        for step in plan.steps:
            print(f"  {step.label}: {' '.join(step_command(step, args.data_root, args.handoffs))}")
        print("  reads: " + ("; ".join(n.label for n in plan.needs) or "nothing from other runs"))
        print("  hand-offs: " + (", ".join(h.name for h in plan.handoffs) or "none"))
        return 0
    if out is None:
        print("give --out (a folder outside the repository); on Kaggle it is /kaggle/working")
        return 2
    if out.is_relative_to(REPO):
        print(f"--out must be outside the repository: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    from medcompress.utils.bundles import (
        VERIFIED_ENV,
        BundleError,
        attach_data,
        attach_handoff,
        git_commit,
        verified_token,
    )

    session = Session(plan, out, now_utc(), commit=git_commit(REPO))
    problems = _environment(session)
    if problems:
        session.unusual += problems
        print("environment check failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return _finish(session, ENVIRONMENT_FAILED)
    env = session_env(out, plan.gpus)
    os.environ.update(env)  # the path roots of this session, for the checks below
    from medcompress.config import load_config

    config = load_config("base.yaml")
    tokens = []
    try:
        if args.data_root is not None:
            shared = attach_data(config, args.data_root)
            tokens.append(verified_token(shared))
            session.inputs.append(f"shared data {shared.root} (commit {shared.meta.get('commit')})")
        for path in args.handoffs:
            bundle = attach_handoff(config, path)
            tokens.append(verified_token(bundle))
            session.inputs.append(f"{bundle.label} {bundle.root}")
    except BundleError as exc:
        session.unusual.append(f"input check failed: {exc}")
        return _finish(session, INPUT_CHECK_FAILED)
    missing = missing_needs(plan, config)
    if missing:
        session.unusual.append("missing inputs (attach with --from): " + "; ".join(missing))
        return _finish(session, INPUT_CHECK_FAILED)
    env[VERIFIED_ENV] = os.pathsep.join(tokens)
    os.chdir(REPO)
    run_steps(session, config, env, repo=REPO, data_root=args.data_root, handoffs=args.handoffs)
    export_handoffs(session, config)
    from medcompress.card_runner import report_step_handoffs

    report_step_handoffs(session)
    collect_results(session, REPO)
    failed = [s for s in session.steps if s.status in ("failed", "stopped: setting open")]
    code = 0 if session.ok else next((s.code for s in failed if s.code), 1)
    return _finish(session, code or 0)


if __name__ == "__main__":
    sys.exit(main())
