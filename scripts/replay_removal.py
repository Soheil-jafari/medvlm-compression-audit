#!/usr/bin/env python
"""Offline replay of every pruning arm's removal set at B1, B2 and B3 (CPU, no model).

Amendment of 2026-11-17. Reads the committed scores (E5 magnitude and activation, E7; E5R's
seeded random scores are drawn here on E7's units, as stage 04 would draw them; E7U uses E7's),
runs stage 05's budget search (:func:`medcompress.compression.recovery.search_budget`) with an
**estimated** footprint (E2's measured peak minus 2 bytes per removed parameter; see
:mod:`medcompress.compression.pruning.replay`) and writes:

* ``results/replay/removal_replay.csv``: one row per arm and budget (k, units by type, removed
  parameters, estimated footprint and its gap to the target);
* ``results/replay/removal_replay_by_layer.csv``: the share of each of the 36 decoder layers;
* ``results/replay/removal_replay.md``: the same as tables.

Exit code 1 if an arm cannot reach a budget (the amendment requires every arm to reach B3).

Example:
    python scripts/replay_removal.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ARMS: tuple[tuple[str, str, str | None, str], ...] = (
    # (label, experiment whose config sets the allocation, variant, scores)
    ("E5:magnitude", "E5", "magnitude", "results/E5/magnitude/nobudget/seed0/unit_scores.csv"),
    ("E5:activation", "E5", "activation", "results/E5/activation/nobudget/seed0/unit_scores.csv"),
    ("E5R", "E5R", None, "random:results/E7/nobudget/seed0/unit_scores.csv"),
    ("E7", "E7", None, "results/E7/nobudget/seed0/unit_scores.csv"),
    ("E7U", "E7U", None, "results/E7/nobudget/seed0/unit_scores.csv"),
)
BUDGETS = ("B1", "B2", "B3")


def _config(exp: str):  # noqa: ANN202
    from medcompress.cli import resolve_config

    namespace = argparse.Namespace(config=None, exp=exp, override=[], data_root=None, handoffs=[])
    return resolve_config(namespace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "results" / "replay")
    args = parser.parse_args(argv)

    from medcompress.budgets import (
        budget_spec_for_run,
        metric_value,
        reference_decoder_bytes,
        reference_report,
        target_value,
    )
    from medcompress.compression.pruning.criteria import get_criterion
    from medcompress.compression.pruning.replay import (
        parameter_measurer,
        replay_arm,
        units_from_scores,
    )
    from medcompress.compression.recovery import RemovalSettings
    from medcompress.utils.io import write_csv

    results = REPO / "results"
    reference = reference_report(results, "E2")
    decoder = reference_decoder_bytes(results, "E2")
    peak = metric_value(reference, "peak_memory")
    measure = parameter_measurer(peak)
    rows, failures = [], []
    for label, exp, _variant, source in ARMS:
        config = _config(exp)
        if source.startswith("random:"):
            units, _ = units_from_scores(REPO / source.split(":", 1)[1])
            scores = get_criterion("random")(config, seed=0).score(None, units, [])  # type: ignore[arg-type]
        else:
            units, scores = units_from_scores(REPO / source)
        removal = RemovalSettings.from_config(config)
        for budget in BUDGETS:
            spec = budget_spec_for_run(config, budget)
            target = target_value(spec, reference, decoder_bytes=decoder)
            try:
                rows.append(replay_arm(label, units, scores, removal, spec, target, measure))
            except ValueError as exc:
                failures.append(f"{label} {budget}: {exc}")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out / "removal_replay.csv",
        ["arm", "allocation", "budget", "uniform_k", "removed_layers", "removed_kv_groups",
         "removed_channel_groups", "removed_params", "removed_units_in_protected_layers",
         "estimated_footprint_bytes", "target_bytes", "relative_gap", "within_tolerance"],
        [(r.arm, r.allocation, r.budget, "" if r.uniform_k is None else r.uniform_k,
          r.removed_layers, r.removed_kv_groups, r.removed_channel_groups, r.removed_params,
          r.removed_in_protected, round(r.footprint), round(r.target), f"{r.relative_gap:.4f}",
          int(r.within_tolerance)) for r in rows],
    )  # fmt: skip
    write_csv(
        args.out / "removal_replay_by_layer.csv",
        ["arm", "budget", "layer_index", "layer_params", "removed_params", "share_of_removed",
         "share_of_layer"],
        [(r.arm, r.budget, i, p, n, f"{s:.4f}", f"{f:.4f}") for r in rows
         for i, p, n, s, f in r.by_layer],
    )  # fmt: skip
    text = markdown(rows, peak, decoder, failures, calibration(results))
    (args.out / "removal_replay.md").write_text(text, encoding="utf-8")
    print(text)
    return 1 if failures else 0


def calibration(results: Path) -> list[str]:
    """The estimate against the measured footprints of the committed superseded runs."""
    from medcompress.compression.pruning.replay import measured_drop_per_parameter

    lines = []
    for folder in sorted((results / "superseded").glob("*/E5/*/B2/seed0")):
        found = measured_drop_per_parameter(folder)
        if found is None:
            continue
        drop, estimate, removed = found
        lines.append(f"| {folder.relative_to(results).as_posix()} | {removed:,} | {drop:.0f} | "
                     f"{estimate:.0f} | {drop / estimate:.4f} |")  # fmt: skip
    if not lines:
        return []
    return ["", "## The estimate against measured footprints",
            "", "Session 1 (superseded design) measured its search's footprints. Measured drop: "
            "the unpruned footprint minus the accepted one. Estimate: 2 bytes per removed "
            "parameter.", "",
            "| run | removed params | measured drop (bytes) | estimate (bytes) | ratio |",
            "|---|---|---|---|---|", *lines]  # fmt: skip


def markdown(rows, peak: float, decoder: float, failures: list[str],  # noqa: ANN001
             checked: list[str]) -> str:  # fmt: skip
    """The two tables of the replay (estimates, labelled as such)."""
    lines = [
        "# Offline replay of the removal sets (estimate)",
        "",
        "Written by `scripts/replay_removal.py` (amendment of 2026-11-17). Footprints are",
        f"**estimates**: E2's measured peak ({peak:.0f} bytes) minus 2 bytes per removed",
        f"parameter; E2's decoder layers hold {decoder:.0f} bytes. The search is stage 05's",
        "(smallest prefix or k at or below target x 1.03). Seed 0 scores.",
        "",
        "k is the number of channel groups removed from every unprotected layer (layers 2-34) of",
        "the `uniform_mlp` allocation, out of the groups each layer has; its share of the MLP",
        'width is k divided by that number. "at target": the smallest k at or below the target',
        'itself; "in band": every k within +-3 % of the target. The search takes the smallest k',
        "at or below target x 1.03, the top of the band.",
        "",
        "| arm | allocation | budget | k (MLP width) | layers | KV groups | channel groups "
        "| removed params | est. gap to target | in band |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        k = (
            ""
            if r.uniform_k is None
            else f"{r.uniform_k} of {r.groups_per_layer} ({r.uniform_k / r.groups_per_layer:.1%})"
        )
        if r.k_nominal is not None:
            band = "" if r.k_band is None else f"; in band {r.k_band[0]}-{r.k_band[1]}"
            k += f"; at target {r.k_nominal}{band}"
        lines.append(
            f"| {r.arm} | {r.allocation} | {r.budget} | {k} | {r.removed_layers} | "
            f"{r.removed_kv_groups} | {r.removed_channel_groups} | {r.removed_params:,} | "
            f"{r.relative_gap:+.2%} | {'yes' if r.within_tolerance else 'no'} |"
        )
    b2 = [r for r in rows if r.budget == "B2"]
    lines += ["", "## Share of each decoder layer removed at B2 (share of the layer's parameters)",
              "", "| layer | " + " | ".join(r.arm for r in b2) + " |",
              "|---|" + "---|" * len(b2)]  # fmt: skip
    for position in range(len(b2[0].by_layer) if b2 else 0):
        index = b2[0].by_layer[position][0]
        cells = [f"{r.by_layer[position][4]:.1%}" for r in b2]
        lines.append(f"| {index} | " + " | ".join(cells) + " |")
    if b2:
        lines += ["", "## Overlap of the B2 removal sets",
                  "", "Share of the row arm's removed units that the column arm also removes.",
                  "", "| arm | " + " | ".join(r.arm for r in b2) + " |",
                  "|---|" + "---|" * len(b2)]  # fmt: skip
        for a in b2:
            cells = [f"{len(a.unit_ids & b.unit_ids) / len(a.unit_ids):.1%}" if a.unit_ids else ""
                     for b in b2]  # fmt: skip
            lines.append(f"| {a.arm} | " + " | ".join(cells) + " |")
    lines += checked
    if failures:
        lines += ["", "## Budgets an arm cannot reach", ""] + [f"- {f}" for f in failures]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
