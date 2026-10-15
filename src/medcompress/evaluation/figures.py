"""Paper figures F1-F5, drawn only from the results summary (plots, never images).

Static print figures, so no hover layer. Colour does one job per figure:

* identity of the method family (3 categorical slots, validated all-pairs for
  colour-vision deficiency): quantization = blue, generic pruning = orange,
  clinical criterion (ours) = aqua; the distillation-only baseline is muted grey.
  Aqua is below 3:1 contrast on white, so every point carries a direct label.
* polarity (F2, change versus E2): diverging blue <-> red with a grey midpoint.

With no results yet, each figure is a labelled placeholder so the paper compiles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from medcompress.evaluation.report import (
    ACCURACY,
    BALANCED_ACCURACY,
    MASK_EFFECT,
    SMALL_RECALL,
    arm_labels,
)
from medcompress.registry import BUDGET_IDS, PRIMARY_BUDGET

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"
FAMILY_COLOURS: Mapping[str, str] = {
    "quantization": "#2a78d6",
    "generic pruning": "#eb6834",
    "clinical (ours)": "#1baf7a",
    "distillation only": MUTED,
}
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")  # negative, zero, positive
RARE_RECALL = "rare_recall:rare"  # pooled recall of the rare prevalence group
ECE = "ece"
PEAK_MEMORY = "peak_memory_bytes"
FLIP_LESION = "lesion_flip_rate"  # written by the lesion-mask audit (image_reliance.py)
FLIP_RANDOM = "random_flip_rate"
RANKING_COMPARISON = "accuracy_vs_clinical"
HEADLINE_LABELS: Mapping[str, str] = {ACCURACY: "Accuracy", BALANCED_ACCURACY: "Balanced accuracy"}


def family_of(arm: str) -> str:
    """Return the method family of an arm label (``E5:taylor`` -> generic pruning)."""
    base = arm.split(":")[0]
    if base in ("E3", "E4", "E9"):
        return "quantization"
    if base in ("E5", "E6"):
        return "generic pruning"
    if base in ("E7", "E8") or base.startswith("A"):
        return "clinical (ours)"
    return "distillation only"


def compressed_arms() -> list[tuple[str, str]]:
    """Return (arm, budget) pairs compared with E2: unmatched arms plus B2 for matched ones."""
    pairs = [("E1", "")] + [(a, "") for e in ("E3", "E4", "E9") for a in arm_labels(e)]
    pairs += [(a, PRIMARY_BUDGET) for e in ("E5", "E6", "E7", "E8") for a in arm_labels(e)]
    return pairs


def _mean(summary: pd.DataFrame, experiment: str, budget: str, metric: str) -> float:
    if summary.empty:
        return math.nan
    hit = summary[
        (summary["experiment"] == experiment)
        & (summary["budget"].fillna("") == budget)
        & (summary["metric"] == metric)
    ]
    return float(hit["mean"].iloc[0]) if len(hit) else math.nan


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def new_figure(width: float = 3.4, height: float = 2.6, ncols: int = 1) -> tuple[Any, Any]:
    """A styled figure with one axes, or a row of ``ncols`` axes (returned as an array)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "text.color": INK})
    fig, ax = plt.subplots(1, ncols, figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    for each in ax if ncols > 1 else [ax]:
        _style(each)
    return fig, ax


def save_figure(fig: Any, path: Path) -> Path:
    """Write ``fig`` as a PDF without a creation date (so the file is reproducible)."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    return path


def placeholder(path: Path, title: str) -> Path:
    """A labelled empty figure, so the paper compiles before there are results."""
    fig, ax = new_figure()
    ax.axis("off")
    ax.text(0.5, 0.5, f"{title}\n(no results yet)", ha="center", va="center", color=MUTED)
    return save_figure(fig, path)


def _family_legend(ax: Any, families: set[str]) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5, color=FAMILY_COLOURS[f], label=f)
        for f in FAMILY_COLOURS
        if f in families
    ]
    if len(handles) >= 2:
        ax.legend(handles=handles, frameon=False, fontsize=7, loc="best")


def headline_vs_hidden(summary: pd.DataFrame, path: Path, headline: str = ACCURACY) -> Path:
    """F1 (H1): drop in the headline accuracy vs drop in small-finding recall, relative to E2."""
    ref_acc = _mean(summary, "E2", "", headline)
    ref_small = _mean(summary, "E2", "", SMALL_RECALL)
    points = []
    for arm, budget in compressed_arms():
        acc = _mean(summary, arm, budget, headline)
        small = _mean(summary, arm, budget, SMALL_RECALL)
        if not any(math.isnan(v) for v in (ref_acc, ref_small, acc, small)):
            points.append((arm, ref_acc - acc, ref_small - small))
    if not points:
        return placeholder(path, "F1 headline vs hidden damage")
    fig, ax = new_figure()
    low = min(0.0, *[min(x, y) for _, x, y in points])
    high = max(0.0, *[max(x, y) for _, x, y in points])
    ax.plot([low, high], [low, high], color=MUTED, linewidth=1, linestyle="--")
    for arm, x, y in points:
        ax.scatter(
            [x],
            [y],
            s=28,
            color=FAMILY_COLOURS[family_of(arm)],
            edgecolors=SURFACE,
            linewidths=1,
            zorder=3,
        )
        ax.annotate(
            arm, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6, color=INK_SECONDARY
        )
    ax.set_xlabel(f"{HEADLINE_LABELS[headline]} drop vs E2")
    ax.set_ylabel("Small-finding recall drop vs E2")
    _family_legend(ax, {family_of(a) for a, _, _ in points})
    return save_figure(fig, path)


def family_damage(summary: pd.DataFrame, path: Path) -> Path:
    """F2 (H2): change vs E2 per metric (rows) and arm (columns), diverging colours."""
    metrics = [ACCURACY, SMALL_RECALL, RARE_RECALL, MASK_EFFECT, ECE]
    arms = compressed_arms()
    grid = []
    for metric in metrics:
        ref = _mean(summary, "E2", "", metric)
        grid.append([_mean(summary, a, b, metric) - ref for a, b in arms])
    frame = pd.DataFrame(grid, index=metrics, columns=[a for a, _ in arms])
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if frame.empty:
        return placeholder(path, "F2 change vs E2 by method")
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("diverging", list(DIVERGING))
    limit = float(frame.abs().max().max()) or 1.0
    fig, ax = new_figure(width=max(3.4, 0.4 * frame.shape[1] + 1.5), height=2.2)
    ax.grid(False)
    image = ax.imshow(
        frame.to_numpy(dtype=float), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto"
    )
    ax.set_xticks(range(frame.shape[1]), frame.columns, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(frame.shape[0]), frame.index, fontsize=6)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            value = frame.iat[i, j]
            if not math.isnan(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=5, color=INK)
    fig.colorbar(image, ax=ax, shrink=0.8).ax.tick_params(labelsize=6)
    return save_figure(fig, path)


def budget_curves(summary: pd.DataFrame, path: Path) -> Path:
    """F3: small-finding recall against measured peak memory (fraction of E2)."""
    ref = _mean(summary, "E2", "", PEAK_MEMORY)
    fig, ax = new_figure()
    drawn = set()
    for exp, dashed in (("E5", False), ("E6", False), ("E7", False), ("E8", True)):
        for arm in arm_labels(exp):
            xs, ys = [], []
            for budget in BUDGET_IDS:
                mem = _mean(summary, arm, budget, PEAK_MEMORY)
                small = _mean(summary, arm, budget, SMALL_RECALL)
                if not (math.isnan(mem) or math.isnan(small) or math.isnan(ref)):
                    xs.append(mem / ref)
                    ys.append(small)
            if xs:
                colour = FAMILY_COLOURS[family_of(arm)]
                ax.plot(
                    xs,
                    ys,
                    color=colour,
                    linewidth=2,
                    linestyle="--" if dashed else "-",
                    marker="o",
                    markersize=4,
                )
                ax.annotate(
                    arm,
                    (xs[-1], ys[-1]),
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=6,
                    color=INK_SECONDARY,
                )
                drawn.add(family_of(arm))
    for arm in arm_labels("E3"):
        mem = _mean(summary, arm, "", PEAK_MEMORY)
        small = _mean(summary, arm, "", SMALL_RECALL)
        if not (math.isnan(mem) or math.isnan(small) or math.isnan(ref)):
            ax.scatter(
                [mem / ref],
                [small],
                s=28,
                marker="s",
                color=FAMILY_COLOURS["quantization"],
                zorder=3,
            )
            ax.annotate(
                arm,
                (mem / ref, small),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
                color=INK_SECONDARY,
            )
            drawn.add("quantization")
    if not drawn:
        import matplotlib.pyplot as plt

        plt.close(fig)
        return placeholder(path, "F3 budget curves")
    ax.set_xlabel("Peak memory (fraction of E2)")
    ax.set_ylabel("Small-finding recall")
    _family_legend(ax, drawn)
    return save_figure(fig, path)


def rank_order(summary: pd.DataFrame, path: Path, headline: str = ACCURACY) -> Path:
    """F4 (H4): rank of each compressed arm by the headline accuracy vs by small-finding recall."""
    rows = []
    for arm, budget in compressed_arms():
        acc = _mean(summary, arm, budget, headline)
        small = _mean(summary, arm, budget, SMALL_RECALL)
        if not (math.isnan(acc) or math.isnan(small)):
            rows.append((arm, acc, small))
    if len(rows) < 2:
        return placeholder(path, "F4 ranking by accuracy vs clinical metrics")
    frame = pd.DataFrame(rows, columns=["arm", "acc", "small"])
    frame["rank_acc"] = frame["acc"].rank(ascending=False, method="min")
    frame["rank_small"] = frame["small"].rank(ascending=False, method="min")
    fig, ax = new_figure()
    top = float(len(frame))
    ax.plot([1, top], [1, top], color=MUTED, linewidth=1, linestyle="--")
    for r in frame.itertuples():
        ax.scatter(
            [r.rank_acc], [r.rank_small], s=28, color=FAMILY_COLOURS[family_of(r.arm)], zorder=3
        )
        ax.annotate(
            r.arm,
            (r.rank_acc, r.rank_small),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color=INK_SECONDARY,
        )
    tau = _mean(summary, RANKING_COMPARISON, "", f"tau:{SMALL_RECALL}")
    if not math.isnan(tau):
        ax.set_title(f"Kendall's tau = {tau:.2f}", fontsize=7, color=INK_SECONDARY)
    ax.set_xlabel(f"Rank by {HEADLINE_LABELS[headline].lower()}")
    ax.set_ylabel("Rank by small-finding recall")
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.invert_xaxis()
    ax.invert_yaxis()
    _family_legend(ax, {family_of(a) for a in frame["arm"]})
    return save_figure(fig, path)


def mask_audit(summary: pd.DataFrame, path: Path) -> Path:
    """F5: answer flip rate with the lesion masked vs an area-matched random region."""
    arms = [("E0", ""), ("E2", "")] + [
        (a, PRIMARY_BUDGET) for e in ("E5", "E6", "E7") for a in arm_labels(e)
    ]
    data = [
        (arm, _mean(summary, arm, budget, FLIP_LESION), _mean(summary, arm, budget, FLIP_RANDOM))
        for arm, budget in arms
    ]
    data = [d for d in data if not (math.isnan(d[1]) or math.isnan(d[2]))]
    if not data:
        return placeholder(path, "F5 mask audit")
    fig, ax = new_figure()
    width = 0.38
    xs = range(len(data))
    ax.bar(
        [x - width / 2 - 0.01 for x in xs],
        [d[1] for d in data],
        width,
        color=FAMILY_COLOURS["quantization"],
        label="lesion masked",
    )
    ax.bar(
        [x + width / 2 + 0.01 for x in xs],
        [d[2] for d in data],
        width,
        color=FAMILY_COLOURS["generic pruning"],
        label="random region masked",
    )
    ax.set_xticks(list(xs), [d[0] for d in data], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Answer flip rate")
    ax.legend(frameon=False, fontsize=7)
    return save_figure(fig, path)


A4_SIZES: tuple[tuple[str, int], ...] = (("n32", 32), ("n64", 64), ("n128", 128), ("n256", 256))


def _sd(summary: pd.DataFrame, experiment: str, budget: str, metric: str) -> float:
    if summary.empty:
        return math.nan
    hit = summary[
        (summary["experiment"] == experiment)
        & (summary["budget"].fillna("") == budget)
        & (summary["metric"] == metric)
    ]
    return float(hit["sd"].iloc[0]) if len(hit) else math.nan


def calibration_size_curve(summary: pd.DataFrame, path: Path) -> Path:
    """Appendix (A4): primary endpoints at B2 against the number of calibration images.

    An A4 size without runs of its own that is identical to E7 (``registry.SAME_AS_E7``) shows
    E7's B2 numbers. Error bars are the SD over seeds.
    """
    from medcompress.registry import SAME_AS_E7

    panels = []
    for metric, label in (
        (SMALL_RECALL, "Small-finding recall"),
        (MASK_EFFECT, "Lesion-mask effect"),
    ):
        points = []
        for variant, size in A4_SIZES:
            arm = f"A4:{variant}"
            if math.isnan(_mean(summary, arm, PRIMARY_BUDGET, metric)) and arm in SAME_AS_E7:
                arm = "E7"
            mean = _mean(summary, arm, PRIMARY_BUDGET, metric)
            if not math.isnan(mean):
                points.append((size, mean, _sd(summary, arm, PRIMARY_BUDGET, metric)))
        panels.append((label, points))
    if not any(points for _, points in panels):
        return placeholder(path, "A4 calibration-set size")
    fig, axes = new_figure(width=5.0, height=2.2, ncols=2)
    colour = FAMILY_COLOURS["clinical (ours)"]
    for ax, (label, points) in zip(axes, panels, strict=True):
        if points:
            xs, ys, sds = zip(*points, strict=True)
            errors = [0.0 if math.isnan(s) else s for s in sds]
            ax.errorbar(xs, ys, yerr=errors, color=colour, marker="o", markersize=4,
                        linewidth=2, capsize=2)  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_xticks([size for _, size in A4_SIZES], [str(size) for _, size in A4_SIZES])
        ax.set_xlabel("Calibration images")
        ax.set_ylabel(label)
    return save_figure(fig, path)


FIGURE_WRITERS = {
    "headline_vs_hidden.pdf": headline_vs_hidden,
    "family_damage.pdf": family_damage,
    "budget_curves.pdf": budget_curves,
    "rank_order.pdf": rank_order,
    "mask_audit.pdf": mask_audit,
    "appendix_a4_calibration_size.pdf": calibration_size_curve,
}


USES_HEADLINE = frozenset({"headline_vs_hidden.pdf", "rank_order.pdf"})


def write_figures(
    summary: pd.DataFrame, figures_dir: Path, *, headline: str = ACCURACY
) -> list[Path]:
    """Write F1-F5 as PDFs to ``figures_dir`` and return their paths.

    ``headline`` is the pre-registered headline accuracy (``evaluation.headline_accuracy``),
    used by F1 and F4.
    """
    if headline not in HEADLINE_LABELS:
        raise ValueError(f"headline must be one of {sorted(HEADLINE_LABELS)}")
    figures_dir = Path(figures_dir)
    return [
        writer(
            summary, figures_dir / name, **({"headline": headline} if name in USES_HEADLINE else {})
        )
        for name, writer in FIGURE_WRITERS.items()
    ]
