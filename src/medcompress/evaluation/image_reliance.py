"""Image-reliance stress tests and the lesion-mask audit.

Stress tests (same questions, modified image):
* blank image             -> a model that still answers "yes" ignores the image,
* lesion masked           -> answer should change for positive findings,
* area-matched random region masked (control) -> answer should mostly NOT change,
* another image with the opposite label -> answer should follow the new image.

Lesion-mask effect (PRIMARY ENDPOINT at B2):
    flip_rate(lesion masked) - flip_rate(area-matched random region masked)
computed on positive, mask-annotated findings.

All tests use the yes/no probes of ``data/probes.py`` and
``VLMWrapper.score_yes_no`` (one forward pass, no free generation). A flip is a
change of the predicted answer relative to the same model's answer on the
unmodified image.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from medcompress.data.masks import area_matched_random_region, load_mask, overlap_fraction
from medcompress.data.probes import format_probe_question

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

DEFAULT_FILL = 128
"""Mid-grey fill used for blank images and for both mask conditions (proposed in D04)."""
_CHUNK = 64


class StressTest(str, Enum):
    """Image perturbations used by the reliance tests."""

    BLANK_IMAGE = "blank_image"
    LESION_MASKED = "lesion_masked"
    RANDOM_REGION_MASKED = "random_region_masked"
    OTHER_PATIENT_IMAGE = "other_patient_image"


# ----------------------------------------------------------------- pure metrics


def answer_flip_rate(original: Sequence[str], perturbed: Sequence[str]) -> float:
    """Fraction of answers that change under a perturbation.

    Answers are compared after stripping whitespace and lower-casing (they are
    normally parsed "yes"/"no" probe answers).
    """
    if len(original) != len(perturbed):
        raise ValueError("Need one perturbed answer per original answer.")
    if not original:
        raise ValueError("Need at least one answer.")
    flips = sum(
        a.strip().lower() != b.strip().lower() for a, b in zip(original, perturbed, strict=True)
    )
    return flips / len(original)


def lesion_mask_effect(
    lesion_masked_flips: Sequence[bool], random_masked_flips: Sequence[bool]
) -> float:
    """Return flip rate (lesion masked) - flip rate (random region masked). Primary endpoint.

    Inputs are per positive, mask-annotated probe: whether the answer flipped when
    the lesion was masked, and when an area-matched random region was masked.
    """
    if not lesion_masked_flips or not random_masked_flips:
        raise ValueError("Need at least one probe per condition.")
    lesion = sum(bool(x) for x in lesion_masked_flips) / len(lesion_masked_flips)
    control = sum(bool(x) for x in random_masked_flips) / len(random_masked_flips)
    return lesion - control


def balanced_accuracy(truth: Sequence[bool], predicted: Sequence[bool]) -> float:
    """Mean of sensitivity and specificity; NaN unless both classes are present."""
    t, p = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    if t.size != p.size:
        raise ValueError("Need one prediction per label.")
    if not t.any() or t.all():
        return math.nan
    return float((p[t].mean() + (~p[~t]).mean()) / 2)


def macro_balanced_accuracy(
    findings: Sequence[str], truth: Sequence[bool], predicted: Sequence[bool]
) -> float:
    """Balanced accuracy per finding, averaged over the findings that have both classes."""
    if not (len(findings) == len(truth) == len(predicted)):
        raise ValueError("findings, truth and predicted must have the same length")
    values = []
    for finding in sorted(set(findings)):
        idx = [i for i, f in enumerate(findings) if f == finding]
        value = balanced_accuracy([truth[i] for i in idx], [predicted[i] for i in idx])
        if not math.isnan(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


# ------------------------------------------------------------------ image edits


def blank_image(like: Any, fill: int = DEFAULT_FILL) -> Any:
    """Return a uniform image of the same size and mode as ``like``."""
    from PIL import Image

    return Image.new("RGB", like.size, (fill, fill, fill))


def fill_region(image: Any, region: Any, fill: int = DEFAULT_FILL) -> Any:
    """Return a copy of ``image`` with the pixels of boolean ``region`` set to ``fill``.

    A region whose shape differs from the image is resized to it (nearest neighbour).
    """
    from PIL import Image

    out = np.array(image.convert("RGB"))
    mask = np.asarray(region, dtype=bool)
    if mask.shape != out.shape[:2]:
        resized = Image.fromarray(mask.astype("uint8") * 255).resize(image.size, Image.NEAREST)
        mask = np.asarray(resized) > 127
    out[mask] = fill
    return Image.fromarray(out)


def stress_image(
    image: Any,
    test: StressTest,
    *,
    mask: Any = None,
    partner: Any = None,
    seed: int = 0,
    fill: int = DEFAULT_FILL,
) -> tuple[Any, float | None]:
    """Return the perturbed image and, for the random-region control, its overlap with the lesion.

    ``mask`` (boolean lesion mask) is required for both mask tests and ``partner``
    (another image) for ``OTHER_PATIENT_IMAGE``.
    """
    if test is StressTest.BLANK_IMAGE:
        return blank_image(image, fill), None
    if test is StressTest.OTHER_PATIENT_IMAGE:
        if partner is None:
            raise ValueError("OTHER_PATIENT_IMAGE needs a partner image")
        return partner.convert("RGB"), None
    if mask is None:
        raise ValueError(f"{test.value} needs a lesion mask")
    if test is StressTest.LESION_MASKED:
        return fill_region(image, mask, fill), None
    mask_arr = np.asarray(mask, dtype=bool)
    region = area_matched_random_region(mask_arr, seed=seed)
    return fill_region(image, region, fill), overlap_fraction(region, mask_arr)


# --------------------------------------------------------------------- probes


@dataclass(frozen=True)
class ImageProbe:
    """One (image, finding) probe with the label of the finding and an optional mask."""

    sample_id: str
    image_path: str
    finding: str
    question: str
    present: bool
    mask_path: str | None = None


def unique_image_probes(
    samples: Sequence[VQASample],
    findings: Sequence[str],
    template: str,
    *,
    display_names: Mapping[str, str] | None = None,
) -> list[ImageProbe]:
    """One probe per (image, finding): an image with several QA rows counts once."""
    names = dict(display_names or {})
    seen: set[tuple[str, str]] = set()
    probes: list[ImageProbe] = []
    for sample in samples:
        for finding in findings:
            key = (sample.image_path, finding)
            if finding not in sample.findings or key in seen:
                continue
            seen.add(key)
            probes.append(
                ImageProbe(
                    sample.sample_id,
                    sample.image_path,
                    finding,
                    format_probe_question(template, names.get(finding, finding)),
                    bool(sample.findings[finding]),
                    sample.mask_path,
                )
            )
    return probes


def pair_opposite_label(
    probes: Sequence[ImageProbe], *, seed: int, groups: Mapping[str, str] | None = None
) -> dict[int, int]:
    """Map probe index to the index of a probe with the same finding but the opposite label.

    ``groups`` optionally maps an image path to a patient or group id; a partner never
    shares the group (or the image) of the probe. Probes without a partner are left out.
    """
    rng = np.random.default_rng(seed)
    by_key: dict[tuple[str, bool], list[int]] = {}
    for i, probe in enumerate(probes):
        by_key.setdefault((probe.finding, probe.present), []).append(i)

    def group(i: int) -> str:
        path = probes[i].image_path
        return (groups or {}).get(path, path)

    pairs: dict[int, int] = {}
    for i, probe in enumerate(probes):
        options = [
            j for j in by_key.get((probe.finding, not probe.present), []) if group(j) != group(i)
        ]
        if options:
            pairs[i] = int(options[int(rng.integers(len(options)))])
    return pairs


def subsample_per_finding(
    probes: Sequence[ImageProbe], max_per_finding: int | None, *, seed: int
) -> list[ImageProbe]:
    """Keep at most ``max_per_finding`` probes of each finding (random, reproducible)."""
    if max_per_finding is None:
        return list(probes)
    rng = np.random.default_rng(seed)
    kept: list[ImageProbe] = []
    for finding in sorted({p.finding for p in probes}):
        group = [p for p in probes if p.finding == finding]
        if len(group) > max_per_finding:
            chosen = sorted(rng.choice(len(group), size=max_per_finding, replace=False))
            group = [group[i] for i in chosen]
        kept.extend(group)
    return kept


def _open(data_root: Path, rel: str) -> Any:
    from PIL import Image

    with Image.open(Path(data_root) / rel) as handle:
        return handle.convert("RGB")


def _scores(
    model: VLMWrapper, images: Sequence[Any], questions: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted yes, yes-minus-no log-probability margin) for each probe."""
    yes, margin = [], []
    for start in range(0, len(images), _CHUNK):
        part = model.score_yes_no(images[start : start + _CHUNK], questions[start : start + _CHUNK])
        yes.extend(s.predicted == "yes" for s in part)
        margin.extend(s.logp_yes - s.logp_no for s in part)
    return np.array(yes, dtype=bool), np.array(margin, dtype=float)


# ------------------------------------------------------------------ mask audit


@dataclass(frozen=True)
class MaskAuditResult:
    """Per-probe outcomes of the lesion-mask audit (kept in memory; never committed)."""

    sample_ids: tuple[str, ...]
    findings: tuple[str, ...]
    original_yes: np.ndarray
    lesion_flip: np.ndarray
    random_flip: np.ndarray
    lesion_margin_drop: np.ndarray
    random_margin_drop: np.ndarray
    control_overlap: np.ndarray

    def summary(self, *, n_resamples: int = 2000, seed: int = 0) -> dict[str, float]:
        """Flip rates, the lesion-mask effect with its paired bootstrap CI, and margin versions."""
        from medcompress.evaluation.statistics import paired_bootstrap_ci

        n = len(self.sample_ids)
        if n == 0:
            raise ValueError("The audit has no probes.")
        lesion, control = self.lesion_flip.astype(float), self.random_flip.astype(float)
        images = np.asarray(self.sample_ids)  # probes of one image are resampled together
        effect, low, high = paired_bootstrap_ci(
            np.mean, lesion, control, n_resamples=n_resamples, seed=seed, clusters=images
        )
        margin_effect, m_low, m_high = paired_bootstrap_ci(
            np.mean,
            self.lesion_margin_drop,
            self.random_margin_drop,
            n_resamples=n_resamples,
            seed=seed,
            clusters=images,
        )
        return {
            "n_probes": float(n),
            "original_yes_rate": float(self.original_yes.mean()),
            "lesion_flip_rate": float(lesion.mean()),
            "random_flip_rate": float(control.mean()),
            "lesion_mask_effect": effect,
            "lesion_mask_effect_ci_low": low,
            "lesion_mask_effect_ci_high": high,
            "lesion_mask_margin_effect": margin_effect,
            "lesion_mask_margin_effect_ci_low": m_low,
            "lesion_mask_margin_effect_ci_high": m_high,
            "control_overlap_mean": float(self.control_overlap.mean()),
        }


def mask_audit(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    *,
    seed: int,
    data_root: Path,
    findings: Sequence[str],
    template: str,
    display_names: Mapping[str, str] | None = None,
    fill: int = DEFAULT_FILL,
    max_probes: int | None = None,
    subsample_seed: int | None = None,
) -> MaskAuditResult:
    """Post-pruning audit: does the model still depend on the lesion region?

    For every positive, mask-annotated (image, finding) probe the lesion is filled in,
    and separately an area-matched random region of the same shape is filled in. A
    flip is a change of the predicted answer relative to the unmodified image.
    ``max_probes`` keeps a random subset drawn with ``subsample_seed`` (default ``seed``);
    give every model the same ``subsample_seed`` so they are audited on the same probes.

    Used by: E5, E6, E7, A1-A6 (after recovery); E0, E2 for reference.
    """
    probes = [
        p
        for p in unique_image_probes(samples, findings, template, display_names=display_names)
        if p.present and p.mask_path
    ]
    if max_probes is not None and len(probes) > max_probes:
        rng = np.random.default_rng(seed if subsample_seed is None else subsample_seed)
        probes = [
            probes[i] for i in sorted(rng.choice(len(probes), size=max_probes, replace=False))
        ]
    if not probes:
        raise ValueError("No positive, mask-annotated probes to audit.")
    originals, lesions, controls, overlaps = [], [], [], []
    for index, probe in enumerate(probes):
        image = _open(data_root, probe.image_path)
        mask = load_mask(str(Path(data_root) / str(probe.mask_path)))
        originals.append(image)
        lesions.append(stress_image(image, StressTest.LESION_MASKED, mask=mask, fill=fill)[0])
        control, overlap = stress_image(
            image,
            StressTest.RANDOM_REGION_MASKED,
            mask=mask,
            seed=seed * 1_000_003 + index,
            fill=fill,
        )
        controls.append(control)
        overlaps.append(overlap)
    questions = [p.question for p in probes]
    base_yes, base_margin = _scores(model, originals, questions)
    les_yes, les_margin = _scores(model, lesions, questions)
    ctl_yes, ctl_margin = _scores(model, controls, questions)
    return MaskAuditResult(
        sample_ids=tuple(p.sample_id for p in probes),
        findings=tuple(p.finding for p in probes),
        original_yes=base_yes,
        lesion_flip=les_yes != base_yes,
        random_flip=ctl_yes != base_yes,
        lesion_margin_drop=base_margin - les_margin,
        random_margin_drop=base_margin - ctl_margin,
        control_overlap=np.array(overlaps, dtype=float),
    )


# ---------------------------------------------------------------------- Gate 1


def reliance_metrics(
    model: VLMWrapper,
    samples: Sequence[VQASample],
    *,
    seed: int,
    data_root: Path,
    findings: Sequence[str],
    template: str,
    display_names: Mapping[str, str] | None = None,
    fill: int = DEFAULT_FILL,
    max_per_finding: int | None = None,
    groups: Mapping[str, str] | None = None,
    subsample_seed: int | None = None,
) -> dict[str, float]:
    """Blank-image and other-image tests on all labelled probes (no masks needed).

    Returns balanced accuracy (macro over findings) on the real images and on blank
    images, the yes rate on blank images, and how often the answer follows an image
    with the opposite label. ``max_per_finding`` keeps a random subset per finding drawn
    with ``subsample_seed`` (default ``seed``); give every model the same one so all are
    tested on the same probes.
    """
    probes = subsample_per_finding(
        unique_image_probes(samples, findings, template, display_names=display_names),
        max_per_finding,
        seed=seed if subsample_seed is None else subsample_seed,
    )
    if not probes:
        raise ValueError("No labelled probes.")
    images = [_open(data_root, p.image_path) for p in probes]
    questions = [p.question for p in probes]
    truth = [p.present for p in probes]
    names = [p.finding for p in probes]
    base_yes, _ = _scores(model, images, questions)
    blank_yes, _ = _scores(model, [blank_image(im, fill) for im in images], questions)
    original = macro_balanced_accuracy(names, truth, base_yes.tolist())
    blank = macro_balanced_accuracy(names, truth, blank_yes.tolist())
    pairs = pair_opposite_label(probes, seed=seed, groups=groups)
    follow = math.nan
    if pairs:
        idx = sorted(pairs)
        partner_images = [images[pairs[i]] for i in idx]
        other_yes, _ = _scores(model, partner_images, [questions[i] for i in idx])
        # "follows the new image" = answers as the partner's label says
        follow = float(
            np.mean([bool(other_yes[k]) == probes[pairs[i]].present for k, i in enumerate(idx)])
        )
    return {
        "n_probes_blank": float(len(probes)),
        "original_balanced_accuracy": original,
        "blank_balanced_accuracy": blank,
        "blank_accuracy_drop": original - blank,
        "blank_yes_rate": float(blank_yes.mean()),
        "other_image_follow_rate": follow,
        "n_probes_other_image": float(len(pairs)),
    }


GATE1_KEYS = (
    "min_original_balanced_accuracy",
    "min_blank_accuracy_drop",
    "min_lesion_mask_effect_ci_low",
)


def gate1_verdict(metrics: Mapping[str, float], thresholds: Mapping[str, Any]) -> dict[str, float]:
    """Apply the approved Gate 1 thresholds to the measured metrics.

    Three checks, all required: the teacher is better than chance on real images, its
    accuracy drops when the image is blanked, and the lower end of the 95 % CI of the
    lesion-mask effect is above ``min_lesion_mask_effect_ci_low`` (zero in the
    proposal). A missing or NaN metric fails its check. Thresholds are pre-registered
    (``docs/decisions/D04_gate1.md``).
    """
    missing = [k for k in GATE1_KEYS if thresholds.get(k) is None]
    if missing:
        raise ValueError(f"Gate 1 thresholds not set: {', '.join(missing)}")

    def at_least(value: float, bound: float) -> bool:
        return not math.isnan(value) and value >= bound

    nan = math.nan
    checks = {
        "gate1_accuracy_ok": at_least(
            metrics.get("original_balanced_accuracy", nan), float(thresholds[GATE1_KEYS[0]])
        ),
        "gate1_blank_drop_ok": at_least(
            metrics.get("blank_accuracy_drop", nan), float(thresholds[GATE1_KEYS[1]])
        ),
        "gate1_lesion_effect_ok": (
            not math.isnan(low := metrics.get("lesion_mask_effect_ci_low", nan))
            and low > float(thresholds[GATE1_KEYS[2]])
        ),
    }
    out = {name: float(ok) for name, ok in checks.items()}
    out["gate1_pass"] = float(all(checks.values()))
    return out


def check_teacher_image_reliance(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    model: VLMWrapper | None = None,
    samples: Sequence[VQASample] | None = None,
) -> None:
    """Stage entry point for ``scripts/00_check_teacher_image_reliance.py`` (run card RC-E0a).

    Go/no-go gate before caching: if the teacher answers almost as well with a
    blank image, the audit is meaningless for this model/dataset pair. Uses the
    validation split only. Writes the metrics and the verdict to
    ``<results_root>/E0/nobudget/seed0/metrics.csv``. Thresholds are pre-registered
    and must be approved before this runs.

    Used by: E0.
    """
    from medcompress.config import ConfigError, find_todos, load_config
    from medcompress.utils.io import upsert_results

    section = config.get("image_reliance_check", {})
    gate = section.get("gate1", {})
    open_keys = {
        *find_todos({"image_reliance_check": section}),
        *(f"gate1.{k}" for k in GATE1_KEYS if gate.get(k) is None),
    }
    domain = load_config(config["components"]["domain"])["domain"]
    if find_todos({"findings": domain.get("findings")}):
        open_keys.add("domain.findings")
    side = config.get("inputs", {}).get("image_longest_side")
    if model is None and (side is None or find_todos({"image_longest_side": side})):
        open_keys.add("inputs.image_longest_side")
    open_keys = sorted(open_keys)
    if open_keys:
        raise ConfigError(f"image-reliance settings still open: {', '.join(open_keys)}")
    paths = config["paths"]
    seed = run.seed if run is not None else 0
    if samples is None:
        from medcompress.data.prepared import load_prepared

        dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
        samples = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))[
            "val"
        ]
    if model is None:
        from medcompress.models.loading import ModelSpec, load_model

        spec = ModelSpec.from_config(load_config(config["components"]["teacher"]))
        model = load_model(spec, image_longest_side=int(side))
    probe = domain["probe"]
    common = {
        "seed": seed,
        "data_root": Path(paths["data_root"]),
        "template": probe["template"],
        "display_names": probe.get("display_names") or {},
        "fill": int(section["fill"]),
    }
    metrics = reliance_metrics(
        model,
        samples,
        findings=list(domain["findings"]),
        max_per_finding=section.get("max_probes_per_finding"),
        **common,
    )
    audit = mask_audit(
        model,
        samples,
        findings=list(section["mask_findings"]),
        max_probes=section.get("max_mask_probes"),
        **common,
    ).summary(seed=seed)
    metrics.update(audit)
    metrics.update(gate1_verdict(metrics, gate))
    tag = run.experiment_tag if run is not None else "E0"
    rows = []
    for name, value in metrics.items():
        if name.endswith(("_ci_low", "_ci_high")):
            continue
        row: dict[str, Any] = {
            "experiment": tag,
            "budget": "",
            "seed": seed,
            "metric": name,
            "value": value,
        }
        if f"{name}_ci_low" in metrics:
            row["ci_low"], row["ci_high"] = metrics[f"{name}_ci_low"], metrics[f"{name}_ci_high"]
        rows.append(row)
    target = Path(paths["results_root"]) / "E0" / "nobudget" / f"seed{seed}" / "metrics.csv"
    upsert_results(rows, target)
    print(f"wrote {len(rows)} rows to {target}; gate1_pass={int(metrics['gate1_pass'])}")
