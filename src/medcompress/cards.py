"""What each run card runs, in order, for the unattended runner (``scripts/run_card.py``).

A :class:`CardPlan` lists a card's steps (the same stage scripts and flags as section 3 of
the card), the hand-offs it writes for other accounts (``handoffs.py``), the hand-offs it
reads (to attach with ``--from``), and how many GPUs it needs. Cards that are split by seed
(E1, E2, E3, E4, E5-E8, A1-A6) take the seed; one account runs one seed in one session.

The plans are data; ``tests/test_cards.py`` checks them against the experiment registry and
the cards themselves. A card whose status is not ``ready`` is refused by the runner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from medcompress.config import RunContext

SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGETS: tuple[str, ...] = ("B1", "B2", "B3")
E5_CRITERIA: tuple[str, ...] = ("magnitude", "activation", "taylor")
E3_VARIANTS: tuple[str, ...] = ("w4", "w8", "w8a8")
ABLATION_VARIANTS: dict[str, tuple[str | None, ...]] = {
    "A1": (None,),
    "A2": (None,),
    "A3": (None,),
    "A4": ("n32", "n64", "n256"),  # n128 is E7 itself (registry.SAME_AS_E7)
    "A5": ("kl",),  # logprob_drop is E7 itself
    "A6": ("layers_only", "heads_channels"),
}
RC_D1_RUN4B: tuple[str, ...] = ("--rule", "full:0.92:or_crop:0.80", "--heldout-filter", "0.86")
"""The rule and the held-out filter of RC-D1 run 4b (D02 amendments 5-7): pass 2 must use them
to rebuild the same split."""
RC_D1_SETTINGS: tuple[str, ...] = (
    "--split-seed", "20261012", "--close-bits", "4", "--mask-threshold", "4",
)  # fmt: skip
"""RC-D1's pre-registered settings (card section 2). The crop settings, the confirmation
threshold and the chaining guard of D02 amendment 2 are in ``configs/data/kvasir_vqa_x1.yaml``
(``near_duplicates``); ``--close-bits`` only names the 0-4 bit band that is reported."""
REFERENCE_FILE = "results/E2/nobudget/seed0/efficiency.csv"
SHARED_DATA_DIR = "medcompress_data"
"""Folder under ``--out`` where RC-D1 writes the shared data (``data/`` + ``private/``)."""
SHEET_DIR = "rc_d1_sheets"
"""Folder under ``--out`` for the contact sheets (private outputs, never committed)."""
EMBED_DIR = "rc_d1_embed"
ADJUDICATION_DIR = "rc_d1_adjudication"
"""Folder under ``--out`` where RC-D1-embed writes its private output (D02 amendment 3)."""


class CardError(ValueError):
    """The card cannot run as asked (unknown card, wrong seed, not runnable here)."""


@dataclass(frozen=True)
class Run:
    """One experiment run of a card."""

    exp: str
    seed: int = 0
    budget: str | None = None
    variant: str | None = None

    @property
    def context(self) -> RunContext:
        return RunContext(self.exp, seed=self.seed, budget=self.budget, variant=self.variant)

    def args(self) -> list[str]:
        out = ["--exp", self.exp, "--seed", str(self.seed)]
        if self.variant:
            out += ["--variant", self.variant]
        if self.budget:
            out += ["--budget", self.budget]
        return out

    @property
    def label(self) -> str:
        parts = [self.exp, self.variant or "", self.budget or "", f"seed{self.seed}"]
        return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class Need:
    """A hand-off the card reads: checked to be present (attached with ``--from``) before step 1.

    ``kind`` is a hand-off kind; ``run`` the run it belongs to (``None`` for the teacher
    cache). An ``optional`` need is used when present (RC-G2's pruned models in the grid's
    seed-0 sessions). ``text`` describes a need that is not a single path.
    """

    kind: str
    run: Run | None = None
    optional: bool = False
    text: str = ""

    @property
    def label(self) -> str:
        if self.text:
            return self.text
        if self.run is None:
            what = "teacher adapter" if self.kind == "teacher_adapter" else "teacher cache"
        else:
            what = f"{self.run.label} {self.kind}"
        return what + (" (optional)" if self.optional else "")


@dataclass(frozen=True)
class Step:
    """One command: a script of ``scripts/`` (or ``-c`` code) with its arguments.

    ``inputs``: the runner adds ``--data-root`` and ``--from`` to it. ``optional``: a setting
    that is still open (exit code 4) skips the step instead of stopping the card.
    ``skip_if_pruned``: the run is skipped when its pruned model is attached and its results
    are committed (a run RC-G2 already made); only its test evaluation then runs.
    ``skip_if_present``: the step is skipped when this output is already there (an E1 or E2
    adapter attached with ``--from``), so a later session runs only what is left.
    """

    label: str
    argv: tuple[str, ...]
    inputs: bool = True
    optional: bool = False
    skip_if_pruned: Run | None = None
    skip_if_present: Need | None = None


@dataclass(frozen=True)
class Handoff:
    """A folder the card writes for another account: one kind, one or more runs."""

    kind: str
    runs: tuple[Run, ...]
    name: str


@dataclass(frozen=True)
class CardPlan:
    card: str
    title: str
    gpus: int
    steps: tuple[Step, ...]
    handoffs: tuple[Handoff, ...] = ()
    needs: tuple[Need, ...] = ()
    seed: int | None = None
    quantization: bool = False
    internet: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tag(self) -> str:
        return self.card if self.seed is None else f"{self.card}_seed{self.seed}"


def _script(name: str, *args: str) -> tuple[str, ...]:
    return (f"scripts/{name}", *args)


def _experiment(run: Run) -> Step:
    return Step(f"run {run.label}", _script("run_experiment.py", *run.args()))


def _test_eval(run: Run, split: str = "test") -> Step:
    name = "test evaluation" if split == "test" else "unfiltered test evaluation"
    return Step(
        f"{name} {run.label}",
        _script("06_evaluate.py", *run.args(), "-o", f"evaluation.split={split}"),
    )


def _test_evals(run: Run) -> tuple[Step, Step]:
    """The test split and the unfiltered test (D02 amendment 6): each model is scored on both."""
    return _test_eval(run), _test_eval(run, "test_unfiltered")


def _training(run: Run) -> list[Step]:
    """Stage 02, the validation pass and the benchmark (it waits while the hardware is open)."""
    return [
        Step(
            f"train {run.label}",
            _script("02_train_student.py", *run.args()),
            skip_if_present=Need("adapter", run),
        ),
        Step(f"validation {run.label}", _script("06_evaluate.py", *run.args())),
        Step(
            f"benchmark {run.label}",
            _script("07_benchmark_efficiency.py", *run.args()),
            optional=True,
        ),
    ]


def _pruning(runs: Sequence[Run]) -> list[Step]:
    steps: list[Step] = []
    for run in runs:
        experiment = _experiment(run)
        steps.append(Step(experiment.label, experiment.argv, skip_if_pruned=run))
        steps += _test_evals(run)
    return steps


def g2_runs() -> tuple[Run, ...]:
    """The five runs of RC-G2: E5 (three criteria), E6 and E7 at B2, seed 0."""
    e5 = tuple(Run("E5", 0, "B2", v) for v in E5_CRITERIA)
    return e5 + (Run("E6", 0, "B2"), Run("E7", 0, "B2"))


def _pick(values: Sequence[str], wanted: Sequence[str] | None) -> tuple[str, ...]:
    if not wanted:
        return tuple(values)
    unknown = sorted(set(wanted) - set(values))
    if unknown:
        raise CardError(f"unknown choice {unknown}; use {list(values)}")
    return tuple(v for v in values if v in wanted)


def _seed(card: str, seed: int | None, allowed: Sequence[int]) -> int:
    if seed is None:
        raise CardError(f"{card} runs one seed per session: give --seed ({list(allowed)})")
    if seed not in allowed:
        raise CardError(f"{card} runs seed {list(allowed)}, not {seed}")
    return seed


def _no_seed(card: str, seed: int | None) -> None:
    if seed is not None:
        raise CardError(f"{card} has no seed; leave --seed out")


def build_plan(
    card: str,
    seed: int | None = None,
    *,
    variants: Sequence[str] | None = None,
    budgets: Sequence[str] | None = None,
    out: Path = Path("/kaggle/working"),
    kvasir_seg: Path | None = None,
    data_root: Path | None = None,
    external_images: Path | None = None,
    embed_dir: Path | None = None,
    labels: Sequence[Path] = (),
    topup_dir: Path | None = None,
) -> CardPlan:
    """The plan of ``card`` (for ``seed`` where the card is split by seed)."""
    card = card.upper()
    picked_budgets = _pick(BUDGETS, budgets)
    if card == "RC-D1":
        _no_seed(card, seed)
        kvasir_seg = kvasir_seg or Path("/kaggle/input")  # searched for images/ + masks/
        shared = Path(out) / SHARED_DATA_DIR
        settings = RC_D1_SETTINGS
        data = ("--data-root", str(shared / "data"))
        bundle_dir, whole = ("--bundle-dir", str(shared)), ("--data-root", str(shared))
        copy = (
            "-c",
            "import sys; sys.path.insert(0, 'src'); from pathlib import Path; "
            "from medcompress.utils.kaggle import copy_kvasir_seg; "
            f"copy_kvasir_seg(Path({str(kvasir_seg)!r}), Path({str(shared / 'data')!r}))",
        )
        sheets = ("--sheet-dir", str(Path(out) / SHEET_DIR))
        if embed_dir is not None:  # run 4: RC-D1-embed's embeddings, not computed again
            sheets = (*sheets, "--embed-dir", str(embed_dir))
        return CardPlan(
            card, "Data preparation, first pass", 0,
            (
                Step("export", _script("prepare_data.py", "export", *data), inputs=False),
                Step("copy Kvasir-SEG", copy, inputs=False),
                Step(
                    "prepare",
                    _script("prepare_data.py", "prepare", "--data-root", str(shared / "data"),
                            "--private-dir", str(shared / "private"), *settings, *sheets),
                    inputs=False,
                ),
                Step("bundle", _script("prepare_data.py", "bundle", *bundle_dir), inputs=False),
                Step("verify", _script("prepare_data.py", "verify", *whole), inputs=False),
            ),
            internet=True,
            notes=(f"upload {shared} as ONE private Kaggle dataset (card section 4)",
                   f"attach {Path(out) / SHEET_DIR / 'rc_d1_final_cross_split_pairs.png'} to "
                   "the report; never commit it"),
        )  # fmt: skip
    if card == "RC-D1-EMBED":
        _no_seed(card, seed)
        target = ("--out-dir", str(Path(out) / EMBED_DIR))
        steps = []
        if data_root is None:  # export the images and copy Kvasir-SEG first (as RC-D1)
            shared = Path(out) / SHARED_DATA_DIR
            data_root = shared / "data"
            source = kvasir_seg or Path("/kaggle/input")
            copy = (
                "-c",
                "import sys; sys.path.insert(0, 'src'); from pathlib import Path; "
                "from medcompress.utils.kaggle import copy_kvasir_seg; "
                f"copy_kvasir_seg(Path({str(source)!r}), Path({str(data_root)!r}))",
            )
            steps += [
                Step("export", _script("prepare_data.py", "export", "--data-root", str(data_root)),
                     inputs=False),
                Step("copy Kvasir-SEG", copy, inputs=False),
            ]  # fmt: skip
        steps.append(
            Step("embed", _script("prepare_data.py", "embed", "--data-root", str(data_root),
                                  *target, *RC_D1_SETTINGS), inputs=False)
        )  # fmt: skip
        return CardPlan(
            "RC-D1-embed", "Embeddings, candidates and the labelling sheets (CPU)", 0,
            tuple(steps), internet=True,
            notes=(f"upload {Path(out) / EMBED_DIR} as ONE private Kaggle dataset; the raters "
                   "need its label_sheets/ and labels_template.csv (card RC-D1-label)",),
        )  # fmt: skip
    if card == "RC-D1-LABEL":
        _no_seed(card, seed)
        if embed_dir is None or not labels:
            raise CardError("RC-D1-label needs --embed-dir (RC-D1-embed's dataset) and --labels")
        extra = ("--topup-dir", str(topup_dir)) if topup_dir is not None else ()
        return CardPlan(
            "RC-D1-label", "Scores of the labelled pairs and the joining rule (CPU)", 0,
            (Step("score the labels",
                  _script("prepare_data.py", "score-labels", "--embed-dir", str(embed_dir),
                          *extra, "--private-dir", str(Path(out) / ADJUDICATION_DIR),
                          "--commitments", "docs/runs/RC-D1-label_commitments.csv",
                          "--labels", *(str(p) for p in labels)), inputs=False),),
            notes=("commit results/data/near_duplicate_labels.csv and near_duplicate_rules.csv: "
                   "RC-D1 run 4 reads the chosen rule from them",
                   f"exit code 4: the third member has pairs to decide (the list is in "
                   f"{Path(out) / ADJUDICATION_DIR}), or a top-up is required (topup-labels, "
                   "card RC-D1-label section 3b)"),
        )  # fmt: skip
    if card == "RC-D1-CALIBRATE":
        _no_seed(card, seed)
        sheets = ("--sheet-dir", str(Path(out) / SHEET_DIR))
        steps: list[Step] = []
        exporting = data_root is None
        if data_root is None:  # no shared data attached: export the images first
            data_root = Path(out) / SHARED_DATA_DIR / "data"
            steps.append(
                Step(
                    "export",
                    _script("prepare_data.py", "export", "--data-root", str(data_root)),
                    inputs=False,
                )
            )
        steps.append(
            Step(
                "calibrate",
                _script(
                    "prepare_data.py",
                    "calibrate",
                    "--data-root",
                    str(data_root),
                    *RC_D1_SETTINGS[:4],
                    *sheets,
                ),
                inputs=False,
            )
        )
        return CardPlan(
            "RC-D1-calibrate", "Near-duplicate calibration (before the split)", 0, tuple(steps),
            internet=exporting,
            notes=("paste the printed distribution back, with "
                   f"{Path(out) / SHEET_DIR / 'rc_d1_calibration_pairs.png'}; never commit it",),
        )  # fmt: skip
    if card == "RC-D1-COUNTS":
        _no_seed(card, seed)
        if data_root is None:
            raise CardError("the second RC-D1 pass needs --data-root (the shared data)")
        counts = _script(
            "prepare_data.py", "counts", "--data-root", str(data_root), *RC_D1_SETTINGS,
            *RC_D1_RUN4B,
        )  # fmt: skip
        if embed_dir is not None:  # run 4's groups need RC-D1-embed's embeddings
            counts = (*counts, "--embed-dir", str(embed_dir))
        return CardPlan(
            card, "Data preparation, second pass (counts with the label values)", 0,
            (Step("counts", counts, inputs=False),),
        )  # fmt: skip
    if card == "RC-E0A":
        _no_seed(card, seed)
        run = Run("E0")
        return CardPlan(
            "RC-E0a", "Gate 1", 2,
            (Step("gate 1", _script("00_check_teacher_image_reliance.py", *run.args())),),
            needs=(Need("teacher_adapter"),), internet=True,
        )  # fmt: skip
    if card == "RC-E0B":
        _no_seed(card, seed)
        run = Run("E0")
        return CardPlan(
            "RC-E0b", "Teacher cache and E0 validation", 2,
            (
                Step("teacher cache", _script("01_cache_teacher_outputs.py", *run.args())),
                Step("E0 validation", _script("06_evaluate.py", *run.args())),
            ),
            handoffs=(Handoff("teacher_cache", (run,), "E0_teacher_cache"),),
            needs=(Need("teacher_adapter"),), internet=True,
        )  # fmt: skip
    if card in ("RC-E1", "RC-E2"):
        exp = card[-2:]
        run = Run(exp, _seed(card, seed, SEEDS))
        return CardPlan(
            card, f"{exp} training, seed {run.seed}", 1, tuple(_training(run)),
            handoffs=(Handoff("adapter", (run,), f"{exp}_seed{run.seed}_adapter"),),
            needs=(Need("teacher_cache"),) if exp == "E2" else (),
            seed=run.seed, internet=True,
        )  # fmt: skip
    if card == "RC-B0":
        _no_seed(card, seed)
        run = Run("E2")
        first = Path(out) / "first_reading.csv"
        copy = (
            "-c",
            f"import shutil; shutil.copy({REFERENCE_FILE!r}, {str(first)!r})",
        )
        bench = _script("07_benchmark_efficiency.py", *run.args())
        return CardPlan(
            card, "E2 reference footprint (two fresh readings)", 1,
            (Step("first reading", bench), Step("keep the first reading", copy, inputs=False),
             Step("second reading", bench)),
            needs=(Need("adapter", Run("E2")),), internet=True,
        )  # fmt: skip
    if card == "RC-E3":
        s = _seed(card, seed, SEEDS)
        runs = [Run("E3", s, variant=v) for v in _pick(E3_VARIANTS, variants)]
        steps = [step for run in runs for step in (_experiment(run), *_test_evals(run))]
        return CardPlan(
            card, f"E3 post-training quantization, seed {s}", 1, tuple(steps),
            handoffs=(Handoff("quantized", tuple(runs), f"E3_seed{s}_quantized"),
                      Handoff("outcomes", tuple(runs), f"E3_seed{s}_outcomes")),
            needs=(Need("adapter", Run("E2", s)),), seed=s, quantization=True, internet=True,
        )  # fmt: skip
    if card == "RC-E4":
        run = Run("E4", _seed(card, seed, SEEDS))
        return CardPlan(
            card, f"E4 quantization-aware distillation, seed {run.seed}", 1,
            (_experiment(run), *_test_evals(run)),
            handoffs=(Handoff("quantized", (run,), f"E4_seed{run.seed}_quantized"),
                      Handoff("outcomes", (run,), f"E4_seed{run.seed}_outcomes")),
            needs=(Need("teacher_cache"),), seed=run.seed, quantization=True, internet=True,
        )  # fmt: skip
    if card == "RC-E9":
        _no_seed(card, seed)
        run = Run("E9")
        return CardPlan(
            card, "E9 the teacher at 4 bit", 2, (_experiment(run), *_test_evals(run)),
            handoffs=(Handoff("quantized", (run,), "E9_seed0_quantized"),),
            needs=(Need("teacher_adapter"),), quantization=True, internet=True,
        )  # fmt: skip
    if card == "RC-G2":
        _no_seed(card, seed)
        runs = list(g2_runs())
        return CardPlan(
            card, "Gate 2 pilot at B2, seed 0 (validation)", 1, tuple(_experiment(r) for r in runs),
            handoffs=(Handoff("pruned", tuple(runs), "G2_B2_seed0_pruned"),),
            needs=(Need("adapter", Run("E2")), Need("teacher_cache")), internet=True,
            notes=("the grid sessions of seed 0 attach this hand-off and skip these five runs",),
        )  # fmt: skip
    if card in ("RC-E5", "RC-E6", "RC-E7"):
        s = _seed(card, seed, SEEDS)
        exp = card[-2:]
        criteria = _pick(E5_CRITERIA, variants) if exp == "E5" else (None,)
        runs = [Run(exp, s, b, v) for v in criteria for b in picked_budgets]
        kept = runs if exp == "E7" else [r for r in runs if r.budget == "B2"]  # E8 / RC-T1
        handoffs = [Handoff("outcomes", tuple(runs), f"{exp}_seed{s}_outcomes")]
        if kept:
            handoffs.insert(0, Handoff("pruned", tuple(kept), f"{exp}_seed{s}_pruned"))
        return CardPlan(
            card, f"{exp} pruning grid, seed {s}", 1, tuple(_pruning(runs)),
            handoffs=tuple(handoffs),
            needs=(Need("adapter", Run("E2", s)), Need("teacher_cache"))
            + tuple(Need("pruned", r, optional=True) for r in runs if r in g2_runs()),
            seed=s, internet=True,
        )  # fmt: skip
    if card == "RC-E8":
        s = _seed(card, seed, SEEDS)
        runs = [Run("E8", s, b) for b in picked_budgets]
        steps = [step for run in runs for step in (_experiment(run), *_test_evals(run))]
        b2 = tuple(r for r in runs if r.budget == "B2")  # RC-T1 times B2
        return CardPlan(
            card, f"E8 clinical pruning + 4-bit, seed {s}", 1, tuple(steps),
            handoffs=(Handoff("quantized", b2, f"E8_seed{s}_quantized"),
                      Handoff("outcomes", tuple(runs), f"E8_seed{s}_outcomes")),
            needs=(Need("adapter", Run("E2", s)),)
            + tuple(Need("pruned", Run("E7", s, r.budget)) for r in runs),
            seed=s, quantization=True,
            internet=True,
        )  # fmt: skip
    if card in {f"RC-A{n}" for n in range(1, 7)}:
        s = _seed(card, seed, SEEDS)
        exp = card[3:]
        choices = ABLATION_VARIANTS[exp]
        if variants:
            choices = _pick([c for c in choices if c], variants)
        runs = [Run(exp, s, "B2", v) for v in choices]
        return CardPlan(
            card, f"{exp} ablation at B2, seed {s}", 1, tuple(_pruning(runs)),
            needs=(Need("adapter", Run("E2", s)), Need("teacher_cache")), seed=s, internet=True,
        )  # fmt: skip
    if card == "RC-R1":
        _no_seed(card, seed)
        runs = [Run("E0")] + [Run(e, s) for e in ("E1", "E2") for s in SEEDS]
        e2 = tuple(r for r in runs if r.exp == "E2")  # RC-C1 compares E2 with every arm
        return CardPlan(
            card, "E0, E1, E2 on the test split", 2,
            tuple(step for r in runs for step in _test_evals(r)),
            handoffs=(Handoff("outcomes", e2, "E2_test_outcomes"),),
            needs=(Need("teacher_adapter"),)
            + tuple(Need("adapter", Run(e, s)) for e in ("E1", "E2") for s in SEEDS),
            internet=True,
        )  # fmt: skip
    if card == "RC-C1":
        _no_seed(card, seed)
        return CardPlan(
            card, "Paired comparisons (CPU)", 0,
            (Step("comparisons", _script("09_compare_arms.py")),
             Step("comparisons, unfiltered test", _script("09_compare_arms.py", "--unfiltered")),
             Step("tables and figures", _script("08_aggregate_results.py"), inputs=False)),
            needs=(Need("outcomes", text="outcomes of E2 (RC-R1) and of every E3-E8 session"),),
        )  # fmt: skip
    if card == "RC-X1":
        _no_seed(card, seed)
        if data_root is None or external_images is None:
            raise CardError("RC-X1 needs --data-root (the shared data) and --external-images")
        return CardPlan(
            card, "EndoBench part A: overlap (CPU)", 0,
            (Step("overlap", _script("prepare_data.py", "overlap", "--data-root", str(data_root),
                                     "--images", str(external_images), "--name", "endobench",
                                     "--dedup-threshold", "4"), inputs=False),),
        )  # fmt: skip
    if card == "RC-TT0":
        _no_seed(card, seed)
        return CardPlan(
            "RC-TT0", "Teacher feasibility: one LoRA step per setup, then rule 3", 2,
            (Step("feasibility, rule 3", _script("teacher_feasibility.py", "run"), inputs=False),),
            internet=True, quantization=True,
            notes=("commit results/smoke/RC-TT0/teacher_training.csv: RC-TT1 reads it",),
        )  # fmt: skip
    if card == "RC-TT1":
        _no_seed(card, seed)
        return CardPlan(
            "RC-TT1", "Our teacher adapter: training (resumable), hand-off, validation", 2,
            (Step("train the teacher", _script("train_teacher.py")),),
            internet=True, quantization=True,
            notes=("the script writes the hand-off E0_teacher_adapter itself",),
        )  # fmt: skip
    if card == "RC-T1":
        raise CardError("RC-T1 runs on the rented GPU, not on Kaggle: follow its card")
    if card in ("RC-L1", "RC-S1", "RC-S2", "RC-S3"):
        raise CardError(f"{card} has no runnable commands (see its card)")
    raise CardError(f"unknown or finished card {card!r}")


RUNNABLE_CARDS: tuple[str, ...] = (
    "RC-D1-calibrate", "RC-D1-embed", "RC-D1-label", "RC-D1", "RC-D1-counts", "RC-E0a",
    "RC-E0b", "RC-E1", "RC-E2", "RC-B0", "RC-E3", "RC-E4", "RC-E9", "RC-G2", "RC-E5", "RC-E6",
    "RC-E7", "RC-E8", "RC-A1", "RC-A2", "RC-A3", "RC-A4", "RC-A5", "RC-A6", "RC-R1", "RC-C1",
    "RC-X1", "RC-TT0", "RC-TT1",
)  # fmt: skip


def card_file(card: str, runs_dir: Path) -> Path:
    """The markdown card of ``card`` (``RC-D1-calibrate`` and ``RC-D1-counts`` are part of
    RC-D1; ``RC-D1-embed`` and ``RC-D1-label`` have cards of their own)."""
    upper = card.upper()
    if upper in ("RC-D1-EMBED", "RC-D1-LABEL"):
        base = "RC-D1-" + upper.rsplit("-", 1)[1].lower()
    else:
        base = "RC-D1" if upper.startswith("RC-D1") else card
    matches = sorted(Path(runs_dir).glob(f"{base}_*.md"))
    if len(matches) != 1:
        raise CardError(f"no single card file for {card} in {runs_dir}")
    return matches[0]


def card_status(path: Path) -> str:
    """The status word of a card (``- Status: `ready` ...``)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- Status:") and "`" in stripped:
            return stripped.split("`")[1].strip().lower()
    raise CardError(f"{path} has no status line")
