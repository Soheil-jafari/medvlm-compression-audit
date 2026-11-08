# Run card RC-A4: calibration set size (A4), 32 / 64 / 256 images x B2, seed 0

## 1. What and why

- **Seeds (seed plan, pre-registration section 11 item 12, Soheil and Amir, 2026-11-08):** seed 0 only, reported as exploratory / descriptive (no confirmatory test); the plan (`scripts/run_card.py`) refuses seeds 1 and 2. This replaces the seed counts written below. A single-seed difference is read against E7's spread over its 3 seeds at B2: smaller than that spread = within run-to-run noise.
- Stage: S10
- Experiment: A4: the clinical criterion scored on 32, 64, 128 or 256 calibration images
  (nested: the 32 images are the first 32 of the 64, and so on).
- Runs: 9 (variants `n32`, `n64`, `n256` x seeds 0, 1, 2 at B2); `n128` is E7 itself and is run
  only if the team decides not to reuse E7 (pre-registration, section 7). Each run is scored on
  the **validation** split and on the **test** split (ablations come after the freeze, so the
  test split is open), and timed with the budget protocol.
- Purpose: how many calibration images the criterion needs (the appendix curve).
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- The pre-registration is frozen (`preregistration.frozen_on` set); the stage scripts refuse
  compression runs before that.
- RC-E2 done for the seeds you run, RC-B0 done and B2 set from it (D07 section 4).
- D07 section 1 answered (how a unit is switched off while scored) and
  `pruning.clinical.ablation_method` set.
- RC-E7 (or at least its B2 runs) done: the ablations are read against E7 at B2.
- `training.steps`, `training.data_subset` (and so the recovery length), `evaluation.max_new_tokens`,
  `inputs.image_longest_side` and `budgets.measurement.hardware` are set.

Platform: Kaggle notebook, one T4, Internet on. Dataset access: your own copy of Kvasir-VQA-x1
and Kvasir-SEG, the private RC-D1 outputs, the teacher cache and the E2 adapters attached as
private datasets. Nothing from those goes into the repository.

Runtime: each run needs a scoring pass, a footprint search, a recovery training and two
evaluation passes. The agent fills in the estimate when it sets the card to `ready` (from the
RC-M1, RC-E2 and RC-E7 times); until then any number is a guess. Every run is independent: split
the runs over sessions and team members freely.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** card owner Amir; seed 0: `n32` Soheil, `n64` and `n256` Amir (docs/schedule.md section 7, 2026-11-08); no seeds 1 and 2.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed<k> adapter; teacher cache.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-A4"` and `SEED = <k>` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-A4 --seed <k>`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run A4 n32 B2 seed<k>
  - test evaluation A4 n32 B2 seed<k>
  - run A4 n64 B2 seed<k>
  - test evaluation A4 n64 B2 seed<k>
  - run A4 n256 B2 seed<k>
  - test evaluation A4 n256 B2 seed<k>
- **Hand-offs it writes:** none (its results reach the others through git).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

One cell (Kaggle settings and inputs of section 2b; **Save Version -> Save & Run All**). It finds the shared data and the attached hand-offs by their manifests (`scripts/find_inputs.py`), stops with a clear message if the shared data are missing, and runs the steps of section 2b with `scripts/run_card.py`, which also checks that every hand-off the card reads is there. Then it writes the hand-offs under `/kaggle/working/handoff/` and the files to commit under `/kaggle/working/results_to_commit/`.

```bash
%%bash
SEED=0   # this session's seed: 0, 1 or 2 (section 2b says who runs which)
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone,
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # so removing the clone keeps
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # every output
python - <<'PY' || pip uninstall -y -q torchao   # peft needs torchao 0.16.0 or newer, or none
import sys
from importlib import metadata
try:
    version = tuple(int(x) for x in metadata.version("torchao").split(".")[:2])
except metadata.PackageNotFoundError:
    sys.exit(0)
sys.exit(0 if version >= (0, 16) else 1)
PY
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
python scripts/find_inputs.py data > /dev/null || exit 1        # stops if the shared data are not attached
python scripts/find_inputs.py handoffs                          # the hand-offs found by their manifests
python scripts/run_card.py --card RC-A4 --seed $SEED 2>&1 | tee /kaggle/working/run_card_RC-A4_seed${SEED}.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

The runner runs score (04), remove and recover (05), evaluate (06) and benchmark (07)
and stops at the first stage that fails; a repeated command reuses existing scores when their
settings file matches. If a stage stops with "configuration incomplete", tell the agent through
the inbox instead of filling a setting in. Pruned models go to `checkpoints/` and per-probe
answers to `outputs/` (never committed): keep the `outcomes*.npz` files as a private dataset,
the paired comparisons of S11 need them.

## 4. What to commit back

For every run (`<run>` = `A4/<variant>`):

- `results/<run>/B2/seed<k>/metrics.csv`, `metrics_test.csv`, `efficiency.csv`,
  `efficiency_setup.json`, `removal.json` and `budget_search.csv`
- `results/<run>/nobudget/seed<k>/unit_scores.csv`, `unit_scores.settings.json` and
  `metrics.csv` (numbers per unit and the scoring settings, no images)
- This card with section 6 filled.

Check before committing: `git status` shows no images, adapters, checkpoints or
`outcomes*.npz`.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Budget met | `budget_within_tolerance` in `metrics.csv` | 1 (a run outside the 3 % band is reported and marked, never hidden or re-tuned) |
| Weights really removed | `parameters_after` < `parameters_before` | always |
| Identical recovery | `recovery_fingerprint` in `removal.json` | the same as in every E5-E7 run |
| Protected layers kept | `removal.json` | no unit of a whole protected layer (first 2, last 1) |
| Calibration size | `scoring_calibration_images` in `results/A4/<variant>/nobudget/seed<k>/metrics.csv` | 32, 64 or 256 as the variant says |

## 6. Run report (filled by the person who ran it)

- Run by / date / seeds and variants run:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Per run: wall time of scoring / search / recovery / evaluation, peak GPU memory:
- Names of the private datasets with the `outcomes*.npz` files:
- Anything unusual (errors, retries, changed settings):
