# Run card RC-E7: clinical pruning criterion (E7), B2 x 3 seeds (main tier), B1 and B3 seed 0

## 1. What and why

- **Seeds (seed plan, pre-registration section 11 item 12, Soheil and Amir, 2026-11-08):** E7 at B2 is in the main-claim tier: seeds 0, 1 and 2. B1 and B3: seed 0 only (exploratory). Seeds 1 and 2 run only after seed 0's futility check (`scripts/choose_comparator.py`, after RC-G2): if E7 at B2 ranks below the comparator on validation, seed 0, the study stops and seed 0 is reported as exploratory. The plan refuses a seed 1 or 2 session before `results/seed_plan/comparator_B2.json` is committed, or when it says stop.
- **Per-finding effects (2026-11-08):** the scoring also writes `results/E7/nobudget/seed<k>/unit_effects.csv` (unit, finding, effect) and its settings file; commit them with `unit_scores.csv`: A1 and A2 of the same seed re-weight them instead of scoring again.
- Stage: S09
- Experiments: E7 (ours), budget B2 with seeds 0, 1, 2, and B1 and B3 with seed 0 only (the trend, pre-registration section 11 item 11; decided 2026-11-06). The cell of a seed 1 or 2 session runs B2 only; the evaluation includes the mask audit
- Runs: 9. Each run is scored on the **validation** split and on the **test** split
  (compression runs come after the freeze, so the test split is open), and timed with the
  budget protocol.
- Purpose: the pruning arms of the audit at every budget, with the shared recovery.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- The pre-registration is frozen (`preregistration.frozen_on` set); the stage scripts refuse
  compression runs before that.
- RC-E2 done for the seeds you run, RC-B0 done and B1 to B3 set from it (D07 section 4).
- D07 section 1 answered (how a unit is switched off while scored) and
  `pruning.clinical.ablation_method` set, and the finding weights frozen in the weight table.
- RC-G2 read (the Gate 2 pilot runs first; it does not stop this card).
- `training.steps`, `training.data_subset` (and so the recovery length), `evaluation.max_new_tokens`,
  `inputs.image_longest_side` and `budgets.measurement.hardware` are set.

Platform: Kaggle notebook, one T4, Internet on. Dataset access: your own copy of Kvasir-VQA-x1
and Kvasir-SEG, the private RC-D1 outputs, the teacher cache and the E2 adapters attached as
private datasets. Nothing from those goes into the repository.

Runtime: the scoring of a (criterion, seed) is done once and reused for all three budgets; each
budget then needs a footprint search, a recovery training and two evaluation passes. The agent
fills in the estimate when it sets the card to `ready` (from the RC-M1 and RC-E2 speeds); until
then any number is a guess. Every run is independent: split the runs over sessions and team
members freely.

Keep the E7 pruned models: RC-E8 quantizes them.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** card owner Amir; seed 0: B2 in RC-G2 (Soheil), B1 and B3 on account 3 (Soheil); seed 1 Soheil on account 3 (lent by Farzan), seed 2 Amir, both after the futility check.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed<k> adapter; teacher cache.
  Seed 0 also attaches the pruned models of RC-G2 (one dataset per run, `g2-b2-seed0-pruned-<run>`, since 2026-11-11): its B2 runs are then not run again, only their test evaluation.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E7"` and `SEED = <k>` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-E7 --seed <k>`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E7 B1 seed<k>
  - test evaluation E7 B1 seed<k>
  - run E7 B2 seed<k>
  - test evaluation E7 B2 seed<k>
  - run E7 B3 seed<k>
  - test evaluation E7 B3 seed<k>
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E7_seed<k>_pruned`, `E7_seed<k>_outcomes`.
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
python scripts/run_card.py --card RC-E7 --seed $SEED 2>&1 | tee /kaggle/working/run_card_RC-E7_seed${SEED}.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

The runner runs score (04), remove and recover (05), evaluate (06) and benchmark (07)
and stops at the first stage that fails; a repeated command reuses existing scores. If a stage
stops with "configuration incomplete", tell the agent through the inbox instead of filling a
setting in. Pruned models go to `checkpoints/` and per-probe answers to `outputs/` (never
committed): keep both as private datasets, the paired comparisons of S11 need the
`outcomes*.npz` files.

## 4. What to commit back

For every run (`<run>` = `E7`):

- `results/<run>/<budget>/seed<k>/metrics.csv`, `metrics_test.csv`, `efficiency.csv`,
  `efficiency_setup.json`, `removal.json` and `budget_search.csv`
- `results/<run>/nobudget/seed<k>/unit_scores.csv`, `unit_scores.settings.json` and
  `metrics.csv` (numbers per unit and the scoring settings, no images; a later run reuses the
  scores only when the settings file matches)
- This card with section 6 filled.

Check before committing: `git status` shows no images, adapters, checkpoints or
`outcomes*.npz`.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Budget met | `budget_within_tolerance` in `metrics.csv` | 1 (a run outside the 3 % band is reported and marked, never hidden or re-tuned) |
| Weights really removed | `parameters_after` < `parameters_before` | always |
| Identical recovery | `recovery_fingerprint` in `removal.json` | the same for every run of every pruning arm |
| Protected layers kept | `removal.json` | no unit of a whole protected layer (first 2, last 1) |

## 6. Run report (filled by the person who ran it)

- Run by / date / seeds and budgets run:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Per run: wall time of scoring / search / recovery / evaluation, peak GPU memory:
- Names of the private datasets with the pruned models and `outcomes*.npz` files:
- Anything unusual (errors, retries, changed settings):

## 7. Training-length check

The B2 seed-0 run of this card is the 10,000-example side of the training-length check
(pre-registration section 6); its 30,000-example counterpart, E7 from E2@30k, is run card RC-L1.
