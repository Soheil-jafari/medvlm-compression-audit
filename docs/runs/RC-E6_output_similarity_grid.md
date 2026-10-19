# Run card RC-E6: output-similarity layer pruning (E6), 3 budgets x 3 seeds

## 1. What and why

- Stage: S09
- Experiments: E6 (layers only, scored once and removed in one shot; D07 deviation), budgets B1, B2, B3, seeds 0, 1, 2
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
  `pruning.clinical.ablation_method` set.
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

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** card owner Amir; seeds: seed 0 Amir (after RC-G2), seed 1 Farzan, seed 2 Amir.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed<k> adapter; teacher cache.
  Seed 0 also attaches the pruned models of RC-G2 (`G2_B2_seed0_pruned`): its B2 runs are then not run again, only their test evaluation.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E6"`, `SEED = <k>`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-E6 --seed <k> --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off> ...`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E6 B1 seed<k>
  - test evaluation E6 B1 seed<k>
  - run E6 B2 seed<k>
  - test evaluation E6 B2 seed<k>
  - run E6 B3 seed<k>
  - test evaluation E6 B3 seed<k>
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E6_seed<k>_pruned`, `E6_seed<k>_outcomes`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data               # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CACHE_ROOT=/kaggle/input/<your-teacher-cache>  # from RC-E0b
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # copy E2/seed<k>/adapter here
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results
export CUDA_VISIBLE_DEVICES=0

python -m pytest -q tests/test_pruning_stage.py tests/test_recovery.py tests/test_budgets.py
SEED=0   # 0, 1 or 2
for B in B1 B2 B3; do
  python scripts/run_experiment.py --exp E6 --budget $B --seed $SEED
  python scripts/06_evaluate.py --exp E6 --budget $B --seed $SEED -o evaluation.split=test
done
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

`run_experiment.py` runs score (04), remove and recover (05), evaluate (06) and benchmark (07)
and stops at the first stage that fails; a repeated command reuses existing scores. If a stage
stops with "configuration incomplete", tell the agent through the inbox instead of filling a
setting in. Pruned models go to `checkpoints/` and per-probe answers to `outputs/` (never
committed): keep both as private datasets, the paired comparisons of S11 need the
`outcomes*.npz` files.

## 4. What to commit back

For every run (`<run>` = `E6`):

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
