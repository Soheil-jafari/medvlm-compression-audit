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
