# Run card RC-A5: definition of the probe effect (A5), KL x B2 x 3 seeds

## 1. What and why

- Stage: S10
- Experiment: A5: the per-probe effect defined as the change of the yes/no prediction (KL)
  instead of the drop of the right answer's log-probability (E7).
- Runs: 3 (variant `kl` x seeds 0, 1, 2 at B2); `logprob_drop` is E7 itself and is run only if
  the team decides not to reuse E7, or if the unit set of A5 has to be reduced (see section 2).
  Each run is scored on the **validation** split and on the **test** split (ablations come after
  the freeze, so the test split is open), and timed with the budget protocol.
- Purpose: does the criterion need the right answer, or does any change of the prediction do?
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
- The KL effect has no first-order estimate (its derivative is zero when the unit is present), so
  every unit of A5 must be scored exactly. If D07 section 1 chooses the first-order estimate for
  some unit type, the agent first changes A5 as D07 says (both variants on layers and KV groups
  only, both run, and `logprob_drop` then is no longer E7 and is run too) and updates this card.

Platform: Kaggle notebook, one T4, Internet on. Dataset access: your own copy of Kvasir-VQA-x1
and Kvasir-SEG, the private RC-D1 outputs, the teacher cache and the E2 adapters attached as
private datasets. Nothing from those goes into the repository.

Runtime: each run needs a scoring pass, a footprint search, a recovery training and two
evaluation passes. The agent fills in the estimate when it sets the card to `ready` (from the
RC-M1, RC-E2 and RC-E7 times); until then any number is a guess. Every run is independent: split
the runs over sessions and team members freely.

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

python -m pytest -q tests/test_pruning_stage.py tests/test_recovery.py tests/test_pruning_ablations.py
SEED=0   # 0, 1 or 2
for V in kl; do
  python scripts/run_experiment.py --exp A5 --variant $V --budget B2 --seed $SEED
  python scripts/06_evaluate.py --exp A5 --variant $V --budget B2 --seed $SEED -o evaluation.split=test
done
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

`run_experiment.py` runs score (04), remove and recover (05), evaluate (06) and benchmark (07)
and stops at the first stage that fails; a repeated command reuses existing scores when their
settings file matches. If a stage stops with "configuration incomplete", tell the agent through
the inbox instead of filling a setting in. Pruned models go to `checkpoints/` and per-probe
answers to `outputs/` (never committed): keep the `outcomes*.npz` files as a private dataset,
the paired comparisons of S11 need them.

## 4. What to commit back

For every run (`<run>` = `A5/<variant>`):

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

## 6. Run report (filled by the person who ran it)

- Run by / date / seeds and variants run:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Per run: wall time of scoring / search / recovery / evaluation, peak GPU memory:
- Names of the private datasets with the `outcomes*.npz` files:
- Anything unusual (errors, retries, changed settings):
