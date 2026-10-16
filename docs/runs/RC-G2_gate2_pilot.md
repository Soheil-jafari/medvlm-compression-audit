# Run card RC-G2: Gate 2 pilot, E5 (three criteria), E6 and E7 at B2, seed 0

## 1. What and why

- Stage: S09
- Experiments: E5 (`magnitude`, `activation`, `taylor`), E6 and E7, budget B2, seed 0, scored on
  the **validation** split only
- Purpose: before the full grid, find out whether the clinical criterion (E7) keeps more of the
  small findings than the best of the generic criteria (E5) and the output-similarity criterion
  (E6) at the primary budget. The answer says which paper shape applies (see
  `docs/FINAL_GOAL.md`). Gate 2 does not stop any run.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent changes the status to `ready` when they are:

- RC-E2 done for seed 0 (the E2 adapter on team storage) and RC-B0 done; the B2 target fraction
  is set from RC-B0 by the rule of D07 section 4 and confirmed by the team.
- D07 section 1 is answered (how a unit is switched off while it is scored), and
  `pruning.clinical.ablation_method` in `configs/experiments/E7.yaml` is set. If the mixed
  method is chosen, the agent adds the check of the estimate on 30 random channel groups to
  this card before setting it to `ready`.
- The pre-registration is frozen (`preregistration.frozen_on` set): the pilot is a compression
  run, and the stage scripts refuse compression runs before the freeze, so the finding weights
  cannot be adjusted after seeing the pilot.
- `training.steps` and `training.data_subset`, `evaluation.max_new_tokens`,
  `inputs.image_longest_side` and `budgets.measurement.hardware` are set.
- RC-D1 done; RC-M1 passed (met 2026-10-15).

Platform: Kaggle notebook, one T4 (`CUDA_VISIBLE_DEVICES=0`), Internet **on**. Dataset access:
your own copy of Kvasir-VQA-x1 and Kvasir-SEG and the private RC-D1 outputs. The E2 adapter and
the teacher cache are attached as private datasets. Nothing from those goes into the repository.

Runtime: five runs; each is scoring of the units, a search for the units that reach B2, a
recovery training and a validation pass. The scoring of E7 is the long part and depends on the
method chosen in D07 section 1. The agent fills in the estimate when it sets the card to `ready`
(it follows from the speeds that RC-M1 and RC-S0 report; until then any number is a guess). Each
run is independent: use one session per run if needed. The scores of a run are reused when the
settings are the same, so a repeated command does not score again.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Amir.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed0 adapter; teacher cache.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-G2"`, `SEED = None`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-G2 --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off> ...`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E5 magnitude B2 seed0
  - run E5 activation B2 seed0
  - run E5 taylor B2 seed0
  - run E6 B2 seed0
  - run E7 B2 seed0
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `G2_B2_seed0_pruned`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits (unknown here; TODO: the team checks on Kaggle):** the session time limit, and whether a T4 x2 session counts once or twice against the weekly quota. A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data               # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CACHE_ROOT=/kaggle/input/<your-teacher-cache>  # from RC-E0b
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/input/<yourname>-e2    # holds E2/seed0/adapter
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results
export CUDA_VISIBLE_DEVICES=0

python -m pytest -q tests/test_pruning_stage.py tests/test_recovery.py tests/test_budgets.py
# the five runs, one per session if needed
for v in magnitude activation taylor; do
  python scripts/run_experiment.py --exp E5 --variant $v --budget B2 --seed 0
done
python scripts/run_experiment.py --exp E6 --budget B2 --seed 0
python scripts/run_experiment.py --exp E7 --budget B2 --seed 0
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

`run_experiment.py` runs the four stages in order (score the units, remove and recover, evaluate,
benchmark) and stops at the first one that fails. If a stage stops with "configuration
incomplete", a setting above is still open: do not fill it in yourself, tell the agent through
the inbox. The pruned model goes to `checkpoints/` (never committed); keep it as a private
dataset in case a later step needs it.

## 4. What to commit back

For each of the five runs (`<run>` is `E5/<variant>`, `E6` or `E7`):

- `results/<run>/B2/seed0/metrics.csv` and `efficiency.csv`
- `results/<run>/B2/seed0/removal.json` and `budget_search.csv`
- `results/<run>/nobudget/seed0/unit_scores.csv`, `unit_scores.settings.json` and
  `metrics.csv` (numbers per unit and the scoring settings, no images; a later run reuses the
  scores only when the settings file matches)
- This card with section 6 filled.

Check before committing: `git status` shows no images, adapters, caches or `outcomes.npz`.

## 5. How the result is read

The agent writes `docs/results/GATE2.md`: E7 against the better of the five E5/E6 arms at B2 on
validation, with the paired image-level bootstrap interval on the primary endpoint of the
pre-registration draft, and the paper shape that follows. Every footprint is listed with its
distance from the target; a run outside the 3 % band is reported as such and is not hidden. The
team records the reading in `docs/decisions_log.md`. Nothing is tuned on these numbers: the
settings are fixed before the run.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Per run: wall time of scoring / search / recovery / evaluation, peak GPU memory:
- Units removed and measured peak memory against the B2 target (from `budget_search.csv`):
- Name of the private datasets with the pruned models and `outcomes.npz` files:
- Anything unusual (errors, retries, changed settings):
