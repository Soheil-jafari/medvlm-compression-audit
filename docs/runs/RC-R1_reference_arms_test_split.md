# Run card RC-R1: test-split evaluation of the reference arms E0, E1 and E2

## 1. What and why

- Stage: S11
- Experiments: E0 (seed 0), E1 and E2 (seeds 0, 1, 2), no budget.
- Runs: 7 evaluations on the **test** split. Nothing is trained.
- Purpose: E0, E1 and E2 are trained and evaluated on the validation split before the
  pre-registration is frozen, while the test split is still locked. Every comparison of the
  paper is on the test split, and E2 is the reference of H1 and of tables T2 and F1 to F5, so
  these three arms need a test-split evaluation after the freeze.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- The pre-registration is frozen (`preregistration.frozen_on` set); before that the evaluation
  refuses the test split.
- RC-E0b, RC-E1 and RC-E2 done; the E1 and E2 adapters kept as private datasets.

Platform: Kaggle notebook, **2 x T4** for E0 (the 7B teacher, as in RC-E0b), one T4 for E1
and E2, Internet on. Dataset access: your own copy of Kvasir-VQA-x1 and Kvasir-SEG, the private
RC-D1 outputs, and the E1 and E2 adapters attached as private datasets. Nothing from those
goes into the repository.

Runtime: one evaluation pass per run, the same work as the validation evaluation in RC-E0b,
RC-E1 and RC-E2 (the split halves are about the same size). The agent fills in the estimate
from those cards' wall times when it sets this card to `ready`.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil.
- **Kaggle settings:** GPU T4 x2 (both T4s); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E1 seed0 adapter; E1 seed<k> adapter; E1 seed2 adapter; E2 seed0 adapter; E2 seed<k> adapter; E2 seed2 adapter.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-R1"`, `SEED = None`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-R1 --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off> ...`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - test evaluation E0 seed0
  - test evaluation E1 seed0
  - test evaluation E1 seed<k>
  - test evaluation E1 seed2
  - test evaluation E2 seed0
  - test evaluation E2 seed<k>
  - test evaluation E2 seed2
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E2_test_outcomes`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits (unknown here; TODO: the team checks on Kaggle):** the session time limit, and whether a T4 x2 session counts once or twice against the weekly quota. This card uses T4 x2. A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data               # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # copy E1/ and E2/ seed<k>/adapter here
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results

python scripts/06_evaluate.py --exp E0 --seed 0 -o evaluation.split=test   # 2 x T4 session
for SEED in 0 1 2; do                                                      # one T4 is enough
  python scripts/06_evaluate.py --exp E1 --seed $SEED -o evaluation.split=test
  python scripts/06_evaluate.py --exp E2 --seed $SEED -o evaluation.split=test
done
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

If a script stops with "configuration incomplete", tell the agent through the inbox instead of
filling a setting in. Keep `outputs/` (the `outcomes_test.npz` files) as a private dataset: the
paired comparisons (RC-C1) need them.

## 4. What to commit back

- `results/E0/nobudget/seed0/metrics_test.csv`
- `results/E1/nobudget/seed<k>/metrics_test.csv` and `results/E2/nobudget/seed<k>/metrics_test.csv`
- This card with section 6 filled.

Check before committing: `git status` shows no images, adapters or `outcomes*.npz`.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| All runs evaluated | `metrics_test.csv` files | 7 files, each with `small_finding_recall` and `lesion_mask_effect` |
| Same probes as every arm | `n_probes_scored` | the same number in every file |

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Wall time per run:
- Names of the private datasets with the `outcomes_test.npz` files:
- Anything unusual (errors, retries, changed settings):
