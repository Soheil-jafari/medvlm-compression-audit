# Run card RC-E8: E7 + 4-bit quantization (E8), 3 budgets x 3 seeds

## 1. What and why

- Stage: S09
- Experiments: E8 = each E7 pruned and recovered model (budgets B1, B2, B3, seeds 0, 1, 2)
  quantized with the E3 `w4` recipe (GPTQ, groups of 128, the same 128 calibration images of
  that seed). 9 runs, scored on validation and test and timed with the budget protocol.
- Purpose: whether pruning and quantization combine, at the same budgets.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- RC-E7 done for the budgets and seeds you run (the pruned models on team storage).
- RC-Q0 passed (`quantization.kernel_class` and the library versions set and pinned). **Met 2026-10-15:** RC-Q0 chose `TorchLinear`; gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned.
- The pre-registration is frozen.

Platform: Kaggle notebook, one T4, Internet on. Dataset access as in RC-E7; attach the E7
pruned models as a private dataset.

Runtime per run: **estimate** under an hour for GPTQ (RC-Q0 and RC-E3 report the real speed),
plus the evaluation and timing.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** card owner Amir; seeds: seed 0 Amir (after RC-G2), seed 1 Farzan, seed 2 Amir.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed<k> adapter; E7 B1 seed<k> pruned; E7 B2 seed<k> pruned; E7 B3 seed<k> pruned.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E8"`, `SEED = <k>`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-E8 --seed <k> --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off> ...`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E8 B1 seed<k>
  - test evaluation E8 B1 seed<k>
  - run E8 B2 seed<k>
  - test evaluation E8 B2 seed<k>
  - run E8 B3 seed<k>
  - test evaluation E8 B3 seed<k>
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E8_seed<k>_quantized`, `E8_seed<k>_outcomes`.
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
# the E7 pruned models must sit at checkpoints/E7/<budget>/seed<k>/pruned

SEED=0   # 0, 1 or 2
for B in B1 B2 B3; do
  python scripts/run_experiment.py --exp E8 --budget $B --seed $SEED
  python scripts/06_evaluate.py --exp E8 --budget $B --seed $SEED -o evaluation.split=test
done
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

Keep `outputs/` (the per-probe `outcomes*.npz` files) as a private dataset too: the paired
comparisons (RC-C1) need the test-split ones. Write its name in section 6.

## 4. What to commit back

- `results/E8/<budget>/seed<k>/metrics.csv`, `metrics_test.csv`, `efficiency.csv`,
  `efficiency_setup.json`
- This card with section 6 filled.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Real low-precision kernels | `low_precision_kernels_ok` | equals 1 |
| Calibration as planned | `quantization_calibration_images` | equals 128 |
| Every remaining decoder matrix quantized | `quantized_matrices` | 7 x the number of decoder layers left after pruning |

## 6. Run report (filled by the person who ran it)

- Run by / date / budgets and seeds:
- Commit actually used:
- Hardware and library versions:
- Wall time and peak GPU memory per run:
- Anything unusual:
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
