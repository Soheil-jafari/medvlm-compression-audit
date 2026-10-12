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
- RC-Q0 passed (`quantization.kernel_class` and the library versions set and pinned).
- The pre-registration is frozen.

Platform: Kaggle notebook, one T4, Internet on. Dataset access as in RC-E7; attach the E7
pruned models as a private dataset.

Runtime per run: **estimate** under an hour for GPTQ (RC-Q0 and RC-E3 report the real speed),
plus the evaluation and timing.

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
