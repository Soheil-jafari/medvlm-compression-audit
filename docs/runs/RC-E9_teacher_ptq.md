# Run card RC-E9: the teacher at 4 bit (supplementary)

## 1. What and why

- Stage: S08
- Experiments: E9, the teacher (Qwen2.5-VL-7B with the Kvasir adapter merged) quantized with
  the E3 `w4` recipe (GPTQ, groups of 128), one run (seed 0), evaluated on validation now and on
  test after the freeze, and timed with the budget protocol.
- Purpose: a supplementary comparison at matched memory: a quantized big model against the
  compressed small ones (D08).
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- RC-Q0 passed; `quantization.kernel_class` and the library versions are set and pinned. **Met 2026-10-15:** RC-Q0 chose `TorchLinear`; gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned.
- RC-D1 done (calibration: the 128 training images of seed 0, the same as the pruning criteria).
- The pre-registration is frozen; `inputs.image_longest_side`, `evaluation.max_new_tokens` and
  `budgets.measurement.hardware` are set.

Platform: Kaggle notebook, **2 x T4** (the 16-bit teacher does not fit one T4 while it is being
quantized), Internet on. Dataset access as in RC-E3.

Runtime: **estimate** one to two hours (GPTQ on the 7B decoder; its MLP matrices are the
largest). If a GPU runs out of memory during the rounding of the MLP matrices, write the error
in section 6 and stop: the agent will move that step to the CPU.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results

python scripts/run_experiment.py --exp E9 --seed 0
python scripts/06_evaluate.py --exp E9 --seed 0 -o evaluation.split=test
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

Keep `outputs/` (the per-probe `outcomes*.npz` files) as a private dataset too: the paired
comparisons (RC-C1) need the test-split ones. Write its name in section 6.

## 4. What to commit back

- `results/E9/nobudget/seed0/metrics.csv`, `metrics_test.csv`, `efficiency.csv` and `efficiency_setup.json`
- This card with section 6 filled.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Real low-precision kernels | `low_precision_kernels_ok` | equals 1 |
| Calibration as planned | `quantization_calibration_images` | equals 128 |

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions:
- Wall time and peak GPU memory:
- Pass criteria: all met / not met (list):
- Anything unusual:
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
