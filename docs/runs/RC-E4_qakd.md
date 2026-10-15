# Run card RC-E4: quantization-aware distillation (E4), 3 seeds

## 1. What and why

- Stage: S08
- Experiments: E4, seeds 0, 1, 2 (3 runs). The student is trained from its pretrained weights
  with 4-bit weights already in the loop: learned step sizes per group of 128 weights, LoRA
  trained through the rounding, and GRACE's confidence-gated decoupled distillation on the
  cached top-20 teacher tokens (D08 option B). Same training pool, steps and optimiser as E2.
  The result is exported in the same 4-bit format as E3 `w4` and evaluated and timed the same
  way.
- Purpose: the "quantize during training" family of the audit.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- RC-Q0 passed; `quantization.kernel_class` and the library versions are set and pinned.
- RC-E0b done (the teacher cache of the training pool on team storage; the manifest hash matches
  the logged one).
- `training.steps`, `training.data_subset` and `teacher_cache.pool_size` are set (the E2 values;
  E4 trains on exactly the E2 pool).
- The pre-registration is frozen before the first E4 run; `inputs.image_longest_side`,
  `evaluation.max_new_tokens` and `budgets.measurement.hardware` are set.

Platform: Kaggle notebook, one T4, Internet on. Dataset access: your own copy of Kvasir-VQA-x1
and the private RC-D1 outputs; the teacher cache attached as a private dataset.

Runtime per seed: **estimate** 1.5 to 3 times an E2 run of the same length (every step rounds
the weights, and the layers are recomputed in the backward pass to save memory). RC-E2 reports
the E2 time; this card then gets a number.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data               # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CACHE_ROOT=/kaggle/input/<your-teacher-cache>  # from RC-E0b
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results
export CUDA_VISIBLE_DEVICES=0

python -m pytest -q tests/test_qakd.py tests/test_qakd_torch.py        # must pass first
SEED=0   # 0, 1 or 2
python scripts/run_experiment.py --exp E4 --seed $SEED
python scripts/06_evaluate.py --exp E4 --seed $SEED -o evaluation.split=test
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

`run_experiment.py` runs the training (02), the evaluation (06) and the benchmark (07). The
codes go to `checkpoints/E4/nobudget/seed<k>/quantized/` (never committed); keep them as a
private dataset.

Keep `outputs/` (the per-probe `outcomes*.npz` files) as a private dataset too: the paired
comparisons (RC-C1) need the test-split ones. Write its name in section 6.

## 4. What to commit back

- `results/E4/nobudget/seed<k>/metrics.csv`, `metrics_test.csv`, `efficiency.csv`, `efficiency_setup.json`
- This card with section 6 filled.

Check before committing: `git status` shows no images, checkpoints, caches or `outcomes*.npz`.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Trained as planned | `train_steps` in `metrics.csv` | equals `training.steps` |
| Every decoder matrix exported | `quantized_matrices` | 7 x the number of decoder layers |
| Real low-precision kernels | `low_precision_kernels_ok` in `efficiency.csv` | equals 1 |
| Training did not diverge | `train_loss_final` | finite (report the value) |

## 6. Run report (filled by the person who ran it)

- Run by / date / seeds:
- Commit actually used:
- Hardware and library versions:
- Wall time and `train_peak_gpu_bytes`:
- Teacher cache manifest hash:
- Pass criteria: all met / not met (list):
- Anything unusual:
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
