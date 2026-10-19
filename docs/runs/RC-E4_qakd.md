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

- **Memory (2026-10-17):** E4 now trains at micro-batch 1 x 16 **without** gradient
  checkpointing, like every training run (D06 amendment). It had checkpointing on for memory
  only, and its peak at 1 x 16 is not measured. Run seed 0 first. If it stops with "CUDA out of
  memory", stop there, keep the last 30 lines of the log and tell the agent: do not turn
  checkpointing back on yourself (the team decides).
- RC-Q0 passed; `quantization.kernel_class` and the library versions are set and pinned. **Met 2026-10-15:** RC-Q0 chose `TorchLinear`; gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned.
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

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil, all three seeds.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher cache.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E4"`, `SEED = <k>`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-E4 --seed <k> --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off>`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E4 seed<k>
  - test evaluation E4 seed<k>
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E4_seed<k>_quantized`, `E4_seed<k>_outcomes`.
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
