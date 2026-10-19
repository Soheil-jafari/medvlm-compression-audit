# Run card RC-E3: post-training quantization of E2 (w4, w8, w8a8), 3 seeds

## 1. What and why

- Stage: S08
- Experiments: E3, variants `w4` and `w8` (GPTQ, groups of 128, GPTQModel kernel) and `w8a8`
  (LLM.int8), one run per E2 seed (0, 1, 2): 9 runs. Each run is evaluated on the
  **validation** split now and on the **test** split after the pre-registration freeze, and
  timed with the budget protocol.
- Purpose: the "quantize after training" family of the audit (D08).
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- RC-Q0 passed; `quantization.kernel_class` and `quantization.library_versions` are set and
  the versions are pinned in `requirements/train.txt`. **Met 2026-10-15:** RC-Q0 chose `TorchLinear`; gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned.
- RC-E2 done for the seed you run (the E2 adapter on team storage).
- RC-D1 done (the calibration images are 128 training images, drawn with the run seed, the
  same ones the pruning criteria use).
- The pre-registration is frozen before the first E3 run (compression runs come after the
  freeze); `inputs.image_longest_side`, `evaluation.max_new_tokens` and
  `budgets.measurement.hardware` are set.

Platform: Kaggle notebook, one T4, Internet on. Dataset access: your own copy of Kvasir-VQA-x1
and the private RC-D1 outputs; the E2 adapter attached as a private dataset. Nothing from those
goes into the repository.

Runtime per run: **estimate** under an hour for GPTQ on 128 calibration images (RC-Q0 reports
the speed on 16 images), plus the evaluation and timing; LLM.int8 needs no calibration. Each
run is independent; one session per seed is fine.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil, all three seeds (seeds 1 and 2 read the E2 hand-offs of Farzan and Amir).
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed<k> adapter.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E3"`, `SEED = <k>`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-E3 --seed <k> --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off>`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E3 w4 seed<k>
  - test evaluation E3 w4 seed<k>
  - run E3 w8 seed<k>
  - test evaluation E3 w8 seed<k>
  - run E3 w8a8 seed<k>
  - test evaluation E3 w8a8 seed<k>
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E3_seed<k>_quantized`, `E3_seed<k>_outcomes`.
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
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # copy E2/seed<k>/adapter here
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results
export CUDA_VISIBLE_DEVICES=0

SEED=0   # 0, 1 or 2
for v in w4 w8 w8a8; do
  python scripts/run_experiment.py --exp E3 --variant $v --seed $SEED
  python scripts/06_evaluate.py --exp E3 --variant $v --seed $SEED -o evaluation.split=test
done
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

`run_experiment.py` runs quantize (03), evaluate (06) and benchmark (07) in order and stops at
the first stage that fails. If a stage stops with "configuration incomplete", a setting above
is still open: do not fill it in yourself, tell the agent through the inbox. The quantized
codes go to `checkpoints/E3/<variant>/nobudget/seed<k>/quantized/` (never committed); keep them
as a private dataset.

Keep `outputs/` (the per-probe `outcomes*.npz` files) as a private dataset too: the paired
comparisons (RC-C1) need the test-split ones. Write its name in section 6.

## 4. What to commit back

For each variant `<v>` and seed `<k>`:

- `results/E3/<v>/nobudget/seed<k>/metrics.csv`, `metrics_test.csv` and `efficiency.csv`
- `results/E3/<v>/nobudget/seed<k>/efficiency_setup.json` (hardware, protocol, kernel classes)
- This card with section 6 filled.

Check before committing: `git status` shows no images, checkpoints, caches or `outcomes*.npz`.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Real low-precision kernels | `low_precision_kernels_ok` in `efficiency.csv` | equals 1 for every run; a run with 0 is not reported as quantized |
| Calibration as planned | `quantization_calibration_images` in `metrics.csv` (w4, w8) | equals 128 |
| Every decoder matrix quantized | `quantized_matrices` (w4, w8) | 7 x the number of decoder layers |

## 6. Run report (filled by the person who ran it)

- Run by / date / seeds:
- Commit actually used:
- Hardware and `pip freeze | grep -E "torch|transformers|peft|gptqmodel|bitsandbytes"`:
- Wall time per variant:
- Pass criteria: all met / not met (list):
- Anything unusual (errors, retries, changed settings):
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
