# Run card RC-E1: student fine-tuned on correct answers (E1), 3 seeds

## 1. What and why

- Stage: S06
- Experiments: E1, no budget, seeds 0, 1 and 2
- Purpose: train the chosen student with LoRA on the ground-truth answers only (KD weight 0),
  then score it on the validation split with the full evaluation suite, so that E2 (the same
  recipe plus distillation) has its control.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent changes the status to `ready` when they are:

- RC-S0 passed for the chosen student (Qwen2.5-VL-3B; the fallback is Qwen3-VL-2B). The
  student is already named in `configs/base.yaml: components.student` and
  `configs/training.yaml: model.config` (D05 amendment); the agent pins its revision once RC-S0
  has passed.
- `training.data_subset` and `training.steps` are set (rule of D06: one pass over the subset,
  `steps = ceil(data_subset / 16)`; the agent proposes the subset from the RC-S0 speed and the
  team confirms). The other training values are already decided (D06).
- RC-D1 done; the finding list, label rules, lesion-size thresholds and
  `image_reliance_check.mask_findings` are written to `configs/domain/endoscopy.yaml`.
- `evaluation.max_new_tokens` and `inputs.image_longest_side` are set (D04 rule, RC-S0 numbers).
- RC-M1 passed.

Platform: Kaggle notebook, one T4 is enough (`CUDA_VISIBLE_DEVICES=0`), Internet **on**.
Dataset access: your own copy of Kvasir-VQA-x1 and Kvasir-SEG and the private RC-D1 outputs
(`splits.json`, `mask_links.json`) attached as a private dataset. Nothing from those goes into
the repository.

Runtime: one seed is one training run plus a validation pass. The agent fills in the estimate
when it sets the card to `ready` (it follows from the RC-S0 speed and `training.steps`).
Use one session per seed; the seeds are independent.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data               # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results
export CUDA_VISIBLE_DEVICES=0

python -m pytest -q tests/test_training.py tests/test_training_config.py     # CPU; must pass first
# one seed per session: 0, then 1, then 2
python scripts/run_experiment.py --exp E1 --seed 0        # trains, evaluates, benchmarks
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

`run_experiment.py` runs the three stages in order (train, evaluate, benchmark) and stops
at the first one that fails. If a stage stops with "configuration incomplete", a setting
above is still open: do not fill it in yourself, tell the agent through the inbox.

Keep `checkpoints/` (the LoRA adapters, `E1/seed<k>/adapter`) and `outputs/` (per-sample
answers) on team storage as private datasets; they are never committed.

## 4. What to commit back

For each seed `k`:

- `results/E1/nobudget/seed<k>/metrics.csv`
- `results/E1/nobudget/seed<k>/efficiency.csv`
- This card with section 6 filled.

Check before committing: `git status` shows no images, adapters, caches or `outcomes.npz`.

## 5. How the result is read

The agent writes `docs/results/E1.md` from the CSVs: clinical metrics, small-finding recall by
lesion-size stratum, rare-finding recall, calibration, image reliance and free-text metrics,
mean and spread over the three seeds. A student that fails the image-reliance checks
(blank image, lesion masked, random region masked, other patient's image) is reported to the
team before anything is compressed.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Per seed: wall time, peak GPU memory, final training loss:
- Name of the private datasets with your adapters and `outcomes.npz` files:
- Anything unusual (errors, retries, changed settings):
