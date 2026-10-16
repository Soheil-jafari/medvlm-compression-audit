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
  has passed. **RC-S0 met 2026-10-16** (both models pass G1); the revision pin is still to do.
- `training.data_subset` and `training.steps` are set (rule of D06: one pass over the subset,
  `steps = ceil(data_subset / 16)`; the agent proposes the subset from the RC-S0 speed and the
  team confirms). The other training values are already decided (D06). **Met 2026-10-16:**
  10,000 examples, 625 steps (Soheil).
- RC-D1 done; the finding list, label rules, lesion-size thresholds and
  `image_reliance_check.mask_findings` are written to `configs/domain/endoscopy.yaml`.
- `evaluation.max_new_tokens` and `inputs.image_longest_side` are set (D04 rule; the side from
  RC-S0b).
- Image side from RC-S0b: **Met 2026-10-17:** `inputs.image_longest_side` = 672 (Soheil, D05).
- RC-M1 passed. **Met 2026-10-15.**

Platform: Kaggle notebook, one T4 is enough (`CUDA_VISIBLE_DEVICES=0`), Internet **on**.
Dataset access: your own copy of Kvasir-VQA-x1 and Kvasir-SEG and the private RC-D1 outputs
(`splits.json`, `mask_links.json`) attached as a private dataset. Nothing from those goes into
the repository.

Runtime: one seed is one training run plus a validation pass. Training at side 672 is 625
updates of 16 examples at micro-batch 1 (D06 amendment, 2026-10-17). RC-S0b run 2 measured
0.849 s per one-example step on synthetic probes, so one update takes about 13.6 s and one seed
about **2.4 h** of training. This is an **estimate**: real examples are longer, and the model
load, the saves (every 50 updates) and the validation pass come on top. The validation pass is
unknown until RC-D1 gives the probe counts. Use one session per seed; the seeds are independent
and run on three accounts (section 2b).

**Memory check (first seed, before the other two).** At 672 the step at micro-batch 1 peaked
at 11,372,899,328 bytes on synthetic probes (RC-S0b run 2; the T4 has 15,636,037,632). Real
training examples are longer (an answer, not "Yes"/"No"), so the first seed confirms it on real
data: `train_peak_gpu_bytes` and `train_seconds` / `train_steps` (seconds per update) in
`metrics.csv`. If training stops with "CUDA out of memory", stop there, keep the last 30 lines of
the log and tell the agent through the inbox. Do **not** change the batch yourself: 1 x 16 is the
decided setting (D06).

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** seeds split across the accounts: seed 0 Soheil, seed 1 Farzan, seed 2 Amir (one session each, in the same week). Card owner: Farzan.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E1"`, `SEED = <k>`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-E1 --seed <k> --data-root /kaggle/input/<shared-data>`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - train E1 seed<k> (skipped when its adapter is attached: a later session then runs only the rest)
  - validation E1 seed<k>
  - benchmark E1 seed<k> (waits if a setting is open; the run goes on)
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E1_seed<k>_adapter`.
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
- Per seed: wall time, peak GPU memory (`train_peak_gpu_bytes`), seconds per update
  (`train_seconds` / `train_steps`), final training loss; for seed 0, whether the micro-batch
  of 4 fitted:
- Name of the private datasets with your adapters and `outcomes.npz` files:
- Anything unusual (errors, retries, changed settings):
