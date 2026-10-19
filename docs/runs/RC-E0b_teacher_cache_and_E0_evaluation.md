# Run card RC-E0b: teacher cache (train) and E0 validation metrics

## 1. What and why

- Stage: S05
- Experiments: E0, no budget, seed 0
- Purpose: (a) generate the teacher's answers and top-20 token log-probabilities for the
  training split once, for every distillation and recovery run; (b) score the teacher
  on the **validation** split with the full evaluation suite (the reference row of
  every comparison).
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent changes the status to `ready` when
they are:

- RC-E0a done, `gate1_pass` = 1, and the team confirmed the verdict in
  `docs/decisions_log.md`. If Gate 1 fails, this card is not run.
- RC-D1 done; finding list, label rules, prevalence groups and lesion-size thresholds
  written into `configs/domain/endoscopy.yaml`; `image_reliance_check.mask_findings` set.
- The open settings of `configs/base.yaml` filled from the RC-D1 counts and the team's
  recipe decisions: `evaluation.bootstrap_resamples`, `evaluation.calibration_bins`,
  `evaluation.max_new_tokens`, `evaluation.generation_samples`,
  `teacher_cache.max_new_tokens`, `teacher_cache.pool_size` (= `training.data_subset`, set
  from the RC-S0 speed), and `inputs.image_longest_side`. Part (a) caches only that training
  pool (drawn with `seeds.data_split`), not all 143,594 training rows.
- Image side from RC-S0b: **Met 2026-10-17:** `inputs.image_longest_side` = 672 (Soheil, D05).
- **Waiting on RC-TT1 (2026-10-18):** the teacher is now Qwen2.5-VL-7B-Instruct with **our**
  LoRA adapter, trained on our training split only (D01 amendment). The published SimulaMet
  adapter has seen almost every image of our test split. Attach the RC-TT1 hand-off
  (`E0_teacher_adapter`) with `--from`; `configs/models/qwen2_5_vl_7b_teacher.yaml` points at
  it, and the run stops before any download if it is missing.
- RC-D1 **run 4** (image-disjoint split, near-duplicates by embeddings with the rule chosen on
  human labels, D02 amendment 3) replaces runs 1 to 3: their splits are not used.
- RC-M1 passed (gives the generation speed that decides how many sessions part (a) needs).
  **Met 2026-10-15:** the teacher generated 4.48 tokens per second (`results/smoke/RC-M1.csv`).

Platform: Kaggle notebook, **2 x T4**, Internet **on**. Dataset access: your own copy of
Kvasir-VQA-x1 and Kvasir-SEG, and the **private** RC-D1 outputs (`splits.json`,
`mask_links.json`) attached as a private Kaggle dataset. Nothing from those goes into
the repository.

Runtime: part (a) depends on the number of training rows and the generation speed
measured in RC-M1; it is **resumable** (finished shards are skipped), so if a session
ends, download `cache/teacher/`, attach it to the next session as a private dataset
(or copy it back into `MEDCOMPRESS_CACHE_ROOT`) and run the same command again.
Part (b) takes minutes to an hour.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil.
- **Kaggle settings:** GPU T4 x2 (both T4s); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher adapter.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E0b"`, `SEED = None`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-E0b --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off>`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - teacher cache
  - E0 validation
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E0_teacher_cache`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). This card uses T4 x2. A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data               # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results

# (a) cache the teacher's answers on the training split (prints the manifest hash at the end)
python scripts/01_cache_teacher_outputs.py --exp E0 --seed 0

# (b) E0 evaluation on the validation split
python scripts/06_evaluate.py --exp E0 --seed 0
python -m pytest -q tests/test_results.py                           # must pass before you commit
```

If a script stops with "configuration incomplete", a setting above is still open: do
not fill it in yourself, tell the agent through the inbox.

Keep `cache/teacher/` on team storage (private Kaggle dataset); it is never committed.

## 4. What to commit back

- `results/E0/nobudget/seed0/metrics.csv` (rows of the Gate 1 check from RC-E0a are
  kept; this run adds its own rows)
- This card with section 6 filled, including the manifest hash.

## 5. How the result is read

The agent writes `docs/results/E0.md` from the CSV: clinical metrics, small-finding
recall by lesion-size stratum, rare-finding recall, calibration, image-reliance and
free-text metrics of the teacher on validation. The manifest hash goes into
`docs/decisions_log.md`; every later cache consumer checks against it.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Number of sessions, wall time per part:
- Cache manifest hash (printed at the end of part (a)):
- Anything unusual (errors, retries, changed settings):
