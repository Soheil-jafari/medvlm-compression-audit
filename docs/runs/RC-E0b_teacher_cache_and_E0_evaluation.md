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
