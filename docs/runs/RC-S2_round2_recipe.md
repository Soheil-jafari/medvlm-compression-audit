# Run card RC-S2: Round 2, top 5 with the full student recipe

## 1. What and why

- Stage: S05b (decision D05, Round 2)
- Experiments: none (selection runs on the validation split; no budget)
- Purpose: retrain the five candidates that advanced from Round 1 with the full student
  recipe (offline distillation from the cached teacher outputs, about twice the Round 1
  number of examples), score them on the **whole** validation split with the lesion-mask
  audit, and keep three. Before training, each candidate's tokenizer must reproduce
  at least 99 % of the cached teacher answers; a candidate that does not is stopped and
  recorded as a gate failure.
- Status: `blocked` (see section 2)

## 2. Requirements

- RC-S1 ranked and committed (`results/selection/R1/ranking.csv`); the five names
  that advance are the ones with `advance` = 1 there. The ranking command of this round
  reads that file, so the list cannot drift.
- RC-E0b part (a) done: the teacher cache, its manifest hash written in
  `docs/decisions_log.md`. Attach the cache as a private Kaggle dataset and set
  `MEDCOMPRESS_CACHE_ROOT` so that `$MEDCOMPRESS_CACHE_ROOT/teacher/manifest.json` exists.
  The loader verifies the manifest hash; compare it with the log before you start.
- `training.kd_weight` and `training.kd_temperature` fixed by the team (they are the E2
  values, the same for every candidate).
- Platform, dataset access and the remote-code rule: as in RC-S1.
- Expected runtime: about twice RC-S1 per candidate plus a full-validation evaluation;
  the agent fills in the estimate when it sets the card to `ready`.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data                # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CACHE_ROOT=/kaggle/input/<teacher-cache>      # contains teacher/
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results

# the five candidates with advance = 1 in results/selection/R1/ranking.csv, shared among you
for c in <candidate> <candidate>; do
  CUDA_VISIBLE_DEVICES=0 python scripts/select_student.py --round R2 --candidate $c
  rm -rf $HF_HOME/hub/models--*
done
python -m pytest -q tests/test_results.py                            # must pass before you commit
```

A candidate whose tokenizer does not reproduce the cache stops after the check and
writes `gate_tokenizer_match` = 0; that is a result, not an error, and the run card needs
no retry. Upload `outputs/selection` as a private dataset (`<yourname>-selection-r2`).

**Ranking (one person).** Same as RC-S1 with `--rank R2` and the `*-selection-r2` datasets.
It reads the Round 1 advance list and stops if an expected candidate has no files.

## 4. What to commit back

- `results/selection/R2/<candidate>/metrics.csv` for each of your candidates
- `results/selection/R2/ranking.csv` and the updated `docs/results/SELECTION.md` (the person who ranks)
- This card with section 6 filled.

## 5. How the result is read

As in RC-S1, with the image-reliance component now the **lesion-mask effect**
(flip rate when the lesion is masked minus flip rate when a control region of the same
size is masked). The three highest points estimates advance to Round 3. Gate G3 (does
the candidate compress: 25 % decoder-layer removal and 4-bit PTQ run and cut measured
memory) is checked by the agent's code once the compression stages exist and is
recorded under `manual_gate_failures` until then.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions:
- Teacher cache manifest hash seen by the loader (must equal the logged one):
- Per candidate: tokenizer match share, wall time, peak GPU memory, errors or retries:
- Name of the private dataset with your `outcomes.npz` files:
- Anything unusual:
