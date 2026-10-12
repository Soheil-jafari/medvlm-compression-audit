# Run card RC-xx: <short title>

> Copy to `docs/runs/RC-xx_<slug>.md`. The agent fills sections 1-4; the person who
> runs it fills section 5 and commits the card together with the result CSVs.
> Never paste images, patient identifiers, tokens or dataset paths that reveal
> restricted content.

## 1. What and why

- Stage: S0x
- Experiments: E? [variant] / budgets / seeds
- Purpose: one sentence.
- Status: `ready` | `running` | `done` | `failed`

## 2. Requirements

- Platform: Kaggle 2x T4 / Colab / rented GPU (type)
- Repository commit to run from: `<sha>` on `research-loop` (or `main`)
- Dataset access needed (each member uses their own): ...
- Inputs from earlier runs (team storage, not git): e.g. teacher cache manifest hash
- Expected runtime and GPU memory: ...

## 3. Commands

```bash
git clone https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git checkout <sha>
pip install -r requirements/base.txt -r requirements/train.txt && pip install -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=...        # your own copy
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=/kaggle/working/results
python scripts/run_experiment.py --exp E? --budget B? --seed 0
```

## 4. What to commit back

Only these files, copied into the repository at the same relative paths:

- `results/<ID>[/<variant>]/<budget|nobudget>/seed<k>/metrics.csv`
- `results/<ID>[/<variant>]/<budget|nobudget>/seed<k>/efficiency.csv`
- This run card with section 5 filled.

Check before committing: the results schema test (`pytest tests/test_results.py`,
added in S02) passes, and
`git status` shows no images, caches, checkpoints or per-sample outputs.

## 5. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware (GPU, driver, library versions from `pip freeze | grep -E "torch|transformers|peft"`):
- Wall time and peak GPU memory:
- Manifest / cache hashes printed by the run:
- Anything unusual (errors, retries, changed settings):
