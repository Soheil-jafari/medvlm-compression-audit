# Run card RC-E2: student with offline distillation (E2), 3 seeds

## 1. What and why

- Stage: S06
- Experiments: E2, no budget, seeds 0, 1 and 2
- Purpose: train the chosen student with the same recipe as E1 plus the teacher's cached
  top-20 token log-probabilities (KD weight 0.5, temperature 2), evaluate it on the validation
  split and measure its footprint. E2 is the starting point of every compression arm and the
  reference of the budgets.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent changes the status to `ready` when they are:

- Everything in RC-E1 section 2.
- RC-E0b done: the teacher cache for the training split exists and its manifest hash is in
  `docs/decisions_log.md`; E0 validation metrics are committed.
- `training.steps` and `training.data_subset` are the same values as for E1 (the only
  difference between E1 and E2 is the training signal).

Platform, dataset access and environment: as in RC-E1. In addition the teacher cache must be
attached as a private dataset (or copied to `MEDCOMPRESS_CACHE_ROOT`, with the shards in
`<cache root>/teacher/`). The run checks every shard against the cache's own manifest and stops
if one was changed. It does not compare the manifest hash with the logged one, so you do that
yourself with the command in section 3 before training.

Runtime: as E1 plus the reading of the cache. The agent fills in the estimate when it sets the
card to `ready`.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** seeds split across the accounts: seed 0 Soheil, seed 1 Farzan, seed 2 Amir (one session each, in the same week). Card owner: Soheil.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher cache.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E2"`, `SEED = <k>`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-E2 --seed <k> --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off>`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - train E2 seed<k> (skipped when its adapter is attached: a later session then runs only the rest)
  - validation E2 seed<k>
  - benchmark E2 seed<k> (waits if a setting is open; the run goes on)
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E2_seed<k>_adapter`.
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

python -m pytest -q tests/test_training.py tests/test_training_config.py tests/test_teacher_cache.py
python -c "from pathlib import Path; from medcompress.models.teacher_cache import cache_manifest_hash as h; print(h(Path('$MEDCOMPRESS_CACHE_ROOT/teacher')))"
# the printed hash must equal the one in docs/decisions_log.md; if not, stop and tell the agent
# one seed per session: 0, then 1, then 2
python scripts/run_experiment.py --exp E2 --seed 0        # trains, evaluates, benchmarks
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

Keep `checkpoints/` as a private dataset named `<yourname>-e2` (the dataset root holds
`E2/seed<k>/adapter`): every pruning and quantization run loads the E2 adapter of its own seed.
It is never committed.

## 4. What to commit back

For each seed `k`:

- `results/E2/nobudget/seed<k>/metrics.csv`
- `results/E2/nobudget/seed<k>/efficiency.csv`
- This card with section 6 filled.

The footprint of seed 0 is the first reading of the budget reference; RC-B0 repeats it on a
clean GPU and its numbers are the ones used.

## 5. How the result is read

The agent writes `docs/results/E2.md` (the same tables as E1, with the difference E2 minus E1
and its paired interval). The agent does not tune anything on these numbers: the recipe is fixed
(D06). A student that fails the image-reliance checks is reported to the team before anything
is compressed.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Teacher cache manifest hash printed by the command in section 3 (must equal the logged one):
- Per seed: wall time, peak GPU memory, final training loss and KD term:
- Name of the private datasets with your adapters and `outcomes.npz` files:
- Anything unusual (errors, retries, changed settings):

## 7. Training-length check

Seed 0 of this card is the 10,000-example side of the training-length check (pre-registration
section 6); its 30,000-example counterpart is run card RC-L1.
