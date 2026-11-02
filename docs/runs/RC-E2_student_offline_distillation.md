# Run card RC-E2: student with offline distillation (E2), 3 seeds

## 1. What and why

- Stage: S06
- Experiments: E2, no budget, seeds 0, 1 and 2
- Purpose: train the chosen student with the same recipe as E1 plus the teacher's cached
  top-20 token log-probabilities (KD weight 0.5, temperature 2), evaluate it on the validation
  split and measure its footprint. E2 is the starting point of every compression arm and the
  reference of the budgets.
- Status: `ready` (2026-11-02), to start **after E1 seed 0's validation row is committed**: the training reads rule 3 from the committed rows and refuses to start until it can (exit code 4, before the model is loaded).

## 2. Requirements

All met except the last, which the training checks itself:
- everything in RC-E1 section 2;
- RC-E0b done: the teacher cache `e0-teacher-cache`, manifest
  `d6ba0cb9d166cb2ae9a5dd8af5a1bbd2eebba904df0ccd795aded0d65b4f15d8`. The hash is pinned in
  `configs/training.yaml` (`teacher_cache_manifest_sha256`, and in `configs/recovery.yaml`):
  every reader of the cache (E2, E4, the recoveries) refuses another cache, before any record is
  read;
- `training.steps` and `training.data_subset` the same as E1's (only the training signal
  differs);
- **rule 3's quality part** (pre-registration section 6): E0's validation macro balanced
  accuracy (0.803) must be at least E1 seed 0's + 0.02. If it is not, the teacher is trained
  once more (+50 % of N) and the cache rebuilt **before** E2, so `02_train_student.py --exp E2`
  reads both committed rows first and stops with `rule 3 ...` until E1 seed 0 is committed, or
  if the teacher fails. After the one extension the teacher is accepted either way.

Hours per seed (estimates): training 625 updates x 13.6 s = **2.4 h** (RC-S0b run 2, side 672,
synthetic probes; real examples are longer, so it may be slower: RC-TT1 ran 3 % over its own
projection), plus the validation pass, **about 0.5 h** (E0's measured pass took 0.78 h; the
student scores probes in 0.414 s instead of 0.584 s and generates about 1.9 times faster), plus
the setup (install, the 7.5 GB student download, loading; not measured). So **about 3 h per
seed** on T4 x2, which counts once against the quota; the student uses one T4.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** seeds split across the accounts: seed 0 Soheil, seed 1 Farzan, seed 2 Amir (one session each, in the same week). Card owner: Soheil.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher cache.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E2"` and `SEED = <k>` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-E2 --seed <k>`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - train E2 seed<k> (skipped when its adapter is attached: a later session then runs only the rest)
  - validation E2 seed<k>
  - benchmark E2 seed<k> (waits if a setting is open; the run goes on)
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E2_seed<k>_adapter`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

Kaggle settings: **GPU T4 x2** (one T4 is used), Internet **on**, **Save Version -> Save & Run
All**. Add Input: `medcompress-data` and `e0-teacher-cache`. One cell; edit only `SEED`:

```bash
%%bash
SEED=0   # Soheil 0, Farzan 1, Amir 2: the only line to edit
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
python - <<'PY' || pip uninstall -y -q torchao   # peft needs torchao 0.16.0 or newer, or none
import sys
from importlib import metadata
try:
    version = tuple(int(x) for x in metadata.version("torchao").split(".")[:2])
except metadata.PackageNotFoundError:
    sys.exit(0)
sys.exit(0 if version >= (0, 16) else 1)
PY
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export CUDA_VISIBLE_DEVICES=0                                   # the student trains on one T4
DATA=$(python scripts/find_inputs.py data) || exit 1          # found by its manifest
python scripts/prepare_data.py verify --data-root $DATA         # must print OK (8,511 files)
CACHE=$(python scripts/find_inputs.py handoff E0_teacher_cache) || exit 1
python scripts/handoff.py verify $CACHE                         # must print OK
python scripts/02_train_student.py --exp E2 --seed $SEED --data-root $DATA --from $CACHE \
    2>&1 | tee /kaggle/working/rc_e2_seed${SEED}_train.txt
code=${PIPESTATUS[0]}; echo "train exit code: $code"
[ "$code" = 0 ] || { echo "STOP: the training did not finish (see above)"; exit 1; }
python scripts/06_evaluate.py --exp E2 --seed $SEED --data-root $DATA \
    2>&1 | tee /kaggle/working/rc_e2_seed${SEED}_validation.txt
echo "validation exit code: ${PIPESTATUS[0]}"
python scripts/handoff.py export --kind adapter --exp E2 --seed $SEED \
    --out /kaggle/working/handoff/E2_seed${SEED}_adapter
python -m pytest -q tests/test_results.py                       # must pass before you commit
mkdir -p /kaggle/working/results_to_commit/results
cp -r results/E2 /kaggle/working/results_to_commit/results/
grep -E "train_|n_probes_scored|macro_balanced_accuracy|sensitivity|specificity|auroc|yes_rate|small_finding_recall|lesion_mask_effect|evaluation_seconds" \
    results/E2/nobudget/seed${SEED}/metrics.csv
```

What it prints, in order:
- `OK: 8511 files match the manifest ...` and `OK: ... files match the manifest ...` (the
  cache);
- `rule 3 (teacher quality): E0 0.8028 >= E1 seed 0 <x> + 0.02 ...`. If it prints
  `configuration incomplete: rule 3 ...` instead, E1 seed 0 is not committed yet, or the
  teacher failed: the cell stops, nothing is trained;
- the training, the validation, the hand-off and the rows, as in RC-E1.

Then: make `/kaggle/working/handoff/E2_seed<k>_adapter` ONE private Kaggle dataset (name it
`e2-seed<k>-adapter`), shared with the team: every compression run of seed `k` starts from it,
and RC-B0, RC-G2 and RC-R1 read it. Commit
`results_to_commit/results/E2/nobudget/seed<k>/metrics.csv`. The benchmark step is not in this
cell (it waits on the hardware setting); RC-B0 measures E2 seed 0's footprint.

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
- Teacher cache: attached `e0-teacher-cache`; the training checks its manifest against the pin
  (d6ba0cb9...) and the rule 3 line it printed:
- Per seed: wall time, peak GPU memory, final training loss and KD term:
- Name of the private datasets with your adapters and `outcomes.npz` files:
- Anything unusual (errors, retries, changed settings):

## 7. Training-length check

Seed 0 of this card is the 10,000-example side of the training-length check (pre-registration
section 6); its 30,000-example counterpart is run card RC-L1.
