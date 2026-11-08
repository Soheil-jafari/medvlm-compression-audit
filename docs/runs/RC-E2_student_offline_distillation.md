# Run card RC-E2: student with offline distillation (E2), 3 seeds

## 1. What and why

- **Seeds (seed plan, pre-registration section 11 item 12, Soheil and Amir, 2026-11-08):** E2 is in the main-claim tier: seeds 0, 1 and 2. Seeds 1 and 2 run only after seed 0's futility check (`scripts/choose_comparator.py`, after RC-G2): if E7 at B2 ranks below the comparator on validation, seed 0, they do not run. The plan refuses a seed 1 or 2 session before `results/seed_plan/comparator_B2.json` is committed, or when it says stop.
- Stage: S06
- Experiments: E2, no budget, seeds 0, 1 and 2
- Purpose: train the chosen student with the same recipe as E1 plus the teacher's cached
  top-20 token log-probabilities (KD weight 0.5, temperature 2), evaluate it on the validation
  split and measure its footprint. E2 is the starting point of every compression arm and the
  reference of the budgets.
- Status: `in progress` (2026-11-08): seed 0 done (section 6); seeds 1 and 2 on the fix commit
  (after 8e0b3d7), after seed 0's futility check (seed plan above). Rule 3 passes (E0 0.803 >= E1 seed 0 0.592 + 0.02).

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

Hours per seed (measured on seed 0, 2026-11-08; seeds 1 and 2 are estimated from it):
training 625 updates x **27.2 s** = **4.72 h** (16,975 s), plus the validation pass, **0.53 h**
(1,906 s), plus the setup (install, the 7.5 GB student download, loading; not measured). So
**about 5.2 h per seed** plus the setup, on T4 x2, which counts once against the quota; the
student uses one T4. The earlier estimate of 13.6 s per update (RC-S0b run 2, about 3 h per
seed) held for E1 (13.5-14.7 s measured) but not for E2: the split distillation step (8e0b3d7)
runs a second forward and backward on every micro-batch whose teacher answer differs from the
ground truth.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** seed 0 Soheil (done 2026-11-08); seed 1 **[team]** (no owner yet), seed 2 Amir, both after the futility check (seed plan of 2026-11-08). Card owner: Soheil.
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
SEED=0   # 0 (done); 1 [team] and 2 Amir only after the futility check: the only line to edit
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone,
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # so removing the clone keeps
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # every output
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

- 2026-11-07 E2 seed 0 OOM on one T4 (torch.OutOfMemoryError, 14.56 GiB, in `_step_loss`: when the teacher's cached answer differs from the ground truth, two forward passes kept both autograd graphs alive until one backward). Fix: the CE term is backpropagated first and its graph freed, then the teacher batch is forwarded and the KL term backpropagated (`offline_kd._step_backward`); no change to loss or settings; E2 seeds 0, 1, 2 all run on the fix commit.
Seed 0 reported by Soheil (2026-11-08); `metrics.csv` and `environment.json` committed
unchanged (b5f6197; the upload `environment.txt` renamed to `environment.json`, content the
same). Seeds 1 and 2: not run yet.

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `train_steps` / examples | 625 / 10,000 | | |
| `train_seconds` | 16,975 (4.72 h; 27.2 s per update) | | |
| `evaluation_seconds` | 1,906 (0.53 h) | | |
| `train_peak_gpu_bytes` | 12,656,672,768 (of the T4's 15,636,037,632) | | |
| `train_loss_final_finite` (finite steps of the last 10) | 0.3244 | | |
| `train_loss_final` (all of the last 10) | NaN (no row: non-finite rows are not written) | | |
| non-finite steps / micro-batches / skipped updates | 16 / 16 / 16 (12 of the skips after step 400) | | |
| validation `macro_balanced_accuracy` (plain) | 0.553 (0.535-0.574) | | |
| `op_macro_balanced_accuracy` (specificity 0.95 on validation) | 0.836 (0.812-0.859) | | |

- Run by / date: seed 0 Soheil, finished 2026-11-08 (after the two fixes of 2026-11-07).
- Commit actually used: a commit after 8e0b3d7 (the split backward). The exact hash is the
  `commit` field of the `e2-seed0-adapter` hand-off's `bundle_manifest.json`; it is not
  copied here until it is read from there (`python scripts/handoff.py verify <folder>` prints
  it as `commit ...`). TODO: Soheil fills it in.
- Hardware and library versions (`results/E2/nobudget/seed0/environment.json`, both stages
  the same): one Tesla T4 (of T4 x2), driver 580.178.04, Python 3.13.16, torch 2.11.0+cu130
  (CUDA 13.0, cuDNN 92700), transformers 5.19.0, peft 0.21.2, accelerate 1.15.0, torchao
  0.18.0, bitsandbytes 0.50.2, optimum 2.3.0, TensorFlow 2.21.0. The same torch build as E1.
- Teacher cache: attached `e0-teacher-cache`; the manifest matched the pin (d6ba0cb9...). Rule
  3 line: `E0 0.8028 >= E1 seed 0 0.5923 + 0.02` (passes).
- Time against E1: E2 training took 16,975 s, **1.9 times** E1's mean of 8,908 s (2.0 times
  E1 seed 0's 8,409 s). The split backward runs a second forward and backward whenever the
  teacher's answer differs from the ground truth, so the extra time is expected; the 13.6 s
  per update of the first estimate held for E1 but not for E2 (section 2 now gives 27.2 s). Peak memory is the same as
  E1's (12.66 GB against 12.65 GB): the fix did what it was for.
- Name of the private dataset with the adapter: `e2-seed0-adapter`. `outcomes.npz` stays in
  the Kaggle version's output (`outputs/E2/nobudget/seed0/`).
- Anything unusual: 16 of 625 updates were skipped because a micro-batch's loss was not finite
  (one micro-batch in each of those steps); the adapter's tensors are finite (the training
  checks them and would have stopped with exit 6). Which examples the skipped steps hold:
  section 6b. No retry, no setting changed.

## 6b. The skipped steps (CPU, report only)

The per-step history is in the adapter folder (`train_history.json`: `losses`,
`nonfinite_steps`, `skipped_steps`). On a CPU notebook with `e2-seed0-adapter`,
`medcompress-data` and `e0-teacher-cache` attached:

```bash
%%bash
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
ADAPTER=$(python scripts/find_inputs.py handoff E2_seed0_adapter) || exit 1
python scripts/handoff.py verify $ADAPTER          # prints OK and the commit used
python scripts/inspect_adapter.py $ADAPTER         # the 16 skipped steps and the losses
DATA=$(python scripts/find_inputs.py data) || exit 1
CACHE=$(python scripts/find_inputs.py handoff E0_teacher_cache) || exit 1
python scripts/skipped_steps.py --exp E2 --seed 0 --data-root $DATA --from $CACHE \
    --history "$(find $ADAPTER -name train_history.json | head -n 1)"
```

`skipped_steps.py` prints, for each skipped step, its 16 examples' count whose cached teacher
answer differs from the ground truth (stripped text; the training compares token ids), the
longest and mean teacher answer in tokens, the longest ground-truth answer in words and which
loss term (CE or KL) was not finite; then the same means over the skipped steps and over all
625 steps, and the mean mid-rank of the skipped steps (about 0.5 = no clustering). Which of
the 16 micro-batches was the non-finite one is not recorded. It writes nothing. Tested here on
its step-to-example mapping only, not end to end (no data here). **Not run yet**: whether the
skips cluster on teacher-differs micro-batches or long answers is not known until it runs.

Loss scale (derived, not measured): the GradScaler starts at 65,536 and halves at every
skipped update; it would double only after 2,000 clean updates in a row, more than the 625
of the run. After 16 skips it ends at 65,536 x 0.5^16 = **1.0**. The 16 skips match the 16
non-finite micro-batches one to one, so each skip comes from a NaN or inf loss in a
micro-batch, not from gradients overflowing at a high scale. At a scale of 1.0, small fp16
gradients can underflow late in the run (12 skips after step 400); whether that affected the
last updates is not known from these numbers.

## 7. Training-length check

Seed 0 of this card is the 10,000-example side of the training-length check (pre-registration
section 6); its 30,000-example counterpart is run card RC-L1.
