# Seed 0, phase B: the remaining cards in sessions of at most about 6 h (2026-11-24)

After the futility stop every remaining run is seed 0 and exploratory. This page groups them into
sessions of at most about 6 h, so the three accounts (Soheil, Amir, account 3) can run in
parallel. **Every hour here is an estimate** from measured passes: a B2 pruning run without its
scoring (budget search, recovery, validation, benchmark) took 141-164 min in RC-G2 (E7 157.8 min
with its scores reused); the two test passes about 32 + 50 min; E7's clinical scoring 5,908 s on
128 calibration images; E2's update 27.2 s. About **115 T4 hours** in all.

## 1. Order and the smoke

1. **No smoke needed:** the B1 / B3 runs of E5 (three criteria), E6 and E7 (the same stages as
   their B2 runs, which ran at 2e795a9 / 3aa4ecb; RC-G2's smoke covers them), and E10's test
   evaluation (the evaluation path the other arms ran).
2. **Smoke first (one session, cell S0):** RC-E8, RC-A1-A6, RC-E4, RC-E10Q. A1, A5 and E10Q have
   never passed it (`run_card.py` refuses them until they are in `smoke_passed.yaml`); E8, A2,
   A3, A4, A6 and E4 passed at 9113632 / a237c5c, before the pruning design of 2026-11-16/17, the
   streamed pruned-model load of 2026-11-18 and the quantization fixes, so they smoke again before
   their runs. The smoke now prints, for every job, the hand-offs it was given and any it lacks.
3. Then the sessions of section 3 in any order across the accounts.

## 2. The cell (every session)

Kaggle settings: GPU T4 x2 for S0 (the smoke's lanes), one T4 is enough for the others (T4 x2
costs the same quota, so keep it); Internet on. Attach the inputs of the session's row, then run
this cell with the row's `ARGS` (and `export` line when the row has one):

```bash
%%bash
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
python scripts/find_inputs.py data > /dev/null || exit 1        # stops if the shared data are not attached
python scripts/find_inputs.py handoffs                          # the hand-offs found by their manifests
ARGS="--card RC-E5 --seed 0 --variant magnitude --budget B1"     # the row's ARGS
python scripts/run_card.py $ARGS 2>&1 | tee /kaggle/working/run_card.txt
echo "exit code: ${PIPESTATUS[0]}"
```

Afterwards save `/kaggle/working/handoff/` as ONE private dataset under the row's name, and send
`/kaggle/working/results_to_commit/` and the run report.

**S0, the smoke** (Soheil, T4 x2, about 2.5 h; inputs `medcompress-data`, `e2-seed0-adapter`,
`e0-teacher-cache`; not the pruned datasets: E8's and A1 / A5's jobs use the smoke's own E7
job, so their "NOT ATTACHED: E7 B2 seed0 pruned" line is expected), instead of the
`run_card.py` line:

```bash
python scripts/smoke_cards.py --card RC-E8 --card RC-A1 --card RC-A2 --card RC-A3 --card RC-A4 \
  --card RC-A5 --card RC-A6 --card RC-E4 --card RC-E10Q 2>&1 | tee /kaggle/working/smoke_phase_b.txt
echo "exit code: ${PIPESTATUS[0]}"
grep -E "hand-offs|NOT ATTACHED|PASS|FAIL" /kaggle/working/smoke_phase_b.txt
```

Send `smoke_report.md`; the agent adds each passing card to `smoke_passed.yaml`.

## 3. Sessions

Inputs: **D** = `medcompress-data`, **A** = `e2-seed0-adapter`, **C** = `e0-teacher-cache`.

| # | Account | ARGS | Inputs | Estimate | Save handoff/ as |
|---|---|---|---|---|---|
| G1 | account 3 | `--card RC-E5 --seed 0 --variant magnitude --budget B1` | D, A, C | 4.0 h | `e5-mag-b1-seed0` |
| G2 | account 3 | `--card RC-E5 --seed 0 --variant magnitude --budget B3` | D, A, C | 4.0 h | `e5-mag-b3-seed0` |
| G3 | Amir | `--card RC-E5 --seed 0 --variant activation --budget B1` | D, A, C | 4.0 h | `e5-act-b1-seed0` |
| G4 | Amir | `--card RC-E5 --seed 0 --variant activation --budget B3` | D, A, C | 4.0 h | `e5-act-b3-seed0` |
| G5 | account 3 | `--card RC-E5 --seed 0 --variant taylor --budget B1` | D, A, C | 4.0 h | `e5-tay-b1-seed0` |
| G6 | Amir | `--card RC-E5 --seed 0 --variant taylor --budget B3` | D, A, C | 4.0 h | `e5-tay-b3-seed0` |
| G7 | account 3 | `--card RC-E6 --seed 0 --budget B1` | D, A, C | 4.2 h | `e6-b1-seed0` |
| G8 | Amir | `--card RC-E6 --seed 0 --budget B3` | D, A, C | 4.2 h | `e6-b3-seed0` |
| G9 | Soheil | `--card RC-E7 --seed 0 --budget B1` | D, A, C | 4.3 h | `e7-b1-seed0` (6-7 GB: E7's pruned model and outcomes) |
| G10 | Soheil | `--card RC-E7 --seed 0 --budget B3` | D, A, C | 4.3 h | `e7-b3-seed0` |
| X1 | any | E10's test evaluation (cell below) | D, `e10-seed0-adapter` | 1.8 h | `/kaggle/working/outputs/` as `e10-test-outcomes` |
| Q1 | Amir (after S0) | `--card RC-E8 --seed 0 --budget B2` | D, A, `g2-b2-seed0-pruned-s4` | 3.5 h | `e8-b2-seed0` |
| P1 | account 3 (after S0) | `--card RC-A1 --seed 0` | D, A, C | 4.0 h (E7's effects re-weighted) | `a1-b2-seed0` |
| P2 | Soheil (after S0) | `--card RC-A2 --seed 0` | D, A, C | 4.0 h (re-weighted) | `a2-b2-seed0` |
| P3 | Amir (after S0) | `--card RC-A3 --seed 0` | D, A, C | 4.0 h (random scores) | `a3-b2-seed0` |
| P4 | account 3 (after S0) | `--card RC-A4 --seed 0 --variant n32` | D, A, C | 4.4 h (scoring 0.4 h) | `a4-n32-b2-seed0` |
| P5 | Soheil (after S0) | `--card RC-A4 --seed 0 --variant n64` | D, A, C | 4.8 h (scoring 0.8 h) | `a4-n64-b2-seed0` |
| P6a | Amir (after S0) | A4 `n256` scoring only (cell below) | D, A | 3.3 h (scoring on 256 images) | nothing; send `results_to_commit/` |
| P6b | Amir (after P6a is committed) | `--card RC-A4 --seed 0 --variant n256` | D, A, C | 4.0 h (its scores reused) | `a4-n256-b2-seed0` |
| P7 | account 3 (after S0) | `--card RC-A5 --seed 0 --variant logprob_drop` | D, A, C | 4.0 h (E7's exact scores reused) | `a5-logprob-b2-seed0` |
| P8 | Soheil (after S0) | `--card RC-A5 --seed 0 --variant kl` | D, A, C | 5.6 h (every unit scored exactly, about 1.6 h) | `a5-kl-b2-seed0` |
| P9 | Amir (after S0) | `--card RC-A6 --seed 0 --variant layers_only` | D, A, C | 4.5 h | `a6-layers-b2-seed0` |
| P10 | account 3 (after S0) | `--card RC-A6 --seed 0 --variant heads_channels` | D, A, C | 5.1 h | `a6-heads-b2-seed0` |
| T1-T4 | Soheil, one after another (after S0) | `--card RC-E4 --seed 0`, with `export MEDCOMPRESS_STOP_AFTER_HOURS=3.8` before it; from T2 on also attach the previous session's `e4-seed0-resume-<n>` | D, C (+ the resume dataset) | about 4.2 h each; 2-4 sessions (training 7.1-14.1 h by estimate); the last adds export, validation, benchmark and the tests (about 2 h) | `e4-seed0-resume-<n>` while paused (the run report says PAUSED); `e4-seed0` at the end |
| Z1 | account 3 (after S0) | `--card RC-E10Q` | D, `e10-seed0-adapter` | 5.0 h (check, GPTQ, validation, benchmark, tests) | `e10q-seed0` |

The E5 / E6 B1 and B3 runs export their outcomes (validation and both tests; the analyses
need the validation file too), not their pruned models (only B2's and E7's are read later).
The A-series now export their outcomes too (2026-11-24). E4 pauses itself on the limit: the
card stops as "paused", exports the resume state, and the next session continues from it.

**X1, E10's test evaluation** (E10's validation and benchmark are committed; this runs only the
two test passes, so the committed benchmark is not measured again):

```bash
%%bash
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
python scripts/find_inputs.py data > /dev/null || exit 1        # stops if the shared data are not attached
python scripts/find_inputs.py handoffs                          # the hand-offs found by their manifests
DATA=$(python scripts/find_inputs.py data) || exit 1
FROM=$(python scripts/find_inputs.py handoffs | sed 's/^/--from /' | tr '\n' ' ')
export HF_HOME=/kaggle/temp/hf CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
for SPLIT in test test_unfiltered; do
  python scripts/06_evaluate.py --exp E10 --seed 0 --data-root "$DATA" $FROM \
    -o evaluation.split=$SPLIT 2>&1 | tee -a /kaggle/working/e10_test.txt
  echo "$SPLIT exit code: ${PIPESTATUS[0]}"
done
mkdir -p /kaggle/working/results_to_commit/results/E10/nobudget/seed0
cp results/E10/nobudget/seed0/metrics_test*.csv /kaggle/working/results_to_commit/results/E10/nobudget/seed0/
```

**P6a, A4 `n256` scoring only** (the card's run would take about 7.3 h; its scores are reused
once committed):

```bash
%%bash
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
python scripts/find_inputs.py data > /dev/null || exit 1        # stops if the shared data are not attached
python scripts/find_inputs.py handoffs                          # the hand-offs found by their manifests
DATA=$(python scripts/find_inputs.py data) || exit 1
FROM=$(python scripts/find_inputs.py handoffs | sed 's/^/--from /' | tr '\n' ' ')
export HF_HOME=/kaggle/temp/hf CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
python scripts/04_score_pruning_units.py --exp A4 --variant n256 --budget B2 --seed 0 \
  --data-root "$DATA" $FROM 2>&1 | tee /kaggle/working/a4_n256_scoring.txt
echo "exit code: ${PIPESTATUS[0]}"
mkdir -p /kaggle/working/results_to_commit/results/A4/n256
cp -r results/A4/n256/nobudget /kaggle/working/results_to_commit/results/A4/n256/
```

## 4. PolypGen (RC-X2)

Status: blocked on the download (Amir's Synapse access). Written and waiting: the loader and the
overlap check (`src/medcompress/data/polypgen.py`, `scripts/polypgen.py describe` / `overlap`).
Once access arrives, Amir runs RC-X2 part A (CPU, minutes): record the licence and the Synapse
terms as the release states them; `describe` (centres, masks, counts; numbers only); `overlap`
against our splits; then the fp16 vision scan of PolypGen's images (non-finite images dropped
for every arm, as the second amendment of 2026-11-13 requires). After that the agent writes the
`external_test` split in the evaluation and part B (the per-test comparisons; centre C4, Oslo,
reported apart). Then every arm's model is evaluated on it (polyp endpoints).

## 5. What to send back

Per session: `results_to_commit/` and the run report; the dataset name. The agent commits the
files unchanged (SHA-256, file list and row counts) and reads no test number.
