# Run card RC-SMOKE: every card still to run, end to end with tiny settings

## 1. What and why

- Stage: before every remaining Kaggle run (2026-11-13)
- Experiments: the real stages of RC-G2 (E5 magnitude, activation, taylor; E6; E7 and its
  30-group check), RC-E3 (w4 and w8a8 end to end; w4 and w8 layer 0 through stage 03's own
  capture with all 128 real calibration examples, then GPTQ on that layer), RC-E4, RC-E8 (B2), RC-E9 (the evaluation path, from the
  attached quantized hand-off), RC-E10 (training), RC-A1-A6 and RC-R1 (E0 validation and test,
  E1 and E2 test), seed 0.
- Purpose: catch every crash, offload, out-of-memory and hand-off refusal in minutes, before a
  real run spends hours. **A gated card runs only after RC-SMOKE passed for it**
  (`scripts/run_card.py` refuses it otherwise; `docs/runs/smoke_passed.yaml`, written by the
  agent from the reported table).
- Status: `ready` (2026-11-13)

## 2. Requirements

- **Who:** Soheil (account 1). **Kaggle settings:** GPU **T4 x2**, Internet **on** (the 2B
  student of E10 is downloaded).
- **Attach (Add Input), all private datasets:** `medcompress-data`, `e2-seed0-adapter`,
  `e0-teacher-cache`, `e0-teacher-adapter`, `e1-seed0-adapter` (RC-R1's E1 test) and
  `e9-seed0-quantized` (shared to Soheil's account from account 3).
- **Target: under 1.5 h** (estimate: two lanes of about 40-50 min each, one per T4, then about
  12 min of teacher jobs on both T4s, plus about 5 min of installation; the loads dominate).
- What is tiny (`src/medcompress/smoke.py`): 2 calibration images; the budget search's two
  end points and at most 2 bisection steps; 2 recovery and 2 training updates; for evaluation
  the first validation images that give every finding 2 positive and 2 negative images plus one
  image with a mask (about 12 probes: fewer cannot fit every finding's operating point), 4
  free-text answers, 20 bootstrap resamples; the benchmark and every footprint of the search
  with 1 sample, no warm-up and 1 timed pass. Everything else is the real code and the real
  settings.
- RC-E3's layer-0 checks (`scripts/check_gptq_inputs.py --layers 0 --gptq`) use the real 128
  calibration examples, not 2: RC-E3 seed 0 stopped on a NaN Hessian at layer 0 (2026-11-13),
  and a tiny calibration could miss the example that caused it. They print every batch's
  finiteness step by step and fail on any non-finite input.
- What is not run in full, and why: E3 `w8` runs only its layer-0 check (its other stages are
  `w4`'s code; RC-Q0 ran its kernels end to end); E5 activation and taylor run stage 04 only (their stages
  05-07 are E5 magnitude's code); A2, A4 (n32), A5 (kl) and A6 (heads_channels) score their
  first 300 units and run stage 04 only (their full scoring takes hours even on 2 images; their
  stages 05-07 are E7's code, which the E7 and A1 jobs run in full). E8 quantizes the smoke's own
  E7 model; A1 re-weights the smoke's own E7 effects.
- Nothing is written to `results/`: the stages use a temporary copy of it under
  `/kaggle/working/smoke/results` (stage 05 reads RC-B0's reference there), and a stage refuses
  the smoke settings with the repository's results root. Nothing to commit.

## 3. Commands

One cell (**Save Version -> Save & Run All**):

```bash
%%bash
set -euo pipefail
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # (smoke_cards.py moves them
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # under /kaggle/working/smoke)
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
python scripts/find_inputs.py data > /dev/null
python scripts/find_inputs.py handoffs
python scripts/smoke_cards.py 2>&1 | tee /kaggle/working/smoke_cards.txt
```

The table is also saved as `/kaggle/working/smoke/smoke_report.md`, and each stage's full log
under `/kaggle/working/smoke/logs/`. One card or more only: `python scripts/smoke_cards.py
--card RC-E9 --card RC-G2`.

## 4. What to send back

`/kaggle/working/smoke/smoke_report.md` (the table: card, job, stage, PASS / WAIT / FAIL,
minutes, the peak GPU memory of every load and of every stage), and for any FAIL the last
lines of its log. Nothing is committed from this card; the agent records the passed cards in
`docs/runs/smoke_passed.yaml`.

## 5. How the result is read

A card passes when every stage of its jobs passed. WAIT means a stage stopped on an open input
or setting (exit code 4): the agent reads the log and decides. A FAIL keeps the card closed
until the agent fixes it and RC-SMOKE passes again for it. The peaks are the planning numbers
for the real runs (the real evaluation and benchmark use more images, so their activations are
larger; the weights and the GPTQ buffers are the same).

## 6. Run report

- Run by / date:
- Commit:
- The table (`smoke_report.md`):
