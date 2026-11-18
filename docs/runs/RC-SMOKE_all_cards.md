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
- First of all, on T4 0, the smoke runs the fp16 calibration check of seed 0 (student, 128
  images; `scripts/check_calibration.py`) into its own temporary results, which every later
  calibration set reads (amendment of 2026-11-13).
- The test evaluations (RC-R1's E0, E1 and E2, RC-E3 `w4`) apply the committed drop list of
  non-finite test images (second amendment of 2026-11-13); without it they wait (exit 4).
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

**Reduced RC-SMOKE (2026-11-13; RC-R1 added 2026-11-14), after RC-CAL0's calibration file and the drop list of non-finite test images are committed:** the same cell
with its last line replaced by

```bash
python scripts/smoke_cards.py --card RC-E3 --card RC-E4 --card RC-A1 --card RC-A4 --card RC-A5 \
  --card RC-A6 --card RC-R1 2>&1 | tee /kaggle/working/smoke_cards_reduced.txt
```

`--card` also runs what those jobs read: the calibration check first, and E7 (A1 re-weights E7's
effects). Each job's checkpoints are deleted once no later job reads them, and the table shows the
disk used and free after every stage. The unit cap of A2, A4, A5 and A6 now keeps whole layers
(the cut through layer 3's channel groups failed A4 and A6 at 9113632); A5 `kl` scores its 108
layers and KV groups. Estimate: about 1 h.

**E7's stage 05 again (2026-11-14), after the host-RAM fix:** the same cell with its last line
replaced by

```bash
python scripts/smoke_cards.py --card RC-G2 --card RC-A5 --sequential 2>&1 | tee /kaggle/working/smoke_cards_g2.txt
```

`--sequential` runs lane B after lane A (half the host RAM, about twice the time). Every stage
now runs under a watcher that samples its resident memory and the machine's available memory
from outside (a stage killed by the kernel still has a reading); the table has a column "Host
RSS peak / lowest available (GiB)" and marks an exit -9 as a likely host-RAM kill. Each model
load and release also prints a `MEDCOMPRESS_MEM` line (GPU and host). Estimate: about 1.5 h.

**E7's stage 05 alone (2026-11-15), before RC-G2 session 4:** the smoke's E7 job (scoring with tiny
settings, then stage 05, validation and benchmark) at the current code, with the calibration
check it reads first. Soheil's account, Kaggle T4 x2, Internet on; attach `medcompress-data`,
`e2-seed0-adapter`, `e0-teacher-cache`. The level check is printed and skipped in the smoke (its
footprints use the shortened protocol). Estimate: about 40 min.

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
python scripts/smoke_cards.py --job E7 2>&1 | tee /kaggle/working/smoke_cards_e7.txt
echo "exit code: ${PIPESTATUS[0]}"
```

**E5 magnitude's stage 05 at the new code (2026-11-16), before the RC-G2 reruns:** the smoke's E5
magnitude job (scoring, stage 05, validation, benchmark with tiny settings) with the calibration
check it reads first. Soheil's account, Kaggle T4 x2, Internet on; attach `medcompress-data`,
`e2-seed0-adapter`, `e0-teacher-cache`. It checks the amendment of 2026-11-16 in the real stage:
`removal_by_layer.csv` must show 0 for layers 0, 1 and 35 and the metrics
`removed_units_in_protected_layers` 0. Estimate: about 20 min.

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
python scripts/smoke_cards.py --job "E5 magnitude" 2>&1 | tee /kaggle/working/smoke_cards_e5_magnitude.txt
echo "exit code: ${PIPESTATUS[0]}"
```

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

**Reduced RC-SMOKE, reported by Soheil on 2026-11-14 (commit a237c5c):** PASS for RC-CAL0, RC-E3,
RC-E4 (QAT peak 11.29 GiB), RC-A4, RC-A6 and A5's `kl` scoring; RC-E3, RC-E4, RC-A4 and RC-A6
added to `smoke_passed.yaml`. RC-G2's E7 stage 05 (prune and recover) was killed with exit -9
after 16.9 min (it passed at 9113632), so RC-A5 (whose `logprob_drop` run reads E7) and RC-A1
did not pass. What the code shows (2026-11-14; the log itself was not attached): with the smoke's
settings stage 05 loads the student 6 times, all before the recovery (stage 05's first load for
the units, the search's two end points and its 2 bisection steps, then the model to recover);
after the recovery stage 05 loads nothing: it merges the adapter and saves the pruned model,
and `safetensors` builds the saved tensors in host memory first (about the model's size).
Likely sequence (an inference, to be confirmed by the new RSS column): the other lane was loading
the student at the same time (its own loads, with hub requests), the two processes together ran
out of host RAM, and the kernel killed the larger one (E7's, during its save). A production
stage 05 runs alone but loads the student about 16 times (the first load, about 14 measurements,
the model to recover), so a per-load leak would hit it as well. Fixed in the loader regardless
of the cause: the release now drops the processor, empties PyTorch's pinned-memory cache and
returns freed heap pages to the system (`malloc_trim`); loads after the first of a pinned model
make no hub requests (`local_files_only`); 15 loads in one process keep the RSS flat
(`tests/test_host_memory.py`; the same test catches loads that are never freed). RC-SCALE
measures it at full size (`budget_search`, and the save in `recovery`).

**RC-SMOKE at 5aa8c69 (cell (a) of RC-G2, 2026-11-17), reported 2026-11-18:** every stage-06
peak was 14.09 GiB and E7's reached 14.40 GiB and died: loading a pruned model kept the replaced
projections allocated and built the whole state dict on the GPU (fixed 2026-11-18: the surgery
frees what it replaces, the weights are streamed one tensor at a time; RC-G2 section 3, "Fix of
2026-11-18"). In the first attempt E7's stage 04 died with `FileExistsError`: the two lanes
attached the teacher cache at the same instant (check, then `symlink_to`); the link creation now
accepts a link made at the same moment by the other lane when it points at the same source. The
full-size check of RC-G2 section 3, cell (a), replaces a new smoke of stage 06: it loads the real
E5R model through the fixed path.

