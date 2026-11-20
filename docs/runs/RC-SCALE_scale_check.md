# Run card RC-SCALE: each heavy step at full size for a few steps, failing on memory growth

## 1. What and why

- Stage: before the next long runs (2026-11-14)
- Experiments: the heavy steps of E7 / A1-A6 (first-order pass, budget search, recovery), E4
  (quantization-aware training), E3 / E8 / E9 (GPTQ) and E9's evaluation, seed 0.
- Purpose: RC-SMOKE runs every stage with tiny settings, so it cannot see memory that grows
  with the real batch, the real depth or the number of steps. RC-G2 session 3 ran out of GPU
  memory in E7's first-order pass after the smoke had passed, and the smoke's E7 stage 05 was
  killed by host RAM. This card runs each heavy step **at its real settings** for a few steps,
  reads the GPU and host memory after every step, and **fails when memory grows step over
  step** or when a step peaks within 1 GiB of a T4. It changes no setting and writes nothing to
  `results/`.
- Status: `ready` (2026-11-14)

## 2. Requirements

- **Who:** Soheil (account 1; the three datasets are there). **Kaggle settings:** GPU **T4
  x2** (only E9's evaluation uses both), Internet on.
- **Attach (Add Input):** `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache`; also
  `e9-seed0-quantized` and `e0-teacher-adapter` if both are on the account (E9's parent is
  rebuilt from the adapter; without them E9's check says WAIT).
- **Target: about 1 h** (estimate: each check loads its model once or a few times; the input
  checks hash the attached files in every check's process).

What runs (`src/medcompress/scale_check.py`), one check per process, one after the other:

| Check | Real settings | Shortened | Fails on |
|---|---|---|---|
| `e7_first_order` | E2 student, seed 0's cleaned calibration set, E7's probes, batch 4, gates on every first-order unit (3,096 channel groups) | the 3 batches with the largest images, then the first 40 batches in production order | GPU memory after each step growing; peak above 13.56 GiB |
| `budget_search` | stage 05's model factory and measurer, a real removal set (magnitude order, B2 level) | 8 loads; the footprint with 2 samples, no warm-up, 1 timed pass | host RSS after each release growing |
| `recovery` | a real pruned 3B (magnitude order, 30 % of the decoder's parameters), the frozen recovery recipe (157-update schedule) | stops after 20 updates; then merges and saves the pruned model as stage 05 does | GPU memory or host RSS after each update growing; peak |
| `e4_training` | E4's quantization-aware training | stops after 10 updates | the same |
| `gptq` | E3 `w4`: the parent, 128 calibration images, the frozen recipe | the first 4 decoder layers | GPU memory at each layer start growing; peak |
| `e9_eval` | E9's quantized teacher on two T4s, validation probes | 30 batches | GPU memory after each batch growing; peak above 2 x 13.56 GiB |

The recovery check also prints **how many of its 20 updates the loss scaler skipped** (and at
which updates). If more than 5 % are skipped it prints a note: the agent then tells Soheil
before anything changes (the scaler is part of the frozen recipe).

## 3. Commands

One cell:

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
python scripts/scale_check.py 2>&1 | tee /kaggle/working/scale_check.txt
echo "scale_check exit code: ${PIPESTATUS[0]}"
```

**Second run (2026-11-14), the four checks that failed or were near a limit at 48097bd:** the
same cell with its last two lines replaced by

```bash
python scripts/scale_check.py --check budget_search --check recovery --check e4_training --check e9_eval 2>&1 | tee /kaggle/working/scale_check_2.txt
echo "scale_check exit code: ${PIPESTATUS[0]}"
```

Attach the same datasets, plus `e9-seed0-quantized` and `e0-teacher-adapter` for E9's check.
Each check now gets only the hand-offs its card reads (`run_card.py`'s selection). Estimate:
about 35 min.

**Third run (2026-11-15), `budget_search` only, before RC-G2 session 1:** the same cell with its
last two lines replaced by

```bash
python scripts/scale_check.py --check budget_search 2>&1 | tee /kaggle/working/scale_check_3.txt
echo "scale_check exit code: ${PIPESTATUS[0]}"
```

Attach `medcompress-data` and `e2-seed0-adapter`. It runs stage 05's first load and 14
measurements (a production search's count), each a fresh load, a real removal set and a release,
and fails if the live GPU memory after a release exceeds 1 GiB or grows, or if host RSS grows.
Every load prints this process's allocated, reserved and reusable memory and the memory outside
it; the first release that leaves more than 256 MiB prints the largest live CUDA tensors and what
holds them (`MEDCOMPRESS_LEFTOVER`). Estimate: about 20 min.

**Fourth run (2026-11-15), `budget_search` with the level check, before RC-G2 session 1:** Soheil's
account, Kaggle T4 x2, Internet on; attach `medcompress-data` and `e2-seed0-adapter`. After its
15 loads it measures the unpruned model once more with the reference's full protocol (32 samples,
3 warm-up passes, 5 timed) and fails unless that footprint matches E2's reference
(7,618,471,936 bytes) within 1 %. Estimate: about 30 min.

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
python scripts/scale_check.py --check budget_search 2>&1 | tee /kaggle/working/scale_check_4.txt
echo "exit code: ${PIPESTATUS[0]}"
```

**Check `awq` (2026-11-20, amendment of that date):** E3 `awq_w4`'s AWQ through GPTQModel at full
size on the first 4 decoder layers (the merged E2 cut to them; the real 128 calibration examples),
then the reload with `AwqTorchLinear` and one forward pass; it prints the seconds per layer, the
peak GPU memory and an estimate for all 36 layers. Run with `--check awq` (RC-E3 section 3b, cell
(a)).

## 4. What to send back

`/kaggle/working/scale/scale_report.md` (the table), `/kaggle/working/scale_check.txt` (the
log), and, if a check failed, its log from `/kaggle/working/scale/logs/`. Nothing to commit
from this card: the agent records the outcome.

## 5. How the result is read

- PASS for a check: memory is flat step over step at full size, with at least 1 GiB of a T4
  left. The table gives the GPU peak, the host RSS peak and the lowest available host RAM of
  each check's process.
- FAIL: the line names the quantity that grows, or the peak. The agent fixes the code before
  the matching production run.
- `e7_first_order` also prints the seconds per step and an estimate of the whole first-order
  pass; `recovery` the seconds per update and the skipped updates.
- Expected E7 first-order peak with the per-layer recomputation (2026-11-14): about 8.5-9.5
  GiB (estimate: 6.99 GiB of weights, about 0.35 GiB of kept layer inputs for a batch of four
  672-px probes, one layer recomputed at a time, the vision tower's forward pass).

## 6. Run report

**First run, reported by Soheil on 2026-11-14 (commit 48097bd, 32 min):**

- PASS `e7_first_order`: GPU memory flat at 7.01 GiB after each step, peak 7.71 GiB (the
  estimate was 8.5-9.5), 3.5 s per step.
- PASS `gptq`: flat, peak 8.89 GiB.
- `recovery`: GPU flat, peak 11.43 GiB; **host RSS 23.6 GiB during the updates and 27.53 GiB at
  the save, lowest available 2.9 GiB** (near the host limit; very likely the smoke's exit -9).
  8 of 20 updates skipped by the loss scaler (updates 1-7 and 11: the scale settling).
- FAIL `budget_search`: 7.12 GiB were still on GPU 0 when the first measurement loaded its
  model (the check's own first load).
- FAIL `e4_training`: `torch.cuda.reset_peak_memory_stats(device)` refused ("Invalid device
  argument") before CUDA was in use, in the check itself.
- FAIL `e9_eval`: the check passed every attached hand-off to E9, among them an unrelated E1
  adapter, which was then compared with the teacher's settings.

What the agent changed (2026-11-14):

- Host RAM of the recovery: every cached teacher record kept a full copy of its shard's arrays
  alive (`NpzFile` reads a whole array on every access, and each record kept a slice of its own
  copy): with 1,000 records per shard, each record held the shard's three token arrays. Now each
  shard's arrays are read once and each record owns only its rows (the values are the same;
  test). This applies to every run that reads the teacher cache (E1, E2, E4, E10, every
  recovery). The save of the pruned model now writes one tensor at a time from the GPU instead of
  building the whole model in host memory first (the same file; test), and the pruned model is
  loaded straight to the GPU. Stage 05 prints host memory after the examples, the recovery and
  the save. Expected: host RSS a few GiB (estimate); the second run measures it.
- `budget_search`: the first load, the removal set and the release now happen in one helper,
  which checks that the GPU is empty again before any measurement and otherwise fails with the
  largest live CUDA tensors. The reference that kept 7.12 GiB alive was not found on the CPU
  (the same sequence there frees everything); the check will name it if it is still there.
  Production stage 05 releases its first load before the search (`release_model(base)`), and
  its 6 loads in one process each loaded at 9113632 (GPU peak 8.14 GB each).
- `e4_training`: CUDA is initialised before the peak counters are reset.
- `e9_eval` and every other check: only the hand-offs of its card (`select_handoffs`).

**Second run, reported by Soheil on 2026-11-14 (commit 2846e81; the four checks above):**

- PASS `recovery`: host RSS 2.7 GiB during the updates (was 23.6) and 8.0 GiB peak at the save
  (was 27.53); GPU flat, peak 11.43 GiB.
- PASS `e4_training`: GPU flat, peak 11.69 GiB; 7 of 10 updates skipped by the loss scaler at the
  start (recorded, as decided).
- PASS `e9_eval`: two T4s, GPU flat, peak 19.85 GiB.
- FAIL `budget_search` again: 7.12 GiB in use on GPU 0 before a measurement's load, 10.7 min in;
  the live-tensor list was not printed. Production stage 05 did its 6 loads in one process in
  both smokes, so Soheil treats it as the check's own path: **it does not block any card**; the
  agent investigates it (section 6, note below).

Verdict: every heavy production step is flat at full size; the recovery's host memory is far
below the limit. Cards are not gated on `budget_search`.

Agent note on `budget_search` (2026-11-14): the first-load check passed (no tensor list was
printed), so the memory stayed after one of the later measurements, not after the check's own
load. The cause is not found yet: the measurement releases its model in a `finally` block, and on
the CPU the same sequence frees everything. Diagnostics added so the next run names it: after
every measurement's release a `MEDCOMPRESS_LEFTOVER` line gives this process's allocated and
reserved GPU memory (with the largest live CUDA tensors and the types that hold them above 1 GiB),
and a refused load now says how much of the memory in use belongs to this process (the rest is
another process or non-PyTorch memory). A production search has about 14 measurements (the smoke
had 4), so if the next log shows memory left by the measurements, it matters for RC-G2 session 4
and the agent will say so before that session.

**Third run, reported by Soheil on 2026-11-15 (commit 74104b4; `budget_search` only):** PASS by
its own criteria: 15 loads (the first and 14 measurements), live GPU memory after every release flat
at 0.59 GiB, host RSS flat (2.63 to 2.71 GiB), peak 7.66 GiB. The leftover is a float16 tensor of
shape (151,936, 2,048), the size of one embedding matrix (0.62e9 bytes), plus small float32 LoRA
tensors (16 x 2,048). **But every footprint was offset:** with 0 units removed the search measured
8.246e9 bytes against E2's fresh-process reference 7.618e9 (`results/E2/nobudget/seed0/
efficiency.csv`); the difference, 0.628e9, is the 0.59 GiB still allocated. The removal itself was
right (275 units: 1.669e9 below the unpruned measurement, the expected 30 % of the decoder's
5.55e9). A search would have aimed at a shifted target and removed about 0.63e9 bytes too many.
Fixed (2026-11-15): the timing sweep's memory meter now records the live bytes at its reset and
subtracts what is not the model's own GPU tensors, so every footprint (stage 05's and stage 07's)
is on a fresh process's footing; stage 05 measures the unpruned model first and stops unless it
matches the reference within 1 %; this check now does the same. The holder of the leftover is not
freed (not found); the measurement no longer depends on it.

