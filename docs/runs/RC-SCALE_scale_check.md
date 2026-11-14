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

- Run by / date / commit:
- Table:
