# Run card RC-E10Q: E10 at 4 bit (exploratory)

## 1. What and why

- Stage: S08 (exploratory baseline; pre-registration section 11 item 14.3)
- Experiments: E10Q, the merged E10 (Qwen3-VL-2B with its E2-recipe adapter, seed 0) quantized
  with the E3 `w4` recipe (GPTQ, groups of 128, the frozen settings of E8 and E9), evaluated on
  validation and timed with the budget protocol.
- Purpose: E10 "at 4 bit against E8" (item 14.3). The code was open at the freeze (section 12
  note) and was written on 2026-11-14. Descriptive, no verdict.
- Status: `ready` **after RC-SMOKE passed for RC-E10Q** (`scripts/run_card.py` refuses it
  before; the smoke job quantizes the smoke's own 2-step E10).

## 2. Requirements

- RC-E10 done (account 3, c6c7cbd, 2026-11-14): its adapter is the private dataset
  `e10-seed0-adapter`.
- The calibration set: 128 training images of seed 0, cleaned by the **2B's own** fp16 check
  (amendment of 2026-11-13: "the model being calibrated"; the 2B has its own vision encoder). The
  card runs that check first (`scripts/check_calibration.py --exp E10Q --seed 0 --size 128`); it
  adds the rows of model `e10` to `results/calibration/seed0_replaced.csv` and keeps the student's
  and the teacher's rows.
- Runtime (estimate): the check about 10 min, GPTQ on the 2B's 28 decoder layers under 1 h, the
  validation pass about 1.7 h, the benchmark about 0.1 h (`docs/schedule.md`); peak GPU memory
  well under one T4 (the 2B's weights are 4.26 GB in 16 bit).

## 2b. Shared data, hand-offs and the unattended run (2026-11-14)

- **Who:** account 3 (where `e10-seed0-adapter` was saved), or any account it is shared with.
- **Kaggle settings:** GPU T4 x2 (the card uses one T4); Internet on (the 2B is downloaded at its
  pinned revision 89644892).
- **Attach (Add Input), all private datasets:** `medcompress-data`, `e10-seed0-adapter`.
- **Unattended:** `python scripts/run_card.py --card RC-E10Q` runs, each in its own process: the
  2B's calibration check, the quantization (skipped when `E10Q_seed0_quantized` is attached), the
  validation pass and the benchmark.
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes one private
  dataset): `E10Q_seed0_quantized` (save it as `e10q-seed0-quantized`).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/`, among them
  `results/calibration/seed0_replaced.csv` with the 2B's rows, `results/E10Q/nobudget/seed0/`.

## 3. Commands

One cell (**Save Version -> Save & Run All**):

```bash
%%bash
set -euo pipefail
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
python scripts/run_card.py --card RC-E10Q 2>&1 | tee /kaggle/working/run_card_RC-E10Q.txt
```

## 4. What to commit back

`results_to_commit/` (the calibration file with the 2B's rows, E10Q's `metrics.csv`,
`efficiency.csv`, `efficiency_setup.json`, `environment.json`) and the run report. The test
evaluations of E10 and E10Q run with the other test evaluations (RC-R1's schedule), then the
comparison step.

## 5. How the result is read

Exploratory and descriptive. The comparison step (`scripts/09_compare_arms.py`) writes, on the
test split at seed 0, `E10_vs_matched_pruned_seed0` (E10 minus the pruned 3B arm, E5 / E6 / E7 at
any budget, whose measured peak memory is nearest to E10's) and `E10Q_vs_E8_seed0` (E10Q minus
the E8 budget whose measured peak memory is nearest to E10Q's): differences with intervals for
the headline accuracy, the small-finding recall, the rare-group recall and the ECE, no p-value.
The arms chosen and their peak memories are in `results/comparisons/choices.json`. Both rules
(E5, E6, E7 only, not the ablations; E8 at the budget nearest E10Q's peak memory) were confirmed
by Soheil and Amir on 2026-11-14 (pre-registration section 10, clarification of 2026-11-14). Both
rows are descriptive, outside H1-H4.

## 6. Run report (filled by the person who ran it)

- Run by / date / commit:
- The 2B's calibration check (images checked, replaced):
- GPTQ wall time, damping retries, peak GPU memory:
- Name of the private dataset with the quantized model:
- Anything unusual:
