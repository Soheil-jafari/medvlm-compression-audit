# Run card RC-FBR: finding-balanced recovery (secondary study, exploratory)

## 1. What and why

- Pre-registration section 10, amendment of 2026-11-22; specification and every fixed value:
  `docs/decisions/fbr_study_2026-11-22.md`. Exploratory, outside H1-H4.
- Developed after the futility stop from seed-0 **validation** results (Gate 2, the plain-answer
  collapse, E5R's oesophagitis drop). No test or PolypGen result informed it.
- The recipe is the registered recovery with each update split into 10 ordinary examples and 6
  labelled training probes, plus a paired Huber margin term against E2's cached margins
  (coefficient 0.25; the sampling-only control has 0). Nothing is tuned.
- Status: `ready` for cells (i), (ii) and (iii); cell (iv) after (ii) and (iii), and only if
  (iii)'s estimate keeps the pilot under the cap.

## 2. Requirements

- **Cap: 30 T4 GPU hours** for the pilot and its checks (cells (i)-(iv) together). Cell (iii)'s
  timed 20-update trial replaces the training estimate below; if the pilot would exceed the cap,
  stop and report (the recipe is not shortened).
- Code: `src/medcompress/compression/fbr.py` (recipe), `fbr_stage.py` (GPU steps),
  `src/medcompress/evaluation/fbr_analysis.py` (diagnostic, cross-fitting, gate),
  `configs/fbr.yaml` (fixed values), scripts `fbr_probe_answers.py`, `fbr_e2_margins.py`,
  `fbr_recover.py`, `fbr_analysis.py`. Tests: `tests/test_fbr.py`, `test_fbr_torch.py`,
  `test_fbr_analysis.py`.
- Outputs: arrays to `<output_root>/fbr/` (keep as private datasets; never commit them, nor
  `fbr_e2_margins.json`: they hold sample ids); rows and summaries to
  `results/exploratory/fbr/` (counts and metrics only).
- The pilot compares each FBR model with its arm's **registered** recovery on validation, so the
  validation `outcomes.npz` of E5 magnitude, E5 activation and E7 at B2 must be attached (the
  RC-G2 notebook versions, if kept; else RC-EX1 section 3b scores them), with an `audit_val.npz`
  from a mask-audit-only rerun (cell (i)) for the operating-threshold audit. Attach one copy of
  each run's `outcomes.npz`: two copies that differ stop the analysis.

## 3. Commands

Estimates below are **estimates** (from measured passes: registered recovery 4,003 s for 157
updates, a validation pass 0.53 h, model loads a few minutes); cell (iii) measures the FBR update.

| Cell | Account | Attach | Estimate | Save afterwards |
|---|---|---|---|---|
| (i) probe answers + audits | where each arm's datasets are | `medcompress-data`, `e2-seed0-adapter`, the arm's pruned dataset (`g2-b2-seed0-pruned-s1` E5 magnitude, `-s1b` E5 activation, `-s2` Taylor, `-s2b` E6, `-s4` E7, `-s5` E5R after its resume, `-s5b` E7U) | about 0.25 h per model (8 models: about 2 h) plus 10 min per audit rerun (3) | `/kaggle/working/outputs/` as `fbr-probe-answers-<account>` |
| (i) CPU diagnostic | any (CPU) | the outcome datasets and the probe-answer datasets | minutes | send `fbr_diagnostic.zip` |
| (ii) E2 margins | Soheil | `medcompress-data`, `e2-seed0-adapter` | about 0.3 h | `/kaggle/working/fbr_margins/` as `fbr-e2-margins` |
| (iii) smoke + trial | Soheil | `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache`, `fbr-e2-margins` | about 0.8 h | send the log and `trial.json` |
| (iv) pilot, one notebook per arm | Soheil (E5 magnitude), account 3 (E5 activation), Amir (E7), any (sampling-only on E5 activation) | as (iii) | about 2.2 h per arm (training about 1.2 h by estimate, validation 0.53 h, probe answers 0.15 h, setup), 4 arms about 9 h | `/kaggle/working/outputs/fbr/` as `fbr-pilot-<label>`; send `results_to_commit/` |

Total by estimate about 13-14 T4 hours (with up to 1.8 h more where a registered validation pass
must be rerun), under the cap.

### (i) Probe answers (GPU), the audit reruns, then the diagnostic (CPU)

One cell per account, with the lines of the arms whose datasets that account holds (E2 in one
of them). The audit-only lines are for the three pilot arms (their validation outcomes predate the
audit margins).

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
for SPEC in "E2" "E5 --variant magnitude --budget B2" "E5 --variant activation --budget B2" \
            "E5 --variant taylor --budget B2" "E6 --budget B2" "E7 --budget B2" \
            "E5R --budget B2" "E7U --budget B2"; do                 # keep this account's lines
  python scripts/fbr_probe_answers.py --exp $SPEC --seed 0 --data-root "$DATA" $FROM \
    2>&1 | tee -a /kaggle/working/fbr_probe_answers.txt
done
for SPEC in "E5 --variant magnitude --budget B2" "E5 --variant activation --budget B2" \
            "E7 --budget B2"; do                                    # the pilot arms, this account's
  python scripts/save_outcomes.py --exp $SPEC --seed 0 --data-root "$DATA" $FROM \
    -o evaluation.split=val --audit-only 2>&1 | tee -a /kaggle/working/fbr_probe_answers.txt
done
grep -E "wrote|parse failures|audit flips|Error|STOP" /kaggle/working/fbr_probe_answers.txt
```

The CPU diagnostic (a CPU notebook; attach the validation outcome datasets and every
`fbr-probe-answers-*` dataset):

```bash
%%bash
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
python scripts/fbr_analysis.py diagnostic --outcomes /kaggle/input --out /kaggle/working/fbr
cd /kaggle/working && zip -qr fbr_diagnostic.zip fbr && ls -l fbr_diagnostic.zip
```

### (ii) E2's margin cache and the answer spelling (GPU, one T4)

It prints each yes / no spelling's tokens, the single-token sets, the first tokens E2 ranks
highest on the schedule's probes and a few decoded examples; it stops if those are not
single-token yes / no spellings (then the study stops and the research lead is told).

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
python scripts/fbr_e2_margins.py --data-root "$DATA" $FROM --out /kaggle/working/fbr_margins \
  2>&1 | tee /kaggle/working/fbr_e2_margins.txt
echo "exit code: ${PIPESTATUS[0]}"
```

### (iii) FBR smoke, then the timed 20-update trial (GPU, one T4)

The smoke runs the whole path with 2 updates and the smoke's evaluation subset, into a scratch
results copy; the trial rebuilds E5 magnitude and times 20 real updates.

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
MARGINS=$(find /kaggle/input -name fbr_e2_margins.json | head -1); echo "margins: $MARGINS"
export HF_HOME=/kaggle/temp/hf CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cp -r results /kaggle/working/smoke_results                       # the smoke never writes results/
MEDCOMPRESS_RESULTS_ROOT=/kaggle/working/smoke_results python scripts/fbr_recover.py \
  --exp E5 --variant magnitude --budget B2 --seed 0 --data-root "$DATA" $FROM \
  --margins "$MARGINS" --out /kaggle/working/fbr_smoke -o recovery.steps=2 \
  -o smoke.eval_images_per_label=2 -o evaluation.bootstrap_resamples=20 \
  -o evaluation.generation_samples=4 2>&1 | tee /kaggle/working/fbr_smoke.txt
echo "smoke exit code: ${PIPESTATUS[0]}"
python scripts/fbr_recover.py --exp E5 --variant magnitude --budget B2 --seed 0 \
  --data-root "$DATA" $FROM --margins "$MARGINS" --out /kaggle/working/fbr_trial --trial 20 \
  2>&1 | tee /kaggle/working/fbr_trial.txt
echo "trial exit code: ${PIPESTATUS[0]}"
cat /kaggle/working/fbr_trial/E5_magnitude/trial.json
```

### (iv) The pilot, one notebook per arm (GPU, one T4)

Set `ARM` to one of `"E5 --variant magnitude"`, `"E5 --variant activation"`, `"E7"`, and for the
control `"E5 --variant activation"` with `CONTROL=--control`.

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
ARM="E5 --variant magnitude"; CONTROL=""                          # edit per notebook
DATA=$(python scripts/find_inputs.py data) || exit 1
FROM=$(python scripts/find_inputs.py handoffs | sed 's/^/--from /' | tr '\n' ' ')
MARGINS=$(find /kaggle/input -name fbr_e2_margins.json | head -1); echo "margins: $MARGINS"
export HF_HOME=/kaggle/temp/hf CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
python scripts/fbr_recover.py --exp $ARM --budget B2 --seed 0 --data-root "$DATA" $FROM \
  --margins "$MARGINS" $CONTROL 2>&1 | tee /kaggle/working/fbr_pilot.txt
echo "exit code: ${PIPESTATUS[0]}"
mkdir -p /kaggle/working/results_to_commit/results/exploratory
cp -r results/exploratory/fbr /kaggle/working/results_to_commit/results/exploratory/
```

After the four runs, the pilot analysis (CPU notebook; attach the four `fbr-pilot-*` datasets,
the registered validation outcomes and audit reruns, and the probe-answer datasets):
`python scripts/fbr_analysis.py pilot --outcomes /kaggle/input --out /kaggle/working/fbr`, then
send the zip of `/kaggle/working/fbr`.

## 4. What to commit back

- `results/exploratory/fbr/`: `diagnostic.csv`, `folds.csv`, `DIAGNOSTIC.md`; per run
  `<label>/metrics.csv`, `training.json` (or `trial.json`); `pilot_*.csv`, `PILOT.md`.
- Never: `fbr_e2_margins.json`, any `.npz`, decoded answer texts.

## 5. How the result is read

The gate of the decision file, section 7, as written. A failed gate is a result; no coefficient,
recipe or threshold is searched. If it passes, a second dated amendment comes before any FBR run
on the other arms or any test evaluation.

## 6. Run report (filled by the person who ran it)

- Run by / date / cell:
- Commit:
- Datasets attached:
- Wall time per cell, peak GPU memory:
- Anything unusual:
