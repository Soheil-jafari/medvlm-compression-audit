# Run card RC-EX1: exploratory analyses A and B from the saved outcomes (CPU)

## 1. What and why

- Stage: after Gate 2 (pre-registration section 10, amendment of 2026-11-20;
  `docs/decisions/exploratory_additions_2026-11-20.md`)
- Experiments: every arm with saved per-probe outcomes; exploratory, outside H1-H4, no verdicts,
  no multiplicity correction
- Purpose: **A** silent collapse and re-calibration cost (plain answers against the operating
  point, the threshold refitted on 8 ... 128 validation images, the lesion-mask audit at the
  operating threshold); **B** what each method forgets (flips against E2 seed 0, their overlap,
  the compression-identified exemplars). It reruns from scratch as arms arrive.
- Status: `ready` for the CPU analysis (it runs on whatever is attached); the outcomes listed
  as missing in section 2 come from the runs named there.

## 2. Requirements: the outcomes

The analysis reads `outcomes.npz` (validation), `outcomes_test.npz`, `outcomes_test_unfiltered.npz`
and `audit_<split>.npz` from every attached dataset whose path ends in
`<ID>[/<variant>]/<budget>/seed<k>/`: the outcome hand-offs (`*_outcomes`) and the output folders
of earlier notebook versions (Kaggle keeps `/kaggle/working/` of a saved version). The evaluation
saves the mask audit's margins from 2026-11-20 on; for runs evaluated before, `save_outcomes.py
--audit-only` adds them (GPU, section 3b), and `save_outcomes.py` without `--audit-only` scores a
split again where its outcomes were not kept. Both write to `outputs/rescored/<run>/`, beside the
attached (read-only) outcomes, and never a results file; an audit rerun that does not reproduce
the saved flips writes nothing.

| Arm | Validation outcomes | Test outcomes (filtered, unfiltered) | Audit margins | How to complete |
|---|---|---|---|---|
| E0 | RC-E0b's version output, if kept | RC-R1 (`E0_test_outcomes`) | no | RC-R1 saves them; validation: audit-only rerun |
| E1 seed 0 | RC-E1's version output, if kept (seed 1's were lost) | RC-R1 (`E1_test_outcomes`, added 2026-11-20) | no | as E0 |
| E2 seed 0 | RC-E2's version output (`outputs/E2/nobudget/seed0/`) | RC-R1 (`E2_test_outcomes`) | no | as E0; **B needs E2's test outcomes** |
| E3 `w4`, `w8`, `w8a8` | `E3_seed0_outcomes` | `E3_seed0_outcomes` | no | audit-only reruns (val, test, unfiltered) |
| E9 | RC-E9's outputs dataset | the same | no | audit-only reruns (two T4s) |
| E10 | RC-E10's version output, if kept | not yet evaluated | no | its test evaluation saves them |
| E5 magnitude / activation / Taylor, E6, E7 at B2 | the RC-G2 sessions' version outputs, if kept | `E5_seed0_outcomes`, `E6_seed0_outcomes`, `E7_seed0_outcomes` (the B2 test evaluations, running 2026-11-20) | no | section 3b |
| E5R, E7U at B2 | RC-G2 session 5 (to come) | none yet: no card evaluates them on the test splits | yes (new code) | `06_evaluate.py --exp E5R --budget B2 --seed 0 -o evaluation.split=test` (and `test_unfiltered`; E7U alike) with the saved pruned model attached, about 1.7 h per arm |
| E3 `awq_w4` | its run (RC-E3) | its run | yes (new code) | - |
| E4, E8, E10Q, A1-A6 | not run yet | not run yet | yes when they run | - |

## 3. Commands

### 3a. The CPU analysis (Kaggle CPU notebook, any account)

Attach every outcome dataset you have (section 2); the shared data are not needed. About
20-40 min for a dozen arms (an estimate: 1,000 draws per n and 10,000 bootstrap resamples per
quantity). Send `/kaggle/working/exploratory.zip` (the CSV files and `SUMMARY.md`).

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
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps   # CPU: no torch needed
python scripts/exploratory_analysis.py --outcomes /kaggle/input 2>&1 | tee /kaggle/working/exploratory.txt
echo "exit code: ${PIPESTATUS[0]}"
cat results/exploratory/SUMMARY.md
cd /kaggle/working && rm -f exploratory.zip && (cd medvlm-compression-audit && zip -qr /kaggle/working/exploratory.zip results/exploratory)
```

### 3b. Completing the outcomes of the B2 pruning arms (GPU, one T4)

Attach `medcompress-data`, `e2-seed0-adapter`, the arm's pruned dataset(s) and its outcomes
dataset from the B2 test evaluation (so the audit rerun is checked against the saved flips).
Estimate: about 1 h per arm (validation in full about 0.6 h, two audit-only passes about 10 min
each); split the arms over accounts by where their datasets are. Drop the validation line for an
arm whose `outcomes.npz` you still have. Save `/kaggle/working/outputs/rescored/` as a private
dataset.
The log prints "largest difference" against the committed rows (expected about 0) and "audit
flips equal the saved ones" for every audit.

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
# one line per run: its arguments; the validation pass in full only where its outcomes.npz was
# not kept (it then saves the audit margins too), the test splits audit-only
for SPEC in "E5 --variant magnitude --budget B2" "E5 --variant activation --budget B2" \
            "E5 --variant taylor --budget B2" "E6 --budget B2" "E7 --budget B2"; do
  python scripts/save_outcomes.py --exp $SPEC --seed 0 --data-root "$DATA" $FROM \
    -o evaluation.split=val 2>&1 | tee -a /kaggle/working/save_outcomes.txt
  for SPLIT in test test_unfiltered; do
    python scripts/save_outcomes.py --exp $SPEC --seed 0 --data-root "$DATA" $FROM \
      -o evaluation.split=$SPLIT --audit-only 2>&1 | tee -a /kaggle/working/save_outcomes.txt
  done
done
grep -E "wrote|largest difference|audit flips|STOP|Error" /kaggle/working/save_outcomes.txt
```

## 4. What to commit back

- `results/exploratory/` (CSV files and `SUMMARY.md`), from `exploratory.zip`.
- Nothing from 3b: it writes outcome files only (keep them as a private dataset).

## 5. Run report (filled by the person who ran it)

- Run by / date:
- Commit:
- Datasets attached:
- Anything unusual:
