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
| E5 magnitude / activation / Taylor, E6, E7 at B2 | the RC-G2 sessions' version outputs, if kept (see below) | E5: two bundles both named `E5_seed0_outcomes` (magnitude and activation; Taylor), `E6_seed0_outcomes`; E7: RC-E7 running (Amir) | test: yes (evaluated at 2ecf302 or later); validation: no | section 3b: the validation pass only where `outcomes.npz` was not kept |
| E5R at B2 | its resume (RC-G2 cell (b)) writes it | section 3c | yes (new code) | section 3c, attach `g2-b2-seed0-pruned-s5` |
| E7U at B2 | the version output of RC-G2 cell (c), 2026-11-20, if kept | section 3c | yes (new code) | section 3c, attach `g2-b2-seed0-pruned-s5b` |
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

**Which B2 validation outcomes exist (2026-11-23).** No RC-G2 session wrote an outcomes
hand-off (its only hand-offs are the pruned models), and the B2 test bundles hold the test
splits only (4 files for magnitude and activation, 2 for Taylor, 2 for E6). So each B2 arm's
validation `outcomes.npz` exists only in the saved version of the RC-G2 notebook that ran its
stage 06, at `/kaggle/working/outputs/<run>/outcomes.npz`, if that version's output was kept.
The agent cannot see Kaggle outputs; the owner checks the Output tab of each:

| Run | Notebook version (stage 06) | Path in its output |
|---|---|---|
| E5 magnitude | Soheil, RC-G2 session 1 resume, 2026-11-19 (commit 2e795a9); not the 2026-11-16 version (old design) | `outputs/E5/magnitude/B2/seed0/outcomes.npz` |
| E5 activation | Soheil, RC-G2 session 1 new, the same notebook version | `outputs/E5/activation/B2/seed0/outcomes.npz` |
| E5 Taylor, E6 | account 3, RC-G2 session 2, 2026-11-19 (commit 3aa4ecb) | `outputs/E5/taylor/B2/seed0/`, `outputs/E6/B2/seed0/` |
| E7 | Amir, RC-G2 session 4, 2026-11-19 (commit 2e795a9) | `outputs/E7/B2/seed0/outcomes.npz` |
| E7U | RC-G2 cell (c), 2026-11-20 (commit 2e795a9) | `outputs/E7U/B2/seed0/outcomes.npz` |
| E5R | not yet: its resume writes it | - |

Where the file is there, add that notebook's output as an input of the analysis notebook and drop
that run's validation line from the cell of section 3b; where it is not, keep the line. The GPU
part of section 3b is unchanged. Its audit-only test lines are not needed for an arm whose test
evaluation ran at 2ecf302 or later (E5's three criteria and E6: their `outcomes_test*.npz` already
hold the audit margins); drop them there to save about 20 min per arm. An attached old-design
`outcomes.npz` (the 2026-11-16 version of session 1) would differ from the 2026-11-19 one for
the same run: the analysis then stops and names both files, rather than picking one.

**Outcome bundles with the same name.** Two sessions of RC-E5 both wrote `E5_seed0_outcomes`
(Soheil: magnitude and activation; account 3: Taylor). Attach both datasets: the analysis keys
every outcome file by its run path inside the bundle (`E5/<criterion>/B2/seed0/...`), never by the
bundle's name, so both are read. A run found twice is kept once when the copies are byte for
byte the same, and stops the analysis when they differ. From 2026-11-23 a session that runs a
chosen subset names its bundles by it (`E5_seed0_magnitude_activation_B2_outcomes`,
`E7_seed0_B2_outcomes`, ...), so new bundles no longer share names.

### 3c. Test evaluations of E5R and E7U at B2 (GPU, one T4; 2026-11-23)

Both arms are exploratory (B2, seed 0). Their test outcomes are saved with the audit margins. The
cells copy the new `metrics_test*.csv` into `results_to_commit/`. Save
`/kaggle/working/outputs/` as a private dataset afterwards (`e5r-b2-test-outcomes`,
`e7u-b2-test-outcomes`); send `results_to_commit/` and the log.

(i) **E5R** (any account). Attach `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache` (the
card's resume asks for it) and `g2-b2-seed0-pruned-s5`. E5R's validation (stage 06) and benchmark
(stage 07) have not run, and the test evaluation needs its validation operating point, so the cell
runs RC-G2 cell (b)'s resume first. Estimate about 2.3 h (an estimate: validation 0.53 h and the
benchmark about 10 min as in RC-G2, then the two test passes about 32 and 50 min, plus setup).

```bash
{setup}
python scripts/run_card.py --card RC-G2 --only E5R --out /kaggle/working/resume 2>&1 | tee /kaggle/working/rc_ex1_e5r.txt
echo "resume exit code: ${{PIPESTATUS[0]}}"
DATA=$(python scripts/find_inputs.py data) || exit 1
FROM=$(python scripts/find_inputs.py handoffs | sed 's/^/--from /' | tr '\n' ' ')
for SPLIT in test test_unfiltered; do
  python scripts/06_evaluate.py --exp E5R --budget B2 --seed 0 --data-root "$DATA" $FROM \
    -o evaluation.split=$SPLIT 2>&1 | tee -a /kaggle/working/rc_ex1_e5r.txt
  echo "$SPLIT exit code: ${{PIPESTATUS[0]}}"
done
mkdir -p /kaggle/working/results_to_commit/results/E5R/B2/seed0
cp results/E5R/B2/seed0/metrics_test*.csv /kaggle/working/results_to_commit/results/E5R/B2/seed0/
ls -l /kaggle/working/resume/results_to_commit/results/E5R/B2/seed0/ /kaggle/working/outputs/E5R/B2/seed0/
```

(ii) **E7U** (any account). Attach `medcompress-data`, `e2-seed0-adapter` and
`g2-b2-seed0-pruned-s5b`. E7U's validation rows are committed (2026-11-23). Estimate about 1.6 h
(the two test passes, about 32 and 50 min, plus setup; an estimate).

```bash
{setup}
DATA=$(python scripts/find_inputs.py data) || exit 1
FROM=$(python scripts/find_inputs.py handoffs | sed 's/^/--from /' | tr '\n' ' ')
for SPLIT in test test_unfiltered; do
  python scripts/06_evaluate.py --exp E7U --budget B2 --seed 0 --data-root "$DATA" $FROM \
    -o evaluation.split=$SPLIT 2>&1 | tee -a /kaggle/working/rc_ex1_e7u.txt
  echo "$SPLIT exit code: ${{PIPESTATUS[0]}}"
done
mkdir -p /kaggle/working/results_to_commit/results/E7U/B2/seed0
cp results/E7U/B2/seed0/metrics_test*.csv /kaggle/working/results_to_commit/results/E7U/B2/seed0/
ls -l /kaggle/working/outputs/E7U/B2/seed0/
```

## 4. What to commit back

- `results/exploratory/` (CSV files and `SUMMARY.md`), from `exploratory.zip`.
- Nothing from 3b: it writes outcome files only (keep them as a private dataset).
- From 3c: `results/E5R/B2/seed0/` (stage 06 and 07 files from the resume, `metrics_test.csv`,
  `metrics_test_unfiltered.csv`) and `results/E7U/B2/seed0/metrics_test*.csv`.

## 5. Run report (filled by the person who ran it)

- Run by / date:
- Commit:
- Datasets attached:
- Anything unusual:
