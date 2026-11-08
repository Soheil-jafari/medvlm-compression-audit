# Run card RC-E1: student fine-tuned on correct answers (E1), 3 seeds

## 1. What and why

- Stage: S06
- Experiments: E1, no budget, seeds 0, 1 and 2
- Purpose: train the chosen student with LoRA on the ground-truth answers only (KD weight 0),
  then score it on the validation split with the full evaluation suite, so that E2 (the same
  recipe plus distillation) has its control.
- Status: `done` (2026-11-06): seeds 0, 1 and 2 trained and scored on validation; rule 3
  passes. Section 3c: seeds 0 and 2 rescored (2026-11-08). Seed 1 is **final** (2026-11-08):
  its outputs were lost, it has no operating-point rows and keeps its committed default-cutoff
  validation rows; E1's test evaluation (RC-R1) uses seed 0 only (E1 is a control outside the
  3-seed tier). Open: seed 0's adapter check (section 3b).

## 2. Requirements

All met (2026-11-02):
- the student Qwen2.5-VL-3B-Instruct at its pinned revision (D05 amendment), image side 672;
- `training.data_subset` 10,000 and `training.steps` 625 (D06); micro-batch 1 x 16;
- RC-D1 done (`medcompress-data`, `split_sha256` 54dfa70a..., checked when attached);
- the evaluation settings frozen (pre-registration section 11): three findings, the probe
  wording, the small-polyp quartiles, `max_new_tokens` 46 by the D04 rule;
- RC-M1 passed.

**Memory check (seed 0 first).** At 672 the step at micro-batch 1 peaked at 11,372,899,328 bytes
on synthetic probes (RC-S0b run 2; the T4 has 15,636,037,632). Real examples are longer, so seed
0 confirms it: `train_peak_gpu_bytes` and `train_seconds` / `train_steps`. If training stops with
"CUDA out of memory", stop there, keep the last 30 lines of the log and tell the agent. Do
**not** change the batch yourself (D06).

Hours per seed (estimates): training 625 updates x 13.6 s = **2.4 h** (RC-S0b run 2, side 672,
synthetic probes; real examples are longer, so it may be slower: RC-TT1 ran 3 % over its own
projection), plus the validation pass, **about 0.5 h** (E0's measured pass took 0.78 h; the
student scores probes in 0.414 s instead of 0.584 s and generates about 1.9 times faster), plus
the setup (install, the 7.5 GB student download, loading; not measured). So **about 3 h per
seed** on T4 x2, which counts once against the quota; the student uses one T4.
Measured (2026-11-06): 13.5-14.7 s per update, training 2.34-2.55 h, validation 0.47-0.53 h,
so the estimate held for E1. E2 is slower (27.2 s per update, RC-E2 section 2).

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** seeds split across the accounts: seed 0 Soheil, seed 1 Farzan, seed 2 Amir (one session each, in the same week). Card owner: Farzan.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E1"` and `SEED = <k>` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-E1 --seed <k>`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - train E1 seed<k> (skipped when its adapter is attached: a later session then runs only the rest)
  - validation E1 seed<k>
  - benchmark E1 seed<k> (waits if a setting is open; the run goes on)
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E1_seed<k>_adapter`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

Kaggle settings: **GPU T4 x2** (one T4 is used), Internet **on**, **Save Version -> Save & Run
All**. Add Input: `medcompress-data` only. One cell; edit only `SEED`:

```bash
%%bash
SEED=0   # Soheil 0, Farzan 1, Amir 2: the only line to edit
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
python scripts/02_train_student.py --exp E1 --seed $SEED --data-root $DATA \
    2>&1 | tee /kaggle/working/rc_e1_seed${SEED}_train.txt
code=${PIPESTATUS[0]}; echo "train exit code: $code"
[ "$code" = 0 ] || { echo "STOP: the training did not finish (see above)"; exit 1; }
python scripts/06_evaluate.py --exp E1 --seed $SEED --data-root $DATA \
    2>&1 | tee /kaggle/working/rc_e1_seed${SEED}_validation.txt
echo "validation exit code: ${PIPESTATUS[0]}"
python scripts/handoff.py export --kind adapter --exp E1 --seed $SEED \
    --out /kaggle/working/handoff/E1_seed${SEED}_adapter
python -m pytest -q tests/test_results.py                       # must pass before you commit
mkdir -p /kaggle/working/results_to_commit/results
cp -r results/E1 /kaggle/working/results_to_commit/results/
grep -E "train_|n_probes_scored|macro_balanced_accuracy|sensitivity|specificity|auroc|yes_rate|small_finding_recall|lesion_mask_effect|evaluation_seconds" \
    results/E1/nobudget/seed${SEED}/metrics.csv
```

What it prints, in order:
- `OK: 8511 files match the manifest ...` (the shared data, found by its manifest);
- the training: the plan, then progress, then the `train_*` rows; it saves its state every
  50 updates;
- the validation: `wrote <n> rows ...`;
- the hand-off: `wrote E1 seed <k> adapter to ...` and `make this folder ONE private Kaggle
  dataset`;
- the rows to read: peak memory, seconds, final loss, `n_probes_scored` (681 expected, as for
  E0), the balanced accuracy, the small-finding recall and the lesion-mask effect.

Then: make `/kaggle/working/handoff/E1_seed<k>_adapter` ONE private Kaggle dataset (name it
`e1-seed<k>-adapter`), shared with the team; RC-R1 needs all three. Commit
`results_to_commit/results/E1/nobudget/seed<k>/metrics.csv`. The benchmark step (stage 07) is
not in this cell: it waits on `budgets.measurement.hardware` (open, Amir).

### 3b. Seed 0's adapter check (CPU, no GPU; 2026-11-06)

Seed 0 logged `final loss nan` (seeds 1 and 2: 0.548, 0.523), but its validation is normal. The
final loss is the mean of the last 10 step losses, so one non-finite step makes it NaN; the fp16
loss scaler then skips that update, which would leave the adapter intact. This cell checks the
adapter. **Accelerator None**, Internet on; Add Input: `e1-seed0-adapter` (and, if you have it,
the output of the E1 seed 0 notebook version). One cell; nothing to edit:

```bash
%%bash
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone,
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # so removing the clone keeps
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # every output
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
ADAPTER=$(python scripts/find_inputs.py handoff E1_seed0_adapter) || exit 1
python scripts/handoff.py verify "$ADAPTER"                     # must print OK
PROGRESS=$(find /kaggle/input -path '*E1/seed0/resume/current/progress.json' | head -n 1)
python scripts/inspect_adapter.py "$ADAPTER" ${PROGRESS:+--progress "$PROGRESS"}
echo "inspect exit code: $? (0 = every tensor finite, 6 = not)"
```

It prints each adapter file's tensors and values and every tensor with NaN or inf, then any loss
history. **This run kept none**: the training's resume state (which holds the per-step losses)
is removed once the adapter is saved, and the run printed no per-step lines. Since 2026-11-06
every training run writes `train_history.json` into its adapter folder (per-step losses, the
non-finite steps, the skipped updates), prints a line every 25 steps, writes
`train_nonfinite_steps`, `train_nonfinite_microbatches`, `train_skipped_steps` and
`train_loss_final_finite` to `metrics.csv`, and stops (exit code 6, nothing saved) if a
trainable weight becomes NaN or inf.

### 3c. The operating point for E1 (CPU if the outcomes were kept; 2026-11-06)

Pre-registration section 11 item 10: each model's thresholds at specificity 0.95 are fitted on
its validation probes and written as `op_threshold:<f>`, with the metrics at them. E1's
validation ran before that, so its `op_*` rows are missing, and its test evaluation (RC-R1)
refuses to run without them. If the E1 session kept `outputs/` outside the clone (cells since
2026-11-04), they come from the saved scores, on CPU, in about a minute per seed. **Accelerator
None**; Add Input: the output of the E1 notebook version of that seed. Edit only `SEED`:

```bash
%%bash
SEED=0   # 0, 1 or 2: the seed whose notebook output is attached
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone,
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # so removing the clone keeps
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # every output
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
OUT=$(find /kaggle/input -path "*outputs/E1/nobudget/seed${SEED}/outcomes.npz" | head -n 1)
[ -n "$OUT" ] || { echo "STOP: E1 seed $SEED's outcomes.npz is not attached or was not kept: evaluate it on validation again (GPU, about 0.5 h)"; exit 1; }
python scripts/rescore_probes.py --exp E1 --seed $SEED --outcomes "$OUT"
python -m pytest -q tests/test_results.py                       # must pass before you commit
mkdir -p /kaggle/working/results_to_commit/results
cp -r results/E1 /kaggle/working/results_to_commit/results/
```

It first checks the outcomes against the committed rows (681 probes, every sensitivity and
specificity), then adds the AUROC, "yes" rate and `op_*` rows; no committed number changes. If
the outcomes were not kept, run the validation again on GPU instead (T4 x2, about 0.5 h per seed,
measured 1,704-1,896 s): the RC-E1 cell above without the training lines (`06_evaluate.py` only,
with the adapter attached: `--from "$(python scripts/find_inputs.py handoff E1_seed${SEED}_adapter)"`).

## 4. What to commit back

For each seed `k`:

- `results/E1/nobudget/seed<k>/metrics.csv`
- `results/E1/nobudget/seed<k>/efficiency.csv`
- This card with section 6 filled.

Check before committing: `git status` shows no images, adapters, caches or `outcomes.npz`.

## 5. How the result is read

The agent writes `docs/results/E1.md` from the CSVs: clinical metrics, small-finding recall by
lesion-size stratum, per-finding sensitivity and specificity, calibration, image reliance and free-text metrics,
mean and spread over the three seeds. A student that fails the image-reliance checks
(blank image, lesion masked, random region masked, other patient's image) is reported to the
team before anything is compressed.

## 6. Run report (filled by the person who ran it)

Reported by Soheil (2026-11-06); the three `metrics.csv` were uploaded as `metrics0/1/2.csv`
(173a3db) and renamed to `metrics.csv` (rows unchanged), so the aggregation and the rule 3
check find them. Seed 0's `train_loss_final` row read `nan`, which the results schema does not
accept: it was removed from the file and is recorded here.

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `train_steps` / examples | 625 / 10,000 | 625 / 10,000 | 625 / 10,000 |
| `train_loss_final` (mean of the last 10 steps) | **nan** (row removed) | 0.548 | 0.523 |
| `train_seconds` | 8,409 (2.34 h) | 9,132 (2.54 h) | 9,183 (2.55 h) |
| `train_peak_gpu_bytes` | 12,651,514,880 | 12,639,033,856 | 12,651,042,816 |
| validation `macro_balanced_accuracy` | 0.592 (0.571-0.614) | 0.546 (0.532-0.560) | 0.557 (0.541-0.575) |
| sensitivity polyp / UC / oesophagitis | 0.48 / 0.06 / 0.05 | 0.28 / 0.00 / 0.00 | 0.32 / 0.00 / 0.03 |
| AUROC polyp / UC / oesophagitis | 0.995 / 0.598 / 0.917 | 0.996 / 0.670 / 0.928 | 0.995 / 0.576 / 0.931 |
| `small_finding_recall` (13 small polyps) | 0.69 (0.42-0.93) | 0.46 (0.18-0.75) | 0.62 (0.33-0.89) |
| `lesion_mask_effect` | 0.28 (0.17-0.38) | 0.09 (0.02-0.16) | 0.19 (0.10-0.27) |
| `evaluation_seconds` | 1,704 | 1,881 | 1,896 |

- Run by / date: seed 0 Soheil, seeds 1 and 2 as split (section 2b); committed 2026-11-06.
- Commit actually used: not reported.
- Hardware and library versions: T4 x2 (one T4 used); **Torch 2.11.0+cu130** (the teacher ran
  on 2.11.0+cu128; TensorFlow 2.20 -> 2.21): the Kaggle image changed between RC-TT1 / RC-E0
  and E1. Recorded in the decisions log; every stage now writes the versions to
  `environment.json` (section 2b).
- Memory: 12.65 GB at peak of the T4's 15.6 GB; micro-batch 1 x 16 fits on real data.
- **Rule 3 (teacher quality) passes:** E0's validation macro balanced accuracy 0.803 >= E1
  seed 0's 0.592 + 0.02 = 0.612 (pre-registration section 6). No teacher extension; E2 may
  start.
- Anything unusual: seed 0's NaN final loss (section 3b); all three students are very
  conservative ("yes" rates: polyp 0.13-0.22, UC 0-0.04, oesophagitis 0-0.01) while their
  scores rank polyp and oesophagitis well, which led to the operating point (pre-registration
  section 11 item 10).

Section 3c (the operating-point rows, CPU):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Run by / date / commit | Soheil, Kaggle CPU notebook, 2026-11-08, commit b5f6197 | **final: none** (the notebook output and adapter were lost; no operating-point rows; not waiting on anyone, 2026-11-08) | the agent, locally on CPU (not Kaggle), 2026-11-08, commit 18c29e5, from Amir's saved outcomes (uploaded as `outcomes1.npz`, SHA-256 63a64a27...; kept outside the repository) |
| Environment | torch 2.11.0+cpu, transformers 5.20.2 (CPU only; the GPU runs used torch 2.11.0+cu130, transformers 5.19.0) | | Python 3.13.16, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, no torch (the rescoring reads the saved scores only) |
| Check against the committed rows | 7 rows (681 probes, the three sensitivities and specificities) identical to the saved outcomes; `tests/test_results.py` 6 passed | | the same 7 rows identical; every one of the 97 committed rows unchanged; `tests/test_results.py` 6 passed |
| Rows written | 31 (the 12 AUROC / "yes" rate / class-count rows, already committed and unchanged, and 19 new `op_*` rows) | | 33 (the same 12, unchanged, and 21 new `op_*` rows: the 19 of seed 0 plus the comparator tie-break rows `op_mask_positive_recall` and `op_small_margin_mean`, added to the code after seed 0's rescoring) |
| `op_threshold:polyp` / `ulcerative_colitis` / `oesophagitis` | -9.241 / -0.083 / -1.705 | | -10.546 / -0.319 / -1.696 |
| `op_macro_balanced_accuracy` | 0.758 (0.728-0.787) | | 0.750 (0.720-0.778) |
| `op_sensitivity:polyp` / `ulcerative_colitis` / `oesophagitis` | 0.990 / 0.082 / 0.622 | | 0.971 / 0.000 / 0.676 |
| `op_specificity` (all findings) | 0.951-0.953 | | 0.951-0.953 |
| `op_small_finding_recall` | 1.00 (13 of 13; interval 1.00-1.00, degenerate) | | 1.00 (13 of 13; degenerate interval) |
| `op_mask_positive_recall` / `op_small_margin_mean` | not written (rescored before these rows existed) | | 0.988 (85 of 86) / 10.275 |

The operating point is fitted on these same validation probes, so these numbers are in-sample
(optimistic); the test split reads the thresholds unchanged. Seed 0's `metrics.csv` committed as
uploaded (rows in the rescoring's order); seed 2's written by `scripts/rescore_probes.py` here
(the `.npz` is never committed).

Adapters: seed 0 `e1-seed0-adapter`; **seed 2 `e1-seed2-adapter`** (Amir made the dataset,
2026-11-08); seed 1 none (lost).
