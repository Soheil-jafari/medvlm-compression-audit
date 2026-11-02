# Run card RC-E1: student fine-tuned on correct answers (E1), 3 seeds

## 1. What and why

- Stage: S06
- Experiments: E1, no budget, seeds 0, 1 and 2
- Purpose: train the chosen student with LoRA on the ground-truth answers only (KD weight 0),
  then score it on the validation split with the full evaluation suite, so that E2 (the same
  recipe plus distillation) has its control.
- Status: `ready` (2026-11-02): the evaluation settings are frozen and E0's validation is committed. **Seed 0 first** (Soheil): E2 waits for its validation row (rule 3). Seeds 1 and 2 can run at any time, in parallel.

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
grep -E "train_|n_probes_scored|macro_balanced_accuracy|small_finding_recall|lesion_mask_effect|evaluation_seconds" \
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

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Per seed: wall time, peak GPU memory (`train_peak_gpu_bytes`), seconds per update
  (`train_seconds` / `train_steps`), final training loss; for seed 0, whether the micro-batch
  of 4 fitted:
- Name of the private datasets with your adapters and `outcomes.npz` files:
- Anything unusual (errors, retries, changed settings):
