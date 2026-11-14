# Run card RC-R1: test-split evaluation of the reference arms E0, E1 and E2

## 1. What and why

- Stage: S11
- Experiments: E0 (seed 0), E1 (**seed 0 only**: seed 1's outputs were lost; E1 is a control outside the 3-seed tier, 2026-11-08) and E2 (seeds 0, 1, 2; seed 0 only after a futility stop, seed plan of 2026-11-08: the plan reads `results/seed_plan/comparator_B2.json`), no budget.
- Runs: 7 evaluations on the **test** split. Nothing is trained.
- Purpose: E0, E1 and E2 are trained and evaluated on the validation split before the
  pre-registration is frozen, while the test split is still locked. Every comparison of the
  paper is on the test split, and E2 is the reference of H1 and of tables T2 and F1 to F5, so
  these three arms need a test-split evaluation after the freeze.
- Status: `blocked` (see section 2). Also before any test evaluation (second amendment of
  2026-11-13): the committed list `results/data/nonfinite_drop.csv` (RC-CAL0 part A: 5 filtered
  and 8 unfiltered test images, from the student's encoder; the code that drops them is in since
  2026-11-14 and every test evaluation refuses without the list), then RC-SMOKE again for RC-R1
  (removed from `smoke_passed.yaml` until it passes with the drop)

## 2. Requirements

- **Operating point first (2026-11-06):** every test evaluation reads the run's validation
  thresholds (`op_threshold:<f>`) from its committed `metrics.csv` and refuses to run without
  them. E0: its validation again (RC-E0a section 3c, about 0.8 h). E1: from the saved validation
  outcomes on CPU (RC-E1 section 3c), or its validation again (about 0.5 h per seed). E2: its
  own validation pass writes them.

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- The pre-registration is frozen (`preregistration.frozen_on` set); before that the evaluation
  refuses the test split.
- RC-E0b, RC-E1 and RC-E2 done; the E1 and E2 adapters kept as private datasets.

Platform: Kaggle notebook, **2 x T4** for E0 (the 7B teacher, as in RC-E0b), one T4 for E1
and E2, Internet on. Dataset access: your own copy of Kvasir-VQA-x1 and Kvasir-SEG, the private
RC-D1 outputs, and the E1 and E2 adapters attached as private datasets. Nothing from those
goes into the repository.

Runtime: one evaluation pass per run, the same work as the validation evaluation in RC-E0b,
RC-E1 and RC-E2 (the split halves are about the same size). The agent fills in the estimate
from those cards' wall times when it sets this card to `ready`.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil.
- **Kaggle settings:** GPU T4 x2 (both T4s); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher adapter; E1 seed0 adapter; E1 seed<k> adapter; E1 seed2 adapter; E2 seed0 adapter; E2 seed<k> adapter; E2 seed2 adapter.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-R1"` and `SEED = None` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-R1`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - test evaluation E0 seed0, then unfiltered test evaluation E0 seed0
  - test evaluation E1 seed0, then unfiltered test evaluation E1 seed0
  - test evaluation E1 seed<k>, then unfiltered test evaluation E1 seed<k>
  - test evaluation E1 seed2, then unfiltered test evaluation E1 seed2
  - test evaluation E2 seed0, then unfiltered test evaluation E2 seed0
  - test evaluation E2 seed<k>, then unfiltered test evaluation E2 seed<k>
  - test evaluation E2 seed2, then unfiltered test evaluation E2 seed2

  (The unfiltered test is the filtered test with the removed test images added back: the
  sensitivity analysis of D02 amendment 6. Its rows go to `metrics_test_unfiltered.csv`.)
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E2_test_outcomes`, `E0_test_outcomes`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). This card uses T4 x2. A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

One cell (Kaggle settings and inputs of section 2b; **Save Version -> Save & Run All**). It finds the shared data and the attached hand-offs by their manifests (`scripts/find_inputs.py`), stops with a clear message if the shared data are missing, and runs the steps of section 2b with `scripts/run_card.py`, which also checks that every hand-off the card reads is there. Then it writes the hand-offs under `/kaggle/working/handoff/` and the files to commit under `/kaggle/working/results_to_commit/`.

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
python scripts/run_card.py --card RC-R1 2>&1 | tee /kaggle/working/run_card_RC-R1.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

If a script stops with "configuration incomplete", tell the agent through the inbox instead of
filling a setting in. Keep `outputs/` (the `outcomes_test.npz` files) as a private dataset: the
paired comparisons (RC-C1) need them.

## 4. What to commit back

- `results/E0/nobudget/seed0/metrics_test.csv`
- `results/E1/nobudget/seed<k>/metrics_test.csv` and `results/E2/nobudget/seed<k>/metrics_test.csv`
- This card with section 6 filled.

Check before committing: `git status` shows no images, adapters or `outcomes*.npz`.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| All runs evaluated | `metrics_test.csv` files | 7 files, each with `small_finding_recall` and `lesion_mask_effect` |
| Same probes as every arm | `n_probes_scored` | the same number in every file |

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Wall time per run:
- Names of the private datasets with the `outcomes_test.npz` files:
- Anything unusual (errors, retries, changed settings):
