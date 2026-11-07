# Run card RC-E8: E7 + 4-bit quantization (E8), B2 x 3 seeds, B1 and B3 seed 0 (trend)

## 1. What and why

- Stage: S09
- Experiments: E8 = each E7 pruned and recovered model (budget B2 with seeds 0, 1, 2, and B1 and B3 with seed 0 only (the trend, pre-registration section 11 item 11; decided 2026-11-06). The cell of a seed 1 or 2 session runs B2 only)
  quantized with the E3 `w4` recipe (GPTQ, groups of 128, the same 128 calibration images of
  that seed). 9 runs, scored on validation and test and timed with the budget protocol.
- Purpose: whether pruning and quantization combine, at the same budgets.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- RC-E7 done for the budgets and seeds you run (the pruned models on team storage).
- RC-Q0 passed (`quantization.kernel_class` and the library versions set and pinned). **Met 2026-10-15:** RC-Q0 chose `TorchLinear`; gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned.
- The pre-registration is frozen.

Platform: Kaggle notebook, one T4, Internet on. Dataset access as in RC-E7; attach the E7
pruned models as a private dataset.

Runtime per run: **estimate** under an hour for GPTQ (RC-Q0 and RC-E3 report the real speed),
plus the evaluation and timing.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** card owner Amir; seeds: seed 0 Amir (after RC-G2), seed 1 Farzan, seed 2 Amir.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed<k> adapter; E7 B1 seed<k> pruned; E7 B2 seed<k> pruned; E7 B3 seed<k> pruned.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E8"` and `SEED = <k>` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-E8 --seed <k>`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E8 B1 seed<k>
  - test evaluation E8 B1 seed<k>
  - run E8 B2 seed<k>
  - test evaluation E8 B2 seed<k>
  - run E8 B3 seed<k>
  - test evaluation E8 B3 seed<k>
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E8_seed<k>_quantized`, `E8_seed<k>_outcomes`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

One cell (Kaggle settings and inputs of section 2b; **Save Version -> Save & Run All**). It finds the shared data and the attached hand-offs by their manifests (`scripts/find_inputs.py`), stops with a clear message if the shared data are missing, and runs the steps of section 2b with `scripts/run_card.py`, which also checks that every hand-off the card reads is there. Then it writes the hand-offs under `/kaggle/working/handoff/` and the files to commit under `/kaggle/working/results_to_commit/`.

```bash
%%bash
SEED=0   # this session's seed: 0, 1 or 2 (section 2b says who runs which)
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
python scripts/run_card.py --card RC-E8 --seed $SEED 2>&1 | tee /kaggle/working/run_card_RC-E8_seed${SEED}.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

Keep `outputs/` (the per-probe `outcomes*.npz` files) as a private dataset too: the paired
comparisons (RC-C1) need the test-split ones. Write its name in section 6.

## 4. What to commit back

- `results/E8/<budget>/seed<k>/metrics.csv`, `metrics_test.csv`, `efficiency.csv`,
  `efficiency_setup.json`
- This card with section 6 filled.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Real low-precision kernels | `low_precision_kernels_ok` | equals 1 |
| Calibration as planned | `quantization_calibration_images` | equals 128 |
| Every remaining decoder matrix quantized | `quantized_matrices` | 7 x the number of decoder layers left after pruning |

## 6. Run report (filled by the person who ran it)

- Run by / date / budgets and seeds:
- Commit actually used:
- Hardware and library versions:
- Wall time and peak GPU memory per run:
- Anything unusual:
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
