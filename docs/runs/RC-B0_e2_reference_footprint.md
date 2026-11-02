# Run card RC-B0: footprint of E2, the reference of the budgets

## 1. What and why

- Stage: S06
- Experiments: E2, seed 0, no budget (measurement only, no training)
- Purpose: measure the peak GPU memory of E2 during inference under the budget protocol (batch
  1, three warm-up and five timed passes, 32 fixed validation images) and how its bytes split
  into decoder layers, vision encoder and the rest. Budgets B1 to B3 are fractions of this
  measured peak, and the split shows how much of it pruning can reach.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent changes the status to `ready` when they are:

- RC-E2 done for seed 0, with the adapter `checkpoints/E2/seed0/adapter` on team storage.
- `budgets.measurement.hardware` names the development GPU and `max_new_tokens` is set
  (the agent sets both; the label must match the GPU you use).
- `inputs.image_longest_side` is set; RC-D1 done (the 32 images come from the validation split).

Platform: Kaggle notebook with **one T4** (`CUDA_VISIBLE_DEVICES=0`) and nothing else running on
the GPU. Peak memory is the budget metric because it should be the same every time it is
measured; this card checks that by measuring twice.

Runtime: minutes.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Amir.
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed0 adapter.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-B0"` and `SEED = None` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-B0`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - first reading
  - keep the first reading
  - second reading
- **Hand-offs it writes:** none (its results reach the others through git).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

One cell (Kaggle settings and inputs of section 2b; **Save Version -> Save & Run All**). It finds the shared data and the attached hand-offs by their manifests (`scripts/find_inputs.py`), stops with a clear message if the shared data are missing, and runs the steps of section 2b with `scripts/run_card.py`, which also checks that every hand-off the card reads is there. Then it writes the hand-offs under `/kaggle/working/handoff/` and the files to commit under `/kaggle/working/results_to_commit/`.

```bash
%%bash
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
python scripts/find_inputs.py data > /dev/null || exit 1        # stops if the shared data are not attached
python scripts/find_inputs.py handoffs                          # the hand-offs found by their manifests
python scripts/run_card.py --card RC-B0 2>&1 | tee /kaggle/working/run_card_RC-B0.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

## 4. What to commit back

- `results/E2/nobudget/seed0/efficiency.csv` (the second reading)
- `results/E2/nobudget/seed0/efficiency_setup.json`
- This card with section 6 filled, including **both** `peak_memory_bytes` values.

## 5. How the result is read

The agent computes the share of the decoder in the peak (`decoder_layers_bytes` relative to the
state bytes and to the peak) and proposes B1 to B3 by the rule of D07 section 4 (remove about
15, 30 and 45 % of the decoder's bytes, written as fractions of the measured peak). The team
confirms the numbers before they go into the pre-registration. If the two peak readings differ
by more than the tolerance of the budgets (3 %), the metric is not repeatable enough for the
matched-budget search and the agent asks the team to change it.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- `peak_memory_bytes`, first reading / second reading:
- Anything unusual (errors, retries, other processes on the GPU):
