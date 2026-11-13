# Run card RC-B0: footprint of E2, the reference of the budgets

## 1. What and why

- Stage: S06
- Experiments: E2, seed 0, no budget (measurement only, no training)
- Purpose: measure the peak GPU memory of E2 during inference under the budget protocol (batch
  1, three warm-up and five timed passes, 32 fixed validation images) and how its bytes split
  into decoder layers, vision encoder and the rest. Budgets B1 to B3 are fractions of this
  measured peak, and the split shows how much of it pruning can reach.
- Status: `done` (2026-11-11, Kaggle T4, commit 1320894): both readings 7,618,471,936 bytes;
  B1-B3 converted to bytes (section 6). An earlier attempt stopped at the input discovery;
  that bug was fixed before this run's commit.

## 2. Requirements

All met on 2026-11-08:

- RC-E2 done for seed 0 (2026-11-08): its adapter is the private dataset `e2-seed0-adapter`;
  **attach it**.
- `budgets.measurement.hardware` = "Kaggle Tesla T4" (frozen) and `max_new_tokens` is 46 (the
  D04 rule); run on a Kaggle T4.
- `inputs.image_longest_side` is 672; RC-D1 done (the 32 images come from the validation split).
- The pre-registration is frozen (2026-11-08), so the budget rule is fixed.

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
python scripts/run_card.py --card RC-B0 2>&1 | tee /kaggle/working/run_card_RC-B0.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

## 4. What to commit back

- `results/E2/nobudget/seed0/efficiency.csv` (the second reading)
- `results/E2/nobudget/seed0/efficiency_setup.json`
- This card with section 6 filled, including **both** `peak_memory_bytes` values.

## 5. How the result is read

B1 to B3 are frozen (2026-11-08) as 15, 30 and 45 % of E2's decoder-layer bytes removed; this
card only converts them into GB: target = `peak_memory_bytes` minus that share of
`decoder_layers_bytes`, both from the committed `efficiency.csv` (`budgets.target_value`).
Nothing is chosen from its numbers. If the two peak readings differ
by more than the tolerance of the budgets (3 %), the metric is not repeatable enough for the
matched-budget search and the agent asks the team to change it.

## 6. Run report (filled by the person who ran it)

- Run by / date: Amir's card, run on Kaggle; the runner's report is dated 2026-11-11 16:40 UTC
  (its "Run by" line was left as the placeholder).
- Commit actually used: 1320894933199f3ef5de27f338b74a1aa2a467bf.
- Hardware and library versions (`results/E2/nobudget/seed0/environment.json`, stage
  `07_benchmark_efficiency`; `efficiency_setup.json`): one Tesla T4 of T4 x2 (capability 7.5,
  15,636,037,632 bytes), driver 580.178.04, Python 3.13.16, torch 2.11.0+cu130 (CUDA 13.0,
  cuDNN 92700), transformers 5.19.0, peft 0.21.2, accelerate 1.15.0, torchao 0.18.0,
  bitsandbytes 0.50.2, optimum 2.3.0, pandas 2.3.3. The same builds as E2 seed 0's training.
- Protocol (`efficiency_setup.json`): batch 1, 3 warm-up and 5 timed passes, 32 validation
  images, `max_new_tokens` 46, hardware "Kaggle Tesla T4"; timing role "development" (the
  latencies are not the paper's timing, RC-T1 is).
- Inputs verified: the shared data (commit 4038a6b) and `e2-seed0-adapter`. An extra attached
  input, `E1_seed2_adapter`, was reported and ignored (RC-B0 does not read it), as it should.
- Steps: first reading 6.5 min, keep the first reading 0.0 min, second reading 5.4 min; all ok.
- `peak_memory_bytes`, first reading / second reading: **7,618,471,936 / 7,618,471,936**
  (identical; the 3 % repeatability rule of section 5 holds, so peak memory stays the budget
  metric).
- The split of E2's bytes (second reading, committed `efficiency.csv`): state 7,509,246,624;
  decoder layers 5,549,543,424; vision encoder 1,337,368,736; other 622,334,464 (the three
  parts sum to the state). Parameters 3,754,622,976.
- Latency: warm 0.9872 s per sample (second reading); cold 6.70 s in the first reading and
  1.91 s in the second (as reported; only the second reading is committed). The cold latency
  depends on what was loaded before and is not a budget metric. Throughput 0.964 samples/s;
  energy 68.9 J per sample (development GPU).
- Anything unusual: none besides the ignored extra input. Results committed unchanged
  (SHA-256 checked against the uploads): `efficiency.csv`, `efficiency_setup.json`,
  `environment.json` (the last now also holds stage 07).

### Budget targets (computed by the code from the committed `efficiency.csv`)

`budgets.target_value`: target = 7,618,471,936 minus the frozen fraction of 5,549,543,424
decoder-layer bytes; matched means within 3 % of the target on either side.

| Budget | Decoder bytes removed | Target (bytes) | 3 % band (bytes) |
|---|---|---|---|
| B1 | 15 % (832,431,513.6) | 6,786,040,422.4 (6.786 GB) | 6,582,459,209.7 to 6,989,621,635.1 |
| B2 (primary) | 30 % (1,664,863,027.2) | 5,953,608,908.8 (5.954 GB) | 5,775,000,641.5 to 6,132,217,176.1 |
| B3 | 45 % (2,497,294,540.8) | 5,121,177,395.2 (5.121 GB) | 4,967,542,073.3 to 5,274,812,717.1 |

The decoder layers are 72.8 % of the measured peak, so even B3 leaves most of the vision
encoder, embeddings and activations untouched; pruning reaches only the decoder layers.
