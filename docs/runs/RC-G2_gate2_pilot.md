# Run card RC-G2: Gate 2 pilot, E5 (three criteria), E6 and E7 at B2, seed 0

## 1. What and why

- Stage: S09
- Experiments: E5 (`magnitude`, `activation`, `taylor`), E6 and E7, budget B2, seed 0, scored on
  the **validation** split only
- Purpose: before the full grid, find out whether the clinical criterion (E7) keeps more of the
  small findings than the best of the generic criteria (E5) and the output-similarity criterion
  (E6) at the primary budget. The answer says which paper shape applies (see
  `docs/FINAL_GOAL.md`). Gate 2 does not stop any run.
- Status: `ready` (2026-11-11): every requirement of section 2 is met (RC-B0 done, B2 =
  5,953,608,908.8 bytes; the mixed scoring method is set, so the 30-group check of the
  first-order estimate is added as the card's last step, section 2c)
  Session 1 at commit 5a25147 failed after 1.5 min (2026-11-13): the scoring stage's student
  stayed on GPU 0 and the recovery stage's second load was offloaded to the CPU. Fixed: every
  stage of a run now runs in its own process, every load checks the GPU is clean and uses an
  explicit device map, and an offloaded model is refused. Run session 1 again on the fix commit

## 2. Requirements

Blocked until all of these are true; the agent changes the status to `ready` when they are:

- RC-E2 done for seed 0 (the E2 adapter on team storage) and RC-B0 done; the B2 target fraction
  is set from RC-B0 by the rule of D07 section 4 and confirmed by the team.
- D07 section 1 is answered (how a unit is switched off while it is scored), and
  `pruning.clinical.ablation_method` in `configs/experiments/E7.yaml` is set. If the mixed
  method is chosen, the agent adds the check of the estimate on 30 random channel groups to
  this card before setting it to `ready`.
- The pre-registration is frozen (`preregistration.frozen_on` set): the pilot is a compression
  run, and the stage scripts refuse compression runs before the freeze, so the finding weights
  cannot be adjusted after seeing the pilot.
- `training.steps` and `training.data_subset`, `evaluation.max_new_tokens`,
  `inputs.image_longest_side` and `budgets.measurement.hardware` are set.
- RC-D1 done; RC-M1 passed (met 2026-10-15).

Platform: Kaggle notebook, one T4 (`CUDA_VISIBLE_DEVICES=0`), Internet **on**. Dataset access:
your own copy of Kvasir-VQA-x1 and Kvasir-SEG and the private RC-D1 outputs. The E2 adapter and
the teacher cache are attached as private datasets. Nothing from those goes into the repository.

Runtime: five runs; each is scoring of the units, a search for the units that reach B2, a
recovery training and a validation pass. **Estimate (2026-11-11), about 19 GPU h plus the five
budget searches, about 25 GPU h with the searches estimated (section 3):** five recoveries of 157 updates x 27.2 s = 1.2 h each (E2's
measured update) and five validation passes of about 0.53 h (E2's measured pass): 8.7 h; E7's
scoring 7.9 h (108 layers and KV groups scored exactly, 640 probes x 0.414 s each; the channel
groups by the first-order estimate in minutes; the probe count is the schedule's assumption);
the 30-group check about 2.2 h (30 exact passes over the same probes); E5 and E6 scoring
minutes. The card runs in four sessions under the 12-hour session limit (section 3). The
scores of a run are reused when the settings are the same and the scores are committed, so a
repeated command does not score again.

## 2c. The 30-group check of the first-order estimate (added 2026-11-11)

D07 section 1 (2026-11-08) scores E7's layers and KV groups exactly and its MLP channel groups
by the first-order estimate (Molchanov et al., CVPR 2019). The check: 30 channel groups drawn at
random (seeded by the run seed, `stage.check_sample`) are also scored exactly, on the same
probes and weights, and Spearman's rank correlation between the two scores is reported
(`scripts/check_first_order.py --seed 0`, E7 by default, after E7's run; it reads the run's
`unit_scores.csv`). It writes `results/E7/nobudget/seed0/first_order_check.csv` (unit id,
first-order score, exact score; no images) and the rows `first_order_check:spearman`,
`:groups`, `:passed`, `:min_rank_correlation` and `:seconds` to that folder's `metrics.csv`.
It is a report, not a gate (pre-registration section 11 item 13, correction 1): the step does
not stop the card. **Below 0.7 the agent stops and asks the team** before any other E7 run
(the trend at B1 and B3, seeds 1 and 2, the ablations that use E7's scores).

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil (account 1; rebalanced schedule of 2026-11-08). Its five validation runs are seed 0 of the main tier's candidates: `scripts/choose_comparator.py` reads them to choose the comparator and apply the futility rule (seed plan of 2026-11-08).
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed0 adapter; teacher cache (not needed in session 3; section 3).
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** the four session cells of section 3 (`python scripts/run_card.py --card RC-G2 --only <part>`; the notebook `notebooks/run_card.ipynb` runs the whole card in one session, which is too long). The runner finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E5 magnitude B2 seed0
  - run E5 activation B2 seed0
  - run E5 taylor B2 seed0
  - run E6 B2 seed0
  - run E7 B2 seed0
  - first-order check E7 seed0 (section 2c)
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): one per pruned run, `G2_B2_seed0_pruned_<run>` (`E5_magnitude`, `E5_activation`, `E5_taylor`, `E6`, `E7`; section 3).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is 12 hours by the sources quoted in section 3 (confirm in the notebook's session panel); RC-G2 therefore runs in four sessions (section 3). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands: four sessions (2026-11-11)

**Why four sessions.** RC-G2 is about 25 GPU h in all (estimate, below), and a Kaggle GPU
session stops at **12 hours** (several sources quoting Kaggle's notebook documentation; the
agent could not open kaggle.com from its session, so check the limit shown in your notebook's
session panel). A stopped session does **not** resume a run where it stopped: E7's scores are
reused once they are committed (same settings and the same E2 adapter), but the budget search
starts again and the recovery's resume state is not handed off. So each session below runs
whole runs only, and stays under about 8 h (estimate).

**How:** `scripts/run_card.py --card RC-G2 --only <part>` (parts: `E5:magnitude`,
`E5:activation`, `E5:taylor`, `E6`, `E7:scores`, `E7`, `E7:check`; one or more per session,
always run in the card's order). Each pruned run hands off **its own** dataset, so no two
sessions write a dataset of the same name; later cards find them by their files, not their
names. Without `--only` the card would run all five runs and the check in one session (too
long).

**Account:** Soheil (account 1). **Kaggle settings:** GPU T4 x2 (the runner uses one T4);
Internet on; **Save Version -> Save & Run All**. Sessions 1, 2 and 3 do not depend on each
other and can run in any order (or the same week); session 4 needs session 3's results
committed first (the runner refuses `--only E7` and `--only E7:check` until
`results/E7/nobudget/seed0/unit_scores.csv` is in the repository).

Hours per session (estimates): a budget search S is about 14 footprint measurements for E5 and
E7 (bisection over about 3,000 units) and about 8 for E6 (36 layers), each about 6 min (RC-B0's
two readings took 6.5 and 5.4 min, model load included): **S about 1.4 h (E5, E7) and 0.8 h
(E6)**. A recovery is 1.2 h and a validation pass 0.53 h (E2's measured figures).

| Session | `--only` | Attach (Add Input) | Estimate | Save as private datasets afterwards | Send to the agent |
|---|---|---|---|---|---|
| 1 | `E5:magnitude`, `E5:activation` | `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache` | about 6.3 h (2 x (1.4 + 1.2 + 0.53)) | `g2-b2-seed0-pruned-e5-magnitude`, `g2-b2-seed0-pruned-e5-activation` (folders `G2_B2_seed0_pruned_E5_magnitude`, `..._E5_activation`) | `results_to_commit/`, the run report |
| 2 | `E5:taylor`, `E6` | the same three | about 5.6 h (3.1 + 0.8 + 1.2 + 0.53) | `g2-b2-seed0-pruned-e5-taylor`, `g2-b2-seed0-pruned-e6` | the same |
| 3 | `E7:scores` | `medcompress-data`, `e2-seed0-adapter` | about 8 h (108 layers and KV groups scored exactly, 640 probes x 0.414 s each, plus loading) | none (the scores are results files) | `results_to_commit/` (E7's `unit_scores.csv`, `unit_effects.csv`, their settings files and `metrics.csv`), the run report: **commit before session 4** |
| 4 | `E7`, `E7:check` | `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache` | about 5.3 h (1.4 + 1.2 + 0.53 + 2.2 for the check; the scores are reused) | `g2-b2-seed0-pruned-e7` | `results_to_commit/` (with `first_order_check.csv`), the run report |

In all about 25 GPU h (estimate). Keep the five datasets on Soheil's account: the cards that
read them run there (the B2 test evaluations, `--card RC-E5 / RC-E6 / RC-E7 --seed 0 --budget
B2` with them attached, and RC-E8 at B2; `docs/schedule.md` section 7, "Datasets").

**Session 3 is the long one.** Its log prints `[clinical] exact ablation: 25/108 units` and so
on. If the first 25 units take more than about 2.5 h, the 108 will not fit in 12 h: stop the
session and tell the agent (the 640 probes are the schedule's assumption, not a measurement).

Session 1:

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
python scripts/run_card.py --card RC-G2 --only E5:magnitude --only E5:activation 2>&1 | tee /kaggle/working/run_card_RC-G2_s1.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

Session 2:

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
python scripts/run_card.py --card RC-G2 --only E5:taylor --only E6 2>&1 | tee /kaggle/working/run_card_RC-G2_s2.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

Session 3 (attach no teacher cache; nothing to hand off):

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
python scripts/run_card.py --card RC-G2 --only E7:scores 2>&1 | tee /kaggle/working/run_card_RC-G2_s3.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

Session 4 (after session 3's results are committed):

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
python scripts/run_card.py --card RC-G2 --only E7 --only E7:check 2>&1 | tee /kaggle/working/run_card_RC-G2_s4.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

The runner runs the stages of each run in order (score the units, remove and recover,
evaluate, benchmark) and stops at the first one that fails. If a stage stops with
"configuration incomplete", a setting above is still open: do not fill it in yourself, tell the
agent through the inbox. Each pruned model goes to `checkpoints/` (never committed) and to its
hand-off folder under `/kaggle/working/handoff/`, which becomes the private dataset named in
the table (lower case, dashes).

## 4. What to commit back

For each of the five runs (`<run>` is `E5/<variant>`, `E6` or `E7`):

- `results/<run>/B2/seed0/metrics.csv` and `efficiency.csv`
- `results/<run>/B2/seed0/removal.json` and `budget_search.csv`
- `results/<run>/nobudget/seed0/unit_scores.csv`, `unit_scores.settings.json` and
  `metrics.csv` (numbers per unit and the scoring settings, no images; a later run reuses the
  scores only when the settings file matches)
- `results/E7/nobudget/seed0/first_order_check.csv` and the `first_order_check:*` rows of
  that folder's `metrics.csv` (section 2c)
- This card with section 6 filled.

Check before committing: `git status` shows no images, adapters, caches or `outcomes.npz`.

## 5. How the result is read

The agent writes `docs/results/GATE2.md`: E7 against the better of the five E5/E6 arms at B2 on
validation, with the paired image-level bootstrap interval on the primary endpoint of the
pre-registration draft, and the paper shape that follows. Every footprint is listed with its
distance from the target; a run outside the 3 % band is reported as such and is not hidden. The
team records the reading in `docs/decisions_log.md`. Nothing is tuned on these numbers: the
settings are fixed before the run.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Per run: wall time of scoring / search / recovery / evaluation, peak GPU memory:
- Units removed and measured peak memory against the B2 target (from `budget_search.csv`):
- Name of the private datasets with the pruned models and `outcomes.npz` files:
- The 30-group check: `first_order_check:spearman` and the step's wall time:
- Anything unusual (errors, retries, changed settings):
