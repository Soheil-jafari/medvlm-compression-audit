# Run card RC-G2: Gate 2 pilot, E5 (three criteria), E6 and E7 at B2, seed 0; E5R and E7U (exploratory)

## 1. What and why

- Stage: S09
- Experiments: E5 (`magnitude`, `activation`, `taylor`), E6 and E7, budget B2, seed 0, scored on
  the **validation** split only; since 2026-11-17 also the exploratory E5R (random scores) and E7U
  (E7's scores), both with E5's design (session 5; not comparator candidates)
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
  explicit device map, and an offloaded model is refused. The rerun at 427c3ce stopped (exit 6)
  in stage 05: the stage still held its first student copy (7.2 GiB) when the budget search
  loaded the next one; the guard refused the load. Fixed: a copy's tensors are freed
  (`loading.release_model`) before any further load, so a stage never holds two copies; each
  footprint is still measured on a freshly loaded and pruned copy. Run session 1 again **after
  RC-SMOKE passed for RC-G2**

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

Added 2026-11-17 (amendment of that date; reports, not gates): the slope and intercept of the
exact effect on the first-order estimate over the 30 groups, with 95 % intervals
(`first_order_check:slope`, `:slope_ci_low`, `:slope_ci_high`, `:intercept`, ...), and E7's B2
selection again with the 30 exact effects in place of their estimates, both selections with the
same offline footprint estimate (`first_order_check_b2.csv`; rows `:b2_switched`,
`:b2_switched_to_removed`, `:b2_switched_to_kept`, `:b2_kv_or_layer_changes`). It runs on CPU
after the exact scoring; `python scripts/check_first_order.py --seed 0 --followup-only` reruns it
from the committed files.

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
| 1 | `E5:magnitude`, `E5:activation` | `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache` | about 6.3 h (2 x (1.4 + 1.2 + 0.53)) | **one** dataset `g2-b2-seed0-pruned-s1` (saved as `g2-b2-seed0-pruned-e5` before the rename of 2026-11-16) holding both folders `G2_B2_seed0_pruned_E5_magnitude` and `G2_B2_seed0_pruned_E5_activation` | `results_to_commit/`, the run report |
| 2 | `E5:taylor`, `E6` | the same three | about 5.6 h (3.1 + 0.8 + 1.2 + 0.53) | **one** dataset `g2-b2-seed0-pruned-s2` holding `G2_B2_seed0_pruned_E5_taylor` and `G2_B2_seed0_pruned_E6` | the same |
| 3 | `E7:scores` | `medcompress-data`, `e2-seed0-adapter` | about 8 h (108 layers and KV groups scored exactly, 640 probes x 0.414 s each, plus loading) | none (the scores are results files) | `results_to_commit/` (E7's `unit_scores.csv`, `unit_effects.csv`, their settings files and `metrics.csv`), the run report: **commit before session 4** |
| 4 | `E7`, `E7:check` | `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache` | about 5.3 h (1.4 + 1.2 + 0.53 + 2.2 for the check; the scores are reused) | **one** dataset `g2-b2-seed0-pruned-s4` holding `G2_B2_seed0_pruned_E7` | `results_to_commit/` (with `first_order_check.csv`), the run report |

**One dataset per session (2026-11-16):** Kaggle makes one dataset from a notebook's output, so a
session's pruned models are saved together: save the whole `/kaggle/working/handoff/` folder as
`g2-b2-seed0-pruned-s<k>`. When that dataset is attached, the runner finds each model folder by its
own manifest and a card reads only the folders it needs (`tests/test_inputs.py`). Session 1's
dataset was saved as `g2-b2-seed0-pruned-e5`; either name works, the manifests decide.

In all about 25 GPU h (estimate). Keep the three datasets (`-s1`, `-s2`, `-s4`) on Soheil's account: the cards that
read them run there (the B2 test evaluations, `--card RC-E5 / RC-E6 / RC-E7 --seed 0 --budget
B2` with them attached, and RC-E8 at B2; `docs/schedule.md` section 7, "Datasets").

**Session 3 is the long one.** Its log prints `[clinical] exact ablation: 25/108 units` and so
on. If the first 25 units take more than about 2.5 h, the 108 will not fit in 12 h: stop the
session and tell the agent (the 640 probes are the schedule's assumption, not a measurement).

**Session 1, attempt at 2846e81 (2026-11-15): stopped in stage 05 at the first budget-search
load (exit 6), 2 min in.** After stage 05's first load was released, the process held 0.59 GiB of
live tensors and about 6.5 GiB of its own allocator cache; the guard counted all 7.12 GiB as "in
use" and refused, although the new load reuses that cache. Fixed (2026-11-15): the guard refuses
only this process's live tensors or memory outside its allocator (above 1.5 GiB each); offload is
still prevented by the explicit device map and the check after every load. The release now also
removes accelerate hooks, clears tensors kept outside parameters and buffers, synchronizes and
empties the cache, prints allocated and reserved memory, and prints the largest survivors once.
The cause of the 0.59 GiB was not reproduced on the CPU (the same loads leave nothing there);
the survivor print names it on the GPU. **Run RC-SCALE's `budget_search` (14 measurements)
first** (RC-SCALE section 3), then session 1 again with the cell below.

**Before session 1 (2026-11-15):** the budget search's footprints were offset by 0.59 GiB left
by earlier loads (RC-SCALE section 6, third run). Fixed: the footprint excludes memory that is not
the model's, and stage 05 measures the unpruned model first and stops unless it matches E2's
reference within 1 % (a line "level check: ..." in the log). Run RC-SCALE's `budget_search`
(fourth run, RC-SCALE section 3) first; when it passes, run session 1 with the cell below (Soheil,
T4 x2, attach `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache`; about 6.3 h, estimate).

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

**Session 3, attempt 1 (Amir, commit 9f04c4f, 2026-11-14): failed after 96 min.** The exact
ablation finished (108 of 108 units); the first-order pass then ran out of GPU memory on one T4
(14.18 GiB allocated, in `scoring.accumulate_gate_gradients` -> `yes_no_logprobs(grad=True)`).
Cause (2026-11-14): the backward pass to the gates kept every decoder layer's intermediate
tensors for all 36 layers at once; with a batch of four 672-px probes that is about 7 GiB on top
of the 6.99 GiB of weights. The smoke's capped unit set and two probes per finding kept it
small there (11.02 GiB). **Fixed without changing a setting** (batch 4 stays): each decoder
layer is recomputed in the backward pass instead of kept (the same gradients, bit for bit on the
tiny model), the weights are checked to collect no gradient, each step's graph is freed before
the next, and a `MEDCOMPRESS_MEM` line gives the GPU memory every 25 steps. Expected peak about
8.5-9.5 GiB (estimate; RC-SCALE measures it). **The exact effects are now saved as soon as the
exact pass ends** (`exact_effects.resume.json` next to the scores, removed once the scores are
written): a rerun with the same settings, units and probes reads them instead of ablating again,
from its own output folder or from an attached input (the failed session's output added with
Add Input). Attempt 1 ran before this code, so its 108 units are not saved: the rerun ablates
them again (attempt 1 reached the first-order pass after less than 96 min, loading included).

Session 3, rerun (run RC-SCALE first; attach `medcompress-data` and `e2-seed0-adapter`; if an
attempt that already had this fix (2026-11-14, the commit that adds RC-SCALE) failed after its
exact pass, also attach that notebook's output):

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
python scripts/run_card.py --card RC-G2 --only E7:scores 2>&1 | tee /kaggle/working/run_card_RC-G2_s3_rerun.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

Session 4 (after session 3's results are committed, and after RC-SMOKE passed again for E7's
stage 05: `python scripts/smoke_cards.py --card RC-G2 --sequential`, see RC-SMOKE section 6):

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

**Reruns under the amendment of 2026-11-16 (no unit of layers 0, 1, 35 is removed).** Session 1's
two runs (and session 2, stopped partway at d9039dc) were made while the units inside the
protected layers were removable; they are superseded (`results/superseded/inner_units_removable/`)
and run again at the new code. E7's scores (session 3) stay: the protected layers' units are left
out of the order, nothing is rescored. Do **not** attach `g2-b2-seed0-pruned-e5` (session 1's old
models, kept for the appendix) to a rerun.

Order and accounts (suggested; the three sessions do not depend on each other and can run at the
same time on three accounts):

1. First, once: the stage-05 smoke of E5 magnitude at the new code (RC-SMOKE section 3,
   `--job "E5 magnitude"`, about 20 min; Soheil). Then, at the same time:
2. Session 1 again (E5 magnitude, activation): **Soheil (account 1)**, about 6.3 h; save the
   handoff folder as `g2-b2-seed0-pruned-s1`.
3. Session 2 again (E5 taylor, E6): **account 3**, about 5.6 h; save as `g2-b2-seed0-pruned-s2`.
4. Session 4 (E7, the 30-group check): **Amir**, about 5.3 h, after the smoke of E7's stage 05
   (RC-SMOKE, `--job E7`) has passed; save as `g2-b2-seed0-pruned-s4`.

Share the three datasets with Soheil's account (the B2 test evaluations and RC-E8 read them
there). Each session attaches `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache` (shared
with accounts 3 and Amir's).

Session 1 again (Soheil):

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
python scripts/run_card.py --card RC-G2 --only E5:magnitude --only E5:activation 2>&1 | tee /kaggle/working/run_card_RC-G2_s1_rerun.txt
echo "exit code: ${PIPESTATUS[0]}"
```

Session 2 again (account 3):

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
python scripts/run_card.py --card RC-G2 --only E5:taylor --only E6 2>&1 | tee /kaggle/working/run_card_RC-G2_s2_rerun.txt
echo "exit code: ${PIPESTATUS[0]}"
```

Session 4 (Amir; after the E7 stage-05 smoke has passed):

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
echo "exit code: ${PIPESTATUS[0]}"
```

Each stage-05 log now shows "level check: ..." and each run writes `removal_by_layer.csv` (the
share of its removed parameters per decoder layer: layers 0, 1 and 35 must read 0) and
`recovery_steps.json`.

**Sessions under the amendment of 2026-11-17 (the locked pruning design; replaces the rerun plan
above).** E5 now removes only MLP channel groups, the same number k from every unprotected layer
(2-34), and its search runs over k; two exploratory arms join at B2, seed 0: E5R (random scores,
E5's design) and E7U (E7's committed scores, E5's design); E7 is unchanged (global, fully
protected). `docs/decisions/pruning_design_2026-11-17.md` has the design. Do **not** attach
`g2-b2-seed0-pruned-e5` or any earlier `-s2` output to these sessions: they hold models of the old
design (a recovery resume state of another removal set is now refused anyway).

Estimates per run (from session 1's measured times: a footprint measurement about 6 min, the
recovery 55-56 min, the validation pass 0.53 h): the k search needs about 9 measurements
(bisection over 0-85, plus the level check), about 0.9 h, against about 14 for a global search.

| Cell | Account | `--only` / job | Attach (Add Input) | Estimate | Save afterwards |
|---|---|---|---|---|---|
| (a) smoke | Soheil | `--job "E5 magnitude" --job E5R --job E7U` | `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache` | about 30-40 min (two T4 lanes) | nothing; send `smoke_report.md` |
| (b) session 1 | Soheil (account 1) | `E5:magnitude`, `E5:activation` | the same three | about 5.5-6 h | `/kaggle/working/handoff/` as `g2-b2-seed0-pruned-s1` |
| (c) session 2 | account 3 | `E5:taylor`, `E6` | the same three | about 5.5 h (Taylor scoring included) | `g2-b2-seed0-pruned-s2` |
| (d) session 4 | Amir | `E7`, `E7:check` | the same three | about 5.3 h | `g2-b2-seed0-pruned-s4` |
| (e) session 5 | any account, after (a) | `E5R`, `E7U` | the same three | about 5.5 h | `g2-b2-seed0-pruned-s5` |

Order: (a) first; when it passes, (b), (c) and (d) at the same time on the three accounts; (e)
after (a) on whichever account is free. Every session sends `results_to_commit/` and its run
report. Each stage-05 log of a `uniform_mlp` arm prints "uniform_mlp allocation: k = ..."; its
`removal_by_layer.csv` must show 0 for layers 0, 1 and 35 and the same removed parameters in every
layer from 2 to 34. The offline estimate is k = 29 at B2 (`results/replay/removal_replay.md`); the
measured search may land a step away. Session 4's check also prints the fit of the exact effects
on the estimates and how many of the 30 groups switch at B2.

(a) The stage-05 smoke of the new design (Soheil, T4 x2):

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
python scripts/smoke_cards.py --job "E5 magnitude" --job E5R --job E7U 2>&1 | tee /kaggle/working/smoke_cards_uniform.txt
echo "exit code: ${PIPESTATUS[0]}"
```

(b) Session 1 (Soheil):

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
python scripts/run_card.py --card RC-G2 --only E5:magnitude --only E5:activation 2>&1 | tee /kaggle/working/run_card_RC-G2_s1_uniform.txt
echo "exit code: ${PIPESTATUS[0]}"
```

(c) Session 2 (account 3):

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
python scripts/run_card.py --card RC-G2 --only E5:taylor --only E6 2>&1 | tee /kaggle/working/run_card_RC-G2_s2_uniform.txt
echo "exit code: ${PIPESTATUS[0]}"
```

(d) Session 4 (Amir):

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
echo "exit code: ${PIPESTATUS[0]}"
```

(e) Session 5 (any account, after the smoke):

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
python scripts/run_card.py --card RC-G2 --only E5R --only E7U 2>&1 | tee /kaggle/working/run_card_RC-G2_s5.txt
echo "exit code: ${PIPESTATUS[0]}"
```

## 4. What to commit back

For each of the five runs (`<run>` is `E5/<variant>`, `E6` or `E7`):

- `results/<run>/B2/seed0/metrics.csv` and `efficiency.csv`
- `results/<run>/B2/seed0/removal.json` and `budget_search.csv`
- `results/<run>/nobudget/seed0/unit_scores.csv`, `unit_scores.settings.json` and
  `metrics.csv` (numbers per unit and the scoring settings, no images; a later run reuses the
  scores only when the settings file matches)
- `results/E7/nobudget/seed0/first_order_check.csv`, `first_order_check_b2.csv` (2026-11-17)
  and the `first_order_check:*` rows of that folder's `metrics.csv` (section 2c)
- Session 5: `results/E5R/nobudget/seed0/` (its random scores) and `results/E5R/B2/seed0/`,
  `results/E7U/B2/seed0/` (E7U has no scores of its own)
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

**Session 3 rerun (`--only E7:scores`), reported by Amir on 2026-11-14 (commit 9069603; T4 x2, the
runner used one T4):** all steps ok.

- Library versions: torch 2.11.0+cu130, transformers 5.19.0, peft 0.21.2, accelerate 1.15.0,
  safetensors 0.8.0.
- Inputs verified: the shared data (commit 4038a6b) and the E2 seed-0 adapter.
- Scoring: 3,204 units (36 layers and 72 KV groups exactly, 3,096 MLP channel groups first-order)
  on 128 calibration images in 5,908 s (100.2 min with loading); scoring peak 8,284,429,824 bytes
  (8.28 GB); the exact effects were saved before the first-order pass; first-order memory flat at
  7.01 GiB. Settings as frozen (`unit_scores.settings.json`: clinical weights, logprob_drop,
  signed effects, 1 negative per positive, batch 4, parent adapter 7a690940...).
- Files committed unchanged (SHA-256 checked against the attached files):
  `results/E7/nobudget/seed0/` `environment.json`, `metrics.csv`, `unit_scores.csv`,
  `unit_scores.settings.json`, `unit_effects.csv`, `unit_effects.settings.json`.
- Hand-offs: none (the scores are results files).
- Note (agent, 2026-11-15) on the protected layers: the code keeps layers 0, 1 and 35 from being
  removed **as whole layers** (`plan_removal` drops them from the removal order and
  `validate_removal_set` refuses them); their KV groups and MLP channel groups stay removable, for
  every arm alike. With these scores the first unit of the order is `lm.layer35.mlp_group65`, and a
  removal sized to B2 by parameter count (an estimate; the search measures memory) holds 64 units
  inside the protected layers: 41 of layer 35's 86 MLP groups, 21 of layer 0's, 2 of layer 1's
  (6.0 % of the removed parameters). Raised with Soheil and Amir before session 4 (decisions log,
  2026-11-15).

**Session 1 (`--only E5:magnitude --only E5:activation`), reported by Soheil on 2026-11-16
(commit 4f54655; T4 x2, one T4 used):** all steps ok (magnitude 206.3 min, activation 181.8 min).
18 files committed unchanged (SHA-256 checked against the attached files) under
`results/E5/{magnitude,activation}/{nobudget,B2}/seed0/`, and the run report as
`docs/runs/reports/run_report_RC-G2_s1.md` (the same file as the attached
`run_report_RC-G2.md`, renamed so later sessions do not overwrite it).

- Level check (stage 05, both runs): the unpruned model measured 7.61257e9 against E2's
  reference 7.61847e9 (-0.08 %, within 1 %).
- Budget search (B2 target 5.954e9; the smallest removal at or below target x 1.03):
  magnitude 220 units (7 layers, 19 KV groups, 194 channel groups), footprint 6.041e9 (+1.47 %),
  13 measurements; activation 281 units (6 layers, 15 KV groups, 260 channel groups), 6.131e9
  (+2.98 %), 12 measurements. Both within the 3 % tolerance. Benchmark peaks: magnitude
  6,041,026,208, activation 6,129,130,144 bytes (about 0.095 GB above their state bytes).
- Units inside the protected layers 0, 1, 35 (removable unit by unit): magnitude 171 of 220
  (138.4 M parameters), activation 172 of 281 (143.1 M).
- Recovery (157 updates): 5 steps with a non-finite loss in each run, 13 (magnitude) and 15
  (activation) updates skipped by the loss scaler (the non-finite steps among them; the rest
  in the first updates while the scale settles); peaks 10.18 and 10.42 GB; 3,281 s and 3,362 s.
  The non-finite steps came at the same points in both runs (one by step 25, two by 50, three by
  75, five by 125): both runs visit the pool in the same order (seed 0). Whether they are the
  steps that hold the 24 training images with a non-finite fp16 vision output (RC-CAL0) is
  checked by `scripts/check_nonfinite_steps.py` (decisions log, 2026-11-16).
- Validation: op macro balanced accuracy magnitude 0.598, activation 0.572 (E2 seed 0: 0.836);
  op small-finding recall 0.692 and 0.538; small-finding recall at the default threshold 0.0 and
  lesion-mask effect 0.0 for both.
- Hand-off: both pruned models in **one** private dataset, `g2-b2-seed0-pruned-e5` (Soheil's
  account), holding `G2_B2_seed0_pruned_E5_magnitude` (5.95 GB) and
  `G2_B2_seed0_pruned_E5_activation` (6.03 GB); the runner finds each by its manifest.

**The non-finite recovery steps, a CPU check (2026-11-16).** Soheil's account, Kaggle CPU (no
GPU needed), Internet on; attach `medcompress-data` and a private dataset holding RC-CAL0's
`vision_scan_student.csv` (for example `rc-cal0-scan`; not in the repository, it lists image ids).
Prints, per session 1 run, the steps that hold one of the 24 listed training images and the
cumulative counts at every 25th step, to compare with the log (1, 2, 3, 3, 5, 5, 5 for both runs).
Estimate: about 5 min.

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
SCAN=$(find /kaggle/input -name vision_scan_student.csv | head -1); echo "scan: $SCAN"
DATA=$(python scripts/find_inputs.py data)
for v in magnitude activation; do
  python scripts/check_nonfinite_steps.py --exp E5 --variant $v --budget B2 --seed 0 \
    --data-root "$DATA" --scan "$SCAN" 2>&1 | tee -a /kaggle/working/nonfinite_steps.txt
done
```

Note (2026-11-16): of the 231.2 M parameters of the protected layers 0, 1 and 35 (3 x 77.1 M),
session 1 removed 138.4 M (magnitude, 59.9 %) and 143.1 M (activation, 61.9 %) unit by unit.

Note (2026-11-16): session 1 above is **superseded** by the amendment of 2026-11-16 (units of
protected layers removable, exploratory, outside H1-H4); its files were moved unchanged to
`results/superseded/inner_units_removable/` and its run report stays here. Session 2 was stopped
partway at d9039dc (nothing committed). Both run again at the new code (section 3).

