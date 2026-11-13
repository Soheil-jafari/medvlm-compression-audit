# Run card RC-C1: paired comparisons on the test split (evidence for H1-H4)

## 1. What and why

- Stage: S11
- Experiments: none trained; reads the per-probe test answers of E2, E3-E8 (every variant,
  budget and seed).
- Purpose: compute the pre-registered paired comparisons (`scripts/09_compare_arms.py`), then
  the tables, figures and verdicts (`scripts/08_aggregate_results.py`).
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- Every run of E2 and E3-E8 has been evaluated on the **test** split (the cards RC-R1,
  RC-E3, RC-E4 and RC-E5 to RC-E8 are `done`) and its `results/` files are committed.
- The `outcomes_test.npz` files of all those runs are kept as private datasets (each card
  says where). They hold per-probe answers, never images, but stay out of the repository.

Platform: a Kaggle **CPU** notebook (no GPU needed). No dataset access is needed: the step
reads only the outcome files and the committed results.

Runtime: the bootstrap took about 2 minutes on the agent's CPU for a synthetic stand-in
(750 test images, 15 findings, 42 arms, 10,000 resamples). This is an **estimate**: the real
number depends on the size of the test split, so allow up to 30 minutes on a slow CPU.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil (CPU).
- **Kaggle settings:** no GPU (CPU); Internet off is fine.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; outcomes of E0 and E2 (RC-R1) and of every E3-E8 session.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-C1"` and `SEED = None` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-C1`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - comparisons
  - comparisons, unfiltered test
  - tables and figures
- **Hand-offs it writes:** none (its results reach the others through git).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

One cell (Kaggle settings and inputs of section 2b; **Save Version -> Save & Run All**). It finds the shared data and the attached hand-offs by their manifests (`scripts/find_inputs.py`), stops with a clear message if the shared data are missing, and runs the steps of section 2b with `scripts/run_card.py`, which also checks that every hand-off the card reads is there. Then it links every attached outcomes hand-off into the outputs folder, and writes the files to commit under `/kaggle/working/results_to_commit/`.

```bash
%%bash
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone,
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # so removing the clone keeps
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # every output
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
python scripts/find_inputs.py data > /dev/null || exit 1        # stops if the shared data are not attached
python scripts/find_inputs.py handoffs                          # the hand-offs found by their manifests
python scripts/run_card.py --card RC-C1 2>&1 | tee /kaggle/working/run_card_RC-C1.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
find -L /kaggle/working/outputs -name outcomes_test.npz | wc -l              # write this in section 6
find -L /kaggle/working/outputs -name outcomes_test_unfiltered.npz | wc -l   # the same number
```

`09_compare_arms.py` stops with "configuration incomplete" if the pre-registration is not
frozen or a setting is open; tell the agent through the inbox instead of filling it in. Seeds
follow the seed plan (pre-registration section 11 item 12, 2026-11-08): the main tier (E2, E7
and the comparator at B2) is compared on its 3 seeds (confirmatory, H1-H3); every other arm is
read at seed 0 and written as `<arm>_vs_E2_seed0` (exploratory); H4 is exploratory; an arm
with **2 seeds stops the step** (`SeedPlanError`). A main-tier arm with a missing seed is read
at seed 0 and listed under `incomplete` in `results/comparisons/choices.json` (with every arm's
tier and seeds under `tiers`); check that list before committing.

## 4. What to commit back

- `results/comparisons/` (every `comparison.csv` and `choices.json`), and the same for the
  unfiltered test, `results/comparisons_unfiltered/`
- `results/summary/`, `paper/tables/`, `paper/figures/` and `docs/results/VERDICTS.md`
- This card with section 6 filled.

Check before committing: `git status` shows no `.npz` files.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Every arm compared | `incomplete` in `choices.json` | empty (otherwise rerun the missing seeds first) |
| Comparator chosen by the rule | `h3.comparator` in `choices.json` | the arm with the highest mean validation small-finding recall at B2 among E5 and E6 |
| Primary endpoints present | `results/comparisons/E7_vs_best/B2/comparison.csv` | `diff:` and `p_holm:` rows for both endpoints |
| Verdicts written | `docs/results/VERDICTS.md` | no hypothesis left `pending` |

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Number of `outcomes_test.npz` files found:
- Wall time of step 09:
- Anything unusual (errors, retries, changed settings):
