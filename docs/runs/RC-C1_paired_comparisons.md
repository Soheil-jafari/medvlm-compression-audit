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
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; outcomes of E2 (RC-R1) and of every E3-E8 session.
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off (add them to each stage command of section 3 when you run it by hand). The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-C1"`, `SEED = None`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-C1 --data-root /kaggle/input/<shared-data> --from /kaggle/input/<hand-off>`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - comparisons
  - comparisons, strict subset
  - comparisons, unfiltered test
  - tables and figures
- **Hand-offs it writes:** none (its results reach the others through git).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results
# Copy every attached outputs dataset into $MEDCOMPRESS_OUTPUT_ROOT, keeping the layout
# <ID>[/<variant>]/<budget|nobudget>/seed<k>/outcomes_test.npz, for example:
#   mkdir -p $MEDCOMPRESS_OUTPUT_ROOT && cp -r /kaggle/input/<outputs-dataset>/* $MEDCOMPRESS_OUTPUT_ROOT/
find $MEDCOMPRESS_OUTPUT_ROOT -name outcomes_test.npz | wc -l   # write this in section 6
find $MEDCOMPRESS_OUTPUT_ROOT -name outcomes_test_unfiltered.npz | wc -l   # the same number

python -m pytest -q tests/test_compare_stage.py
python scripts/09_compare_arms.py
python scripts/09_compare_arms.py --strict       # the strict subset (D02 amendment 3)
python scripts/09_compare_arms.py --unfiltered   # the unfiltered test (D02 amendment 6)
python scripts/08_aggregate_results.py
python -m pytest -q tests/test_results.py tests/test_paper.py   # must pass before you commit
```

`09_compare_arms.py` stops with "configuration incomplete" if the pre-registration is not
frozen or a setting is open; tell the agent through the inbox instead of filling it in. An arm
with a missing seed is left out and listed under `incomplete` in
`results/comparisons/choices.json`; check that list before committing.

## 4. What to commit back

- `results/comparisons/` (every `comparison.csv` and `choices.json`), and the same for the two
  sensitivity analyses, `results/comparisons_strict/` and `results/comparisons_unfiltered/`
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
