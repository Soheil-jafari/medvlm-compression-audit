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

python -m pytest -q tests/test_compare_stage.py
python scripts/09_compare_arms.py
python scripts/08_aggregate_results.py
python -m pytest -q tests/test_results.py tests/test_paper.py   # must pass before you commit
```

`09_compare_arms.py` stops with "configuration incomplete" if the pre-registration is not
frozen or a setting is open; tell the agent through the inbox instead of filling it in. An arm
with a missing seed is left out and listed under `incomplete` in
`results/comparisons/choices.json`; check that list before committing.

## 4. What to commit back

- `results/comparisons/` (every `comparison.csv` and `choices.json`)
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
