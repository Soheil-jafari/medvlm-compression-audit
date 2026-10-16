# Run card RC-X1: external test set (EndoBench), part A: terms, format and overlap

## 1. What and why

- Stage: S11
- Experiments: none. This part checks the external set before any model sees it.
- Purpose: table T6 reports the main arms on an external set, deduplicated against our
  training images (D02: EndoBench). Before the agent writes a loader, the team must confirm
  that we may use it, record what its files look like, and count how many of its images are
  near-copies of Kvasir images we train or test on (EndoBench draws on public endoscopy sets,
  so overlap is likely).
- Status: `blocked` (see section 2)
- Part B (the evaluation of E0, E2, E3, E5, E6 and E7 at B2) is written by the agent from what
  this part reports; it is not written yet.

## 2. Requirements

- RC-D1 done (the overlap is counted against our splits).
- No GPU. Any machine where you may store the data under your own access (a Kaggle CPU notebook
  is enough).

Runtime: minutes for the overlap count (it fingerprints each image once); the reading of the
terms takes as long as it takes.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Farzan (CPU).
- **Kaggle settings:** no GPU (CPU); Internet off is fine.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-X1"`, `SEED = None`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-X1`. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - overlap
- **Hand-offs it writes:** none (its results reach the others through git).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits (unknown here; TODO: the team checks on Kaggle):** the session time limit, and whether a T4 x2 session counts once or twice against the weekly quota. A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Steps

1. **Terms.** Find the official EndoBench release from the paper (arXiv 2505.23601) and read
   its licence and access conditions. Write in section 6 the exact URL or hub ID you opened,
   its version or revision, the licence name and whether research use, use with other
   datasets, and reporting results are allowed. If any of these is not allowed or not clear,
   **stop here** and tell the agent through the inbox: we then decide on another external set
   (D02 lists ColonEval as the alternative).
2. **Download** with your own access, outside the repository.
3. **Format.** Look at the files and write in section 6 (numbers and field names only, no
   images, no patient identifiers):
   - the number of images and of questions;
   - the question types and the answer format (yes/no, multiple choice with how many options,
     free text);
   - which tasks ask about findings that are on our finding list (`configs/domain/endoscopy.yaml`),
     and with which wording;
   - whether any images come with lesion masks or boxes;
   - the source datasets of the images, as the release lists them.
4. **Overlap** against our splits (counts only):

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
python scripts/prepare_data.py overlap --data-root /kaggle/working/data \
    --private-dir /kaggle/input/<your-rc-d1-private> \
    --images /path/to/endobench/images --name endobench --dedup-threshold 4
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

The threshold 4 (of 64 bits) is the one RC-D1 used for validation and test.

## 4. What to commit back

- `results/data/endobench_overlap.csv` (counts only)
- This card with section 6 filled. The agent then fills `docs/data_cards/endobench.md` from it.

Check before committing: `git status` shows no images, data files or file lists of the data.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Use allowed | section 6, step 1 | research use, use with our datasets and publishing results are all allowed |
| Something left to test | `images_after_dedup` | more than 0; the agent proposes a minimum per finding with part B |

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Step 1, terms (URL or hub ID opened, version, licence, the three permissions):
- Step 3, format (counts, question types, answer format, tasks on our finding list, masks or
  boxes, source datasets):
- Step 4, overlap (the numbers printed):
- Anything unusual:
