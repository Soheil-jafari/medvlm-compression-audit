# Run card RC-D1-label: label the pairs, then score them (two raters; CPU)

## 1. What and why

- Stage: S03 (D02 amendment 3 of 2026-10-19, its additions of the same day, and amendment 4
  of 2026-10-22; all before any model result, amendment 4 before the scoring)
- Experiments: none (data only)
- Why: the rule that decides which images are copies of each other is set from your labels.
  The rule, fixed in advance, is the one that catches the most copies while at least half of
  the pairs it joins are real copies, and that does not chain more than 10 % of the images into
  one group.
- Who:
  - **two team members** labelled every pair (done, 2026-10-22; how, including where blinding
    was not complete, is in D02 amendment 4);
  - **no third member** (D02 amendment 4: the team has two members for this step);
  - Soheil runs the scoring.
- Status: `ready` (both label files are committed by SHA-256 in
  `docs/runs/RC-D1-label_commitments.csv`; the scoring has not run).

## 2. For the two raters, in plain words

Done on 2026-10-22: both files are labelled and committed by SHA-256. Kept as the record
of what the raters were told.

**What you get:** a private Kaggle dataset with only these three things (for example
`rc-d1-sheets`, made from RC-D1-embed's output):
- `label_sheets/`: 30 pictures, 10 pairs each. Each pair is two images side by side, with its
  number ("pair 17"). No score is shown, on purpose.
- `labels_template.csv`: one row per pair, with columns `pair_id`, `rater_1`, `rater_2`.
- `label_images.csv`: which two image files each pair is, if you want to open them at full
  size.

Do **not** open anything else of RC-D1-embed's dataset, above all `label_pairs.csv` and the
`.npz` files: they hold the scores.

**What to do:**
1. Make your own copy of `labels_template.csv`, named `labels_rater_1.csv` (the first rater)
   or `labels_rater_2.csv` (the second).
2. Fill **only your column** (`rater_1` or `rater_2`) with one word per pair:
   `same_scene`, `different` or `unsure`.
3. Look at each pair for as long as you need. Open the images at full size when a pair is hard.
4. When you are done, **commit your file** (section 2b) before anything else.
5. Do not talk about the pairs with the other rater, and do not look at the other's file, until
   **both** commitments are on `research-loop`.

**What counts as `same_scene`:**
- You can point to the **same structures** in both images: the same polyp or lesion, the
  same instrument in the same place, the same folds, the same vessel pattern, the same debris
  or bubbles.
- That holds even if:
  - one image is a re-saved, resized, cropped, brightened or blurred copy of the other;
  - the camera moved a little, zoomed, or the frame is a moment earlier or later;
  - the highlights (white reflections) sit in different places.
- The same polyp with the same instrument, seen from a slightly different position, is
  `same_scene`.

**What counts as `different`:**
- You cannot point to the same structures.
- Looking alike is not enough: the same colour, the same brightness, the lumen (the dark
  opening) in the same place, the same kind of smooth or pale mucosa (for example two pale
  oesophagus images with a similar texture), or the same text, green box or black border
  drawn on the image.

**What counts as `unsure`:**
- You looked carefully and still cannot tell, for example very smooth or very blurred mucosa
  with nothing to point to.
- Use it rarely. In the main scoring a pair called `unsure` counts as "not a copy"; a second
  analysis counts it the other way where that is less favourable.

**The overlays** (the text at the top left, the green box at the bottom left, the black border)
never decide anything: ignore them.

Time (**estimate**): about 10 to 20 seconds per pair, so about 50 to 100 minutes for 298 pairs.

### 2b. Commit your file (blinding)

Your labels never go into the repository. What goes in is your file's **SHA-256**, a
fingerprint: it shows later that the file was not changed, and the history shows who committed
first. In your own clone of `research-loop`, with your own account:

```bash
git pull --ff-only origin research-loop
python scripts/prepare_data.py commit-labels --labels labels_rater_1.csv --role rater_1 \
    --round first
git add docs/runs/RC-D1-label_commitments.csv
git commit -m "Commit the rater_1 labels of round first"
git push origin research-loop
```

(The second rater uses `labels_rater_2.csv` and `--role rater_2`.) The command refuses a file
that holds labels in another rater's column, and a second file for the same role and round.
After you have pushed, do not change your file: the scoring refuses a changed file.

Then put your file in ONE private Kaggle dataset for the labels (for example
`rc-d1-labels`), shared with the team only. Never commit it: it holds per-pair labels.

## 3. Scoring (Soheil, Kaggle CPU, minutes): D02 amendment 4

Two raters, no third member.

**Before you run:**
- Notebook: **CPU**, Internet **on** (the clone and pip).
- Attach (Add Input), private datasets only:
  - **RC-D1-embed's output dataset** (below: `/kaggle/input/rc-d1-embed`; use its slug if it
    is different);
  - **`rc-d1-labels-final`**, holding only the two label files, `labels_rater_1.csv` and
    `labels_rater_2.csv`. They must be byte for byte the committed ones: the scoring checks
    their SHA-256.
- Open question (decisions log): the kappa pairs. The cell uses `--kappa-pairs 11-298` as
  decided. Rater 1 labelled pairs 291-298 after the files were exchanged; if the team changes
  the range to 11-290, edit that one argument.

**The cell** (one cell; run it, then download the two files it copies):

```bash
%%bash
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
ls /kaggle/input/rc-d1-labels-final
python scripts/prepare_data.py score-labels --no-third-member --kappa-pairs 11-298 \
    --embed-dir /kaggle/input/rc-d1-embed \
    --commitments docs/runs/RC-D1-label_commitments.csv \
    --labels /kaggle/input/rc-d1-labels-final/labels_rater_1.csv \
             /kaggle/input/rc-d1-labels-final/labels_rater_2.csv \
    2>&1 | tee /kaggle/working/rc_d1_label.txt
echo "score-labels exit code: ${PIPESTATUS[0]} (0 a rule is chosen; 4 a top-up is required or no rule qualifies; 5 a check failed)"
mkdir -p /kaggle/working/results_to_commit/results/data
cp results/data/near_duplicate_labels.csv results/data/near_duplicate_rules.csv \
    /kaggle/working/results_to_commit/results/data/ 2>/dev/null
python - <<'PY'
import csv
from pathlib import Path

path = Path("results/data/near_duplicate_labels.csv")
if not path.is_file():
    print("no near_duplicate_labels.csv was written: read the printout above")
else:
    rows = {r["metric"]: r for r in csv.DictReader(path.open())}

    def show(metric):
        r = rows.get(metric)
        if r is None:
            return f"{metric}: not written"
        ci = f" ({r['ci_low']} to {r['ci_high']})" if r.get("ci_low") else ""
        return f"{metric}: {r['value']}{ci}"

    for metric in ("labels:kappa", "labels:kappa:first_pair", "labels:kappa:last_pair",
                   "labels:kappa:pairs", "labels:agreement", "labels:positives",
                   "labels:either_same_scene", "labels:chosen:view_code",
                   "labels:chosen:threshold", "labels:chosen:crop_threshold",
                   "labels:chosen:sensitivity", "labels:chosen:precision",
                   "labels:in_band_labelled", "labels:topup_required"):
        print(show(metric))
    needed = rows.get("labels:topup_required", {}).get("value")
    print("TOP-UP NEEDED: see section 3b" if needed and float(needed) > 0
          else "top-up not needed" if needed is not None else "top-up: not decided")
PY
```

- `labels:chosen:view_code`: 1 full, 2 centre, 3 the larger of the two.
- `labels:chosen:crop_threshold` -1: no crop part.

**What it does** (`data/labelling.py`, `data/embed_step.py`), in order:
1. **The commitments:** each label file must match its committed SHA-256, else it stops with
   exit code 5. A file changed after its commitment is refused.
2. **The model:** RC-D1-embed's output must hold the pinned model, revision and licence (else
   exit code 5).
3. **Cohen's kappa on pairs 11-298 only** (both blind), over the three labels, and for "same
   scene" against the rest; the raw agreement on the same pairs. Reported only: no threshold
   triggers anything. There is no adjudication.
4. **The positives (primary rule):** the pairs **both** raters call `same_scene`. `unsure`
   counts as not `same_scene`. All 298 pairs count for the rule.
5. **Every candidate rule, fixed in advance** (amendment 3):
   - a cosine of at least t (t from 0.50 to 0.99 in steps of 0.01), in the full view, the
     centre view or the larger of the two;
   - alone, or OR a crop correlation of at least 0.65 or 0.80;
   - 450 rules in all. For each, the sensitivity and the precision on the labelled pairs, with
     Wilson 95 % intervals.
6. **The choice:**
   - the highest sensitivity among the rules with precision >= 0.5 whose largest group over
     all 6,449 images stays within 10 %;
   - ties go to the higher precision, then to one view before a crop combination, then to the
     view order full, centre, larger, then to the lower threshold;
   - the guard is checked over all pairs of images, not only the labelled ones, one rule at a
     time in the order above until one passes.

     Measured here on synthetic embeddings (4 CPU cores): one check took 0.3 to 15 s, with
     up to 7.8 million joined pairs, under 1 GB. How many checks the real labels need is not
     known in advance.
7. **For the chosen rule:** the same two numbers weighted to all candidate pairs (each pair
   counts for its stratum's size over the pairs drawn), with stratified bootstrap 95 %
   intervals. These are for the data card.
8. **Reported, not used for the choice:**
   - the **either-rater** reading (`same_scene` for either rater): the chosen rule's
     sensitivity and precision under it, and the rule it would choose
     (`labels:either_rater:*`);
   - the **unsure** analysis: `unsure` as not `same_scene` for the precision and as
     `same_scene` for the sensitivity (both raters `same_scene` or `unsure`), and the rule it
     would choose (`labels:unsure_analysis:*`).
9. **Is a top-up required?** If fewer than **30** labelled pairs lie within **+-0.03** of the
   chosen threshold (in the chosen view), it prints `TOP-UP REQUIRED`, writes the files with
   `labels:topup_required` = 1 and stops with exit code 4. Run 4 waits. Go to section 3b.
   Either way it prints a `top-up:` line.

If no rule qualifies, it says so and stops with exit code 4: the team decides.

Do not use `CARD = "RC-D1-label"` in `notebooks/run_card.ipynb` for this scoring: its plan is
amendment 3's (a third member).

### 3b. The top-up (only if the scoring says `TOP-UP REQUIRED`)

First commit the two results files of the first scoring (section 4): the top-up reads the
first chosen rule from them. Then, on CPU, with RC-D1-embed's dataset and the shared data
attached (for the images of the sheets):

```bash
python scripts/prepare_data.py topup-labels --data-root /kaggle/input/<shared-data> \
    --embed-dir /kaggle/input/rc-d1-embed --out-dir /kaggle/working/rc_d1_topup \
    2>&1 | tee rc_d1_topup.txt
```

It does no new embedding. It draws:
- 100 candidate pairs from the band around the chosen threshold, seed 20261013;
- leaving out the pairs already labelled;
- all of them if the band holds fewer.

The pairs are numbered from 299 on, with new sheets without scores, a template and
`results/data/near_duplicate_topup.csv` (counts only: commit it).

Upload `/kaggle/working/rc_d1_topup` as ONE private dataset (for example `rc-d1-topup`). Give
the raters only its `label_sheets/`, `label_images.csv` and `labels_template.csv`. Then:
1. **Both** raters label all its pairs, as in section 2 (`labels_rater_1_topup.csv`,
   `labels_rater_2_topup.csv`).
2. Each commits their file with `--round topup` (section 2b).
3. They do not talk until both are pushed.

Then score once more, on all labels, with no third member there either:

```bash
python scripts/prepare_data.py score-labels --no-third-member --kappa-pairs 11-298 \
    --embed-dir /kaggle/input/rc-d1-embed --topup-dir /kaggle/input/rc-d1-topup \
    --commitments docs/runs/RC-D1-label_commitments.csv \
    --labels /kaggle/input/rc-d1-labels-final/labels_rater_1.csv \
    /kaggle/input/rc-d1-labels-final/labels_rater_2.csv \
    /kaggle/input/rc-d1-labels-final/labels_rater_1_topup.csv \
    /kaggle/input/rc-d1-labels-final/labels_rater_2_topup.csv 2>&1 | tee rc_d1_label_2.txt
```

The rule is chosen again **once**, on all labels, by the same criterion; there is no second
top-up (`topup-labels` refuses). The top-up round's kappa is reported on its own
(`labels:kappa:topup`).

## 4. What to paste back and commit

- Paste back the whole printout (`rc_d1_label.txt`, and `rc_d1_topup.txt` and
  `rc_d1_label_2.txt` after a top-up), and the summary lines the cell prints. Above all:
  - the kappa line (pairs 11-298) and the positives line;
  - the `chosen rule` line;
  - the `either-rater analysis`, `unsure analysis` and `top-up:` lines.
- Commit, from `/kaggle/working/results_to_commit/results/data/`:
  - `results/data/near_duplicate_labels.csv`: kappa, counts and the chosen rule as numbers;
  - `results/data/near_duplicate_rules.csv`: every rule's sensitivity and precision;
  - after a top-up, `results/data/near_duplicate_topup.csv` too.

  All are aggregate only. RC-D1 run 4 reads the chosen rule from the first.
- Never commit a label file or the sheets.

## 5. Pass criteria

| Check | Where | Pass if |
|---|---|---|
| The files are the committed ones | the `matches the commitment` lines | one per label file, no "is not in" stop |
| Both raters labelled every pair | the run | no "has no label" stop |
| The kappa is on the decided pairs | `labels:kappa:first_pair`, `labels:kappa:last_pair` | 11 and 298 (or the range the team set) |
| Agreement | `labels:kappa` | reported as it is; nothing depends on it |
| A rule was chosen | `labels:chosen:*` | present, with `labels:topup_required` = 0 |

## 6. Run report (filled by the person who ran it)

- Raters (names) and dates:
- Commit actually used (the `commit:` line):
- The commitments (D02 amendment 4): both rows pushed in commit 92b78e3; the `rater_2` row
  replaced an earlier one (288 labels, pairs 1-10 empty)
- The kappa line (pairs, kappa, the same-scene kappa, raw agreement):
- The positives line (both, either, both same_scene or unsure):
- The chosen rule line (rule, sensitivity and precision with their intervals, weighted values,
  largest group):
- The either-rater and unsure analysis lines:
- Top-up: required or not; if so, the `topup-labels` lines and the second scoring's lines:
- The scoring's run time:
- Anything unusual:
