# Run card RC-D1-label: label the pairs, then score them (two raters, a third member; CPU)

## 1. What and why

- Stage: S03 (D02 amendment 3 of 2026-10-19 and its additions of the same day, before any
  model result and before any label)
- Experiments: none (data only)
- Why: the rule that decides which images are copies of each other is set from your labels.
  The rule, fixed in advance, is the one that catches the most copies while at least half of
  the pairs it joins are real copies, and that does not chain more than 10 % of the images into
  one group.
- Who:
  - **two team members** label every pair, each on their own (proposal: Soheil and one of Amir
    or Farzan);
  - **the third team member** (the one who is not a rater) decides the disagreements;
  - Soheil runs the scoring.
- Status: `ready` once RC-D1-embed's output is shared (RC-D1-embed ran on 2026-10-19: 298
  pairs to label).

## 2. For the two raters, in plain words

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

## 3. Scoring (Soheil, CPU, minutes)

Attach RC-D1-embed's dataset and the labels dataset. Internet is not needed. Pull
`research-loop` first, so the commitments are there.

**Unattended (Save & Run All):** `notebooks/run_card.ipynb`, cell 1:

```python
CARD = "RC-D1-label"
SEED = None
DATA = None
FROM = []
COMMIT = None
EXTRA = ["--embed-dir", "/kaggle/input/rc-d1-embed",
         "--labels", "/kaggle/input/rc-d1-labels/labels_rater_1.csv",
         "/kaggle/input/rc-d1-labels/labels_rater_2.csv"]
```

Or by hand:

```bash
python scripts/prepare_data.py score-labels --embed-dir /kaggle/input/rc-d1-embed \
    --commitments docs/runs/RC-D1-label_commitments.csv \
    --private-dir /kaggle/working/rc_d1_adjudication \
    --labels /kaggle/input/rc-d1-labels/labels_rater_1.csv \
    /kaggle/input/rc-d1-labels/labels_rater_2.csv 2>&1 | tee rc_d1_label.txt
```

What it does (`data/labelling.py`, `data/embed_step.py`), in order:
1. **The commitments:** every label file must match a committed SHA-256 (else exit code 5).
2. **Cohen's kappa before adjudication**, over the three labels, and for "same scene" against
   the rest; per round (the first pairs; the top-up).
3. **The third member** (D02 amendment 3, additions):
   - they decide every pair the two raters label differently (`unsure` included);
   - if a round's kappa is below **0.40**, they decide **every** pair of that round;
   - if any pair is still undecided, the step writes their list,
     `adjudication_template.csv` (`pair_id, rater_3`), to `--private-dir`, prints
     `ADJUDICATION NEEDED`, writes nothing to the repository and stops with exit code 4.

   The third member then:
   - fills the `rater_3` column (same three words, same instructions as section 2), looking
     the pairs up on the sheets. The list shows no score and neither rater's label;
   - saves it as `labels_rater_3.csv`;
   - commits it as in section 2b with `--role rater_3` (the same `--round`);
   - adds it to the labels dataset.

   Run the scoring again with that file added to `--labels`. A decided pair is a copy only if
   the third member calls it `same_scene`.
4. **The positives (primary rule):** the pairs that are `same_scene` for both raters, or for the
   third member where they decided. `unsure` counts as not `same_scene`.
5. **Every candidate rule, fixed in advance:**
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
   - the guard is checked over all pairs of images, not only the labelled ones.
7. **For the chosen rule:** the same two numbers weighted to all candidate pairs (each pair of
   the first round counts for its stratum's size over the pairs drawn), with stratified
   bootstrap 95 % intervals. These are for the data card.
8. **The unsure analysis:** for the chosen rule, `unsure` counted as not `same_scene` for the
   precision and as `same_scene` for the sensitivity; and the rule this reading would choose.
   Reported only: run 4 uses the primary rule.
9. **Is a top-up required?** If fewer than **30** labelled pairs lie within **+-0.03** of the
   chosen threshold (in the chosen view), it prints `TOP-UP REQUIRED`, writes the files with
   `labels:topup_required` = 1, and stops with exit code 4. Run 4 waits. Go to section 3b.

If no rule qualifies, it says so and stops with exit code 4: the team decides.

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
1. Both raters label the new pairs as in section 2 (`labels_rater_1_topup.csv`,
   `labels_rater_2_topup.csv`).
2. Each commits their file with `--round topup` (section 2b).
3. They do not talk until both are pushed.

Then score once more, on all labels:

```bash
python scripts/prepare_data.py score-labels --embed-dir /kaggle/input/rc-d1-embed \
    --topup-dir /kaggle/input/rc-d1-topup \
    --commitments docs/runs/RC-D1-label_commitments.csv \
    --private-dir /kaggle/working/rc_d1_adjudication \
    --labels /kaggle/input/rc-d1-labels/labels_rater_1.csv \
    /kaggle/input/rc-d1-labels/labels_rater_2.csv \
    /kaggle/input/rc-d1-labels/labels_rater_3.csv \
    /kaggle/input/rc-d1-labels/labels_rater_1_topup.csv \
    /kaggle/input/rc-d1-labels/labels_rater_2_topup.csv 2>&1 | tee rc_d1_label_2.txt
```

(Unattended: add `"--topup-dir", "/kaggle/input/rc-d1-topup"` and the extra files to `EXTRA`.)
- The third member decides the top-up's disagreements the same way, with `--round topup` and
  their own new file.
- The rule is chosen again **once**, on all labels, by the same criterion; there is no second
  top-up (`topup-labels` refuses).

## 4. What to paste back and commit

- Paste back the whole printout (`rc_d1_label.txt`, and `rc_d1_topup.txt` and
  `rc_d1_label_2.txt` after a top-up). Above all:
  - the kappa line and the round lines (the kappa before adjudication, how many pairs the third
    member decided);
  - the `chosen rule` line and the `unsure analysis` line.
- Commit:
  - `results/data/near_duplicate_labels.csv`: kappa, counts and the chosen rule as numbers;
  - `results/data/near_duplicate_rules.csv`: every rule's sensitivity and precision;
  - after a top-up, `results/data/near_duplicate_topup.csv` too.

  All are aggregate only. RC-D1 run 4 reads the chosen rule from the first.
- Never commit a label file, the adjudication list or the sheets.

## 5. Pass criteria

| Check | Where | Pass if |
|---|---|---|
| The files are the committed ones | the `matches the commitment` lines | one per label file, no "is not in" stop |
| Both raters labelled every pair | the run | no "has no label" stop |
| The third member decided what they had to | `labels:adjudicated` | no `ADJUDICATION NEEDED` left |
| Agreement | `labels:kappa` (before adjudication) | reported as it is. Below 0.40 in a round, the third member decided every pair of it |
| A rule was chosen | `labels:chosen:*` | present, with `labels:topup_required` = 0 |

## 6. Run report (filled by the person who ran it)

- Raters (names), the third member (name), and dates:
- Time spent labelling (each):
- Commit order of the commitments (commit hash, role, round, in the order they were pushed):
- The kappa line and the round lines (kappa before adjudication, pairs decided):
- The chosen rule line (rule, sensitivity and precision with their intervals, weighted values,
  largest group):
- The unsure analysis line:
- Top-up: required or not; if so, the `topup-labels` lines and the second scoring's lines:
- Anything unusual (pairs that were hard, disagreements):
