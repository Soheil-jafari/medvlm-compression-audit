# Run card RC-D1-label: label the pairs, then score them (two raters; CPU)

## 1. What and why

- Stage: S03 (D02 amendment 3 of 2026-10-19, before any model result)
- Experiments: none (data only)
- Why: the rule that decides which images are copies of each other is set from your labels.
  The rule, fixed in advance, is the one that catches the most copies while at least half of
  the pairs it joins are real copies, and that does not chain more than 10 % of the images into
  one group.
- Who: **two team members**, each on their own (proposal: Soheil and one of Amir or Farzan).
  Then Soheil runs the scoring.
- Status: `ready` once RC-D1-embed's output is shared.

## 2. For the two raters, in plain words

**What you get** (from RC-D1-embed's private dataset):
- `label_sheets/`: about 30 pictures, 10 pairs each. Each pair is two images side by side,
  with its number ("pair 17"). No score is shown, on purpose.
- `labels_template.csv`: one row per pair, with columns `pair_id`, `rater_1`, `rater_2`.
- `label_images.csv`: which two image files each pair is, if you want to open them at full
  size. Use this file only. Do **not** open `label_pairs.csv`: it holds the scores.

**What to do:**
1. Make your own copy of `labels_template.csv`, named `labels_rater_1.csv` (the first rater)
   or `labels_rater_2.csv` (the second).
2. Fill **only your column** (`rater_1` or `rater_2`) with one word per pair:
   `same_scene`, `different` or `unsure`.
3. Do not talk about the pairs with the other rater until you have both finished, and do not
   look at the other's file.
4. Look at each pair for as long as you need. Open the images at full size when a pair is hard.

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
- Use it rarely: a pair you call `unsure` counts as "not a copy" in the scoring.

**The overlays** (the text at the top left, the green box at the bottom left, the black border)
never decide anything: ignore them.

Time (**estimate**): about 10 to 20 seconds per pair, so about 50 to 100 minutes for about 300
pairs.

**When done:** put both filled files (`labels_rater_1.csv`, `labels_rater_2.csv`) in ONE
private Kaggle dataset (for example `rc-d1-labels`), shared with the team only. Never commit
them: they are per-pair labels.

## 3. Scoring (Soheil, CPU, minutes)

Attach RC-D1-embed's dataset and the labels dataset. Internet is not needed.

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
    --labels /kaggle/input/rc-d1-labels/labels_rater_1.csv \
    /kaggle/input/rc-d1-labels/labels_rater_2.csv 2>&1 | tee rc_d1_label.txt
```

What it computes (`data/labelling.py`, `data/embed_step.py`):
- **Cohen's kappa** of the two raters over the three labels, and for "same scene" against the
  rest.
- **The positives:** the pairs both raters call `same_scene`.
- **Every candidate rule, fixed in advance:** a cosine of at least t (t from 0.50 to 0.99 in
  steps of 0.01), in the full view, the centre view or the larger of the two, alone or OR a
  crop correlation of at least 0.65 or 0.80. That is 450 rules. For each, the sensitivity and
  the precision on the labelled pairs, with Wilson 95 % intervals.
- **The choice:**
  - the highest sensitivity among the rules with precision >= 0.5 whose largest group over
    all 6,449 images stays within 10 %;
  - ties go to the higher precision, then to one view before a crop combination, then to the
    view order full, centre, larger, then to the lower threshold;
  - the guard is checked over all pairs of images, not only the labelled ones.
- **For the chosen rule:** the same two numbers weighted to all candidate pairs (each labelled
  pair counts for its stratum's size over the pairs drawn), with stratified bootstrap 95 %
  intervals. These are for the data card.

If no rule qualifies, it says so and stops with exit code 4: the team decides.

## 4. What to paste back and commit

- Paste back the whole printout (`rc_d1_label.txt`), above all the kappa line and the
  `chosen rule` line.
- Commit `results/data/near_duplicate_labels.csv` (kappa, counts and the chosen rule as
  numbers) and `results/data/near_duplicate_rules.csv` (every rule's sensitivity and
  precision). Both are aggregate only. RC-D1 run 4 reads the chosen rule from the first.

## 5. Pass criteria

| Check | Where | Pass if |
|---|---|---|
| Both raters labelled every pair | the run | no "has no label" stop |
| Agreement is usable | `labels:kappa` | reported as it is. A kappa below about 0.4 (an agent proposal, not pre-registered) is a reason to talk before run 4 |
| A rule was chosen | `labels:chosen:*` | present |

## 6. Run report (filled by the person who ran it)

- Raters (names) and dates:
- Time spent labelling (each):
- The kappa line:
- The chosen rule line (rule, sensitivity and precision with their intervals, weighted values,
  largest group):
- Anything unusual (pairs that were hard, disagreements):
