# Run card RC-D1: data preparation (Kvasir-VQA-x1 + Kvasir-SEG)

## 1. What and why

- Stage: S03
- Experiments: none (data only)
- Purpose:
  - download the data into your own Kaggle session;
  - **re-split all 6,449 images by image** (D02 amendment of 2026-10-18; the official split is
    by question, so 3,821 of its 4,058 test images are also training images): near-duplicate
    images are grouped first, and whole groups go to train / validation / test, 70 / 15 / 15 %
    of the images, stratified by finding answer and by Kvasir-SEG mask;
  - group near-duplicates in **two stages** (D02 amendment 2 of 2026-10-18, final values
    2026-10-19; run 2 showed that the 4-bit dHash rule misses adjacent frames, and the
    calibration that it also joins different images, section 5): dHash candidates within 12
    bits, each joined only if its correlation on the centre of the image, where no overlay is
    drawn, is at least 0.65;
  - link Kvasir-SEG masks to the VQA images;
  - report **aggregate counts only**, so the finding list, prevalence groups, label rules and
    size thresholds can be pre-registered;
  - write **one shared-data folder** (the images, the rows, Kvasir-SEG and the private RC-D1
    files, with a SHA-256 manifest) that every later card attaches as a private Kaggle dataset.
- Who: **Soheil** (decision of 2026-10-17).
- Status: `ready` for **run 3** (section 3b). The calibration step ran on 2026-10-19 and the
  threshold is decided (section 5). Runs 1 and 2 are recorded in section 5; nothing from them
  is used.

## 2. Requirements

- Platform: Kaggle notebook, **CPU is enough**, Internet **on** (Settings -> Internet).
- Repository commit to run from: latest `research-loop` (write the commit in section 5).
- Dataset access and licences:
  - **Kvasir-VQA and Kvasir-VQA-x1** download from the Hugging Face hub in the notebook.
    Licence: CC BY-NC 4.0 (non-commercial use, with attribution).
  - **Kvasir-SEG**, by the browser route (run 1: `wget` from datasets.simula.no failed on Kaggle,
    because the certificate chain could not be verified):
    1. Download `kvasir-seg.zip` in your browser from <https://datasets.simula.no/kvasir-seg/>.
    2. Upload it as a **private** Kaggle dataset.
    3. Attach it to the notebook (**Add Input**).

    The notebook finds the folder that holds `images/` and `masks/` with `find`: no dataset
    name is written anywhere. Run 1 found it at
    `/kaggle/input/datasets/<user>/kvasir-seg/Kvasir-SEG`, with 1,000 images and 1,000 masks.
    Licence: research and educational use only.
  - The shared-data folder holds copies of these images. It stays a **private** dataset,
    shared with the team's three Kaggle accounts only (Soheil, Amir, Farzan), never public. The
    manifest records both licences and this rule.
- **Three steps.**
  1. **Calibration** (CPU; section 3a): **done 2026-10-19** (commit 73f31bc; results in
     `results/data/near_duplicate_calibration.csv`, commit b4cf740). The threshold 0.65 is in
     `configs/data/kvasir_vqa_x1.yaml` (`near_duplicates.min_correlation`) with its reason
     (D02 amendment 2, final values).
  2. **Run 3** (section 3b): as run 2, with the two-stage grouping. It writes the shared data,
     the counts and a final contact sheet with three bands of cross-split pairs (section 3b).
     Attach that sheet to the report. **It is the deciding check:** if a person sees same-scene
     pairs on it, the next step is the pre-registered fallback (decisions log, 2026-10-19), not
     a new threshold.
  3. **Pass 2** (CPU, minutes): the label values of the abnormality question
     (`configs/data/kvasir_vqa_x1.yaml`, `values: {}`) are still empty, so run 3 writes the
     answer values per split (`answer_count:<rule>:<value>:<split>`) but no `positives:*` or
     `prevalence:*`. The agent proposes the label values from these counts; the team confirms;
     pass 2 then recounts from the attached shared data with the same settings. It stops if
     the splits do not hash as in run 3. The splits themselves do not depend on the label
     values: they are stratified by the raw answer values.
- Expected runtime of run 3 (**estimate**): about 15 to 25 minutes, as run 2. The export took
  about 9 minutes in run 1. The calibration step scored the 211,329 candidate pairs (its wall
  time was not reported); on a 4-core CPU here, 100,000 synthetic pairs took about 7 s and
  the training-against-held-out matrix about 12 s. Hashing the folder for the manifest: not
  measured.

Settings (pre-registration items; decided before any model result, D02 amendments of
2026-10-18, final values 2026-10-19):

| Setting | Value | Why |
|---|---|---|
| `--split-seed` | `20261012` | any fixed integer; recorded in the pre-registration |
| `--close-bits` | `4` | names the band of very close pairs (0-4 of 64 dHash bits) that is reported; since 2026-10-19 they are **not** joined automatically (the calibration showed 0-4 bit pairs of different images). `--dedup-threshold` is its old name and still works |
| `candidate_bits` | `12` | every pair within 12 bits, 0-4 bits included, is a candidate for the crop check |
| `crop_fraction` | `0.60` | the central 60 % of width and of height: 20 % is cut from each side, where the text (top left), the green box (bottom left) and the black border are drawn |
| `crop_side` | `32` | the crop, grey-scale, area-averaged to 32 x 32 pixels |
| `crop_highpass_sigma` | `4.0` | a Gaussian blur of 4 px (at 32) is subtracted: the measure follows the mucosa's detail, not the lighting or the lumen position, which made unrelated images look alike at 9 x 8 |
| `crop_max_shift` | `3` | the crop of one image is compared with the other shifted by up to 3 px (5.6 % of the image side) each way: adjacent frames move a little |
| measure | highest Pearson correlation over the shifts, both ways round | insensitive to brightness and contrast; 1.0 for identical crops |
| `min_correlation` | `0.65` | candidate pairs at or above it are joined: the valley of the bimodal 5-8 bit distribution; 2 of 2,000 random controls reach it (calibration, section 5) |
| `max_group_fraction` | `0.10` | chaining guard: run 3 STOPS (exit code 7) if one group holds more than 10 % of the images (644 of 6,449); a group needs at least 3 images to chain. Below 0.60 the largest group jumps above 30 %: that is what chaining looks like |
| `--mask-threshold` | `4` | max differing bits to link a Kvasir-SEG image to a VQA image |
| split ratios | 70 / 15 / 15 % of the images | `split_ratios` in `configs/data/kvasir_vqa_x1.yaml` |

The crop settings, the threshold and the guard are in `configs/data/kvasir_vqa_x1.yaml`
(`near_duplicates`); the seed and the two bit settings are on the command line.

How the groups are made (`data/dedup.py: two_stage_groups`, CPU-tested):
1. Candidates: every pair of images within 12 bits of the 64-bit dHash.
2. Joined: every candidate whose crop correlation is at least 0.65, and no other pair. Nothing
   is joined on the dHash alone; the 0-4 bit pairs left apart are counted and shown.
3. Union-find over the joined pairs: if a~b and b~c, a, b and c are one group.
4. The chaining guard: STOP and report if the largest group holds more than 10 % of the images.

From the calibration (with the old automatic join), the largest group at 0.65 was 349 images
(5.4 %). Without the automatic join it can only be smaller or equal: fewer pairs are joined.

How the groups are split (`data/splits.py: assign_groups_stratified`, CPU-tested; unchanged):
1. The groups are put in a seeded order.
2. Labels (the answer values of the finding question, and "has a Kvasir-SEG mask") are taken
   from the rarest to the most common. A label carried by at least three groups first gets one
   group in each split that has none.
3. Each remaining group of a label goes to the split furthest below its share of that label.
4. Groups without a label fill the image targets.

Checks after the split (in `summary.csv`, section 3b):
- no validation or test image has a joined pair with a training image (zero by construction;
  the run stops otherwise);
- the pairs within 4 bits that are not joined, in all and across splits, and the held-out
  images with a training image within 4 bits (reported, not filtered);
- the dHash tail (0 to 17 bits) and the crop-correlation tail to the training split;
- how many validation and test images have a training image **beyond 12 bits** whose crop
  correlation reaches the threshold: pairs the candidate stage cannot see. A number for the
  data card, not a filter.

Limitation: HyperKvasir has no patient IDs. Images of the same patient taken at different
moments can still fall into different splits (data card).

## 3. Commands

### 3a. The calibration step (before run 3): done 2026-10-19

Kept for the record and for a rerun. Attach run 2's shared-data dataset if you saved it (it
holds the same 6,449 images). If not, attach nothing: the step exports the images first.
Internet on. Since 2026-10-19 the table and the sheet follow the final rule (no automatic join
within 4 bits); the committed calibration file was made with the old rule.

**Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-D1-calibrate"`,
`SEED = None`, `DATA = "/kaggle/input/<run-2-shared-data>"` (or `None` to export first),
`FROM = []`, `EXTRA = []`. Or by hand, in one CPU notebook:

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 5
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps && pip install -q datasets

DATA=/kaggle/input/<run-2-shared-data>                        # verified first (SHA-256)
# without run 2's shared data, export first (about 9 minutes in run 1):
#   DATA=/kaggle/working/medcompress_data/data
#   python scripts/prepare_data.py export --data-root $DATA
python scripts/prepare_data.py calibrate --data-root $DATA --split-seed 20261012 \
    --close-bits 4 --sheet-dir /kaggle/working/rc_d1_sheets 2>&1 | tee rc_d1_calibration.txt
```

What it prints (aggregate counts only): the settings; the candidate pairs by dHash distance
(0 to 12 bits); the crop correlation in bins of 0.05 for pairs at 0-4, 5-8 and 9-12 bits and
for 2,000 random pairs beyond 12 bits (the controls); the controls' median, 99th percentile and
maximum; for each threshold from 0.30 to 0.95: the joined pairs, the 0-4 bit pairs left apart,
the groups, the largest group and its share, the guard, and the controls at or above it; a
provisional proposal (a starting point to check by eye, not a decision).

The contact sheet `/kaggle/working/rc_d1_sheets/rc_d1_calibration_pairs.png` (private: it
shows dataset images; **never commit it**) has four bands, each pair side by side with the crop
drawn as a yellow box: the 8 candidate pairs just below the threshold (not joined), the 8 just
above it (joined), 4 random controls, and the 0-4 bit pairs left apart (highest correlation
first). `--threshold 0.80` redraws it at another value.

How 0.65 was chosen (2026-10-19, section 5): a missed duplicate puts a copy of a training image
into validation or test (expensive); a wrong join only puts two images in the same split
(cheap, as long as no giant group forms). So the threshold sits in the valley of the bimodal
5-8 bit distribution, not at the provisional proposal 0.80, which missed a same-scene pair.

### 3b. Run 3

Attach Kvasir-SEG only (not run 2's shared data: the copy step looks for a folder with
`images/` and `masks/` under `/kaggle/input`). Internet on.

**Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-D1"`,
`SEED = None`, `DATA = None`, `FROM = []`, `EXTRA = []`. The runner finds the attached Kvasir-SEG
folder itself, runs the five steps below and prints section 5. Or by hand, in one CPU notebook:

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 5
grep -E "min_correlation:|max_group_fraction:" configs/data/kvasir_vqa_x1.yaml  # 0.65, 0.10
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps && pip install -q datasets

export SHARED=/kaggle/working/medcompress_data                # data/ + private/, outside the repo
python scripts/prepare_data.py export --data-root $SHARED/data

# Kvasir-SEG: the attached folder that holds images/ and masks/ (found, not named)
SEG=$(dirname "$(find /kaggle/input -type d -name masks -path '*Kvasir-SEG*' | head -n 1)")
echo "Kvasir-SEG at: $SEG"; ls "$SEG/images" | wc -l; ls "$SEG/masks" | wc -l   # 1000 and 1000
python -c "import sys; sys.path.insert(0, 'src'); from pathlib import Path; \
from medcompress.utils.kaggle import copy_kvasir_seg; \
copy_kvasir_seg(Path('$SEG'), Path('$SHARED/data'))"

python scripts/prepare_data.py prepare --data-root $SHARED/data --private-dir $SHARED/private \
    --split-seed 20261012 --close-bits 4 --mask-threshold 4 \
    --sheet-dir /kaggle/working/rc_d1_sheets                  # exit 7 = the chaining guard: STOP
python scripts/prepare_data.py bundle --bundle-dir $SHARED    # SHA-256 manifest of every file
python scripts/prepare_data.py verify --data-root $SHARED     # must print OK
python -m pytest -q tests/test_results.py                     # must pass before you commit
```

If `prepare` stops with exit code 7 ("STOP: the largest near-duplicate group holds ..."), do
not change any setting: paste the message back. Exit code 4 means the clone does not hold the
threshold (an old commit).

**Paste back:**
- the line `groups ..., largest ... (...%); held-out images with a confirmed duplicate in
  train: 0; pairs within 4 bits not joined: ... (... across splits)`;
- the hashes printed at the end (`split_sha256`, manifests, the bundle's file count);
- the final contact sheet `/kaggle/working/rc_d1_sheets/rc_d1_final_cross_split_pairs.png`,
  attached to the report (private; **never commit it**), with one line per band: how many
  pairs, if any, show the same scene.

The final contact sheet has three bands, all pairs across splits, each with its crop
correlation, dHash distance and splits:
1. the 24 most similar pairs by the crop correlation;
2. the 24 closest pairs by the dHash distance;
3. the 0-4 bit pairs the crop left apart: the count in all and across splits in the title, and
   the 12 across splits with the highest crop correlation.

All of them should be different scenes. **This sheet is the deciding check.** If a person sees
same-scene pairs on it, the split is not used: the next step is the pre-registered fallback
(embeddings from a small pretrained vision model, compared by cosine similarity on CPU;
decisions log, 2026-10-19), not another threshold.

What `results/data/summary.csv` holds after run 3 (aggregate counts only; experiment `DATA`):

| Rows | What |
|---|---|
| `images:all`, `groups:total`, `groups:multi_image`, `groups:largest`, `groups:largest_fraction`, `images_in_multi_image_groups`, `groups:size:<1, 2, 3, 4, 5, 6_to_10, 11_to_50, 51_to_100, 101_or_more>`, `groups:<split>` | the near-duplicate groups |
| `near_duplicates:<setting>` | the settings used (`close_bits`, `candidate_bits`, `min_correlation`, `max_group_fraction`, the crop) |
| `near_duplicates:candidate_pairs`, `:joined_pairs`, `:candidate_pairs_not_joined` | the two stages |
| `near_duplicates:close_pairs`, `:close_pairs_joined`, `:close_pairs_not_joined`, `:close_pairs_not_joined:cross_split` | the 0-4 bit pairs, and those the crop left apart (in all and across splits) |
| `qa_rows:<split>`, `images:<split>`, `image_fraction:<split>`, `masked_images:<split>` | the size of each split |
| `val_test_confirmed_duplicates_of_train`, `confirmed_duplicates_of_train:<val/test>` | must be 0 |
| `val_test_close_dhash_to_train` | held-out images with a training image within 4 bits that the crop left apart (reported, not filtered) |
| `min_distance_to_train:<val/test>:0` ... `:17`, `:18_or_more` | the dHash distance of each held-out image to the training split |
| `max_score_to_train:<val/test>:<bin>` | each held-out image's highest crop correlation with a training image, in bins of 0.05 |
| `similar_to_train_beyond_candidates:<val/test>` | held-out images with a training image beyond 12 bits at or above the threshold (not filtered) |
| `cross_split_top:max_score`, `:min_score`, `:within_candidate_bits`; `cross_split_closest_dhash:min_bits`, `:max_bits`, `:max_score` | bands 1 and 2 of the final contact sheet |
| `answer_count:<rule>:<value>` and `answer_count:<rule>:<value>:<split>` | the answer values of the finding question, overall and per split (images) |
| `positives:<finding>:<split>`, `prevalence:<finding>` | after the label values are filled (pass 2) |
| `answer_words:p50/p90/p99/max` | on the training split (sets `max_new_tokens`, D04) |
| `masks:linked/ambiguous/unmatched`, `lesion_area_threshold:q1..q3` | Kvasir-SEG links; size strata from the training split |
| `official:images:*`, `official:test_rows:*`, `official:test_only_min_distance_to_train:*` | the official split's overlap (data card, appendix item) |

**Pass 2** (after the agent has committed the label values and the team has confirmed them), a
new CPU notebook with run 3's shared data attached:

```bash
python scripts/run_card.py --card RC-D1-counts --data-root /kaggle/input/<shared-data>
# by hand: python scripts/prepare_data.py counts --data-root /kaggle/input/<shared-data> \
#     --split-seed 20261012 --close-bits 4 --mask-threshold 4
```

## 4. Make the shared data a private Kaggle dataset, and what to commit back

Upload, step by step. Kaggle's button names may differ slightly; the dataset must stay
**Private** throughout.

1. Run run 3 with **Save Version -> Save & Run All** (the unattended notebook), so that
   `/kaggle/working/medcompress_data/` is saved in the version's output.
2. Open the finished version, then its **Output** tab, and create a dataset from the
   `medcompress_data` folder (**New Dataset**). Name it, for example, `medcompress-data`, and set
   **Private**. Do not use a public dataset, and do not upload it anywhere else. If run 2's
   dataset exists, give run 3's a new name (for example `medcompress-data-run3`) or delete run
   2's, so nobody attaches run 2's splits by mistake.
3. On the dataset's page, open its settings (**Sharing** / **Collaborators**). Add Amir's and
   Farzan's Kaggle usernames with view access. Keep it Private.
4. Check it from a new notebook: **Add Input** -> the dataset, then run
   `python scripts/prepare_data.py verify --data-root /kaggle/input/<dataset>`. It must print
   `OK`; the runner of every later card does the same check before it starts.
5. Write the dataset's name and the `split_sha256` that `bundle` prints into section 5.

Not known here: the size limit of a notebook output and of a dataset (TODO: check on Kaggle;
the folder is about the size of the images).

What to commit back:
- `results/data/near_duplicate_calibration.csv` (the calibration step; aggregate counts only).
- `results/data/summary.csv` (aggregate counts only; no image names or per-image rows). Pass 2
  replaces it.
- This card with section 5 filled.

Never commit the contact sheets: they show dataset images. Keep them in the notebook output
and attach them to the report.

Never commit the shared-data folder or anything in it. Every later run needs it as the attached
private dataset.

## 5. Run report (filled by the person who ran it)

### Run 1 (2026-10-16/17, Soheil, Kaggle CPU): old split; nothing committed

Run 1 is recorded for the data card; its splits are not used.
- `export`: 6,500 images and both splits, 422 MB on disk, about 9 minutes.
- Kvasir-SEG: `wget` from datasets.simula.no failed on Kaggle (the certificate chain could not
  be verified). The zip was downloaded in a browser and uploaded as a private Kaggle dataset,
  found at `/kaggle/input/datasets/<user>/kvasir-seg/Kvasir-SEG` (1,000 images, 1,000 masks).
- `prepare` with the old split (seed 20261012, dedup 4, mask 4):

  | Split | QA rows | Images | Masked images |
  |---|---|---|---|
  | train | 143,316 | 6,212 | 1,010 |
  | validation | 39 | 39 | 0 |
  | test | 30 | 30 | 0 |

  `duplicates_removed:val_test` 3,989; masks linked 1,010, ambiguous 5, unmatched 12; answer
  words p50 9, p90 18, p99 25, max 36.
- Diagnosis: the official Kvasir-VQA-x1 split is by question, not by image.
  - Images: train 6,212; test 4,058; in both 3,821; test only 237; all 6,449.
  - Test rows: 15,718 of the 15,955 are on shared images; 237 are on test-only images (one row
    each).
  - Smallest dHash distance from each test-only image to any training image (distance: number
    of images): 0: 43, 1: 43, 2: 43, 3: 21, 4: 18, 5: 14, 6: 14, 7: 10, 8: 5, 9: 7, 10: 5,
    11: 1, 12: 6, 13: 1, 14: 4, 15: 1, 17: 1.
  - The dataset paper (arXiv 2506.09958, section 2.3): "Each image-question pair is assigned to
    either the training or test set".
  - The published SimulaMet adapter was trained on the official train split, so it has seen
    almost every image. Hence the re-split, and our own teacher (D01 and D02 amendments).

### Run 2 (reported 2026-10-18, Soheil, Kaggle CPU): 4-bit grouping; nothing committed

Run 2 is recorded for D02 amendment 2; its splits are not used.
- `split_sha256` 024f4abf05a81dffaa1c2b120e3e7bb99456b03fd248ddb6419284a8cf57fde9.

  | Split | Images | Share of images | QA rows | Masked images |
  |---|---|---|---|---|
  | train | 4,518 | 70.06 % | 112,370 | 710 |
  | validation | 965 | 14.96 % | 24,379 | 152 |
  | test | 966 | 14.98 % | 22,458 | 148 |

- Groups (dHash <= 4 bits): 4,457 in all; 398 with more than one image; the largest holds 255
  images; 2,390 images are in multi-image groups.
- `val_test_near_duplicates_of_train` 0 (within 4 bits).
- Smallest dHash distance from each held-out image to any training image (bits: images); every
  held-out image is in these lists:
  - validation: 5: 119, 6: 126, 7: 148, 8: 169, 9: 136, 10: 100, 11: 60, 12: 41, 13: 29,
    14: 13, 15: 16, 16: 6, 17: 2;
  - test: 5: 99, 6: 113, 7: 138, 8: 126, 9: 168, 10: 121, 11: 82, 12: 56, 13: 30, 14: 15,
    15: 13, 16: 4, 18 or more: 1.
- Visual check of the pairs:
  - random pairs have a median distance of 26 bits, and a 5th percentile of 17;
  - 1,038 of the 1,931 validation and test images have a training image within 8 bits;
  - at 5-6 bits, 3 of 8 pairs were the same scene; at 7-8 bits, 2 to 3 of 8; at 9-12 bits, 0
    of 4; random controls 0 of 4;
  - the other pairs looked alike only at 9 x 8 pixels: colour, lumen position, and the
    overlays (text at the top left, a green box at the bottom left).
- Conclusion: the 4-bit rule misses real duplicates, and a larger dHash threshold alone would
  chain unrelated images. Hence the two-stage grouping (D02 amendment 2).
- Rare answer values (images, train / validation / test), for the freeze list only: barretts
  15 / 4 / 3, short-segment barretts 8 / 4 / 1, hemorrhoids 1 / 0 / 0. Run 3's counts replace
  them.

### Calibration step (2026-10-19, Soheil, Kaggle CPU, commit 73f31bc)

Results: `results/data/near_duplicate_calibration.csv` (commit b4cf740). Made with the rule of
that commit: pairs within 4 bits were still joined automatically, so the table's groups include
them.
- 6,449 images; 211,329 candidate pairs within 12 bits (by distance 0 to 12: 337, 620, 899,
  1,274, 1,756, 2,643, 4,218, 7,240, 11,845, 20,005, 31,980, 50,967, 77,545).
- Crop correlation of the 5-8 bit pairs (25,946): bimodal, one mode at 0.15-0.30 (different
  images) and one at 0.80-0.90 (same scene); the valley at 0.60-0.70 (bins 0.60: 731, 0.65:
  671, 0.70: 809).
- 0-4 bit pairs (4,886): 929 score below 0.65 (sum of the committed bins under 0.65).
- Random controls (2,000 pairs beyond 12 bits): median 0.137, 99th percentile 0.420, maximum
  0.808; 2 reach 0.65.
- Largest group: 2,059 images (31.9 %) at 0.55; 428 (6.6 %) at 0.60; 349 (5.4 %) at 0.65; 335
  (5.2 %) at 0.70 and 0.75; 321 (5.0 %) at 0.80. The provisional proposal was 0.80.
- Contact sheet, by eye:
  - just below 0.80 (not joined): 1 of 8 pairs is the same scene (#5, 7 bits, practically
    identical colon frames), so 0.80 misses it;
  - just above 0.80 (joined): 2 of 8 are the same scene (#6, #7); 6 of 8 are different images,
    mostly pale oesophagus images with a similar texture;
  - always joined within 4 bits, lowest correlation: all 4 pairs are clearly different images
    (r = 0.009, 0.042, 0.063, 0.065), so the automatic join links unrelated images and feeds
    chaining;
  - random controls: all different.
- Chosen (D02 amendment 2, final values, 2026-10-19): `min_correlation` 0.65, the valley of the
  bimodal 5-8 bit distribution. A missed duplicate leaks test images into training
  (expensive); a wrong join only puts two images in the same split (cheap, while no giant group
  forms). The automatic join within 4 bits is removed; the guard is 10 %.
- Overlays inside the yellow box: not stated in the report (add it if checked).

### Run 3

- Run by / date:
- Commit actually used (it must contain `min_correlation: 0.65`):
- Hardware / library versions (`pip freeze | grep -E "datasets|pillow|numpy"`):
- Wall time:
- The line `groups ..., largest ...; held-out images with a confirmed duplicate in train: ...;
  pairs within 4 bits not joined: ...`:
- Hashes printed at the end (`split_sha256`, manifest hashes, the bundle's file count):
- The final contact sheet: attached (yes / no); same-scene pairs seen in band 1, band 2,
  band 3 (number, or none):
- Name of the private shared-data dataset, and who it is shared with:
- Anything unusual (errors, retries, changed settings):
