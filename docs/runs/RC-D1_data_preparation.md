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
  - group near-duplicates in **two stages** (D02 amendment 2 of 2026-10-18; run 2 showed that
    the 4-bit dHash rule misses adjacent frames, section 5): dHash candidates within 12 bits,
    confirmed by a correlation on the centre of the image, where no overlay is drawn;
  - link Kvasir-SEG masks to the VQA images;
  - report **aggregate counts only**, so the finding list, prevalence groups, label rules and
    size thresholds can be pre-registered;
  - write **one shared-data folder** (the images, the rows, Kvasir-SEG and the private RC-D1
    files, with a SHA-256 manifest) that every later card attaches as a private Kaggle dataset.
- Who: **Soheil** (decision of 2026-10-17).
- Status: `ready` for the **calibration step** (section 3a), then **run 3** (section 3b). Run 3
  stops with exit code 4 until the confirmation threshold chosen from the calibration step is
  committed. Runs 1 and 2 are recorded in section 5; nothing from them is used.

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
  1. **Calibration** (CPU, minutes; section 3a). It scores every candidate pair with the crop
     measure and prints the distribution, what each threshold would give, and a provisional
     proposal; it draws a contact sheet of the pairs just above and below the threshold and of
     random controls. Paste the printout and the sheet back. The threshold is then chosen
     (Soheil, with the agent), written into `configs/data/kvasir_vqa_x1.yaml`
     (`near_duplicates.min_correlation`) with its reason (D02 amendment 2), and committed.
  2. **Run 3** (section 3b): as run 2, with the two-stage grouping. It writes the shared data,
     the counts and a final contact sheet of the 24 most similar pairs across splits. Attach
     that sheet to the report.
  3. **Pass 2** (CPU, minutes): the label values of the abnormality question
     (`configs/data/kvasir_vqa_x1.yaml`, `values: {}`) are still empty, so run 3 writes the
     answer values per split (`answer_count:<rule>:<value>:<split>`) but no `positives:*` or
     `prevalence:*`. The agent proposes the label values from these counts; the team confirms;
     pass 2 then recounts from the attached shared data with the same settings. It stops if
     the splits do not hash as in run 3. The splits themselves do not depend on the label
     values: they are stratified by the raw answer values.
- Expected runtime (**estimates**):
  - calibration with run 2's shared data attached: about 5 to 10 minutes. Verifying the
    shared data (about a minute, not measured), reading the 6,449 images (a few minutes, not
    measured), then the scores: on a 4-core CPU here, 100,000 pairs took about 7 s and the
    training-against-held-out matrix about 12 s (synthetic signatures). The number of
    candidate pairs on the real images is not known yet. Without run 2's shared data, the
    export comes first (about 9 minutes in run 1);
  - run 3: about 15 to 25 minutes, as run 2 (the export, about 9 minutes in run 1; reading,
    grouping and splitting; hashing the folder for the manifest).

Settings (pre-registration items; decided before any model result, D02 amendments of
2026-10-18):

| Setting | Value | Why |
|---|---|---|
| `--split-seed` | `20261012` | any fixed integer; recorded in the pre-registration |
| `--dedup-threshold` | `4` | pairs within 4 of 64 dHash bits are always one group (as in run 2) |
| `candidate_bits` | `12` | pairs within 12 bits are candidates for the crop check |
| `crop_fraction` | `0.60` | the central 60 % of width and of height: 20 % is cut from each side, where the text (top left), the green box (bottom left) and the black border are drawn |
| `crop_side` | `32` | the crop, grey-scale, area-averaged to 32 x 32 pixels |
| `crop_highpass_sigma` | `4.0` | a Gaussian blur of 4 px (at 32) is subtracted: the measure follows the mucosa's detail, not the lighting or the lumen position, which made unrelated images look alike at 9 x 8 |
| `crop_max_shift` | `3` | the crop of one image is compared with the other shifted by up to 3 px (5.6 % of the image side) each way: adjacent frames move a little |
| measure | highest Pearson correlation over the shifts, both ways round | insensitive to brightness and contrast; 1.0 for identical crops |
| `min_correlation` | **open: from the calibration step** | candidate pairs at or above it are joined |
| `max_group_fraction` | `0.05` | chaining guard: run 3 STOPS (exit code 7) if one group holds more than 5 % of the images (322 of 6,449); a group needs at least 3 images to chain |
| `--mask-threshold` | `4` | max differing bits to link a Kvasir-SEG image to a VQA image |
| split ratios | 70 / 15 / 15 % of the images | `split_ratios` in `configs/data/kvasir_vqa_x1.yaml` |

The crop settings and the threshold are in `configs/data/kvasir_vqa_x1.yaml`
(`near_duplicates`); the seed and the two bit thresholds are on the command line.

How the groups are made (`data/dedup.py: two_stage_groups`, CPU-tested):
1. Candidates: every pair of images within 12 bits of the 64-bit dHash.
2. Joined: every pair within 4 bits, and every candidate whose crop correlation is at least
   `min_correlation`.
3. Union-find over the joined pairs: if a~b and b~c, a, b and c are one group.
4. The chaining guard: STOP and report if the largest group holds more than 5 % of the images.

How the groups are split (`data/splits.py: assign_groups_stratified`, CPU-tested; unchanged):
1. The groups are put in a seeded order.
2. Labels (the answer values of the finding question, and "has a Kvasir-SEG mask") are taken
   from the rarest to the most common. A label carried by at least three groups first gets one
   group in each split that has none.
3. Each remaining group of a label goes to the split furthest below its share of that label.
4. Groups without a label fill the image targets.

Checks after the split (in `summary.csv`, section 3b):
- no validation or test image has a joined pair (within 4 bits, or confirmed by the crop) with
  a training image (zero by construction; the run stops otherwise);
- the dHash tail (5 to 17 bits) and the crop-correlation tail to the training split;
- how many validation and test images have a training image **beyond 12 bits** whose crop
  correlation reaches the threshold: pairs the candidate stage cannot see. A number for the
  data card, not a filter.

Limitation: HyperKvasir has no patient IDs. Images of the same patient taken at different
moments can still fall into different splits (data card).

## 3. Commands

### 3a. The calibration step (before run 3)

Attach run 2's shared-data dataset if you saved it (it holds the same 6,449 images). If not,
attach nothing: the step exports the images first. Internet on.

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
    --dedup-threshold 4 --sheet-dir /kaggle/working/rc_d1_sheets 2>&1 | tee rc_d1_calibration.txt
```

What it prints (aggregate counts only):
- the settings;
- the candidate pairs by dHash distance, 0 to 12 bits;
- the crop correlation in bins of 0.05, in four columns: pairs at 0-4 bits (always joined),
  5-8 bits, 9-12 bits, and 2,000 random pairs beyond 12 bits (the controls, seeded);
- the controls' median, 99th percentile and maximum;
- for each threshold from 0.30 to 0.95: the confirmed pairs at 5-12 bits, the groups, the
  multi-image groups, the largest group and its share, the guard (`ok` or `STOP`), and how many
  controls reach the threshold;
- the provisional proposal: the smallest threshold that passes the guard and that at most 2 of
  the 2,000 controls reach (a control can itself be a moved frame the dHash missed). It is a
  starting point to check by eye, not a decision.

The contact sheet `/kaggle/working/rc_d1_sheets/rc_d1_calibration_pairs.png` (private: it
shows dataset images; **never commit it**) has four bands, each pair side by side with the crop
drawn as a yellow box:
1. just below the threshold: the 8 highest-scoring candidate pairs at 5-12 bits that are NOT
   joined;
2. just above it: the 8 lowest-scoring candidate pairs at 5-12 bits that are JOINED;
3. 4 of the random controls;
4. the 4 pairs within 4 bits (always joined) with the lowest crop correlation, to see whether
   the 4-bit rule itself joins different scenes.

To redraw the sheet at another value, run the same command with `--threshold 0.80` (any
value; it is added to the table).

**How the threshold is chosen** (record the value and one sentence of reason in section 5):
1. The guard must say `ok` at the chosen value.
2. Every pair in the "just above" band must be the same scene: identical, re-encoded, or an
   adjacent frame (the same mucosa, polyp or instrument, slightly moved). If one is a
   different scene, raise the threshold and redraw.
3. The "just below" band should be different scenes. If it holds clear same-scene pairs, lower
   the threshold and redraw.
4. If a band stays mixed, take the lower value that passes the guard (agent proposal): a wrong
   join moves a few images together, a missed duplicate puts a copy of a training image into
   validation or test.
5. Check on the sheet that the yellow box leaves out the text, the green box and the border.
   If an overlay reaches into the box, report it (the crop fraction then needs a change, made
   before run 3).

**Paste back:** the whole printout (`rc_d1_calibration.txt`) and the contact sheet (download it
from the notebook's Output). Commit `results/data/near_duplicate_calibration.csv` (aggregate
counts only). The agent then writes the chosen value into `configs/data/kvasir_vqa_x1.yaml`
(`near_duplicates.min_correlation`) and D02 amendment 2, and commits it. Run 3 runs from that
commit.

### 3b. Run 3

Attach Kvasir-SEG only (not run 2's shared data: the copy step looks for a folder with
`images/` and `masks/` under `/kaggle/input`). Internet on.

**Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-D1"`,
`SEED = None`, `DATA = None`, `FROM = []`, `EXTRA = []`. The runner finds the attached Kvasir-SEG
folder itself, runs the five steps below and prints section 5. Or by hand, in one CPU notebook:

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 5
grep "min_correlation:" configs/data/kvasir_vqa_x1.yaml       # must be a number, not TODO
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
    --split-seed 20261012 --dedup-threshold 4 --mask-threshold 4 \
    --sheet-dir /kaggle/working/rc_d1_sheets                  # exit 7 = the chaining guard: STOP
python scripts/prepare_data.py bundle --bundle-dir $SHARED    # SHA-256 manifest of every file
python scripts/prepare_data.py verify --data-root $SHARED     # must print OK
python -m pytest -q tests/test_results.py                     # must pass before you commit
```

If `prepare` stops with exit code 7 ("STOP: the largest near-duplicate group holds ..."), do
not change any setting: paste the message back. If it stops with exit code 4, the threshold is
not committed yet (section 3a).

**Paste back:** the line `groups ..., largest ...; held-out images with a confirmed duplicate in
train: 0`, the hashes printed at the end, and attach the final contact sheet
`/kaggle/working/rc_d1_sheets/rc_d1_final_cross_split_pairs.png` to the report (private;
**never commit it**). It shows the 24 most similar pairs across splits by the crop
correlation, with their dHash distance and splits: they should all be different scenes.

What `results/data/summary.csv` holds after run 3 (aggregate counts only; experiment `DATA`):

| Rows | What |
|---|---|
| `images:all`, `groups:total`, `groups:multi_image`, `groups:largest`, `groups:largest_fraction`, `images_in_multi_image_groups`, `groups:size:<1, 2, 3, 4, 5, 6_to_10, 11_to_50, 51_to_100, 101_or_more>`, `groups:<split>` | the near-duplicate groups |
| `grouping_threshold_bits`, `near_duplicates:<setting>` | the settings used (the crop, the threshold, the guard) |
| `near_duplicates:candidate_pairs`, `near_duplicates:joined_pairs:always`, `near_duplicates:joined_pairs:confirmed`, `near_duplicates:candidate_pairs_not_joined` | the two stages |
| `qa_rows:<split>`, `images:<split>`, `image_fraction:<split>`, `masked_images:<split>` | the size of each split |
| `val_test_near_duplicates_of_train`, `val_test_confirmed_duplicates_of_train`, `confirmed_duplicates_of_train:<val/test>` | must be 0 |
| `min_distance_to_train:<val/test>:5` ... `:17`, `:18_or_more` | the dHash tail above the always-joined distance |
| `max_score_to_train:<val/test>:<bin>` | each held-out image's highest crop correlation with a training image, in bins of 0.05 |
| `similar_to_train_beyond_candidates:<val/test>` | held-out images with a training image beyond 12 bits at or above the threshold (not filtered) |
| `cross_split_top:max_score`, `:min_score`, `:within_candidate_bits` | the 24 pairs of the final contact sheet |
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
#     --split-seed 20261012 --dedup-threshold 4 --mask-threshold 4
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

### Calibration step

- Run by / date:
- Commit actually used:
- Input (run 2's shared data, or export):
- Wall time:
- The printout (`rc_d1_calibration.txt`, all of it):
- The contact sheet: attached (yes / no); threshold it was drawn at:
- Chosen `min_correlation` and the reason (one sentence, from the sheet):
- Overlays inside the yellow box (yes / no):

### Run 3

- Run by / date:
- Commit actually used (it must contain the chosen `min_correlation`):
- Hardware / library versions (`pip freeze | grep -E "datasets|pillow|numpy"`):
- Wall time:
- The line `groups ..., largest ...; held-out images with a confirmed duplicate in train: ...`:
- Hashes printed at the end (`split_sha256`, manifest hashes, the bundle's file count):
- The final contact sheet: attached (yes / no); are all 24 pairs different scenes?
- Name of the private shared-data dataset, and who it is shared with:
- Anything unusual (errors, retries, changed settings):
