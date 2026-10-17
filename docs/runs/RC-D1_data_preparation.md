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
  - link Kvasir-SEG masks to the VQA images;
  - report **aggregate counts only**, so the finding list, prevalence groups, label rules and
    size thresholds can be pre-registered;
  - write **one shared-data folder** (the images, the rows, Kvasir-SEG and the private RC-D1
    files, with a SHA-256 manifest) that every later card attaches as a private Kaggle dataset.
- Who: **Soheil** (decision of 2026-10-17).
- Status: `ready` for **run 2** (run 1 used the old split; nothing from it is kept, section 5).

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
- **Two passes.**
  1. Run 2 writes the shared data and the counts.
  2. The label values of the abnormality question (`configs/data/kvasir_vqa_x1.yaml`,
     `values: {}`) are still empty, so run 2 writes the answer values per split
     (`answer_count:<rule>:<value>:<split>`) but no `positives:*` or `prevalence:*`. The agent
     proposes the label values from these counts; the team confirms.
  3. Pass 2 (CPU, minutes) then recounts from the attached shared data with the same settings.
     It stops if the splits do not hash as in run 2. The splits themselves do not depend on the
     label values: they are stratified by the raw answer values.
- Expected runtime of run 2 (**estimate**):
  - the export took about 9 minutes in run 1 (6,500 images, 422 MB);
  - then a few minutes to fingerprint the 6,449 images and the 1,000 Kvasir-SEG images, group
    them and split them (not measured);
  - and about a minute to hash the folder for the manifest (not measured).

  About 15 to 25 minutes in all.

Settings (pre-registration items; decided before any result, D02 amendment of 2026-10-18):

| Setting | Value | Why |
|---|---|---|
| `--split-seed` | `20261012` | any fixed integer; recorded in the pre-registration |
| `--dedup-threshold` | `4` | max differing bits (of 64): images this close are one group, so they never cross splits |
| `--mask-threshold` | `4` | max differing bits to link a Kvasir-SEG image to a VQA image |
| split ratios | 70 / 15 / 15 % of the images | `split_ratios` in `configs/data/kvasir_vqa_x1.yaml` |

How the groups are split (`data/splits.py: assign_groups_stratified`, CPU-tested):
1. The groups are put in a seeded order.
2. Labels (the answer values of the finding question, and "has a Kvasir-SEG mask") are taken
   from the rarest to the most common. A label carried by at least three groups first gets one
   group in each split that has none.
3. Each remaining group of a label goes to the split furthest below its share of that label.
4. Groups without a label fill the image targets.

The run then checks that no validation or test image is within 4 bits of a training image
(zero by construction) and counts the distances from 5 to 17 bits (a data-card number, not a
filter).

## 3. Commands

**Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-D1"`,
`SEED = None`, `DATA = None`, `FROM = []`, `EXTRA = []`. The runner finds the attached Kvasir-SEG
folder itself, runs the five steps below and prints section 5. Or by hand, in one CPU notebook:

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 5
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
    --split-seed 20261012 --dedup-threshold 4 --mask-threshold 4
python scripts/prepare_data.py bundle --bundle-dir $SHARED    # SHA-256 manifest of every file
python scripts/prepare_data.py verify --data-root $SHARED     # must print OK
python -m pytest -q tests/test_results.py                     # must pass before you commit
```

What `results/data/summary.csv` holds after run 2 (aggregate counts only; experiment `DATA`):

| Rows | What |
|---|---|
| `images:all`, `groups:total`, `groups:multi_image`, `groups:largest`, `images_in_multi_image_groups`, `groups:<split>` | the near-duplicate groups |
| `qa_rows:<split>`, `images:<split>`, `image_fraction:<split>`, `masked_images:<split>` | the size of each split |
| `val_test_near_duplicates_of_train` | must be 0 |
| `min_distance_to_train:<val/test>:5` ... `:17`, `:18_or_more` | the tail just above the grouping threshold |
| `answer_count:<rule>:<value>` and `answer_count:<rule>:<value>:<split>` | the answer values of the finding question, overall and per split (images) |
| `positives:<finding>:<split>`, `prevalence:<finding>` | after the label values are filled (pass 2) |
| `answer_words:p50/p90/p99/max` | on the training split (sets `max_new_tokens`, D04) |
| `masks:linked/ambiguous/unmatched`, `lesion_area_threshold:q1..q3` | Kvasir-SEG links; size strata from the training split |
| `official:images:*`, `official:test_rows:*`, `official:test_only_min_distance_to_train:*` | the official split's overlap (data card, appendix item) |

**Pass 2** (after the agent has committed the label values and the team has confirmed them), a
new CPU notebook with the shared data attached:

```bash
python scripts/run_card.py --card RC-D1-counts --data-root /kaggle/input/<shared-data>
# by hand: python scripts/prepare_data.py counts --data-root /kaggle/input/<shared-data> \
#     --split-seed 20261012 --dedup-threshold 4 --mask-threshold 4
```

## 4. Make the shared data a private Kaggle dataset, and what to commit back

Upload, step by step. Kaggle's button names may differ slightly; the dataset must stay
**Private** throughout.

1. Run pass 1 with **Save Version -> Save & Run All** (the unattended notebook), so that
   `/kaggle/working/medcompress_data/` is saved in the version's output.
2. Open the finished version, then its **Output** tab, and create a dataset from the
   `medcompress_data` folder (**New Dataset**). Name it, for example, `medcompress-data`, and set
   **Private**. Do not use a public dataset, and do not upload it anywhere else.
3. On the dataset's page, open its settings (**Sharing** / **Collaborators**). Add Amir's and
   Farzan's Kaggle usernames with view access. Keep it Private.
4. Check it from a new notebook: **Add Input** -> the dataset, then run
   `python scripts/prepare_data.py verify --data-root /kaggle/input/<dataset>`. It must print
   `OK`; the runner of every later card does the same check before it starts.
5. Write the dataset's name and the `split_sha256` that `bundle` prints into section 5.

Not known here: the size limit of a notebook output and of a dataset (TODO: check on Kaggle;
the folder is about the size of the images).

What to commit back:
- `results/data/summary.csv` (aggregate counts only; no image names or per-image rows). Pass 2
  replaces it.
- This card with section 5 filled.

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

### Run 2

- Run by / date:
- Commit actually used:
- Hardware / library versions (`pip freeze | grep -E "datasets|pillow|numpy"`):
- Wall time:
- Hashes printed at the end (`split_sha256`, manifest hashes, the bundle's file count):
- Name of the private shared-data dataset, and who it is shared with:
- Anything unusual (errors, retries, changed settings):
