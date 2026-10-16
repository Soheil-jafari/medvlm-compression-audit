# Run card RC-D1: data preparation (Kvasir-VQA-x1 + Kvasir-SEG)

## 1. What and why

- Stage: S03
- Experiments: none (data only)
- Purpose: download the data to your own Kaggle session, make the validation/test
  halves of the official test split, remove near-duplicates, link Kvasir-SEG masks to
  VQA images, report **aggregate counts only** so the finding list, prevalence groups,
  label rules and size thresholds can be pre-registered, and write **one shared-data folder**
  (the images, the rows, Kvasir-SEG and the private RC-D1 files, with a SHA-256 manifest) that
  every later card attaches as a private Kaggle dataset.
- Who: **Soheil** (decision of 2026-10-17).
- Status: `ready`

## 2. Requirements

- Platform: Kaggle notebook, **CPU is enough**, Internet **on** (Settings -> Internet).
- Repository commit to run from: latest `research-loop` (write the commit in section 5).
- Dataset access and licences:
  - **Kvasir-VQA and Kvasir-VQA-x1** download from the Hugging Face hub. Licence: CC BY-NC 4.0
    (non-commercial use, with attribution).
  - **Kvasir-SEG:** download `kvasir-seg.zip` yourself from
    <https://datasets.simula.no/kvasir-seg/>. Licence: research and educational use only. Attach
    it to the notebook as a **private** Kaggle dataset.
  - The shared-data folder holds copies of these images. It stays a **private** dataset,
    shared with the team's three Kaggle accounts only (Soheil, Amir, Farzan), never public. The
    manifest records both licences and this rule.
- **Two passes.**
  1. Pass 1 (now) writes the shared data and the first counts. The label values of the
     abnormality question (`configs/data/kvasir_vqa_x1.yaml`, `values: {}`) are still empty,
     so pass 1 writes the answer counts (`answer_count:*`) but no `positives:*` or
     `prevalence:*`.
  2. The agent then proposes the label values from those answer counts, and the team confirms.
  3. Pass 2 (CPU, minutes) recounts from the attached shared data with the same settings. It
     stops if the splits do not hash as in pass 1.
- Expected runtime of pass 1: roughly the download time (about 1.5 GB of images, per the
  Kvasir-VQA card), plus a few minutes of hashing for the splits and again for the manifest.
  Not measured.

Settings proposed by the agent (pre-registration items; change them here **before**
running if the team prefers others, and never after seeing any result):

| Setting | Value | Why |
|---|---|---|
| `--split-seed` | `20261012` | any fixed integer; recorded in the pre-registration |
| `--dedup-threshold` | `4` | max differing bits (of 64) for a near-duplicate |
| `--mask-threshold` | `4` | max differing bits to link a Kvasir-SEG image to a VQA image |

## 3. Commands

**Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-D1"`,
`SEED = None`, `DATA = None`, `EXTRA = ["--kvasir-seg", "/kaggle/input/<your-kvasir-seg>"]`. It
runs the five steps below and prints section 5. Or by hand:

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 5
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps && pip install -q datasets

export SHARED=/kaggle/working/medcompress_data                # data/ + private/, outside the repo
python scripts/prepare_data.py export --data-root $SHARED/data

# Kvasir-SEG from your private Kaggle dataset (the folder that holds images/ and masks/):
python -c "import sys; sys.path.insert(0, 'src'); from pathlib import Path; \
from medcompress.utils.kaggle import copy_kvasir_seg; \
copy_kvasir_seg(Path('/kaggle/input/<your-kvasir-seg>'), Path('$SHARED/data'))"

python scripts/prepare_data.py prepare --data-root $SHARED/data --private-dir $SHARED/private \
    --split-seed 20261012 --dedup-threshold 4 --mask-threshold 4
python scripts/prepare_data.py bundle --bundle-dir $SHARED    # SHA-256 manifest of every file
python scripts/prepare_data.py verify --data-root $SHARED     # must print OK
python -m pytest -q tests/test_results.py                     # must pass before you commit
```

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

- Run by / date:
- Commit actually used:
- Hardware / library versions (`pip freeze | grep -E "datasets|pillow|numpy"`):
- Wall time:
- Hashes printed at the end (`split_sha256`, manifest hashes, the bundle's file count):
- Name of the private shared-data dataset, and who it is shared with:
- Anything unusual (errors, retries, changed settings):
