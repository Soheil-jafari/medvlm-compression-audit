# Run card RC-D1: data preparation (Kvasir-VQA-x1 + Kvasir-SEG)

## 1. What and why

- Stage: S03
- Experiments: none (data only)
- Purpose: download the data to your own Kaggle session, make the validation/test
  halves of the official test split, remove near-duplicates, link Kvasir-SEG masks to
  VQA images, and report **aggregate counts only** so the finding list, prevalence
  groups, label rules and size thresholds can be pre-registered.
- Status: `ready`

## 2. Requirements

- Platform: Kaggle notebook, **CPU is enough**, Internet **on** (Settings -> Internet).
- Repository commit to run from: latest `research-loop` (write the commit in section 5).
- Dataset access: Kvasir-VQA and Kvasir-VQA-x1 download from the Hugging Face hub
  (CC BY-NC 4.0). Kvasir-SEG: download `kvasir-seg.zip` yourself from
  <https://datasets.simula.no/kvasir-seg/> (research and educational use only) and
  attach it to the notebook as a **private** Kaggle dataset.
- Expected runtime: roughly the download time (about 1.5 GB of images, per the
  Kvasir-VQA card) plus a few minutes of hashing.

Settings proposed by the agent (pre-registration items; change them here **before**
running if the team prefers others, and never after seeing any result):

| Setting | Value | Why |
|---|---|---|
| `--split-seed` | `20261012` | any fixed integer; recorded in the pre-registration |
| `--dedup-threshold` | `4` | max differing bits (of 64) for a near-duplicate |
| `--mask-threshold` | `4` | max differing bits to link a Kvasir-SEG image to a VQA image |

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 5
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps && pip install -q datasets

export DATA=/kaggle/working/data
python scripts/prepare_data.py export --data-root $DATA

# Kvasir-SEG from your private Kaggle dataset (adjust the input path):
mkdir -p $DATA/kvasir_seg
cp -r /kaggle/input/<your-kvasir-seg>/Kvasir-SEG/images $DATA/kvasir_seg/images
cp -r /kaggle/input/<your-kvasir-seg>/Kvasir-SEG/masks  $DATA/kvasir_seg/masks

python scripts/prepare_data.py prepare --data-root $DATA \
    --private-dir /kaggle/working/private \
    --split-seed 20261012 --dedup-threshold 4 --mask-threshold 4
python -m pytest -q tests/test_results.py    # must pass before you commit
```

## 4. What to commit back

- `results/data/summary.csv` (aggregate counts only; no image names or per-image rows).
- This card with section 5 filled.

Keep `/kaggle/working/private` (splits, exclusions, mask links, manifests) as a
**private** Kaggle dataset or notebook output: every later run needs it. Never make
it public and never commit it.

## 5. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware / library versions (`pip freeze | grep -E "datasets|pillow|numpy"`):
- Wall time:
- Hashes printed at the end (`split_sha256`, manifest hashes):
- Anything unusual (errors, retries, changed settings):
