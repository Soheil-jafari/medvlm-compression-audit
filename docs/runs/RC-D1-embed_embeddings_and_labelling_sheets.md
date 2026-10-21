# Run card RC-D1-embed: embeddings, candidate pairs and the labelling sheets (CPU)

## 1. What and why

- Stage: S03 (D02 amendment 3 of 2026-10-19, before any model result)
- Experiments: none (data only)
- Why: RC-D1 run 3 was not accepted. Its final contact sheet showed same-scene pairs across
  splits that both earlier measures missed:
  - 3 of the 24 highest crop scores were the same scene at 14-32 dHash bits;
  - about 5 of the 24 closest by dHash;
  - about 9 of 12 pairs within 4 bits that the crop measure left apart.

  The joining measure is now the cosine of image embeddings. Its rule is set from human
  labels, not from score distributions.
- What the run does (`python scripts/prepare_data.py embed`):
  1. Reads every image once (dHash and the crop signature).
  2. Reproduces RC-D1 run 3 on the same data with the same code and seed, and prints whether
     its `split_sha256` starts with run 3's `4a595fdf`. The pairs of run 3's final contact
     sheet become candidates.
  3. Loads the embedding model from the Hugging Face hub and prints its id, the revision (the
     commit hash the hub resolved) and the licence of its hub record: DINOv2-small
     (`facebook/dinov2-small`), now pinned at revision
     `ed25f3a31f01632728cabb09d1542f84ab7b0056`, licence `apache-2.0` (the hub record, read by
     the run of 2026-10-19).
  4. Checks the model on synthetic copies: a shifted, a blurred, a rescaled and a brightened
     copy must each be nearer to the original than three different scenes, in both views. If
     not, it stops (exit code 8).
  5. Embeds every image in two views (`data/embeddings.py`):
     - `full`: the whole image, black border and overlays included;
     - `centre`: the central 60 % of width and height, which leaves out the corner text, the
       green box and the border.

     Each view is resized to 224 x 224 and normalised with the model's own mean and standard
     deviation. The feature is the pooled output, L2-normalised; similarity is the cosine.
  6. Builds the candidate pairs (`data/candidates.py`):
     - the 20 nearest neighbours of every image by cosine, in each view;
     - every pair within 12 dHash bits;
     - the 5 highest crop correlations of every image, at any dHash distance;
     - run 3's sheet pairs.

     Each pair carries all four scores: the dHash distance, the crop correlation and both
     cosines.
  7. Draws about 300 pairs to label, seed 20261012, stratified (`data/labelling.py`):
     - all run-3 sheet pairs;
     - 20 + 20 pairs within 0-2 and 3-4 dHash bits;
     - 25 + 25 + 20 pairs at crop correlation 0.50-0.65, 0.65-0.80 and 0.80 or more;
     - 12 pairs in each cosine bin of 0.05 from below 0.50 to 1.00.

     The pairs are numbered in a random order and drawn side by side on sheets **without any
     score**.
- Who: **Soheil**.
- Status: `done` (Kaggle T4, 2026-10-19, commit cb5de07; report in section 6)

## 2. Requirements

- Platform: Kaggle notebook, **CPU**, Internet **on** (the model downloads from the hub).
- Attach (Add Input), private datasets only:
  - the shared data of RC-D1 run 3, if it was saved, or run 2's: the same 6,449 images, with
    Kvasir-SEG inside. Then nothing is downloaded but the model;
  - or, if neither was saved, the Kvasir-SEG dataset: the run then exports the images first
    (about 9 minutes in run 1).
- Runtime (**estimates**; the model has not run here):
  - reading the images and the dHash and crop signatures: a few minutes (not measured);
  - reproducing run 3: a few minutes (as run 3);
  - **embedding: about 20 to 50 minutes on Kaggle's CPU** for 6,449 images x 2 views. This
    comes from the model's size (about 12 billion operations per view for a ViT-S/14 at 224,
    an estimate) at an assumed 50 to 150 billion per second; not measured. The step prints its
    progress every tenth of the images with the time so far, so the real speed shows within
    minutes. On a T4 GPU the same step would take a minute or two (also an estimate); the code
    uses a GPU when the notebook has one;
  - the candidates and the all-pairs crop pass: about 3 to 4 minutes (measured here: 199 s for
    6,449 synthetic signatures on 4 CPU cores);
  - the sheets: under a minute (not measured).

  In all, about 30 to 70 minutes with the shared data attached; add about 9 minutes for the
  export without it.
- The output folder (`/kaggle/working/rc_d1_embed`) is **private**: it names and shows images.
  It holds:
  - `embeddings.npz` (+ `.json`: model, revision, licence, settings);
  - `candidates.npz` and `crop_pairs.npz`;
  - `label_pairs.csv`: the pairs with their scores, for the scoring step only, **not for the
    raters**;
  - `label_images.csv`: which two images each pair is, without scores, for the raters;
  - `labels_template.csv` and `label_sheets/`;
  - `embed_meta.json` and `bundle_manifest.json`.

  Upload it as ONE private Kaggle dataset (for example `rc-d1-embed`), shared with the two
  raters and the team only. Never commit any of it.

## 3. Commands

**Unattended (Save & Run All):** `notebooks/run_card.ipynb`, cell 1:

```python
CARD = "RC-D1-embed"
SEED = None
DATA = "/kaggle/input/<run-3-or-run-2-shared-data>"   # None if neither was saved (attach Kvasir-SEG)
FROM = []
COMMIT = None
EXTRA = []
```

Or by hand, in one CPU notebook cell:

```bash
%%bash
cd /kaggle/working && git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
python scripts/prepare_data.py embed --data-root /kaggle/input/<shared-data> \
    --out-dir /kaggle/working/rc_d1_embed --split-seed 20261012 --close-bits 4 \
    --mask-threshold 4 2>&1 | tee /kaggle/working/rc_d1_embed.txt
```

(If Kaggle's torchao is older than 0.16.0, run `pip uninstall -y torchao` first, as in the other
cards; the notebook does this itself.)

## 4. What to paste back and commit

Paste back these printed lines:
- `read ... images`;
- `run 3 reproduced: split_sha256 ... (matches / DIFFERS FROM ...)`;
- `embedding model: ... at revision ...; licence (hub record): ...` and the `RECORD:` line;
- the two `self-check` lines;
- `embedded ... images, two views, in ... s`;
- `candidate pairs: ...`;
- `... pairs to label`.

The revision, the model id and the licence are pinned in `configs/data/kvasir_vqa_x1.yaml`
(`near_duplicates.embedding`) and in D02 (additions of 2026-10-19). A run with another model,
revision or licence is refused by the scoring, the top-up and run 4.

Commit `results/data/near_duplicate_embed.csv` (aggregate counts only). Upload the output
folder as ONE private dataset (section 2), then hand the sheets to the raters (RC-D1-label).

## 5. Pass criteria

| Check | Where | Pass if |
|---|---|---|
| Run 3 reproduced | the `run 3 reproduced` line; `embed:run3:split_hash_matches` | `matches` (else report: the run-3 pairs may differ from the sheet that was looked at) |
| The model separates copies | the `self-check` lines | the run did not stop with exit code 8 |
| Every source gave pairs | `embed:candidate_pairs:<source>` | all five above 0 (`run3_band` only with the shared data) |
| A sample to label | `embed:labelling_pairs` | about 300 |

## 6. Run report (filled by the person who ran it)

Run of 2026-10-19, as reported by Soheil from the printout:
- Run by / date: reported by Soheil, 2026-10-19
- Commit actually used: cb5de07
- Input (which shared data, or the export): not reported here; run 3 reproduced
  (`split_sha256` 4a595fdf...), so its sheet pairs are among the candidates
- Model id, revision and licence as printed: `facebook/dinov2-small`, revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, licence `apache-2.0` (the hub record)
- Self-check lines (cosine to the original):
  - synthetic copies: 0.937-0.995 (full), 0.988-0.999 (centre);
  - synthetic different images: 0.843-0.937 (full), 0.879-0.942 (centre).

  The cosines are compressed into the high range; the decision boundary is likely high
  (around 0.95). The sample has only 24 of its 298 pairs there (bins 0.90-0.95 and
  0.95-1.00), hence the top-up of D02 amendment 3 (card RC-D1-label, section 3b).
- Embedding time (the printed seconds) and device: reported as 143 s for 6,449 images, two
  views, on a T4; the committed counts file says 156 s (`embed:seconds` 156.28), so the file
  may come from another run of the card or the printout was misread. The other counts agree.
  The CPU estimate of section 2 (20 to 50 minutes) was not tested.
- Candidate pairs per source; pairs to label: 366,028 candidate pairs; per source (from the
  committed counts file): `knn_full` 97,040, `knn_centre` 98,578, `dhash` 211,329, `crop_knn`
  26,290, `run3_band` 56 (all five above 0); 298 pairs to label
- Name of the private dataset with the output, and who it is shared with: not reported here
- `results/data/near_duplicate_embed.csv`: committed on 2026-10-20 (commit 69799ff)
- Anything unusual: none reported
