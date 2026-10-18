# Data card: Kvasir-VQA-x1 (with Kvasir-VQA images and Kvasir-SEG masks)

> No images, no patient identifiers, no per-image rows. Counts marked **[RC-D1]** come from
> `results/data/summary.csv` once RC-D1 run 3 is committed.

| Field | Value |
|---|---|
| Name and version | Kvasir-VQA-x1, revision `61e41148c3214bc5140ad0ab4c28520a512e2a73`; images from Kvasir-VQA (SimulaMet-HOST) |
| Owner on the team | Soheil (RC-D1) |
| Source | <https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1> |
| Reference | arXiv 2506.09958 (to add to `paper/references.bib`) |
| Licence and terms | CC BY-NC 4.0 (Kvasir-VQA and Kvasir-VQA-x1); Kvasir-SEG: research and educational use only. Our shared copy is a private Kaggle dataset, shared within the team only |
| Domain and modality | GI endoscopy, RGB images with questions and answers |
| Role in this project | main QA (training, validation, test), rare findings |
| Config | `configs/data/kvasir_vqa_x1.yaml` |

## The official split overlaps by image

The official Kvasir-VQA-x1 split is by **question**, not by image. The dataset paper (arXiv
2506.09958, section 2.3) says so: "Each image-question pair is assigned to either the training
or test set". RC-D1 run 1 (2026-10-16/17) counted:

- **Images:** official train 6,212; official test 4,058; in both 3,821; test only 237; all
  6,449.
- **Test rows:** 15,718 of the 15,955 official test rows are on images that are also training
  images. 237 are on test-only images, one row each.
- **The test-only images are mostly near-copies of training images.** Smallest dHash distance
  (bits of 64) from each test-only image to any training image, with the number of images:

  | Distance | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 17 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Images | 43 | 43 | 43 | 21 | 18 | 14 | 14 | 10 | 5 | 7 | 5 | 1 | 6 | 1 | 4 | 1 | 1 |

- **Consequence:** with the old split (the official test split halved), deduplication left 39
  validation and 30 test images, none with a Kvasir-SEG mask. The published SimulaMet adapter
  was trained on the official train split, so it has seen almost every image of any
  image-disjoint test split. Hence the image-disjoint re-split (D02 amendment) and our own
  teacher adapter (D01 amendment), both 2026-10-18.

RC-D1 run 3 writes these numbers again from the code (`official:*` rows in `summary.csv`).
The appendix item reports them next to the published adapter's test results.

## Our splits (RC-D1 run 3)

- **Why two stages (D02 amendment 2).** RC-D1 run 2 grouped images within 4 dHash bits only.
  A look at the pairs showed that this misses real duplicates: at 5-6 bits, 3 of 8 pairs were
  the same scene, and at 7-8 bits, 2 to 3 of 8. At 9-12 bits, 0 of 4 were; those pairs looked
  alike only at 9 x 8 pixels (colour, lumen position, and the text and green box drawn on the
  image). 1,038 of run 2's 1,931 validation and test images had a training image within 8
  bits. A larger dHash threshold alone would chain unrelated images.
- **Method:**
  1. Every image of both official splits is fingerprinted (dHash, 64 bits).
  2. Candidates: pairs within 12 bits.
  3. Confirmation on the centre of the image: the central 60 % of width and height, grey-scale,
     32 x 32 pixels, with a Gaussian blur (4 px) subtracted; the measure is the highest Pearson
     correlation over shifts of up to 3 px (5.6 % of the side). Pairs at or above the
     threshold **[RC-D1 calibration]** `min_correlation` are joined. The threshold is chosen
     on the data before the split, from the distribution over all candidate pairs and a
     contact sheet checked by eye (RC-D1 section 3a).
  4. Pairs within 4 bits are always joined; union-find over all joined pairs gives the groups.
     The run stops if one group holds more than 5 % of the images (chaining guard).
  5. Whole groups are assigned to train / validation / test = 70 / 15 / 15 % of the images,
     seed 20261012, stratified by the answer values of the finding question and by the
     Kvasir-SEG mask (`data/splits.py: assign_groups_stratified`).
- **Images and QA rows per split:** **[RC-D1]** `images:<split>`, `qa_rows:<split>`,
  `image_fraction:<split>`.
- **Groups:** **[RC-D1]** `groups:total`, `groups:multi_image`, `groups:largest`,
  `groups:largest_fraction`, `groups:size:*`; pairs **[RC-D1]**
  `near_duplicates:joined_pairs:always` and `:confirmed`.
- **Near-duplicates of training images in validation and test:** 0 by construction
  (**[RC-D1]** `val_test_near_duplicates_of_train`, `val_test_confirmed_duplicates_of_train`).
  Reported, not filtered: the dHash distances above 4 bits (**[RC-D1]**
  `min_distance_to_train:<split>:5` ... `:17`), each held-out image's highest crop correlation
  with a training image (**[RC-D1]** `max_score_to_train:<split>:*`), and the held-out images
  whose crop correlation with a training image beyond 12 bits reaches the threshold
  (**[RC-D1]** `similar_to_train_beyond_candidates:<split>`). The 24 most similar pairs across
  splits are checked by eye on a contact sheet (kept private).
- **Limitation: no patient identifiers.** HyperKvasir, the source of the Kvasir-VQA images, has
  no patient IDs, only image IDs. The grouping joins identical images and adjacent frames of
  one scene, but images of the same patient taken at different moments (another part of the
  bowel, another angle) can still fall into different splits. Validation and
  test are therefore image-disjoint and near-duplicate-disjoint, **not patient-disjoint**, and
  results on them may be optimistic by an unknown amount.
- **Kvasir-SEG masks per split:** **[RC-D1]** `masked_images:<split>`; links
  `masks:linked/ambiguous/unmatched`.
- **Split hash:** **[RC-D1]** `split_sha256` (printed by `prepare_data.py bundle`).
