# Data card: Kvasir-VQA-x1 (with Kvasir-VQA images and Kvasir-SEG masks)

> No images, no patient identifiers, no per-image rows. Counts marked **[RC-D1]** come from
> `results/data/summary.csv` once RC-D1 run 4 is committed, and from
> `results/data/near_duplicate_labels.csv` (RC-D1-label).

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

RC-D1 run 4 writes these numbers again from the code (`official:*` rows in `summary.csv`).
The appendix item reports them next to the published adapter's test results.

## Our splits (RC-D1 run 4)

- **Why embeddings (D02 amendments 2 and 3).**
  - RC-D1 run 2 grouped images within 4 dHash bits only; that missed adjacent frames.
  - Run 3 joined dHash candidates by a crop correlation of at least 0.65. Its final contact
    sheet still showed same-scene pairs across splits:
    - 3 of the 24 highest crop scores, at 14-32 bits, where the dHash never looked;
    - about 5 of the 24 closest by dHash;
    - about 9 of 12 pairs within 4 bits that the crop measure left apart.
  - The likely causes: Kvasir-VQA mixes HyperKvasir and Kvasir-Instrument, so one frame can
    exist at another size or crop; and the crop correlation under-scores copies of smooth or
    blurred mucosa, or when the highlights move.
- **Method:**
  1. Every image is embedded by a small pretrained self-supervised vision model (proposal
     DINOv2-small; id, revision and licence: **[RC-D1-embed]**). Two views: the whole image
     and the central 60 % of width and height, 224 x 224. The features are L2-normalised and
     compared by cosine.
  2. About 300 candidate pairs, stratified with the border zones oversampled, are labelled
     by two raters, independently, as same scene, different or unsure, on sheets without
     scores. Agreement: Cohen's kappa **[RC-D1-label]** `labels:kappa`.
  3. The joining rule is chosen on these labels by a rule fixed in advance: the highest
     sensitivity with precision >= 0.5, with no group over 10 % of the images. The chosen
     rule: **[RC-D1-label]** `labels:chosen:*`.
     - On the labelled pairs: sensitivity **[RC-D1-label]** `labels:chosen:sensitivity` and
       precision `labels:chosen:precision`, with Wilson 95 % intervals.
     - Weighted to all candidate pairs: `labels:chosen:sensitivity_weighted` and
       `:precision_weighted`.
  4. The rule is applied to all pairs of images; the joined pairs form groups.
  5. Whole groups are assigned to train / validation / test = 70 / 15 / 15 % of the images,
     seed 20261012, stratified by the answer values of the finding question and by the
     Kvasir-SEG mask (`data/splits.py: assign_groups_stratified`).
  6. A contact sheet of the closest cross-split pairs (by cosine, crop correlation and dHash)
     is checked by eye (kept private).
- **Strict test subset** (a pre-registered sensitivity analysis): the test images whose most
  similar training image is below the chosen threshold minus 0.05: **[RC-D1]**
  `strict_test:images`, `strict_test:qa_rows`. The main comparisons are reported on it too.
- **Images and QA rows per split:** **[RC-D1]** `images:<split>`, `qa_rows:<split>`,
  `image_fraction:<split>`.
- **Groups:** **[RC-D1]** `groups:total`, `groups:multi_image`, `groups:largest`,
  `groups:largest_fraction`, `groups:size:*`; joined pairs **[RC-D1]**
  `near_duplicates:joined_pairs`.
- **Near-duplicates of training images in validation and test:** 0 by construction
  (**[RC-D1]** `val_test_confirmed_duplicates_of_train`). Reported, not filtered: the 0-4 bit
  pairs the rule left apart (**[RC-D1]** `near_duplicates:close_pairs_not_joined`, in all and
  across splits); each held-out image's dHash distance, crop correlation and cosine to the
  training split (**[RC-D1]** `min_distance_to_train:*`, `max_score_to_train:*`,
  `max_cosine_to_train:*`).
- **Limitation: no patient identifiers.** HyperKvasir, the source of the Kvasir-VQA images, has
  no patient IDs, only image IDs. The grouping joins identical images and adjacent frames of
  one scene, but images of the same patient taken at different moments (another part of the
  bowel, another angle) can still fall into different splits. Validation and
  test are therefore image-disjoint and near-duplicate-disjoint, **not patient-disjoint**, and
  results on them may be optimistic by an unknown amount. The near-duplicate rule itself misses
  some copies: its sensitivity on the labelled pairs is reported above, with its interval.
- **Kvasir-SEG masks per split:** **[RC-D1]** `masked_images:<split>`; links
  `masks:linked/ambiguous/unmatched`.
- **Split hash:** **[RC-D1]** `split_sha256` (printed by `prepare_data.py bundle`).
