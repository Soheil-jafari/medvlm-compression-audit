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
  1. Every image is embedded by DINOv2-small (`facebook/dinov2-small`, revision
     `ed25f3a31f01632728cabb09d1542f84ab7b0056`, licence apache-2.0 per the hub record;
     RC-D1-embed, 2026-10-19). Two views: the whole image and the central 60 % of width and
     height, 224 x 224. The features are L2-normalised and compared by cosine.
  2. 298 candidate pairs, stratified with the border zones oversampled, are labelled by two
     raters as same scene, different or unsure, on sheets without scores. There is no third
     member (D02 amendment 4).
     - Blinding was not complete:
       - rater 2 labelled pairs 1-10 after seeing rater 1's labels for them;
       - rater 1 labelled pairs 1-10 and 291-298 after the two files were exchanged.
     - Agreement: Cohen's kappa on pairs 11-290 (both blind) **[RC-D1-label]** `labels:kappa`.
     - A pair is a copy when both raters call it the same scene; `unsure` counts as not the
       same scene.
     - Two readings are reported but do not set the rule:
       - the same scene for either rater (`labels:either_rater:*`);
       - `unsure` counted the less favourable way (`labels:unsure_analysis:*`).
     - If fewer than 30 labelled pairs lie within 0.03 of the chosen threshold, 100 more from
       that band are labelled and the rule is chosen again once.
  3. The joining rule is chosen on these labels by a rule fixed in advance: the highest
     sensitivity with precision >= 0.5, with no group over 10 % of the images. The chosen
     rule: **[RC-D1-label]** `labels:chosen:*`.
     - On the labelled pairs: sensitivity **[RC-D1-label]** `labels:chosen:sensitivity` and
       precision `labels:chosen:precision`, with Wilson 95 % intervals.
     - Weighted to all candidate pairs: `labels:chosen:sensitivity_weighted` and
       `:precision_weighted`.
     - **It failed the deciding check** (RC-D1 run 4, D02 amendment 5):
       - the rule chosen on the both-raters truth, `full:0.93`, passed every automatic check;
       - but the final contact sheet showed many same-scene pairs across splits (consecutive
         frames of one video): about 17, 4 and 13 of the 24 pairs in its three bands;
       - run 4 is not accepted. The likely cause: rater 2's first written legend called the
         code "identical", narrower than the same scene, so near-identical frames counted as
         negatives.
     - **Post hoc** (amendment 5, after that check):
       - the truth becomes "the same scene for either rater"; its rule by the same criterion
         is `full:0.92` OR crop >= 0.80 (`labels:either_rater:rule:*`);
       - a sweep of `full:T` OR crop >= 0.80, T from 0.92 down to 0.84 by 0.02, was fixed
         before any new sheet was seen;
       - the rule used is the highest T within the guard whose contact sheet shows no
         same-scene pair: **[RC-D1 sweep]** `results/data/near_duplicate_sweep.csv` and the
         team's counts per band. If none is clean, the preparation stops.
     - **The sweep stopped** (D02 amendment 6, 2026-10-28):
       - only `full:0.92` OR crop >= 0.80 stayed within the guard (largest group 447 images,
         6.9 %);
       - its sheet still showed same-scene pairs across splits: about 12, 3 and 1 of 24 per
         band.
     - **Post hoc** (amendment 6): the groups of that rule are kept, and a held-out filter is
       applied after the split (step 5b).
  4. The rule (`full:0.92` OR crop >= 0.80) is applied to all pairs of images; the joined
     pairs form groups.
  5. Whole groups are assigned to train / validation / test = 70 / 15 / 15 % of the images,
     seed 20261012, stratified by the answer values of the finding question and by the
     Kvasir-SEG mask (`data/splits.py: assign_groups_stratified`).
  5b. **Held-out filter** (D02 amendment 6). After the split, a validation or test image is
     removed if:
     - its nearest training image has a full-view cosine of at least F;
     - or a crop correlation of at least 0.70;
     - or a training image lies within 4 dHash bits with a crop correlation of at least 0.50.

     The filter never moves images between splits and never touches train. Validation/test
     pairs are counted, not filtered (`heldout_filter:val_test_pairs_matching`).
     - F is the highest of 0.90 to 0.80 (by 0.02) whose sheet of the closest remaining
       train/val and train/test pairs shows no same-scene pair, if the filtered test keeps at
       least half of the test images: **[RC-D1 filter sweep]**
       `results/data/near_duplicate_filter_sweep.csv` and the team's counts.
     - **No F was clean** (D02 amendment 7, 2026-10-28). Same-scene pairs per band (cosine /
       crop / dHash, of 24 each):
       - F 0.90: about 3 / 0 / 0;
       - F 0.88: about 5-7 / 0 / 0;
       - F 0.86: about 4-6 / 0 / 0;
       - 0.84 to 0.80 keep less than half of the test images.

       The remaining pairs are frames of the same examination with the camera moved, not
       copies. **F = 0.86** was chosen (post hoc), the lowest F that keeps half of the test:
       - the test keeps 512 of 968 images (52.9 %) and 12,067 of 21,728 QA rows;
       - validation keeps 390 of 968 images (40.3 %);
       - masked test images: 86 of 147.
     - What it removes and keeps: **[RC-D1]** `heldout_filter:removed:<split>`,
       `heldout_filter:kept_fraction:<images, qa_rows, masked_images>:<split>`.
     - **The internal test is a "reduced-leakage" test**, reported with those residual counts.
       It is one of two co-primary tests; the other is an external test from another centre
       (PolypGen and/or GastroVision, D02 amendment 7; not part of this card).
     - Every model is scored on three tests:
       - the filtered test;
       - the unfiltered test (the filtered test with the removed images added back, a second
         private test file, `test_unfiltered`);
       - the external test.

       The conclusions are reported for each. The strict subset is no longer a separate
       analysis (2026-10-29): at F = 0.86 it is the whole filtered test (512 of 512 images).
     - RC-D1 run 4b (2026-10-29) applied it: `split_sha256`
       `54dfa70a9aa99ea86262447acfd16fe4d4e871272781a31f39c9f330f4f7d090`, pinned in the
       dataset config.
  6. A contact sheet of the closest cross-split pairs (by cosine, crop correlation and dHash)
     is checked by eye (kept private). After the filter it shows train/val and train/test
     pairs only.
- **Strict test subset** (pre-registered as a sensitivity analysis; **dropped** on 2026-10-29,
  because at F = 0.86 it is the whole filtered test): the test images whose most similar
  training image is below the threshold of the rule used (run 4b) minus 0.05, within the
  filtered test: **[RC-D1]**
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
  After the held-out filter, same-examination pairs (the camera moved) remain across splits:
  about 4-6 of the 24 most similar train/val and train/test pairs by cosine at F = 0.86. No
  image-level method can remove them without patient or video ids. Hence the test is called
  reduced-leakage, and an external test from another centre is co-primary (D02 amendment 7).
- **Kvasir-SEG masks per split:** **[RC-D1]** `masked_images:<split>`; links
  `masks:linked/ambiguous/unmatched`.
- **Split hash:** **[RC-D1]** `split_sha256` (printed by `prepare_data.py bundle`).
