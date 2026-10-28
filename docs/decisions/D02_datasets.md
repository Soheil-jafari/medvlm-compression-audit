# D02: Datasets and their roles

**Status:** decided 2026-10-12 (Soheil): Q1 = A (validation and test are the two
halves of the official test split, by image), Q2 = match Kvasir-SEG by fingerprint
(default), Q3 = EndoBench (licence and download still to verify), Q4 = Kvasir-VQA-x1
classes only to start (default), Q5 = retinal kept for later.
**Stage:** S01. **Prepared:** 2026-10-12 (iteration 7).
**Facts:** read from Hugging Face dataset records, dataset pages, papers and READMEs
on 2026-10-12 (links below). Unconfirmed facts are marked `TODO: verify`.

## Question

Which datasets play which role? The study needs:

1. **Main QA** (train / validation / test): image + question -> answer.
2. **Masks**: lesion outlines, for small-finding strata and the lesion-mask audit
   (both primary endpoints).
3. **Rare findings**: enough positives of uncommon findings to measure rare recall.
4. **Stress tests**: perturbed images (blank, lesion masked, random region masked,
   other image). These we generate ourselves; no extra dataset is required.
5. **External test**: a different source, deduplicated against our training images.

## What exists

| Dataset | What it is | Size | Masks | Patient IDs | Licence / access | Link |
|---|---|---|---|---|---|---|
| Kvasir-VQA-x1 | GI endoscopy VQA built on Kvasir-VQA; questions stratified by complexity 1-3; classes such as abnormality presence, polyp type / size / count, instrument presence / location, landmark location | **143,594 train / 15,955 test** QA rows | No (only `img_id`) | No | CC BY-NC 4.0, not gated | [HF](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1), [paper](https://arxiv.org/abs/2506.09958) |
| Kvasir-VQA (parent) | 6,500 images from HyperKvasir and Kvasir-Instrument; yes/no, choice, location, count questions | 58.8k QA rows | No | No | CC BY-NC 4.0 | [HF](https://huggingface.co/datasets/SimulaMet-HOST/Kvasir-VQA), [paper](https://arxiv.org/abs/2409.01437) |
| Kvasir-SEG | 1,000 polyp images from Kvasir v2 with masks and bounding boxes; 332x487 to 1920x1072 px | 1,000 | **Yes** (polyps) | No | Research and educational use only; no commercial use without permission | [page](https://datasets.simula.no/kvasir-seg/) |
| Colon-X: ColonVQA | Colonoscopy VQA consolidated from public sources (Kvasir, HyperKvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-Larib, PolypGen, ...); 76 findings, 18 tasks | 1,100,786 queries, 212,742 images | TODO: verify | TODO | Google form for full access; no licence in the README (HF annotation repo tagged apache-2.0) | [paper](https://arxiv.org/abs/2512.03667v2), [GitHub](https://github.com/ai4colonoscopy/Colon-X) |
| Colon-X: ColonEval, ColonPert | Evaluation subset and a human-perturbation robustness set of Colon-X | TODO: verify | TODO | TODO | as above | as above |
| EndoBench | Endoscopy MLLM benchmark (NeurIPS 2025 Datasets and Benchmarks): 4 scenarios, 12 tasks, 5 levels of visual prompting | 6,832 VQA pairs from 21 datasets | TODO: verify | TODO | Licence and download location TODO: verify | [paper](https://arxiv.org/abs/2505.23601) |
| IDRiD (retinal) | Fundus images, 4288 x 2848 px | 81 images with masks | **Yes**: microaneurysms (81), hard exudates (81), haemorrhages (80), soft exudates (40) | No | CC BY 4.0 | [page](https://idrid.grand-challenge.org/Data/) |
| EyeVQA (retinal) | Ophthalmic VQA benchmark from 21 public datasets (Sept 2026) | 20,000 QA | Targets from masks / boxes; release of the annotations unclear | No | Metadata CC BY 4.0; images must be fetched from the source datasets | [paper](https://arxiv.org/abs/2609.32352), [GitHub](https://github.com/PKUTHM/EyeVQA) |
| FundusGround (retinal) | Fundus VQA with lesions mapped to retinal regions (May 2026) | 10,719 images, 72,706 questions | Region-level | No | Release TODO: verify | [paper](https://arxiv.org/abs/2605.22414v1) |

### Facts that shape the design

- **The teacher saw the whole Kvasir-VQA-x1 train file** (3 epochs, D01). A
  validation split cut from train would flatter the teacher in Gate 1 and in E0.
- **No patient or procedure IDs** in Kvasir-VQA-x1 or Kvasir-SEG. Splits can only be
  image-level, so near-duplicate frames across splits must be removed by image
  fingerprint (`data/dedup.py`). This is a stated limitation.
- **Kvasir-VQA-x1 has no masks, Kvasir-SEG has no questions.** Kvasir-SEG images come
  from Kvasir v2, and Kvasir-VQA images from HyperKvasir; matching the two by image
  fingerprint should give polyp masks for some VQA images. How many match is
  unknown until RC-D1 runs on the team's copies (TODO).
- **Overlap risk everywhere:** ColonVQA and probably EndoBench reuse Kvasir and
  HyperKvasir images, so any external set must be deduplicated against our training
  images before it counts as external.

## Options

**Q1. Validation split** (used for Gate 1, student selection, Gate 2, choosing the
best of E5/E6):

- **A. Split the official test set by image into validation and test halves
  (recommended).** Neither the teacher nor any student has seen either half. Test
  shrinks to about 8k QA rows, still large.
- **B. Cut validation from the train file.** Keeps the full test set, but the teacher
  has trained on those images, so Gate 1 and E0 numbers are inflated.

**Q2. Masks:** Kvasir-SEG matched to Kvasir-VQA-x1 images by fingerprint
(recommended; only option for endoscopy polyps). If too few matches, the mask-based
endpoints are computed on Kvasir-SEG images with our own templated yes/no probes
("Is there a polyp in this image?").

**Q3. External test:** EndoBench (recommended, if its licence and download are
confirmed), ColonEval, or none.

**Q4. Rare findings:** Kvasir-VQA-x1's own abnormality classes only (recommended to
start), or add ColonVQA (more findings, but a heavier download behind a form and a
large overlap with Kvasir).

**Q5. Retinal:** keep as a later second-domain study (IDRiD masks; EyeVQA or
FundusGround questions once their release is clear), or drop.

## Recommendation

| Role | Dataset |
|---|---|
| Main QA | Kvasir-VQA-x1: train file for training; official test split halved by image into **validation** and **test** (Q1 = A) |
| Masks | Kvasir-SEG, matched to Kvasir-VQA-x1 images by fingerprint; fallback: Kvasir-SEG with templated probes |
| Rare findings | Kvasir-VQA-x1 abnormality classes (prevalence from the train file); ColonVQA later only if rare groups are too thin |
| Stress tests | Generated by us from the main test images and Kvasir-SEG masks |
| External test | EndoBench, deduplicated against our training images (licence to confirm) |
| Second domain (later, optional) | Retinal: IDRiD masks plus EyeVQA or FundusGround questions, once their release is confirmed |

First draft of the finding list (from the Kvasir-VQA-x1 question classes; final list
from the RC-D1 counts): polyp (presence, size, type), and each abnormality type that
`abnormality_presence` answers name (TODO: list from RC-D1). Instruments and
landmarks are kept as **control** questions, not findings.

## Risks

- **Non-commercial terms:** CC BY-NC 4.0 and Kvasir-SEG's research-only terms allow
  this research and publishing results, but no commercial use, and no data in the
  repository (already our policy).
- **Image-level splits only:** without patient IDs, near-duplicates may leak across
  splits; fingerprint deduplication is mandatory and its threshold is pre-registered.
- **Few masked test images:** if few Kvasir-SEG images match the test half, the
  small-finding stratum (smallest quarter of lesion areas) may be small; CIs will
  show it. Fallback in Q2.
- **External set terms unclear:** EndoBench's licence and download are not confirmed.
- **Label quality:** the Kvasir-VQA-x1 viewer shows typos and some answers that
  contradict their class (for example `box_artifact_presence`); answer parsing must
  normalise them, and the rules are pre-registered.

## What changes in the repository once decided

- `configs/data/*.yaml`: chosen datasets `DECIDED` with verified sources, licences
  and revisions; `configs/base.yaml: components.datasets` filled.
- `docs/data_cards/` for each chosen dataset.
- S03 tasks (readers, fingerprint matching of Kvasir-SEG, splits) and run card RC-D1.

## Amendment (2026-10-18, Soheil): image-disjoint re-split of all images

**Why.** RC-D1 run 1 showed that the official Kvasir-VQA-x1 split is by question, not by image.
- Images: train 6,212; test 4,058; in both 3,821; test only 237; all 6,449.
- Test rows: 15,718 of the 15,955 are on images that are also in train.
- The dataset paper (arXiv 2506.09958, section 2.3) says so: "Each image-question pair is
  assigned to either the training or test set".
- The test-only images are close to training images too. Smallest dHash distance (bits) to a
  training image, with the number of images: 0: 43, 1: 43, 2: 43, 3: 21, 4: 18, 5: 14, 6: 14,
  7: 10, 8: 5, 9: 7, 10: 5, 11: 1, 12: 6, 13: 1, 14: 4, 15: 1, 17: 1.

Halving the official test split (Q1 = A) therefore left 39 validation and 30 test images after
deduplication, and none with a Kvasir-SEG mask.

**Decision.** This replaces Q1 = A.
1. All 6,449 images are re-split, with the questions of both official splits, by image: train /
   validation / test = 70 / 15 / 15 % of the images, seed 20261012.
2. Near-duplicate images (dHash distance <= 4 bits, transitive, union-find) are grouped first,
   and whole groups are assigned.
3. The assignment is stratified, so that every finding with positives and the Kvasir-SEG-linked
   images are in validation and test in proportion. The method is a group version of iterative
   stratification (`data/splits.py: assign_groups_stratified`, CPU-tested):
   - labels are the answer values of the finding question, and "has a mask";
   - the labels are taken from the rarest to the most common, and one group goes to each split
     first where a label has three groups or more;
   - the rest of a label's groups go to the split furthest below its share;
   - unlabelled groups fill the image targets.
4. RC-D1 checks that no validation or test image is within 4 bits of a training image (zero by
   construction). It reports the distances from 5 to 17 bits as a data-card number, not a
   filter.
5. The official split is kept only for the data card and the appendix item of D01: the
   published adapter has seen these images.

Applied in `data/prepare.py`, `data/splits.py`, `data/dedup.py`,
`configs/data/kvasir_vqa_x1.yaml` (`split_ratios`) and run card RC-D1 (run 2).

## Amendment 2 (2026-10-18, Soheil): two-stage near-duplicate detection

Decided before any model result, from RC-D1 run 2 (4-bit grouping; split_sha256
`024f4abf05a81dffaa1c2b120e3e7bb99456b03fd248ddb6419284a8cf57fde9`; nothing committed):
- Images 4,518 / 965 / 966 (70.06 / 14.96 / 14.98 %); QA rows 112,370 / 24,379 / 22,458;
  masked images 710 / 152 / 148.
- Groups: 4,457 in all, 398 with more than one image, the largest 255 images; 2,390 images in
  multi-image groups. No held-out image within 4 bits of a training image.
- Smallest dHash distance to a training image (bits: images): validation 5: 119, 6: 126, 7: 148,
  8: 169, 9: 136, 10: 100, 11: 60, 12: 41, 13: 29, 14: 13, 15: 16, 16: 6, 17: 2; test 5: 99,
  6: 113, 7: 138, 8: 126, 9: 168, 10: 121, 11: 82, 12: 56, 13: 30, 14: 15, 15: 13, 16: 4,
  18 or more: 1.
- Visual check: random pairs have a median distance of 26 bits and a 5th percentile of 17;
  1,038 of the 1,931 validation and test images have a training image within 8 bits. At 5-6
  bits, 3 of 8 pairs were the same scene; at 7-8 bits, 2 to 3 of 8; at 9-12 bits, 0 of 4; random
  controls 0 of 4. The other pairs looked alike only at 9 x 8 pixels: colour, lumen position, and
  the overlays (text at the top left, a green box at the bottom left).

So the 4-bit rule misses real duplicates, and a larger dHash threshold alone would chain
unrelated images.

**Decision.** Two stages, then group, then split exactly as in the first amendment.
1. Candidates: pairs within 12 bits of the 64-bit dHash.
2. Confirmation with a finer measure that ignores borders and overlays:
   - the central 60 % of the width and of the height (20 % cut from each side), grey-scale,
     resized by area average to 32 x 32 pixels, with a Gaussian blur of 4 px (at 32)
     subtracted, so the measure follows the mucosa's detail rather than the lighting and the
     lumen position;
   - the measure: the highest Pearson correlation between the crop of one image and the other
     image shifted by up to 3 px (5.6 % of the side) in each direction, both ways round;
   - the threshold `min_correlation` is calibrated on the data by `prepare_data.py calibrate`
     (RC-D1 section 3a, CPU): the distribution over all candidate pairs, what each threshold
     from 0.30 to 0.95 gives, and a contact sheet of the pairs just above and below the
     proposed value plus random controls. It is chosen so that identical scenes and adjacent
     frames are joined and different scenes are not; the value and the reason are recorded
     here before run 3. **Open until the calibration step has run.**
3. Union-find over the confirmed pairs plus every pair within 4 bits; whole groups to 70 / 15 /
   15 %, the same stratification, seed 20261012.
4. Safety checks in `summary.csv`: the number and sizes of the groups and the largest group;
   the run STOPS and reports if the largest group holds more than 5 % of all images (322 of
   6,449; a group of one or two images cannot chain, so only groups of three or more count,
   which matters only for tiny test sets); the validation and test images with a confirmed
   duplicate in train (must be 0); the new tail. Also reported, not filtered: each held-out
   image's highest crop correlation with a training image, and how many reach the threshold
   with a training image beyond 12 bits (what the candidate stage cannot see).
5. After the split, a final contact sheet of the 24 most similar cross-split pairs by the
   crop measure (a PNG, private, never committed), attached to the run report.
6. Data card limitation: HyperKvasir has no patient IDs, so images of the same patient at
   different moments can still fall into different splits.

The crop fraction, the size, the blur and the shift are the agent's choice within this
decision:
- 60 % is the lower end of the decided 60-70 %, to keep the text and the green box out; the
  contact sheets draw the crop so the runner can check it;
- the blur removes what made unrelated images look alike at 9 x 8;
- the shifts let adjacent frames match.

On synthetic test images (`tests/test_near_duplicates.py`), identical, re-encoded, brighter
and 3 %-shifted copies score above 0.95, different images below 0.6, and overlays outside the
crop do not change the score. Scores on the real images are unknown until the calibration step.

The calibration step's provisional proposal is the smallest grid value that passes the guard
and that at most 0.1 % of 2,000 random pairs beyond 12 bits reach. It is a starting point to
check on the contact sheet, not the decision.

Applied in `data/dedup.py` (`two_stage_groups`, `crop_similarity`), `data/prepare.py`
(`prepare`, `calibrate`), `data/contact_sheet.py`, `scripts/prepare_data.py` (`calibrate`,
exit code 7 for the guard), `configs/data/kvasir_vqa_x1.yaml` (`near_duplicates`) and run card
RC-D1 (calibration step, run 3).

### Final values (2026-10-19, Soheil), before any model result

From the calibration step (Kaggle CPU, 2026-10-19, commit 73f31bc;
`results/data/near_duplicate_calibration.csv`, commit b4cf740). It ran with the rule of that
commit, in which pairs within 4 bits were still joined automatically:
- 211,329 candidate pairs within 12 bits.
- The crop correlation of the 5-8 bit pairs is bimodal: one mode at 0.15-0.30 (different
  images), one at 0.80-0.90 (same scene), with the valley at 0.60-0.70 (bins 0.60: 731,
  0.65: 671, 0.70: 809).
- Random controls (2,000 pairs beyond 12 bits): median 0.137, 99th percentile 0.420, maximum
  0.808; 2 of 2,000 reach 0.65.
- Largest group: 2,059 images (31.9 %) at 0.55, 428 (6.6 %) at 0.60, 349 (5.4 %) at 0.65, 335
  (5.2 %) at 0.70 and 0.75, 321 (5.0 %) at 0.80.
- The contact sheet at the provisional proposal 0.80:
  - just below 0.80 (not joined): 1 of 8 pairs is the same scene (7 bits, practically
    identical colon frames), so 0.80 misses it;
  - just above 0.80 (joined): 2 of 8 are the same scene; 6 of 8 are different images, mostly
    pale oesophagus images with a similar texture;
  - the four 0-4 bit pairs with the lowest correlation (r = 0.009, 0.042, 0.063, 0.065) are
    all clearly different images: the automatic join links unrelated images and feeds
    chaining;
  - random controls: all different.

Reasoning: a missed duplicate leaks test images into training (expensive); a wrong join only
puts two images in the same split (cheap, as long as no giant group forms). So the threshold
goes to the valley, not to 0.80, and the automatic join is removed.

**Final values:**
1. `min_correlation` = **0.65**: the valley of the bimodal 5-8 bit distribution; 2 of 2,000
   random controls reach it.
2. **No automatic join within 4 bits.** Every candidate pair within 12 bits, 0-4 bits
   included, is joined only if its crop correlation is at least 0.65. The 0-4 bit pairs that
   are not joined are reported: their number, in all and across splits, and a band of the
   final contact sheet (`--close-bits 4` names that band; `--dedup-threshold` is its old name).
3. **Chaining guard at 10 %** of all images (was 5 %); the jump above 30 % below 0.60 shows
   what real chaining looks like. The run STOPS (exit code 7) above 10 %, as before. With the
   automatic join removed, the largest group at 0.65 can only be smaller than or equal to the
   calibration's 349 (fewer pairs are joined).
4. **The deciding check** stays the final contact sheet, now with the 24 most similar
   cross-split pairs by the crop correlation and the 24 closest by the dHash distance (and the
   0-4 bit pairs left apart). If a person sees same-scene pairs there, the next step is a
   stronger near-duplicate measure, pre-registered as the fallback in the decisions log
   (2026-10-19): embeddings from a small pretrained vision model, compared by cosine
   similarity, run on CPU. It is not implemented now.

Applied in `configs/data/kvasir_vqa_x1.yaml` (`near_duplicates`), `data/dedup.py`,
`data/prepare.py`, `scripts/prepare_data.py`, `cards.py` and run card RC-D1 (run 3).

## Amendment 3 (2026-10-19, Soheil): embeddings, with the rule set from human labels

Decided before any model result, from RC-D1 run 3 (Kaggle CPU, commit 103fbb0; **not
accepted**, nothing committed):
- `split_sha256` 4a595fdf...; images 4,515 / 967 / 967; QA rows 102,639 / 30,438 / 26,130;
  masked images 658 / 143 / 209.
- Groups 4,176, the largest 241 (3.7 %); joined pairs 18,034; 929 pairs within 4 bits not
  joined (252 across splits).
- `similar_to_train_beyond_candidates`: validation 125, test 119.
- Cross-split crop scores, top 24: 0.781 to 0.893. The closest cross-split pairs by dHash:
  1-2 bits, crop at most 0.588.
- The final sheet, by eye:
  - the 24 highest crop scores (14-32 bits, never candidates): 3 the same scene;
  - the 24 closest by dHash: about 5 the same scene, about 5 unsure;
  - 12 pairs within 4 bits that the crop left apart: about 9 the same scene.

  Likely causes: Kvasir-VQA mixes HyperKvasir and Kvasir-Instrument, so the same frame can
  exist at another size or crop (its dHash changes); and the crop correlation under-scores
  copies of smooth or blurred mucosa, or when the highlights move.

Both measures miss true duplicates, so the split is not clean.

**Decision.** The pre-registered fallback becomes the **primary** joining measure, and its rule
is set from human labels, not from score distributions.
1. **Embeddings** (`data/embeddings.py`):
   - the model: a small pretrained self-supervised vision model, DINOv2-small from the
     Hugging Face hub (`facebook/dinov2-small`); RC-D1-embed read its hub record. Pinned below
     (additions of 2026-10-19): revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, licence
     `apache-2.0` (the hub record);
   - two views of every image, each resized to 224 x 224 (aspect not kept, the same for every
     image): `full`, the whole image with its border and overlays; `centre`, the central 60 %
     of width and height, which leaves out the corner overlays. Both are tested on the labels;
   - the model's own normalisation; the pooled output, L2-normalised; cosine similarity;
   - CPU: about 20 to 50 minutes for 6,449 images x 2 views (an **estimate** from the model's
     size; not measured; the run prints its speed). A GPU is used if the notebook has one;
   - a check before the run: synthetic shifted, blurred, rescaled and brightened copies must be
     nearer to their original than different images, in both views (else STOP, exit code 8).
2. **Candidates** (`data/candidates.py`): the 20 nearest neighbours of every image by cosine
   (each view); every pair within 12 dHash bits; the 5 highest crop correlations of every image
   at any distance; and the pairs of run 3's final sheet, reproduced from the same data, code
   and seed (its split hash is checked against `4a595fdf`).
3. **The labelled set** (`data/labelling.py`), seed 20261012, about 300 pairs from a partition
   of the candidates:
   - all run-3 sheet pairs;
   - 20 + 20 pairs within 0-2 and 3-4 dHash bits;
   - 25 + 25 + 20 pairs at crop correlation 0.50-0.65, 0.65-0.80 and 0.80 or more;
   - 12 pairs in each 0.05-wide bin of the larger cosine, from below 0.50 to 1.00, which
     oversamples the rare high-cosine pairs;
   - drawn at random within each stratum; each pair weighs its stratum's size over the pairs
     drawn;
   - numbered side-by-side sheets without any score, and a CSV template (`pair_id, rater_1,
     rater_2`, labels `same_scene / different / unsure`);
   - two team members label independently (card RC-D1-label says what counts as the same
     scene). The scoring computes Cohen's kappa and, for every rule, the sensitivity and
     precision for "same scene" with Wilson 95 % intervals; a pair is positive when both raters
     call it `same_scene`.

   Only aggregate numbers are committed; never images, image names, labels per pair or sheets.
4. **The rule, fixed in advance:**
   - the family: a cosine of at least t, t from 0.50 to 0.99 by 0.01, in the full view, the
     centre view or the larger of the two, alone or OR a crop correlation of at least 0.65 or
     0.80 (450 rules);
   - the choice: the highest sensitivity among the rules with precision >= 0.5 on the labelled
     set whose largest group over all images stays within the guard (10 %);
   - ties go to the higher precision, then to one view before a crop combination, then to the
     view order full, centre, larger, then to the **lower** threshold. Catching copies counts
     more than precision: a wrong join only keeps two images together;
   - the chosen rule's sensitivity and precision with intervals go to the data card, and also
     weighted to all candidate pairs (stratified bootstrap intervals).
5. **Run 4:**
   - the chosen rule over **all** pairs of images, exactly (not only candidates); connected
     components; the same stratified 70 / 15 / 15 split with seed 20261012; the guard at 10 %;
   - the final cross-split sheet, checked by eye: the 24 highest cosines, the 24 highest crop
     correlations, the 24 closest by dHash;
   - the **strict test subset** (pre-registered sensitivity analysis): the test images whose
     most similar training image is below the chosen threshold minus **0.05** (cosine units;
     for a combined rule also below its crop level minus 0.05). The main comparisons are
     reported again on it (`scripts/09_compare_arms.py --strict`, `results/comparisons_strict/`);
   - no patient IDs exist in HyperKvasir: the splits are not patient-disjoint, which the
     limitations say.

RC-TT1 (our teacher's training) waits for run 4.

Applied in `configs/data/kvasir_vqa_x1.yaml` (`near_duplicates.rule`, `embedding`,
`candidates`, `labelling`, `strict_margin`), `data/embeddings.py`, `data/candidates.py`,
`data/labelling.py`, `data/embed_step.py`, `data/prepare.py`, `evaluation/compare_stage.py`,
`scripts/prepare_data.py` (`embed`, `score-labels`), `cards.py`, and run cards RC-D1-embed,
RC-D1-label and RC-D1 (run 4).

Recorded later, from the runs: the kappa, the chosen rule and its sensitivity and precision
(RC-D1-label); run 4's counts. The model id, revision and licence are recorded below.

### Amendment 3, additions (2026-10-19, Soheil), after RC-D1-embed and before any label

**RC-D1-embed** (Kaggle T4, 2026-10-19, commit cb5de07), as reported from its printout:
- run 3 reproduced: `split_sha256` 4a595fdf...;
- model `facebook/dinov2-small` at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`,
  licence `apache-2.0` (the hub record);
- 6,449 images, two views, embedded in 143 s on the GPU as reported; the counts file
  committed later (`results/data/near_duplicate_embed.csv`, commit 69799ff) says 156 s
  (`embed:seconds` 156.28), and its other counts agree with the report;
- 366,028 candidate pairs; 298 pairs to label;
- self-check (cosine to the original):
  - synthetic copies: 0.937-0.995 (full view), 0.988-0.999 (centre view);
  - synthetic different images: 0.843-0.937 (full), 0.879-0.942 (centre).

The cosines are compressed into the high range. The decision boundary is therefore likely high
(around 0.95, Soheil's reading), where the sample has only 24 of its 298 pairs (the bins
0.90-0.95 and 0.95-1.00).

Recorded now, before any label exists:

1. **Pinned:** `near_duplicates.embedding` in `configs/data/kvasir_vqa_x1.yaml` holds the model,
   the revision and the licence above. Run 4, the scoring step and the top-up refuse an
   RC-D1-embed output whose recorded model, revision or licence differ (exit code 5).
2. **Adaptive top-up** (`labelling.topup_sample`, `prepare_data.py topup-labels`):
   - after the first 298 pairs are labelled and the rule is chosen: if fewer than **30**
     labelled pairs lie within **+-0.03** of the chosen cosine threshold (in the chosen view;
     the band's edges included), **100** more pairs are drawn from that band;
   - drawn at random, seed **20261013**, from RC-D1-embed's candidate pairs (the population of
     the first sample), leaving out the pairs already labelled; if the band holds fewer, all of
     them; if it holds none, there is no top-up and the rule stands;
   - numbered after the first pairs (299 on), on new sheets without scores; both raters label
     them under the same instructions and the same blinding;
   - the rule is then chosen again, **once**, on all labels (the first pairs and the top-up),
     by the same criterion. No second top-up;
   - the scoring step says when the top-up is required (exit code 4); run 4 waits until the
     rule is chosen again. The command reuses RC-D1-embed's outputs: no new embedding run;
   - the weighted estimates over all candidates use the first pairs only (their stratified
     design covers every candidate). The top-up pairs count in the labelled-pair sensitivity
     and precision and in the choice;
   - reported (`results/data/near_duplicate_topup.csv`): the first rule, the labelled pairs in
     the band, the candidate pairs in the band and the pairs of all images in the band.
3. **Disagreements** (`labelling.adjudicate`):
   - the **third team member** (not one of the two raters) decides:
     - every pair the two raters label differently, `unsure` included;
     - and, if a round's Cohen's kappa (three labels, before adjudication) is **below 0.40**,
       every pair of that round. This is the agent's reading of "if kappa < 0.40": the third
       member then decides the agreed pairs too. A round is the first pairs, or the top-up;
   - a decided pair is positive only if the third member calls it `same_scene`;
   - the third member gets only the list of pair numbers (written privately by the scoring
     step) and the sheets: no score and neither rater's label;
   - reported: Cohen's kappa before adjudication (all pairs and per round) and the number of
     decided pairs (`labels:kappa`, `labels:kappa:<round>`, `labels:adjudicated`).
4. **Unsure:**
   - **primary rule:** a pair is a copy only if it is `same_scene` for both raters, or for the
     third member where they decided. `unsure` counts as not `same_scene`. The rule chosen
     this way sets run 4;
   - **sensitivity analysis:** `unsure` counts as not `same_scene` for the precision and as
     `same_scene` for the sensitivity (the less favourable reading of each). Reported: the
     chosen rule's sensitivity and precision this way, and the rule this way would choose
     (`labels:unsure_analysis:*`). It does not change run 4.
5. **Rater blinding:**
   - the raters see no score: they get the sheets, `label_images.csv` and the template only,
     never `label_pairs.csv` or the candidate files;
   - they do not discuss labels until both label files are committed;
   - per-pair labels never enter the repository, so "committed" means a commitment of the file:
     each rater runs `prepare_data.py commit-labels` on their own file. It checks that only
     their column is filled and adds the file's SHA-256 to
     `docs/runs/RC-D1-label_commitments.csv`. The rater commits that change alone;
   - the commit order is in the history and goes into the run report;
   - the scoring step checks every label file against these hashes (`--commitments`): a file
     changed after its commitment is refused (exit code 5);
   - the same holds for the top-up round and for the third member's file.

Applied in `configs/data/kvasir_vqa_x1.yaml` (`embedding`, `labelling.topup`),
`data/labelling.py`, `data/embed_step.py`, `data/embeddings.py`, `data/prepare.py`,
`utils/bundles.py`, `scripts/prepare_data.py` (`topup-labels`, `commit-labels`,
`score-labels --topup-dir --private-dir --commitments`), `cards.py`, `scripts/run_card.py`,
and the run cards RC-D1-embed (report) and RC-D1-label.

## Amendment 4 (2026-10-22, Soheil): two raters, no third member

Recorded before any label was compared with any cosine or rule, and before the scoring.

**Facts:**
- The commitments in `docs/runs/RC-D1-label_commitments.csv`, round `first`:
  - `labels_rater_1.csv`, SHA-256
    `02fc7492c1a9516f262344f4aa20fae4b772a7009207842c2f7e20527502156d`, 298 labels;
  - `labels_rater_2.csv`, SHA-256
    `ffb003276629eaa21335426d7cbf549dc8dbc64ffc9373565b388b6fc7ea08fb`, 298 labels.

  Both rows reached `research-loop` in one commit (92b78e3, 2026-10-21 22:22 UTC), so the
  history does not show an order between the two raters.
- Rater 2 labelled pairs 11-298 blind, in letters. They were converted with the legend given
  in writing before any comparison: T = `different`, F = `same_scene`, U = `unsure`.
  - The legend was once stated the other way round. Raw agreement was then looked at under
    both readings (0.17 and 0.78), and the team confirmed the first written legend.
  - The scoring reports the raw agreement on pairs 11-298 under the confirmed legend
    (`labels:agreement`).
- Rater 2's labels for pairs 1-10 were given after rater 2 saw rater 1's labels for them;
  rater 2 agreed with all 10.
- Rater 1 labelled pairs 1-10 and 291-298 after the files were exchanged.
- **The recorded decision for a second commitment:** the `rater_2` row replaced an earlier
  `rater_2` row for the same round (288 labels, pairs 1-10 empty). `commit-labels` refuses a
  second file for a role and round without a recorded decision; this is that decision. The
  earlier row is not in the repository's history (it was never pushed).

**Decision:**
1. **Two raters, no third member:** the team has two members for this step. Amendment 3's
   third-member steps are dropped:
   - no adjudication of disagreements;
   - no rule at kappa below 0.40.
2. **Kappa** is computed on pairs **11-298 only** (both blind) and reported. No kappa
   threshold triggers anything.
3. **Primary truth:** a pair is a copy only if **both** raters call it `same_scene`;
   `unsure` counts as not `same_scene`. The rule is chosen on this truth, over all 298 pairs,
   with amendment 3's 450 rules, criterion, tie-breaks, Wilson intervals and weighting.
4. **Reported, not used for the choice:**
   - the truth "`same_scene` for **either** rater": the chosen rule's sensitivity and
     precision under it, and the rule it would choose (`labels:either_rater:*`);
   - the unsure analysis stays (`labels:unsure_analysis:*`). With two raters, its sensitivity
     side counts a pair as positive when both raters say `same_scene` or `unsure`. This reads
     each rater's `unsure` as `same_scene` (the agent's reading).
5. **Top-up:** the rule of amendment 3 is unchanged. If it is needed, both raters label all
   its pairs, and the third-member steps are skipped there too. A top-up round's kappa is
   reported on all its pairs (`labels:kappa:topup`; reported only, the agent's choice).

**Open question for the team:** rater 1 labelled pairs 291-298 after the files were exchanged,
so these 8 pairs may not be blind for rater 1. The pairs both labelled before the exchange are
11-290. Item 2 says 11-298. The card runs `--kappa-pairs 11-298` as decided; confirm, or
change the range, before the scoring.

(Settled by the scoring itself: the committed labels file has the kappa on pairs 11-290; see
amendment 5.)

Applied in `scripts/prepare_data.py` (`score-labels --no-third-member --kappa-pairs
FIRST-LAST`) and `data/embed_step.py`. Without the flag, the scoring is amendment 3's.

The rows written in addition (`results/data/near_duplicate_labels.csv`):
- `labels:kappa:first_pair`, `labels:kappa:last_pair`, `labels:kappa:pairs`: the kappa's
  pairs;
- `labels:third_member` = 0;
- `labels:either_rater:chosen:*`, `labels:either_rater:rule:*` and
  `labels:either_rater:same_rule`.

`labels:kappa`, `labels:kappa_same_scene_vs_rest` and `labels:agreement` are on the kappa's
pairs only. Run card RC-D1-label has the Kaggle cell.

## Amendment 5 (2026-10-28, Soheil): RC-D1 run 4 not accepted; a rule sweep (post hoc)

**Run 4 is not accepted.** Run 4 was run from commit 5e08c51 with the rule `full:0.93`, chosen
on amendment 4's both-raters truth. It passed every automatic check:
- `val_test_confirmed_duplicates_of_train` 0;
- largest group 370 images (5.7 %);
- `split_sha256` `4bc426fea4965a4a8408c6dffa7c9ad60be7b0f974d7eb624385db5e089de8d3`.

The deciding check failed. On `rc_d1_final_cross_split_pairs.png`, many cross-split pairs are
the same scene: consecutive frames of one video. The reviewer counted, from a screenshot:

| Band of the final contact sheet | Same scene |
|---|---|
| 1. the 24 highest cosines (full view) | about 17 of 24 |
| 2. the 24 highest crop correlations | about 4 of 24 |
| 3. the 24 closest by the dHash | about 13 of 24 |

Nothing of run 4's split is used. Its `results/data/summary.csv` stays in the repository
until run 4b replaces it, as the record of the run.

The labels the rule was chosen on (`results/data/near_duplicate_labels.csv`, committed in
5e08c51) were scored on pairs 11-290 for the kappa (`labels:kappa:first_pair` 11,
`labels:kappa:last_pair` 290). This settles amendment 4's open question in favour of 11-290.

**Likely cause (recorded as such, not tested):**
- the primary truth required both raters to say `same_scene`;
- rater 2's first written legend called the code "identical", a narrower notion than
  `same_scene`;
- so near-identical frames counted as negatives, which pushed the threshold up.

**Decision (made after the deciding check; post hoc, and reported as such):**
1. **Truth = `same_scene` for either rater** (amendment 4's secondary reading). The rule chosen
   on it by amendment 3's criterion is the one RC-D1-label's scoring wrote to the committed
   labels file (`labels:either_rater:rule:*`): `full:0.92` OR crop >= 0.80. The sweep reads it
   from that file, prints it, and refuses to run if its first rule is not this one.
2. **Rule sweep, fixed before any new sheet is seen:**
   - the candidate rules are `full:T` OR crop >= 0.80, T in {0.92, 0.90, 0.88, 0.86, 0.84};
   - for each, the split is built exactly as `prepare` builds it (`prepare` itself runs for
     each rule): seed 20261012, close bits 4, mask threshold 4, the attached RC-D1-embed
     embeddings;
   - recorded per rule: the largest group and its fraction, the joined pairs, the val/test
     confirmed duplicates (must be 0), and the three-band cross-split contact sheet. The bands
     are as now, 24 pairs each; there is one PNG per T, written privately outside the
     repository;
   - a rule over the 10 % guard is marked and gets no sheet.
3. **Selection, fixed now:**
   - the highest T (the least joining) within the guard whose three bands show no same-scene
     pair, judged by the team on the sheets;
   - the count of same-scene pairs per band per T is recorded;
   - if none is clean, stop and report. Nothing lower is tried without a new recorded
     decision.
4. **Run 4b** = `prepare` with the selected rule (`--rule`), then `bundle` and `verify`. Its
   final contact sheet is checked again the same way.
5. **In the paper:** the label-chosen rule, its failure on the deciding check, the sweep and
   the selected rule. The strict test subset stays as a sensitivity analysis.

Applied in:
- `data/rule_sweep.py` and `scripts/prepare_data.py rule-sweep --data-root ... --embed-dir ...
  --out-dir ... --thresholds 0.92 0.90 0.88 0.86 0.84 --crop 0.80`. It is one CPU run. The
  counts per T go to stdout and to `results/data/near_duplicate_sweep.csv` (aggregate only:
  `sweep:<rule>:*`); the PNGs go to `--out-dir`, outside the repository;
- `prepare --rule full:T:or_crop:0.80` (and `counts --rule`, the same rule in pass 2). Only a
  rule listed in the committed sweep file within the guard is accepted; anything else stops
  with exit code 5. `results/data/summary.csv` then records the rule
  (`near_duplicates:rule:*`) and `near_duplicates:rule_from_sweep` = 1;
- `data/prepare.py` (`prepare` can take the sweep's one crop pass) and `data/candidates.py`
  (`JoinRule.from_label`);
- the RC-D1 card: section 3c (the sweep) and the run 4b section (section 3e since amendment 6),
  with their Kaggle cells.

## Amendment 6 (2026-10-28, Soheil): the sweep's only rule within the guard is not clean; a held-out filter (post hoc)

**The rule sweep of amendment 5** (run from commit 8268bef), as reported:

| Rule | Largest group | Within the 10 % guard |
|---|---|---|
| `full:0.92:or_crop:0.80` | 447 images (6.9 %) | yes |
| `full:0.90:or_crop:0.80` | 23.4 % | no |
| `full:0.88:or_crop:0.80` | 41.6 % | no |
| `full:0.86:or_crop:0.80` | 54.0 % | no |
| `full:0.84:or_crop:0.80` | 70.8 % | no |

For `full:0.92:or_crop:0.80`:
- `split_sha256` `becacdaebcdee6424318785ed7e32ffc783ee6ce99817e5c7a2fe916d59fc245`;
- val/test confirmed duplicates of train 0;
- images train / val / test 4,513 / 968 / 968.

Its sheet is not clean. The reviewer counted, from the sheet:
- band 1 (the highest cosines): about 12 of 24 the same scene;
- band 2 (the highest crop correlations): about 3 of 24;
- band 3 (the closest by the dHash): about 1 of 24.

So, per amendment 5, **the sweep stops**: no rule of it is used as it stands.

The sweep's counts file (`results/data/near_duplicate_sweep.csv`) was not on `research-loop`
when this was recorded. `prepare --rule` and the filter sweep read it, so it is committed
first.

**Decision (recorded as post hoc; fixed before any new sheet is seen):**
1. **Groups:** `full:0.92:or_crop:0.80`, the swept rule within the guard, with its split
   unchanged.
2. **Held-out filter** (`data/heldout_filter.py`). After the split, every validation and test
   image is removed if:
   - its nearest training image has a full-view cosine of at least **F**;
   - or its nearest training image has a crop correlation of at least **0.70**;
   - or a training image lies within **4** dHash bits with a crop correlation of at least
     **0.50**. The agent reads this condition as one training image meeting both.

   The filter:
   - never moves an image between splits and never touches train;
   - does not filter the validation/test pairs: those meeting a condition are only counted
     (`heldout_filter:val_test_pairs_matching`).
3. **Filter sweep:** F in {0.90, 0.88, 0.86, 0.84, 0.82, 0.80}. For each F it reports:
   - the validation and test images and QA rows kept, and their fraction of the unfiltered
     split;
   - the masked test images kept;
   - the per-finding test positives kept, where the label values exist (not yet: pass 2);
   - a three-band sheet of the closest remaining train/val and train/test pairs (24 each, the
     same bands as now), private, outside the repository.
4. **Selection, fixed now:**
   - the highest F whose three bands show no same-scene pair, judged by the team;
   - the counts are recorded per F and band;
   - and the filtered test must keep at least **50 %** of the unfiltered test images.

   If none qualifies, stop and report.
5. **The test set:**
   - **primary:** the filtered test;
   - **sensitivity analyses:** the unfiltered test, and the strict subset (taken within the
     filtered test, the test split as written; the agent's reading).

   The filter is applied once, here; no model result is seen before it is fixed.

Not built yet (open item): the unfiltered-test sensitivity analysis needs the removed test
images evaluated too. Their img_ids are in the private `heldout_filter_removed.json`; the
evaluation reads only the split files, so this step is still to be added.

Applied in:
- `data/heldout_filter.py`;
- `data/prepare.py`: `prepare(..., heldout_filter=...)`. The rows (`heldout_filter:*`), the
  strict subset, the sheet bands (then train/val and train/test pairs only) and
  `split_sha256` are of the filtered splits. The removed img_ids go to the private
  `heldout_filter_removed.json` next to `splits.json`;
- `data/rule_sweep.py` and `scripts/prepare_data.py filter-sweep --data-root ... --embed-dir ...
  --rule full:0.92:or_crop:0.80 --out-dir ... --thresholds 0.90 0.88 0.86 0.84 0.82 0.80`.
  The levels default to crop 0.70, 4 bits, crop 0.50. The counts go to stdout and to
  `results/data/near_duplicate_filter_sweep.csv`; one private PNG per F;
- `prepare --rule ... --heldout-filter F` and `counts` with the same flags. Only an F that the
  committed filter sweep lists is accepted: with the same rule, at or above the 50 % test
  floor, with two decimals. Anything else stops with exit code 5;
- the RC-D1 card: the sweep's report, and sections 3d (the filter sweep) and 3e (run 4b).

## Answer format for the inbox

`- YYYY-MM-DD (name): D02 decided: Q1 <A|B>, Q2 <match|probes>, Q3 <EndoBench|ColonEval|none>, Q4 <x1 only|add ColonVQA>, Q5 <later|drop>.`
