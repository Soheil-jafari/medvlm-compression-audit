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

Chosen threshold (filled after the calibration step): `min_correlation` = TODO; reason: TODO.

## Answer format for the inbox

`- YYYY-MM-DD (name): D02 decided: Q1 <A|B>, Q2 <match|probes>, Q3 <EndoBench|ColonEval|none>, Q4 <x1 only|add ColonVQA>, Q5 <later|drop>.`
