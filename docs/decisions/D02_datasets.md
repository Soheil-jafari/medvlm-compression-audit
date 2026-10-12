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

## Answer format for the inbox

`- YYYY-MM-DD (name): D02 decided: Q1 <A|B>, Q2 <match|probes>, Q3 <EndoBench|ColonEval|none>, Q4 <x1 only|add ColonVQA>, Q5 <later|drop>.`
