# Data card: EndoBench

> Filled from D02 and run card RC-X1. Facts not yet checked by a team member are marked
> "TODO: verify". No images or patient identifiers.

| Field | Value |
|---|---|
| Name and version | EndoBench; version TODO: verify (RC-X1 step 1) |
| Owner on the team | Farzan |
| Source (verified URL / hub ID) | TODO: verify (RC-X1 step 1); paper: arXiv 2505.23601 |
| Reference (key in `paper/references.bib`) | TODO (added with the related-work pass, S12) |
| Licence and terms | TODO: verify (RC-X1 step 1) |
| Access type (open / registration / restricted) | TODO: verify |
| Domain and modality | endoscopy, images with questions |
| Role in this project | external test (D02), after deduplication against our images |
| Config | `configs/data/endobench.yaml` |

## Content

- Number of images / questions / patients: 6,832 question-answer pairs from 21 datasets, as the
  paper reports (D02); images and patients TODO: verify (RC-X1 step 3)
- Question types and answer format: TODO: verify (RC-X1 step 3)
- Finding labels and mapping to the domain finding list: TODO (agent, after RC-X1)
- Lesion masks (yes/no, which findings): TODO: verify (RC-X1 step 3)
- Patient / procedure IDs available: TODO: verify

## Splits

- Official splits: TODO: verify
- Our use: test only, never trained or tuned on.

## Preprocessing

TODO (loader written after RC-X1 part A).

## Deduplication

- Fingerprint method and threshold: 64-bit difference hash, at most 4 differing bits, the
  same as for our validation and test splits (`data/dedup.py`, RC-D1).
- Images removed as near-duplicates of our images: `results/data/endobench_overlap.csv`
  (RC-X1 step 4).

## Known limitations and biases

- Built from public endoscopy datasets, some of which may also be the source of Kvasir images:
  the overlap count decides how much is left to test.
