# Exploratory additions after the futility stop (2026-11-20)

**Status:** decided 2026-11-20 (research lead). Pre-registration section 10, amendment of
2026-11-20. Fixed before any test-split result of a pruned arm was committed or opened (the
seed-0 B2 test evaluations of RC-E5, RC-E6 and RC-E7 were running).
**Scope:** all three additions are **exploratory**: outside H1-H4, no verdicts, no multiplicity
correction, reported as such. No pre-registered metric, the comparator choice (E5 magnitude) or
the futility outcome changes.

## Why

The futility stop removed the confirmatory claims. Gate 2 (`docs/results/GATE2.md`, validation,
seed 0, B2) showed two things worth measuring on every arm:

- the plain answers of E5 magnitude, E5 activation and E6 say "no" to every mask-positive polyp
  probe, so their plain macro balanced accuracy is 0.500, while E7's is 0.688;
- E7 has the lowest calibration error (ECE 0.081, against 0.25-0.33 for the generic arms).

The operating-point metrics re-tune a threshold per arm on validation, which hides both.

**Definitions used throughout.** *Plain answer:* the one-forward-pass yes/no decision of a probe,
"yes" when log p(yes) >= log p(no) (margin threshold 0; `YesNoScore.predicted`); it is not a
free-text generation. *Operating point:* the pre-registered per-finding threshold on the margin
with specificity 0.95 on the arm's own validation probes (`probe_suites.fit_thresholds`); a finding
with no negatives keeps its plain answers (`operating_point_answers`). *Probe:* one (image,
finding) question; every interval resamples images (all probes of an image together).

## A. Silent collapse and re-calibration cost (CPU, from saved outcomes)

- **Arms:** every arm with saved validation and test outcomes: E2, E1, E10, E10Q, E3 `w4` / `w8` /
  `w8a8`, E3 `awq_w4`, E4, E8, E9, E5 magnitude / activation / Taylor, E5R, E6, E7, E7U, the A-series,
  every budget. The analysis runs on what exists and is rerun as arms arrive.
- **Per finding and macro, on each test (filtered, unfiltered, external when it exists):**
  plain-answer yes rate, sensitivity, specificity and balanced accuracy; the operating-point
  ones; the gap (operating point minus plain); ECE (the pre-registered calibration metric,
  `probe_suites.calibration_metrics`).
- **Re-calibration curve:** the operating threshold refitted by the same rule on n labelled
  validation images, n = 8, 16, 32, 64, 128 and all.
  - *Draws:* images drawn without replacement, stratified by finding presence: the stratum of an
    image is the set of findings present in it; each stratum gets its share of n by largest
    remainder (proportional to its size in the validation split). 1,000 draws per n, seed 0.
  - *Use:* each draw's thresholds are applied to the test split (a finding without negatives in
    the draw keeps its plain answers, the pre-registered rule).
  - *Report:* per n, the median and the 2.5-97.5 % range of sensitivity, specificity and balanced
    accuracy (per finding and macro); the smallest n whose median macro balanced accuracy is
    within 0.02 of the full-validation threshold's. n = 0 is the plain answer; n = all is the
    pre-registered operating point (one value, no draws).
- **Lesion-mask audit at the operating threshold:** the same flip-rate difference as primary
  endpoint 2 (lesion masked minus an area-matched random region masked, on the mask-positive polyp
  probes), but a flip is a change of the operating-point decision (margin against the arm's
  threshold) instead of the plain answer. Reported next to the pre-registered plain-answer
  version, which stays as it is. Labelled exploratory, with the yes-rate confound stated: an arm
  whose plain answer is always "no" cannot flip, so its plain-answer effect is 0 by construction.
  *Data:* this needs the audit's per-probe margins (original, lesion masked, random masked). Until
  this amendment the evaluation saved only the plain-answer flips; from now on it saves the
  margins too (`audit_original_margin`, `audit_lesion_margin`, `audit_random_margin`). For runs
  evaluated before, a mask-audit-only rerun on the GPU writes them; the rerun must reproduce the
  saved plain-answer flips exactly (checked; the control regions are seeded).

## B. What each method forgets (CPU, from saved outcomes)

- **Flips** per arm against E2 seed 0, at the operating point and on plain answers, on the probes
  both share: *forgotten* = E2 correct and the arm wrong; *gained* = E2 wrong and the arm right.
  Broken down by finding, size stratum (small polyps, other polyps; the pre-registered small
  threshold) and mask presence. Counts are per probe; images with at least one forgotten probe are
  reported too.
- **Overlap:** the Jaccard overlap of the forgotten probe sets for every pair of arms, and between
  families (the union of the family's forgotten sets):
  - pruning: E5 (each criterion), E5R, E6, E7, E7U, A1-A6;
  - quantization: E3 (`w8`, `w4`, `w8a8`, `awq_w4`), E4;
  - pruning and quantization: E8;
  - distillation: E1 (the student without distillation; E2 is the reference);
  - native small model: E10, E10Q.
  E9 (the teacher at 4 bit) and E0 are reported against E2 too but belong to no family.
- **Compression-identified exemplars** (after Hooker et al., "What do compressed deep neural
  networks forget?"; Holste et al., MICCAI 2023): the probes forgotten by at least half (rounded
  up) of the compressed student arms present at B2, or at no budget for the arms without one
  (E3, E4); the compressed student arms are the pruning, quantization and pruning-and-quantization
  arms above (not E1, E9, E10, E10Q). Their counts by finding, the small-polyp share and the mask
  area (`probe_area`), against all test probes.
- **Intervals:** an image-bootstrap 95 % interval, 10,000 resamples, seed 0, for every count and
  rate.

## C. AWQ as a second 4-bit post-training quantization arm (GPU)

- **The arm:** E3 `awq_w4`, seed 0, exploratory. AWQ (Lin et al., MLSys 2024) on E2.
- **Settings:** 4-bit weights, groups of 128, the same 128 calibration images as GPTQ (the cleaned
  calibration set of seed 0, amendment of 2026-11-13), decoder linear layers only (the same
  targets as E3 `w4`), everything else as E3 `w4`; then the same validation, tests and benchmark.
- **Library and kernel** (checked 2026-11-20 in the source of the pinned GPTQModel 7.6.0; AutoAWQ
  is archived): GPTQModel 7.6.0 implements AWQ quantization (`METHOD.AWQ`) and has a Qwen2.5-VL
  definition whose decoder targets are q/k/v/o and gate/up/down, as E3's. Its AWQ kernels:
  `AwqTorchLinear` (pure PyTorch, any device, 2-8 bit, groups up to 128) runs on a T4; AWQ Marlin
  needs compute capability 8.0 or more and Machete newer, so neither runs on a T4 (7.5); the GEMM,
  GEMV and ExLlamaV2 AWQ kernels need GPTQModel's compiled CUDA extensions. The arm uses
  `AwqTorchLinear`, the AWQ counterpart of the `TorchLinear` kernel E3 uses (kept on 2026-11-16),
  so the latency comparison with E3 `w4` stays kernel-for-kernel. No new library is pinned.
- **Before the real run:** a smoke (RC-SMOKE job "E3 awq_w4") and a full-size scale check of the
  quantization step on the first 4 decoder layers (RC-SCALE check `awq`).

## Outputs

`results/exploratory/` (CSV and `SUMMARY.md`), from `scripts/exploratory_analysis.py`; E3
`awq_w4` under `results/E3/awq_w4/nobudget/seed0/` like every E3 variant.
