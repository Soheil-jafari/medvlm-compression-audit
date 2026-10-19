# D05: Student selection (10 -> 5 -> 3 -> 1)

**Status:** answered 2026-10-12 (Soheil). **Amended 2026-10-12 (Soheil): pick one student**
(see "Amendment" below); the staged protocol is kept for the record and is not run. The freeze
now waits for the Round 0 numbers of the chosen student (image side, G2 threshold, training
length).

## Decision (2026-10-12)

- **Q1: teacher family only.** Only models from the teacher's family, i.e. sharing
  its tokenizer, are candidates at all (stricter than option C below: no
  other-family model runs in any round). With the Qwen teacher (D01) six of the first
  ten stayed and four more Qwen-family architectures were added (table below). If fewer than 10 exist, Round 1
  starts with all of them and still keeps 5.
- **Q2: keep the size.** Each Round 2 model is the Round 1 checkpoint retrained.
- **Q3: weights accepted.** S = 0.4 probe macro-F1 + 0.3 small-finding recall +
  0.3 image reliance.

D01 was decided the same day (T1), and the pool below is final.
**Stage:** S05b (`docs/stages/S05b_student_selection.md`). **Prepared:** 2026-10-12.
**Depends on:** D01 (teacher) for the tokenizer rule; D02 (datasets) for the
validation split; Gate 1 for Rounds 2-3.
**Model facts:** read from each model's Hugging Face API record and `config.json`
on 2026-10-12 (links in the table). Facts not in those records are marked
"per model card" or `TODO: verify`.

## Question

Once the teacher is fixed, which small vision-language model becomes the student
that every compression arm (E1-E9) starts from? The team asked for a staged
selection: compare 10 candidates with different architectures in a simplified
common test, rebuild the top 5 as new models, keep the top 3, then pick one.

## How this memo reads the request

- **"Simplified to be compared with each other"** = the same cheap protocol for all
  10 candidates (same data, same small training budget, same scoring), not cut-down
  copies of the models.
- **"Use the architecture of the top 5 for 5 new models"** = the five winning
  architectures are trained again from their pretrained checkpoints with the full
  student recipe (all training data, distillation from the teacher). The team may
  instead pick another size of the same family for a new model, if the size is
  fixed before Round 2. Building new hybrid architectures (for example one model's
  vision encoder on another's language model) is **not** proposed: the joined parts
  would need their own pre-training, which does not fit Kaggle T4s.
- This is **successive halving** (Jamieson and Talwalkar 2016; Hyperband, Li et al.
  2018): many cheap trials, then fewer and more expensive ones, with roughly equal
  compute per round.

## Candidates (final pool, fixed 2026-10-12 after D01)

The teacher is Qwen2.5-VL-7B with the Kvasir-VQA-x1 LoRA (D01), and only
teacher-family models may compete (decision above), so every candidate uses the
Qwen tokenizer. One model per architecture, all at or below about 5B stored
parameters so that LoRA training can fit one 16 GB T4 in fp16 (checked in Round 0).
Tokenizer column: hub blob hashes of `vocab.json` / `merges.txt` compared with the
teacher's (D01 has the full table); the runtime check is in S04.

| # | Model (hub ID) | Stored params | Vision side | Connector / visual tokens | Language model | Tokenizer vs teacher | Licence (hub tag) | What it adds |
|---|---|---|---|---|---|---|---|---|
| 1 | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct) | 3.75B | Qwen ViT, 32 layers x 1280, native resolution | MLP merger | Qwen2.5 decoder, 36 layers, 16 heads / 2 KV heads | identical files | qwen-research | The teacher's own smaller sibling: the safe baseline |
| 2 | [Qwen3-VL-2B-Instruct](https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct) | 2.13B | Qwen3-VL ViT, 24 x 1024, patch 16 | DeepStack: features from ViT layers 5, 11, 17 fed into the LLM | Qwen3 decoder, 28 layers, 16 / 8 | same vocab and merges; extra special tokens | apache-2.0 | Multi-level visual features (may help small lesions) |
| 3 | [InternVL3.5-2B](https://huggingface.co/api/models/OpenGVLab/InternVL3_5-2B) | 2.35B | InternViT-300M, 24 x 1024, 448 px tiles | Pixel shuffle (0.5) + MLP | Qwen3-1.7B, 28 layers, 16 / 8 | same vocab; merges differ by a header line | apache-2.0 | High-resolution tiling |
| 4 | [Ovis2.5-2B](https://huggingface.co/api/models/AIDC-AI/Ovis2.5-2B) | 2.57B | SigLIP2 NaViT (so400m, patch 16), native resolution | Visual embedding table (65,536 visual words) | Qwen3-1.7B, 28 layers, 16 / 8 | same vocab; merges differ by a header line | apache-2.0 | A different way of turning patches into tokens |
| 5 | [FastVLM-1.5B](https://huggingface.co/api/models/apple/FastVLM-1.5B) | 1.91B | MobileCLIP-L 1024 (FastViTHD hybrid conv encoder) | 2-layer MLP; few visual tokens | Qwen2-family decoder, 28 layers, 12 / 2 | files byte-different (TODO: compare contents) | apple-amlr (research only) | Efficient encoder with few tokens: does it miss small lesions? |
| 6 | [Hulu-Med-4B](https://huggingface.co/api/models/ZJU-AI4H/Hulu-Med-4B) | 4.83B | SigLIP-based encoder with 2D RoPE, 27 x 1152, patch 14 | 2-layer MLP; medical-aware visual-token reduction (per paper) | Qwen3-4B dimensions: 36 layers, 2560, 32 / 8 | vocab byte-different (TODO), merges as Qwen3 | apache-2.0; needs remote code | Medical pre-training |
| 7 | [LLaVA-OneVision-1.5-4B-Instruct](https://huggingface.co/api/models/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct) | 4.74B | RICE-ViT (region-aware), 24 x 1024, patch 14 | Spatial merge 2 | Qwen3-4B dimensions: 36 layers, 2560, 32 / 8 | same vocab; merges differ by a header line | apache-2.0; needs remote code | Region-aware encoder; fully open training data |
| 8 | [MedVLM-R1](https://huggingface.co/api/models/JZPeterPan/MedVLM-R1) | 2.21B | Qwen2-VL ViT, native dynamic resolution | Qwen2-VL patch merger | Qwen2-family decoder, 28 layers, 12 / 2 | same vocab; merges differ by a header line | apache-2.0 | Medical reasoning learned by reinforcement learning (GRPO on MRI VQA, per card) |
| 9 | [SAIL-VL2-2B](https://huggingface.co/api/models/BytedanceDouyinContent/SAIL-VL2-2B) | 2.73B | SAIL-ViT, 24 x 1536, 448 px | Pixel shuffle (0.5) + MLP | Qwen3-1.7B, 28 layers, 16 / 8 | same vocab; merges differ by a header line | apache-2.0 | Wider vision encoder than the other 2B models |
| 10 | [LLaVA-OneVision-0.5B](https://huggingface.co/api/models/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 0.89B | SigLIP so400m, 384 px | AnyRes tiling up to 9 crops + 2-layer MLP | Qwen2-0.5B, 24 layers, 14 / 2 | same vocab; merges differ by a header line | apache-2.0 | Smallest candidate, many tiles per image |

**Reserves** (replace a candidate that fails Round 0, in this order):
[InternVL3-2B](https://huggingface.co/api/models/OpenGVLab/InternVL3-2B) (2.09B,
apache-2.0), [Qwen2-VL-2B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2-VL-2B-Instruct)
(2.21B, apache-2.0), [Qwen3-VL-4B-Instruct](https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct)
(4.44B, apache-2.0; other size of candidate 2),
[Eagle2-2B](https://huggingface.co/api/models/nvidia/Eagle2-2B) (2.21B, cc-by-nc-4.0;
tokenizer files byte-different). NVILA-Lite-2B was left out: its hub page has no
model card or licence.

**Dropped by the teacher-family rule** (verified configs kept for the record, marked
not eligible): Qwen3.5-2B (new 248k tokenizer), SmolVLM2-2.2B (SmolLM2 tokenizer),
LFM2.5-VL-1.6B (LFM tokenizer), MedGemma 1.5 4B (Gemma tokenizer).

Contamination flags for gate G5: Hulu-Med (trained on public data including
endoscopy) and LLaVA-OneVision-1.5 (large open training mix) must be checked for
Kvasir images before either can win.

## Options

**Q1. Tokenizer rule** (the main decision):

- **A. Compatible final only (recommended).** All 10 run Round 1. Only
  teacher-compatible candidates can enter Round 2. The project keeps the logit
  distillation it already planned. If an incompatible candidate wins Round 1 by more
  than its confidence interval, the team is asked once whether to switch to B.
- **B. Sequence-level distillation for everyone.** Students learn from the
  teacher's generated answers instead of its probabilities. Every candidate can win,
  but the E2 and recovery recipes change and the distillation signal is weaker.
- **C. Pre-filter.** Only compatible candidates enter Round 1. Cheaper, but loses
  the architecture comparison the team asked for.

**Q2. Round 2 size:** keep each winner's checkpoint (recommended) or allow another
size of the same family, fixed before Round 2 starts.

**Q3. Score weights:** the proposal below, or different weights. Must be fixed
before Round 1.

## Selection protocol

All rounds use the **validation split only**. The test split is never touched until
the compression runs, so selecting the student cannot leak into the results. The
protocol below is frozen (date in the inbox answer) before Round 1 runs.

| Round | Models | What each gets | Measured | Advance |
|---|---|---|---|---|
| R0 feasibility | 10 (+ reserves as needed) | No training. Load in fp16 on a Kaggle T4, 20 probes, one LoRA training step | Loads (native or remote code), no NaN/inf in fp16, peak memory, latency, visual tokens per image, yes/no scoring works, a LoRA step fits | Pass/fail; a failure is replaced by the next reserve |
| R1 screen | 10 | Zero-shot test, then the same short LoRA fine-tune for all: the same `n1` training examples, same steps, same LoRA rank on the decoder, vision encoder frozen, 1 seed | On a fixed validation subset: macro-F1 over finding probes, small-finding recall, blank-image accuracy drop, memory, latency | Top 5 by score S |
| R2 new models | 5 | Retrained from the pretrained checkpoint with the full student recipe: all training data, distillation from the teacher per Q1, about 2x the R1 compute each, 1 seed | Full validation suite: clinical, small and rare findings, lesion-mask effect, calibration; plus the compressibility gate | Top 3 by score S |
| R3 final | 3 | The exact E2 recipe that the study will use, 2 seeds each | As R2, mean and spread over seeds | Top 1 by score S |
| Final | 1 | Becomes the student for E1 and E2 (S06, 3 seeds) | - | - |

Successive-halving budget: each round costs about the same (10 x b, 5 x 2b,
3 x 2 seeds x about 1.7b). The unit `b` is set after R0 has measured each
candidate's training throughput, so that Round 1 fits one week of the team's Kaggle
GPU quota (TODO: check the current weekly quota per account).

### Gates (pass/fail, checked in the round named)

| Gate | Round | Rule |
|---|---|---|
| G1 runs on the hardware | R0 | Loads, scores and trains one LoRA step on one T4 in fp16 |
| G2 uses the image | R1 | Blank-image accuracy drop above a threshold fixed before R1 (a student that ignores the image has nothing for compression to forget) |
| G3 can be compressed by our methods | R2 | Removing 25% of decoder layers (no recovery) and 4-bit PTQ both run, reduce measured memory, and change validation metrics measurably; used as pass/fail only, never as a ranking score, so the study does not pick a student that looks robust or fragile on purpose |
| G4 licence | R0 | Research use and publication of results allowed (all 10 appear to be; qwen-research and apple-amlr limit commercial use and may limit releasing compressed weights) |
| G5 no test exposure | R0 | No documented use of our test images in the model's training data. Hulu-Med's card says it was trained on public data that includes endoscopy, so Kvasir images may be in it; LLaVA-OneVision-1.5 was trained on a large open mix. The team checks both before either can win |

### Score S (proposal for Q3)

S = 0.4 x macro-F1 (finding probes) + 0.3 x small-finding recall + 0.3 x image
reliance (blank-image drop in R1; lesion-mask effect in R2 and R3), each scaled
0-1 across the candidates in that round.

Clinical metrics are in the score on purpose: the study measures what compression
removes, so the uncompressed student must first be good at small findings and must
use the image, otherwise there is nothing to lose (floor effect). Efficiency is a
tie-breaker, not part of S, because all compression budgets are relative to the
student's own footprint.

Ties: paired bootstrap 95% CI of the score difference between neighbouring ranks
is reported. Advancement follows the point estimate (fixed rule); a final pick
within the CI of the runner-up is broken by: teacher compatibility, then lower fp16
memory, then the simpler licence.

### Fairness controls

Same images, questions and order for every model; same number of training samples
seen; LoRA on the same module types (decoder attention and MLP); vision encoder
frozen; same longest image side (TODO: 448 or 512 px), with each model's own
visual-token count recorded as a covariate; each model's own chat template; greedy
decoding with the same answer limit; yes/no probes scored by log-probability; fp16
for every model, because T4s have no bf16.

## Recommendation

(As written before the decision; the team chose Q1 = teacher family only, Q2 =
keep, Q3 = accept.)

1. Run the 10 -> 5 -> 3 -> 1 protocol with the 10 candidates and reserves above.
2. Q1: originally A (all architectures in Round 1, compatible ones from Round 2).
   The team chose the stricter rule, so all 10 candidates are now Qwen-family.
3. Q2: keep the Round 1 checkpoints.
4. Q3: accept the score above, and fix the G2 threshold and `n1` from the R0
   numbers before Round 1.
5. Report all four rounds in the paper's appendix. The Round 1 table across 10
   architectures that share one tokenizer is a small result in its own right: which
   design choices hold up on small findings.

## Risks

- **Engineering:** ten architectures means ten processors and chat templates. Seven
  probably need remote code because their configs name custom architectures
  (InternVL3.5, Ovis2.5, FastVLM, Hulu-Med, LLaVA-OneVision-1.5, SAIL-VL2,
  LLaVA-OneVision-0.5B; Hulu-Med's and LLaVA-OneVision-1.5's cards confirm it).
  Mitigation: a generic loader with per-family adapters for scoring and LoRA only;
  pruning surgery is written only for Round 2 models.
- **Noise in Round 1:** short training can reorder close candidates. Keeping 5 of 10
  leaves margin; CIs are reported.
- **Teacher exposure and contamination:** the medical candidates may have seen
  public endoscopy images (G5).
- **Unusual parts:** Ovis2.5's visual embedding table and Qwen3-VL's DeepStack
  links change what "a layer" is for pruning. G3 catches a model that cannot be
  compressed by our methods.
- **Time:** about three rounds of Kaggle runs before E1 and E2 can start. Round 1
  does not need the teacher, so it can run while S05 (Gate 1, teacher cache) runs.
- **fp16 stability:** some families are trained in bf16 and may overflow in fp16
  on a T4; R0 checks this per model.

## Amendment (2026-10-12, answered by Soheil: pick one)

**Status:** accepted by Soheil on 2026-10-12 with the words "let's use the pick-one
approach". The protocol above is not run. Amir and Farzan, who asked for the staged selection,
still have to be told.

Soheil's question: the paper asks what compression does to a student, not which student is
best, so why compare ten candidates? The answer is that the comparison is not needed for the
research question. The student only has to meet two conditions:

1. **Something to lose.** It must load and train on one T4 in fp16, use the image (a
   blank-image test), and not be at zero on small findings. Otherwise compression has nothing
   to forget (floor effect).
2. **A short, honest reason for the choice,** stated before any result.

Proposal:

- Student: **Qwen2.5-VL-3B-Instruct**. Reasons: the smaller sibling of the teacher, identical
  tokenizer files (table above), loads in the library without downloaded model code, and the
  unit surgery of stage 04 is tested on its structure.
- Fallback, named now: **Qwen3-VL-2B-Instruct**, used only if the student fails condition 1.
  Its DeepStack links change what a "layer" is for pruning, so it is the second choice, not
  the first.
- Checks: Round 0 of this memo for these two models only (run card RC-S0), plus the
  blank-image gate G2 at the threshold that is fixed before the numbers are read. No score S,
  no ranking.
- The image-use checks also run on the trained E1/E2 students: stage 06 includes the blank
  image, lesion-masked, random-region-masked and other-patient-image tests
  (`image_reliance_check` in `configs/base.yaml`), so a student that ignores the image is
  caught before any compression starts. This is a pass/fail check on one student, not a reason
  to train several.
- Wording in the paper: "chosen because", never "the best student".
- Kept in the repository, unused unless the team asks for it: the ranking code, the rounds
  R1 to R3 and the run cards RC-S1 to RC-S3.
- Training several students would not show that the choice of student matters. That needs
  every student compressed the same way and compared, which is a second research question and
  a decision for the whole team.
- Optional later: a second student (Qwen3-VL-2B) to show whether the findings hold for more
  than one student. That answers a reviewer better than a ten-model comparison and costs one
  more set of E1/E2/pruning runs, so it is decided after the first student's budget is known.

Not decided by this proposal: GPU hours. They are unmeasured until RC-S0 and RC-M1 report.

Applied on 2026-10-12: `configs/base.yaml` and `configs/training.yaml` point at the chosen
student (status stays `CANDIDATE` until RC-S0 passes, then the revision is pinned),
`configs/student_selection.yaml` carries the amendment, run card RC-S0 covers the two models
only, RC-S1 to RC-S3 are kept but not used. Still to do: tell Amir and Farzan.

Applied to the ranking on 2026-10-16: the selection code ran and ranked
`student_selection.candidates`, which still listed the ten models, so `select_student.py --rank R0`
stopped on the first model that was never run (internvl3_5_2b). `candidates` now holds only
Qwen2.5-VL-3B and Qwen3-VL-2B; the ten of the full protocol are kept as the record
`candidates_full_protocol` (not read by the code), and their model configs stay in
`configs/models/`. Round 0 result (RC-S0, Kaggle T4): both models pass gate G1
(`results/selection/R0/ranking.csv`).

## Round 0 numbers and proposals (2026-10-16, agent; the team decides)

Measured by RC-S0 on one Kaggle T4 (15,636,037,632 bytes), synthetic images at the image side
448, from `results/selection/R0/*/metrics.csv`. Both models pass gate G1.

| | Qwen2.5-VL-3B (student) | Qwen3-VL-2B (fallback) |
|---|---|---|
| Visual tokens per image (side 448) | 194 | 142 |
| Seconds per yes/no probe | 0.179 | 0.074 |
| Peak bytes, scoring | 7,843,901,952 | 4,505,061,376 |
| Peak bytes, one LoRA step at one example | 9,140,717,056 | 5,058,956,288 |
| Seconds per LoRA step at one example | 0.74 | 0.56 |

RC-M1 measured the 3B at 0.199 s per probe on the same side; D07 keeps the larger number.

Proposals, using only these numbers (every time below is an **estimate**):

1. **`inputs.image_longest_side` = 448** (not accepted on 2026-10-16: decided after RC-S0b,
   see below). It is the only side measured so far, for the student, the
   fallback and the teacher (RC-M1: 0.314 s per probe, 17,181,603,328 bytes peak over two T4s),
   and the side is shared by all of them. Visual tokens grow about with the image area, so 672
   would give about 2.25 times as many (about 436 for the 3B, an estimate) with more memory and
   time per example; that would need its own measurement first. Whether 448 leaves enough of
   the small lesions is answered by condition 1 above (small-finding recall of the trained
   student not at zero), not by a guess.
2. **G2 threshold = 0.10**, the blank-image balanced-accuracy drop the teacher's Gate 1 already
   uses (D04, `configs/experiments/E0.yaml`). RC-S0 cannot measure it (synthetic images, no
   labels); it is read from the validation evaluation of the trained students (stage 06,
   E1/E2) and fixed before those numbers exist.
3. **Training length (`training.data_subset`, `training.steps` = ceil(data_subset / 16), D06).**
   At 0.74 s per example (the one-example step; the cost at micro-batch 4 is not measured) one
   run costs about data_subset x 0.74 s. The student-side training of the whole study is about
   24 x data_subset examples: E1 and E2 with 3 seeds each (6 x data_subset) and about 72
   recoveries of a quarter each (45 pruning-grid runs and 27 ablation runs, 18 x data_subset);
   E4 is left out (its quantization-aware step is not measured).

   | data_subset | steps | one E1/E2 run | student training of the study |
   |---|---|---|---|
   | 5,000 | 313 | about 1.0 h | about 25 h |
   | 10,000 | 625 | about 2.1 h | about 49 h |
   | 20,000 | 1,250 | about 4.1 h | about 99 h |

   Provisional proposal: **10,000** (accepted on 2026-10-16). Two numbers were still missing: the
   teacher cache costs data_subset x (answer tokens / 4.48 tokens per second, RC-M1), where the
   answer length comes from RC-D1, and the weekly GPU quota per account is not yet checked
   (TODO above). The team picks the size once both are known.
4. **`compute_unit_b` and the Round 1 validation subset size:** not needed. They belong to
   Round 1, which the amendment does not run; they stay `TODO` and unused.

**Memory of the D06 batch (flag, not a change to D06).** At one example the 3B's LoRA step
needs 1,631,448,064 bytes above the loaded weights (7,509,268,992 bytes). If all of that grew
with the number of examples, a micro-batch of 4 would need about 14.0 GB, 90 % of the T4: an
estimate with two errors in opposite directions (part of the extra is the adapter and its
optimizer state, which do not grow; real training examples are longer than a Round 0 probe,
whose text is one short question). So the D06 micro-batch of 4 x 4 accumulation **may not fit
one T4 for the 3B**; for the 2B the same estimate is 7.5 GB (48 %). The first student training
(RC-E1) measures it; if it does not fit, the team decides (for example 2 x 8 or 1 x 16, which
keeps 16 examples per update, or gradient checkpointing, as E4 uses).

### Decisions on these proposals (2026-10-16, Soheil)

- **Training size: accepted.** 10,000 examples per E1/E2 run, 625 steps at the D06 effective
  batch of 16 (`configs/training.yaml`; the teacher cache holds the same 10,000).
- **Image side: not decided.** 448 stays provisional. Reason: the study is about small
  findings; at 448 a polyp of 5 % of the image width is about 22 pixels, smaller than one
  28 x 28 visual-token patch, so a reviewer could say that the input resolution, not the
  compression, removed small findings. The side is decided from a cheap measurement on
  synthetic images (RC-S0b), before Gate 1 or any real-image run, by the rule below.
- **Training-length check added** to the pre-registration (section 6): E2 and E7 with 30,000
  examples at B2, seed 0, next to the main 10,000 runs.
- **G2 threshold:** unchanged, still the proposal of 0.10 awaiting the team.

### Image side: measurement and decision rule (written 2026-10-16, before RC-S0b runs)

RC-S0b measures the student (one T4) and the teacher (two T4s) at the sides 448, 560 and 672 on
synthetic images: visual tokens, seconds per probe, peak memory, the teacher's generation
speed, and one LoRA step of the student at micro-batch 1, 2 and 4. It also prints what the
published teacher adapter and the Qwen processors state about image size ("not stated" where
nothing is).

For each side the agent then projects the GPU hours of the whole study. Inputs from RC-S0b at
that side: the student's seconds per probe `p_s` and per update `u` (16 / m times
`student_lora_step_seconds_mb<m>`, the warm step, at the largest micro-batch m that fits; for
the gradient-checkpointing option, `student_lora_gc_step_seconds_mb<m>` instead), the teacher's seconds per probe `p_t` and
tokens per second `g_t`. Inputs from RC-D1: the validation and test probe counts and the
answer lengths that fix `teacher_cache.max_new_tokens` (`T`).

| Block | Projected hours at one side |
|---|---|
| Teacher cache | 10,000 x (p_t + T / g_t), an upper bound (answers stop at their end token) |
| Gate 1 and the E0 evaluations | the teacher's probe passes over the validation (and test) probes, x p_t |
| E1 and E2 | 6 runs x 625 updates x u |
| Pruning scores (D07) | 108 exactly scored units x 640 probes x p_s per clinical scoring run, for E7 and the ablations that score |
| Recoveries | about 72 runs x the recovery updates x u |
| Evaluations | every run's validation and test passes, x p_s per probe and condition, plus 1,000 free-text answers each |
| Training-length check | the cache for 20,000 more examples, 1,875 updates x u, one scoring run, one recovery, four evaluations |

The team states the GPU hours it has (90 Kaggle GPU hours per week, 30 per person, stated
2026-10-17; a T4 x2 session counts once against the Kaggle quota, measured 2026-10-19) and
picks
the **largest side whose projected total fits**. If the micro-batch of 4 does not fit one T4 at
that side, the D06 amendment proposal applies (micro-batch 2 or 1, accumulation 8 or 16, 16
examples per update, or gradient checkpointing at 4 x 4; D06 memo). Nothing in this rule reads a real image or a result of the
study.

### Image side and G2: decided (2026-10-17, Soheil)

- **Image side 672** (`configs/base.yaml: inputs.image_longest_side`), from RC-S0b run 2
  (`results/smoke/RC-S0b/`, synthetic probes):

  | Side | Student s per probe | Student LoRA step mb1 / mb2 / mb4 | Teacher s per probe | Teacher tokens per s |
  |---|---|---|---|---|
  | 448 | 0.177 | 0.47 s / 0.69 s / 1.35 s | 0.273 | 5.14 |
  | 560 | 0.282 | 0.59 s / 1.07 s / out of memory | 0.421 | 3.50 |
  | 672 | 0.414 | 0.85 s / 1.54 s / out of memory | 0.584 | 2.66 |

  At 672 a polyp of 5 % of the image width is about 34 pixels, more than one 28 x 28 patch.
  Training runs at micro-batch 1 x 16 (the D06 amendment of the same day).
- **G2 threshold 0.10**, the blank-image drop of the teacher's Gate 1
  (`configs/student_selection.yaml`).
- **GPU budget:** 90 Kaggle GPU hours per week, 30 per person. The projected hours of the study
  at 672 are in `docs/schedule.md` (estimates; the evaluation passes and the teacher cache are
  unknown until RC-D1 and RC-E0b).

## What changes in the repository once decided

- `configs/student_selection.yaml`: freeze date, weights, `n1`, `b`, G2 threshold,
  image side; mark the rule chosen for Q1.
- `configs/models/*_student.yaml`: the finalist becomes `DECIDED` with a pinned
  revision; the others stay candidates.
- `docs/stages/S05b_student_selection.md` and run cards RC-S0 to RC-S3.
- `docs/decisions_log.md`: the selection protocol freeze, then each round's result.

## Answer format for the inbox

`- YYYY-MM-DD (name): D05 decided: Q1 <A|B|C>, Q2 <keep|resize>, Q3 <accept|weights ...>; protocol frozen.`
