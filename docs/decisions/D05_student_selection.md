# D05: Student selection (10 -> 5 -> 3 -> 1)

**Status:** answered 2026-10-12 (Soheil). Protocol freeze waits for the Round 0
numbers (`train_examples`, `compute_unit_b`, G2 threshold, image side).

## Decision (2026-10-12)

- **Q1: teacher family only.** Only models from the teacher's family, i.e. sharing
  its tokenizer, are candidates at all (stricter than option C below: no
  other-family model runs in any round). With a Qwen teacher this keeps candidates
  1, 2, 4, 5, 6 and 10; the pool is topped up to 10 with further verified
  architectures from the same tokenizer family. If fewer than 10 exist, Round 1
  starts with all of them and still keeps 5.
- **Q2: keep the size.** Each Round 2 model is the Round 1 checkpoint retrained.
- **Q3: weights accepted.** S = 0.4 probe macro-F1 + 0.3 small-finding recall +
  0.3 image reliance.

The final pool is fixed once D01 (teacher) is decided.
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

## Candidates

One model per architecture family, all at or below about 5B stored parameters so
that LoRA training can fit one 16 GB T4 in fp16 (checked in Round 0).

| # | Model (hub ID) | Stored params | Vision side | Connector / visual tokens | Language model | Tokenizer (vocab) | Licence (hub tag) | What it adds |
|---|---|---|---|---|---|---|---|---|
| 1 | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct) | 3.75B | Qwen ViT, 32 layers x 1280, native resolution | MLP merger | Qwen2.5 decoder, 36 layers, 16 heads / 2 KV heads | Qwen (151,936 rows; the Qwen2.5-VL-7B teacher has 152,064, so ids must be aligned) | qwen-research | Same family as the lead teacher: the safe baseline |
| 2 | [Qwen3-VL-2B-Instruct](https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct) | 2.13B | Qwen3-VL ViT, 24 x 1024, patch 16 | DeepStack: features from ViT layers 5, 11, 17 fed into the LLM | Qwen3 decoder, 28 layers, 16 / 8 | Qwen (151,936) | apache-2.0 | Multi-level visual features (may help small lesions) |
| 3 | [Qwen3.5-2B](https://huggingface.co/api/models/Qwen/Qwen3.5-2B) | 2.27B | ViT, 24 x 1024, patch 16 | Vision built in from pre-training (per release notes, TODO: verify) | Hybrid decoder, 24 layers: 18 linear-attention + 6 full-attention (every 4th) | New Qwen tokenizer (248,320) | apache-2.0 | Non-standard attention; newest Qwen |
| 4 | [InternVL3.5-2B](https://huggingface.co/api/models/OpenGVLab/InternVL3_5-2B) | 2.35B | InternViT-300M, 24 x 1024, 448 px tiles | Pixel shuffle (0.5) + MLP | Qwen3-1.7B, 28 layers, 16 / 8 | Qwen (151,936) | apache-2.0 | High-resolution tiling |
| 5 | [Ovis2.5-2B](https://huggingface.co/api/models/AIDC-AI/Ovis2.5-2B) | 2.57B | SigLIP2 NaViT (so400m, patch 16), native resolution | Visual embedding table (65,536 visual words) | Qwen3-1.7B, 28 layers, 16 / 8 | Qwen (151,936) | apache-2.0 | A different way of turning patches into tokens |
| 6 | [FastVLM-1.5B](https://huggingface.co/api/models/apple/FastVLM-1.5B) | 1.91B | MobileCLIP-L 1024 (FastViTHD hybrid conv encoder) | 2-layer MLP; few visual tokens | Qwen2-family decoder, 28 layers, 12 / 2 | Qwen (151,936) | apple-amlr (research only) | Efficient encoder with few tokens: does it miss small lesions? |
| 7 | [SmolVLM2-2.2B-Instruct](https://huggingface.co/api/models/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | 2.25B | SigLIP, 27 layers, 384 px, patch 14 | Pixel shuffle x3 + MLP (strong token compression) | SmolLM2-1.7B, 24 layers | SmolLM2 (49,280) | apache-2.0 | Heavy visual-token compression, small vocabulary |
| 8 | [LFM2.5-VL-1.6B](https://huggingface.co/api/models/LiquidAI/LFM2.5-VL-1.6B) | 1.60B | SigLIP2 NaFlex 400M, 27 layers, 512 px tiles + thumbnail | 2-layer MLP, downsample 2 | LFM2.5-1.2B hybrid, 16 layers: 10 short-convolution + 6 attention | LFM (65,536) | lfm1.0 (read the terms) | A language model that is not a plain transformer |
| 9 | [MedGemma 1.5 4B-it](https://huggingface.co/api/models/google/medgemma-1.5-4b-it) | 4.30B | SigLIP pre-trained on medical images (per model card) | 256 tokens per image at 896 x 896 (per model card) | Gemma 3 4B decoder | Gemma (config is gated, TODO: verify) | Health AI Developer Foundations terms; gated | Medical pre-training, Gemma family |
| 10 | [Hulu-Med-4B](https://huggingface.co/api/models/ZJU-AI4H/Hulu-Med-4B) | 4.83B | SigLIP-based encoder with 2D RoPE, 27 x 1152, patch 14 | 2-layer MLP; medical-aware visual-token reduction (per paper) | Qwen3-4B dimensions: 36 layers, 2560, 32 / 8 | Qwen (151,936) | apache-2.0; needs remote code | Medical pre-training **and** a Qwen tokenizer |

**Reserves** (replace a candidate that fails Round 0, in this order):
[PaliGemma 2 3B](https://huggingface.co/api/models/google/paligemma2-3b-pt-448)
(3.03B, prefix-LM built for fine-tuning; gemma licence, gated),
[Granite Vision 3.3 2B](https://huggingface.co/api/models/ibm-granite/granite-vision-3.3-2b)
(2.98B, LLaVA-NeXT tiling; apache-2.0),
[MiniCPM-V 4](https://huggingface.co/api/models/openbmb/MiniCPM-V-4) (4.06B;
apache-2.0 tag), [Gemma 4 E2B-it](https://huggingface.co/api/models/google/gemma-4-E2B-it)
(5.12B stored, the name refers to an effective 2B; also takes audio; apache-2.0;
tight on a T4),
[DeepSeek-VL2-Tiny](https://huggingface.co/api/models/deepseek-ai/deepseek-vl2-tiny)
(3.37B mixture-of-experts; deepseek licence; experts make structured pruning harder).

Left out on purpose: other sizes of the same families (one size per family in
Round 1; Round 2 may change size), and anything that cannot train on a T4.

### Tokenizer compatibility with the teacher

Offline distillation in this project uses cached top-k teacher token
probabilities, which needs the same tokenizer (`configs/models/candidate_pairs.yaml`).

| Teacher (D01) | Candidates that can use logit distillation (pending a token-by-token check in S04) |
|---|---|
| Qwen2.5-VL-7B or another Qwen2.5 / Qwen3 family teacher | 1, 2, 4, 5, 6, 10 |
| Qwen3.5 family teacher | 3 |
| MedGemma (Gemma) teacher | 9 |
| none | 7, 8 |

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
| G5 no test exposure | R0 | No documented use of our test images in the model's training data. Hulu-Med's card says it was trained on public data that includes endoscopy, so Kvasir images may be in it; MedGemma's card lists radiology, pathology, ophthalmology and dermatology data and does not mention endoscopy. The team checks before either can win |

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

1. Run the 10 -> 5 -> 3 -> 1 protocol with the 10 candidates and 5 reserves above.
2. **Q1 = A**: all 10 in Round 1, only teacher-compatible models from Round 2 on.
   With the lead teacher (Qwen2.5-VL-7B) that leaves six possible finalists
   (1, 2, 4, 5, 6, 10).
3. **Q2**: keep the Round 1 checkpoints.
4. **Q3**: accept the score above, and fix the G2 threshold and `n1` from the R0
   numbers before Round 1.
5. Report all four rounds in the paper's appendix. The Round 1 table across 10
   architectures is a small result in its own right: which design choices hold up
   on small findings.

## Risks

- **Engineering:** ten families means ten processors and chat templates. Four
  probably need remote code because their configs name custom architectures
  (InternVL3.5, Ovis2.5, FastVLM, Hulu-Med; Hulu-Med's card confirms it).
  Mitigation: a generic loader with per-family adapters for scoring and LoRA only;
  pruning surgery is written only for Round 2 models.
- **Noise in Round 1:** short training can reorder close candidates. Keeping 5 of 10
  leaves margin; CIs are reported.
- **Teacher exposure and contamination:** the medical candidates may have seen
  public endoscopy images (G5).
- **Unusual layers:** if Qwen3.5 (linear attention) or LFM2.5 (convolution layers)
  reach Round 2, our unit definitions for head and channel pruning must be extended.
  G3 catches a model that cannot be compressed by our methods.
- **Time:** about three rounds of Kaggle runs before E1 and E2 can start. Round 1
  does not need the teacher, so it can run while S05 (Gate 1, teacher cache) runs.
- **fp16 stability:** some families are trained in bf16 and may overflow in fp16
  on a T4; R0 checks this per model.

## What changes in the repository once decided

- `configs/student_selection.yaml`: freeze date, weights, `n1`, `b`, G2 threshold,
  image side; mark the rule chosen for Q1.
- `configs/models/*_student.yaml`: the finalist becomes `DECIDED` with a pinned
  revision; the others stay candidates.
- `docs/stages/S05b_student_selection.md` and run cards RC-S0 to RC-S3.
- `docs/decisions_log.md`: the selection protocol freeze, then each round's result.

## Answer format for the inbox

`- YYYY-MM-DD (name): D05 decided: Q1 <A|B|C>, Q2 <keep|resize>, Q3 <accept|weights ...>; protocol frozen.`
