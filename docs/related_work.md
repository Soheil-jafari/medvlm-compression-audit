# Related work and positioning

Written 2026-11-08 (pre-registration section 11 item 13). Every claim about another paper below
is what the team recorded at that date; the bibliographic details are to be checked against the
papers before the paper text cites them.

## Closest prior work on weighted importance

| Work | What it weights | Granularity | Difference from ours |
|---|---|---|---|
| FairPrune (MICCAI 2022, arXiv 2203.02110) | importance balanced across patient groups (fairness) | weights | weights groups of patients, not clinical findings; no lesion-size or lesion-mask evaluation |
| Class-aware pruning (arXiv 2312.05875) | importance per class | structured | classes of an image classifier, not findings probed in a vision-language model |
| LTAP (ICML 2025) | importance aware of the long tail | | long-tail weighting; our A2 (rarity-only weights) is its in-framework analogue |
| D-Pruner (NAACL Findings 2024, arXiv 2405.06275) | importance for a domain (and general ability) | **unstructured** | unstructured, language models; ours is structured (layers, KV groups, channel groups) |
| MixCal (arXiv 2502.18424) | the calibration data for compressing to a domain | unstructured / 2:4 | changes the calibration set, not the importance; not structured |

**In-framework analogues.** A1 (uniform finding weights: the probe effects alone, the
task-aware case) and A2 (rarity-only weights: the long-tail-aware case, LTAP-style) are the
nearest counterparts of these ideas inside our framework, with everything else equal (units,
calibration images, recovery, budgets). E7 against A1 is our pre-specified secondary comparison
(pre-registration section 5).

## Compression of medical multimodal models

Khan et al. 2025 (arXiv 2507.21976) compress LLaVA for medical use: layer removal by output
similarity, then AWQ quantization. Their evaluation scores answers with an LLM judge on 39 test
images and has no per-finding or small-lesion analysis. Our E6 follows their layer-removal
criterion (re-implemented, see below); our evaluation is per finding, at a fixed operating
point, with the small-lesion recall and the lesion-mask reliance as primary endpoints, on two
co-primary test sets.

## What is new here (stated precisely)

1. **Structured pruning of a medical vision-language model** (decoder layers, KV groups, MLP
   channel groups), matched on **measured** peak memory, not parameter counts.
2. **Importance = the drop in the correct answer's probability per clinical finding**, weighted
   by a **pre-registered clinical-priority rule** (inverse prevalence with a cap, times a
   clinical tier), fixed before any result.
3. **Evaluation on small-lesion recall and lesion-mask reliance** (does the model still change
   its answer when the lesion is masked, more than for an area-matched random region), per
   finding, at an operating point fixed on validation.
4. **A matched-memory comparison of pruning, quantization and distillation** in one framework
   with one recovery recipe (E2 distillation, E3 / E8 / E9 post-training quantization, E4
   quantization-aware distillation, E5-E7 pruning; E10 a natively small model).

## Our generic arms are re-implementations

The generic pruning arms are **re-implementations in one framework with identical recovery**
(the same LoRA, distillation from the same teacher cache, the same steps and data), so they
differ from the clinical criterion only in the importance. They are named for what they compute,
never as the original methods:

| Arm | What it is | Not |
|---|---|---|
| E5 `magnitude` | weight magnitude per unit | |
| E5 `taylor` | first-order (Taylor) importance, in the LLM-Pruner style | not LLM-Pruner itself (its grouping, recovery and data differ) |
| E5 `activation` | activation-scaled weight importance (Wanda-style) | not Wanda itself |
| E6 | output-similarity layer removal, in the ShortGPT / Khan et al. style, scored once and removed in one shot (a declared deviation) | not ShortGPT or Khan et al. themselves |

The comparator is the best of these four on validation (pre-registration section 5).

## Not included, with reasons

- **FLAP, SliceGPT:** the comparator is already the best of four standard criteria in one
  framework; more structured-pruning methods would add runs, not a different question.
- **AWQ:** AutoAWQ is deprecated; GPTQ `w4` / `w8` and LLM.int8 cover post-training
  quantization (D08).
- **Visual-token pruning:** a different axis (fewer image tokens at inference, not fewer
  weights); it does not change the stored model and is not matched on the same footprint.
