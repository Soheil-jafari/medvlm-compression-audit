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

## Mixed-precision quantization (E11, E12: exploratory, not built)

HAQ and HAWQ (2019) assign bit widths per layer from a sensitivity measure (hardware feedback,
the Hessian). Our planned E11 does the same with a **clinically weighted** sensitivity: each
decoder layer alone is quantized to 4 bit and the drop in the correct answer is measured with
the same probes, weights and calibration images as E7; the most sensitive layers stay at 8 bit
until the memory target (the midpoint between E3 `w4` and `w8`) is reached. E12 is the same
procedure with an unweighted sensitivity. Both run only if a feasibility check shows that our
evaluation stack runs per-layer bit widths with real low-precision kernels; otherwise they are
reported as not run and listed as future work (pre-registration section 12).

## Not included, with reasons

- **FLAP, SliceGPT:** the comparator is already the best of four standard criteria in one
  framework; more structured-pruning methods would add runs, not a different question.
- **AWQ:** AutoAWQ is deprecated; GPTQ `w4` / `w8` and LLM.int8 cover post-training
  quantization (D08).
- **Visual-token pruning:** a different axis (fewer image tokens at inference, not fewer
  weights); it does not change the stored model and is not matched on the same footprint.

## Reading list (kept from the earlier version of this file, unchanged)

> **Re-check before submission.** Verify titles, author lists, venues and arXiv
> versions against the sources, and search again for newer work. Entries are
> preprints unless a venue is listed. Fields marked TODO were not available when
> this list was written; do not fill them from memory.

| # | Work | Venue / status | arXiv | Relevance to this project | Re-checked |
|---|---|---|---|---|---|
| 1 | Khan et al., *Compression Strategies for Efficient Multimodal LLMs in Medical Contexts* | preprint | 2507.21976 | Basis of our output-similarity layer-pruning baseline (E6, "Khan et al. style"). | TODO |
| 2 | Lin et al., *MoPE-CLIP* (full title TODO) | CVPR 2024 | 2403.07839 | TODO: summarise after reading. | TODO |
| 3 | Holste et al., *How Does Pruning Impact Long-Tailed Multi-Label Medical Image Classifiers?* | MICCAI 2023 | 2308.09180 | Effect of pruning on long-tailed medical classifiers (from the title); related to our rare-finding analysis. | TODO |
| 4 | Khalid et al., title TODO (topic: model pruning, long-tail forgetting and explanation reliability in medical imaging) | preprint | 2609.07803 | Topic: pruning, long-tail forgetting and explanation reliability in medical imaging. | TODO |
| 5 | GHOST-Q, authors and title TODO | preprint | 2609.29999 | TODO: summarise after reading. | TODO |
| 6 | Kim, Kim, Goebel, *When Quantization Preserves Accuracy but Not Evidence* | preprint | 2609.24799 | Accuracy vs evidence under quantization (from the title); related to our "hidden harm" framing. | TODO |
| 7 | Huang et al., title TODO (topic: structural pruning and recovery of large vision-language models) | preprint | 2604.24380 | Topic: structural pruning and recovery of large VLMs; related to E5 and our recovery design. | TODO |
| 8 | Cai et al., *LLaVA-KD* (full title TODO) | ICCV 2025 | 2410.16236 | TODO: summarise after reading; listed as related to our distillation arm (E2). | TODO |
| 9 | Qian et al., *Med-OPD* (full title TODO) | preprint | 2607.16303 | TODO: summarise after reading. | TODO |
| 10 | Chen et al., *GRACE* (full title TODO) | ICML 2026 | 2601.22709 | Template for our quantization-aware distillation arm (E4). | TODO |
| 11 | MedTokenBudget, authors and title TODO | preprint | 2609.30613 | TODO: summarise after reading. | TODO |
| 12 | Colon-X (ColonVQA, ColonPert), authors and title TODO | preprint | 2512.03667 | Source of candidate rare-finding / stress-test datasets. | TODO |
| 13 | Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin, Andrea Frome, *What Do Compressed Deep Neural Networks Forget?* | arXiv preprint (2019) | 1911.05248 | The cost of compression falls on a small subset of examples, hidden by top-line accuracy. Basis of our pre-registered secondary flip analysis (2026-10-31): the test probes whose answer flips between E0 and each compressed arm, by finding and lesion-size stratum. | TODO |

## Paper structure for related work

Planned subsections (assign each work after reading it; TODO):

1. Compression of vision-language models.
2. Medical vision-language models and their compression.
3. What compression forgets.

BibTeX keys are in `paper/references.bib`.
