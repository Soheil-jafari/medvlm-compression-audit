# Project definition

**Working title:** What Compression Forgets: Pruning vs. Quantization vs. Distillation
in Medical Vision-Language Models, and a Clinically Weighted Pruning Criterion for
Preserving Small Findings.

**Team (roles provisional):** Amir (pruning, clinical criterion, efficiency
measurements), Soheil (teacher/student setup, distillation, quantization, recovery
training), Farzan (datasets, masks, evaluation harness, statistics).

## Idea

Take one generative medical vision-language model (image + question -> answer) and
compress it in several ways to the same **measured** memory or latency budget.
Compare which compression method silently damages small-lesion recall, rare-finding
recall, image reliance, calibration and hallucination, even when average accuracy
holds. Our one new method is a clinically weighted structured-pruning criterion.

## Our method: clinical pruning criterion

- **Candidate units j:** decoder layers, attention heads (respecting grouped
  key-value heads), MLP channel groups. Vision encoder blocks optional. Projector kept.
- **Importance:**
  `I_j = sum_{c in C} w_c * mean_{x in X_cal} Delta_j(c, x)`
- **Delta_j(c, x):** drop in log-probability of the CORRECT yes/no answer to a fixed
  finding probe ("Is there a <finding> in this image? Answer yes or no.") when unit j
  is ablated. Alternative variant: change in prediction (KL). Probes use a single
  forward pass, no free generation.
- **w_c:** capped inverse prevalence x clinical importance tier, from a pre-registered
  table (`configs/preregistration/finding_weights.yaml`).
- **Procedure:** probe -> ablate each unit -> score -> rank -> physically remove the
  lowest-scoring units until the budget is met -> shared recovery training -> mask
  audit (does the pruned model still change its answer when the lesion is masked,
  more than when an area-matched random region is masked?).
- **Recovery:** identical for all pruning arms (LoRA fine-tuning + offline
  distillation from cached teacher outputs, same data subset, same steps).

Full definitions: `src/medcompress/compression/pruning/criteria/clinical.py`.

## Experiments

E0-E9 and ablations A1-A6, budgets B1-B3, 3 seeds per training arm. See
`docs/experiment_plan.md`.

**Primary endpoints (pre-registered):** small-finding recall and lesion-mask effect
at budget B2, comparing E7 against the best of E5 and E6.

## Evaluation

- Clinical: per-finding sensitivity/specificity, macro-F1, overall accuracy
- Small findings: recall stratified by lesion area from masks (smallest quartile = "small")
- Rare findings: recall by prevalence group (common / medium / rare), groups fixed in advance
- Image reliance: blank image; lesion masked vs area-matched random region masked;
  another patient's image
- Calibration: ECE, Brier score, risk-coverage curves
- Hallucination / omission / negation errors
- Efficiency on real hardware: parameters, file size, peak memory, cold and warm
  latency, throughput, energy if measurable, plus a check that low-precision kernels
  actually run
- Statistics: paired bootstrap 95% CIs, McNemar per finding, Holm correction, mean
  and SD over 3 seeds

## Still undecided (configurable, never hard-coded)

- Medical domain: lead candidate GI endoscopy; alternative retinal fundus.
- Teacher and student (must share a tokenizer). Candidates: Qwen2.5-VL-7B with a
  published Kvasir-VQA-x1 adapter -> Qwen2.5-VL-3B; Qwen3-VL-8B -> Qwen3-VL-2B;
  MedGemma family.
- Datasets. Candidates: Kvasir-VQA-x1 (main QA), Kvasir-SEG (polyp masks),
  ColonVQA/ColonEval and ColonPert (rare findings, stress tests), EndoBench
  (external test); IDRiD and FGADR for retinal.

## Compute context

Development on Kaggle (2x T4, 16 GB each) and Google Colab; teacher outputs cached to
disk once; LoRA training; a small rented-GPU budget for final timing.
