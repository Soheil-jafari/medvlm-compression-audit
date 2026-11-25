# Decisions log

Dated, append-only. Newest entries at the bottom. Changes to pre-registered items
must be logged here.

## Decisions

- **2026-10-12:** chose the merged A+B direction.
- **2026-10-12:** created the initial repository scaffolding (structure, configs,
  documented stubs, docs, paper outline). No methods implemented.
- **2026-10-12:** fixed the `.gitignore` rule that had kept `src/medcompress/data/` and
  `configs/data/` out of git (CI was failing on import); restored both.
- **2026-10-12:** set up the research loop: definition of done in
  `docs/FINAL_GOAL.md`, stages S00-S12 in `docs/stages/`, live state in
  `docs/agent/STATUS.md`. The loop works only on the `research-loop` branch; the
  team reviews it and merges into `main` through pull requests.
- **2026-10-12 (Soheil):** the student will be chosen by a staged selection after the
  teacher is fixed: 10 architectures compared in a simplified common screen, the top
  5 rebuilt as new models, then top 3, then one. Protocol drafted in
  `docs/decisions/D05_student_selection.md` (answers pending); D01 now covers the
  teacher only.
- **2026-10-12 (Soheil):** D05 answered: students only from the teacher's family
  (same tokenizer), each winner keeps its size, score weights 0.4 / 0.3 / 0.3.
- **2026-10-12 (Soheil):** D01 decided: teacher = Qwen2.5-VL-7B-Instruct
  (revision cc594898) + SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft LoRA (revision 79e77755).
  Optional robustness check at the end: a fine-tuned Qwen3-VL-8B teacher.
  Consequence (D05): the student pool is 10 Qwen-tokenizer models.
- **2026-10-12 (Soheil):** D02 decided: domain = GI endoscopy; main QA = Kvasir-VQA-x1
  (revision 61e41148), validation and test = the two halves of its official test
  split by image; masks = Kvasir-SEG matched by fingerprint; rare findings from
  Kvasir-VQA-x1 classes; external test = EndoBench (licence to verify); retinal kept
  as an optional later second domain.
- **2026-10-12 (Soheil):** D03 decided: MIT licence for the code.
- **2026-10-12 (Soheil):** D04 answered: Gate 1 moderate rule (balanced accuracy at
  least 0.65, blank-image drop at least 0.10, lesion-mask effect CI above zero), mid-grey
  fill (128), teacher cache `top_k` 20. `max_new_tokens` follows the rule in D04 once
  the RC-D1 answer lengths are known.
- **2026-10-12 (Soheil):** no scheduled runs; the research loop runs back to back in
  an interactive session until the team stops it.
- **2026-10-12 (Soheil):** D06 answered, all recommendations: LoRA rank 16 / alpha 32 /
  dropout 0.05; AdamW lr 1e-4, linear decay, warm-up 20, micro-batch 4 x 4, KD weight 0.5,
  temperature 2; recovery = 25 % of the E2 steps; 10 000 bootstrap resamples, 15 ECE bins,
  1 000 free-generation rows. `steps` and `data_subset` follow at the selection freeze.
- **2026-10-12 (Soheil):** D07 partly answered: rank by score per parameter for every
  criterion; first 2 and last 1 decoder layers protected for every arm; MLP channel groups of
  128, 128 calibration images, one-shot removal, 1 negative per positive, signed effects,
  probe batch 4. Open: how units are switched off while scored (cost question, waits for the
  RC-M1 speed) and the budgets. Soheil also asked that nothing heavy runs in the agent session:
  all GPU work stays in run cards for the team.
- **2026-10-12 (Soheil):** D07 section 4 partly answered: the budget metric is peak GPU
  memory during inference, the tolerance is 3 % of the target, and the protocol is batch 1,
  3 warm-up and 5 timed passes, 32 validation images drawn by a fixed seed (the same for every
  arm). Written to `configs/budgets.yaml`. Still open: the timing hardware, B1 to B3 (rule after
  RC-B0), `max_new_tokens` (rule of D04) and the scoring method (not needed before E2 exists).
- **2026-10-12 (Soheil):** asked whether the 10 -> 5 -> 3 -> 1 selection is needed, because the
  paper asks what compression does to a student, not which student is best. Agreed that it is
  not needed for the research question. The numbers 1 to 10 in D05 are the order of listing,
  not a ranking: nothing has been measured. An amendment is proposed in D05 (Qwen2.5-VL-3B
  with Round 0 and the image checks, Qwen3-VL-2B as the named fallback, no ranking); it is
  applied only after Soheil confirms it and Amir and Farzan are told.
- **2026-10-12 (Soheil):** D05 amendment accepted: pick one student. Qwen2.5-VL-3B-Instruct
  with Round 0 (RC-S0) and the image-use checks; Qwen3-VL-2B-Instruct is the named fallback
  and the optional second student. Rounds R1 to R3 are not run. Applied to the base and
  training configs; the training length and the image size follow from the RC-S0 numbers. Amir
  and Farzan are to be told.
- **2026-10-12 (Soheil):** D08 answered: engine `transformers` for every arm; E3 `w4` and `w8`
  with GPTQ (GPTQModel, group size 128), `w8a8` with bitsandbytes LLM.int8 (reported as 8-bit
  with 16-bit outliers); only the decoder's linear layers are quantized; calibration = the same
  128 images as the pruning criteria; E8 and E9 use the `w4` recipe; E4 = option B (4-bit
  learned-step-size quantization trained with LoRA and GRACE's confidence-gated decoupled
  distillation on the cached top-20). A smoke test (RC-Q0) pins the library versions first.
- **2026-10-15 (Kaggle runs RC-M1 and RC-Q0, Soheil):** both passed all their checks.
  RC-Q0 chose the GPTQModel kernel class `gptqmodel.nn_modules.qlinear.torch.TorchLinear` (the
  fastest at `w4` among the classes that passed `w4` and `w8`); `quantization.kernel_class` and
  `quantization.library_versions` are set, and gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned in `requirements/train.txt` with `torchao>=0.16.0` (Kaggle's 0.10.0 breaks peft
  0.21.2). RC-M1 measured 0.199 s per probe for the 3B student (D07 had assumed 0.12 s): the
  proposed scoring mix of D07 section 1 now costs about 3.8 h per seed instead of 2.3 h
  (estimate). D07 section 1 itself is still open. RC-S0 is ready. (Agent, from
  `results/smoke/RC-M1.csv`, `results/smoke/RC-Q0.csv` and the two cards.)
- **2026-10-16 (Kaggle run RC-S0, Soheil):** Qwen2.5-VL-3B and the fallback Qwen3-VL-2B both
  pass gate G1 on one T4 (`results/selection/R0/ranking.csv`). The ranking still used the
  ten-model pool and stopped on a model that was never run; the D05 amendment is now applied to
  the pool (`student_selection.candidates` = the two models, the ten kept as
  `candidates_full_protocol`). peft 0.21.2 needs `optimum>=1.24.0` next to gptqmodel (33 test
  failures without it; tested with 2.3.0), now in `requirements/train.txt`; the full suite then
  passed on the T4 (895 passed). Measured at side 448: 3B 194 visual tokens, 0.179 s per probe,
  9,140,717,056 bytes for one LoRA step at one example; 2B 142, 0.074 s, 5,058,956,288 bytes.
  D07's 2B column now uses 0.074 s (proposed scoring mix about 3.3 h per seed, estimate).
  Agent proposals for the team in D05: image side 448, G2 threshold 0.10, data_subset 10,000
  (provisional); flag that the D06 micro-batch of 4 may not fit the T4 for the 3B (RC-E1
  measures it).
- **2026-10-16 (Soheil, answers to the RC-S0 proposals):**
  1. *Training size accepted:* 10,000 examples per E1/E2 run, 625 steps
     (`training.data_subset`, `training.steps`; `teacher_cache.pool_size` = 10,000). Still open:
     the rounding of the recovery steps (25 % of 625 = 156.25; proposal 157, D06 memo).
  2. *Image side not decided:* 448 stays provisional (`inputs.image_longest_side` stays TODO).
     Reason: at 448 a polyp of 5 % of the image width is about 22 pixels, smaller than one
     28 x 28 visual-token patch, so the input resolution rather than the compression could
     remove small findings. The side is chosen before Gate 1 by run card RC-S0b (sides 448, 560
     and 672 on synthetic images) and the rule written before it runs (D05, "Image side:
     measurement and decision rule"): the largest side whose projected GPU hours fit the hours
     the team has. The D06 micro-batch amendment (2 x 8 or 1 x 16) is a proposal only.
     RC-E0a, RC-E0b and RC-E1 wait on RC-S0b.
  3. *Training-length check added:* pre-registration section 6, secondary and not
     confirmatory: E2 and E7 at B2, seed 0, with 30,000 examples (the first 30,000 of the same
     seeded pool, which contain the 10,000; 1,875 steps); E7@30k starts from E2@30k and recovers
     for 25 % of its steps on the first 7,500 examples. Run card RC-L1 (blocked; code first).
  4. *G2 threshold unchanged:* still the proposal of 0.10, not decided.
- **2026-10-16 (RC-S0b run 1, Soheil):** stopped by a script bug. Every step loaded a fresh
  model in one process and a second 7.5 GB copy stayed on the GPU, so the micro-batch 2 and 4
  out-of-memory results and the crash at side 560 are not valid; valid are the stated settings
  (the adapter was trained in bf16 with gradient checkpointing, `max_pixels` null; the
  processors set no `min_pixels` / `max_pixels`), the side-448 scoring rows and the
  micro-batch-1 step (fits, 9,377,473,536 bytes). Fixes: every (side, step) now runs in its own
  process and any failure is a recorded row; the LoRA step is warmed up, then timed like RC-S0;
  steps with gradient checkpointing are measured too (measurement only; added to the D06
  micro-batch proposal as an alternative, D06 unchanged). Student revision pinned to
  `66285546d2b821cf421d4f5eb2576359d3770cd3`, the revision downloaded on Kaggle (Soheil); the
  fallback Qwen3-VL-2B stays unpinned and RC-S0b prints the revision the hub serves for it.
  RC-S0b stays `ready` for run 2.
- **2026-10-17 (Soheil, from RC-S0b run 2):**
  1. *Image side 672* (`inputs.image_longest_side: 672`), chosen from RC-S0b run 2
     (`results/smoke/RC-S0b/`). At 672 the student shows 434 visual tokens and 0.414 s per probe;
     its LoRA step takes 0.849 s at micro-batch 1 (peak 11,372,899,328 bytes) and 1.541 s at
     micro-batch 2 (peak 13,674,425,344 bytes); micro-batch 4 runs out of memory. The teacher takes
     0.584 s per probe and generates 2.66 tokens per second (peak 17,856,403,456 bytes over two T4s).
     All on synthetic probes. RC-E0a, RC-E0b and RC-E1 no longer wait on RC-S0b.
  2. *G2 threshold 0.10*, the same blank-image drop as the teacher's Gate 1.
  3. *D06 amendment: micro-batch 1 with 16 accumulation steps* (still 16 examples per update),
     gradient checkpointing off, for every training run at side 672 (E1, E2, E4, every recovery,
     RC-L1). Reasons:
     - 2 x 8 left about 2 GB free on synthetic probes, and real examples are longer;
     - one fixed setting for the whole study.
     Measured cost: 16 x 0.849 s = about 13.6 s per update, against 8 x 1.541 s = about 12.3 s for
     2 x 8. E4 had gradient checkpointing on, for memory only; it is now off as well, and its memory
     at 1 x 16 is not measured yet (RC-E4 seed 0 checks it first).
  4. *GPU budget:* 90 Kaggle GPU hours per week in total, 30 per person (Soheil, Amir, Farzan).
     Not known yet: whether a T4 x2 session counts once or twice against the quota (TODO: the team
     checks on Kaggle).
  5. *RC-D1 is run by Soheil* (not Farzan).

  Found by the agent while applying these (iteration 50):
  - The compression arms (E3, E5-E8, A1-A6) had no `model.config`, so they would have stopped
    before loading the E2 student. They now use `components.student`.
  - The first RC-D1 pass writes no positives or prevalences, because the label values of
    `configs/data/kvasir_vqa_x1.yaml` are still empty. A second, CPU-only counts pass on the
    shared data follows after the values are filled from the first pass's answer counts.
  - D07 says A1 and A2 reuse E7's per-probe effects, but the code scores them again. At side 672
    that costs about 7.9 h per run (estimate).
- **2026-10-18 (decided under Soheil's delegation, from RC-D1 run 1):**
  1. *D02 amendment: image-disjoint re-split.*
     - The official Kvasir-VQA-x1 split is by question: 3,821 of its 4,058 test images are
       training images; 15,718 of the 15,955 test rows are on shared images.
     - All 6,449 images are re-split by image, 70 / 15 / 15 %, seed 20261012.
     - Near-duplicate groups (dHash <= 4 bits, transitive) are assigned whole, stratified by
       finding answer and Kvasir-SEG mask.
     - No validation or test image is within 4 bits of a training image (checked); the 5-17
       bit tail is a data-card number.
     - Run 1's splits are discarded; RC-D1 run 2 makes the new ones.
  2. *D01 amendment: our own teacher adapter.* Qwen2.5-VL-7B-Instruct (same revision) + a LoRA
     we train on our training split only, at side 672. The recipe follows the SimulaMet
     `args.json` where it applies. Run cards RC-TT0 (feasibility) and RC-TT1 (training);
     `configs/teacher_training.yaml`.
  3. *Rule 3* (pre-registered before any validation result):
     - setup (a), fp16 over 2 x T4, unless it runs out of memory or is more than 2x slower than
       (b), QLoRA on 1 x T4;
     - N = the largest multiple of 1,000 examples within 15 GPU hours, capped at one pass;
     - examples stratified by question type;
     - quality: Gate 1 passes and macro balanced accuracy >= E1 seed 0 + 0.02, else one
       extension of +50 % of N.
  4. *Appendix item:* the published SimulaMet adapter on our test split, labelled "has seen
     these images in training", and the official split's overlap in the data card.

  Recorded in D01, D02, D04, D06, the configs, RC-D1, RC-TT0, RC-TT1, RC-E0a, RC-E0b and the
  pre-registration draft. Open: whether Kaggle counts a T4 x2 hour once or twice (it changes N
  by a factor of 2; the conservative reading is used until checked).
- **2026-10-18 (Soheil, from RC-D1 run 2), D02 amendment 2: two-stage near-duplicate
  detection,** before any model result.
  - Run 2 (4-bit grouping) missed real duplicates: at 5-6 bits, 3 of 8 pairs were the same
    scene, and at 7-8 bits, 2 to 3 of 8; 1,038 of the 1,931 validation and test images have a
    training image within 8 bits. At 9-12 bits the pairs looked alike only at 9 x 8 pixels, so
    a larger dHash threshold alone would chain unrelated images.
  - Candidates within 12 dHash bits are confirmed by the highest Pearson correlation of the
    central 60 % (32 x 32, a 4 px blur subtracted, shifts up to 3 px). The threshold is chosen
    from a calibration step on the data (RC-D1 section 3a). Pairs within 4 bits stay joined.
    Union-find, then the split as before.
  - The run STOPS if one group holds more than 5 % of the images. Validation and test images
    with a confirmed duplicate in train must be 0. A final contact sheet of the 24 most similar
    cross-split pairs is attached to the report.
  - Data card: no patient IDs in HyperKvasir, so the splits are not patient-disjoint.
  - RC-TT0 does not wait for RC-D1 (synthetic images); RC-TT1 waits for RC-D1 run 3.

  Recorded in D02, the dataset config, RC-D1, RC-TT0, RC-TT1, RC-E0a, RC-E0b, RC-X1, the data
  card and the pre-registration draft.
- **2026-10-19 (Soheil, from the RC-D1 calibration step and RC-TT0), before any model
  result:**
  1. *D02 amendment 2, final values.* `min_correlation` = 0.65, the valley of the bimodal 5-8
     bit distribution (2 of 2,000 random controls reach it); 0.80 missed a same-scene pair and
     joined 6 different-image pairs out of 8.
  2. *No automatic join within 4 bits:* the four 0-4 bit pairs with the lowest crop correlation
     were clearly different images. Every candidate within 12 bits is joined only by the crop;
     the 0-4 bit pairs left apart are counted and shown on the final contact sheet.
  3. *Chaining guard at 10 %* of the images (was 5 %): the largest group jumps above 30 % below
     0.60, which is what chaining looks like. STOP (exit code 7) above 10 %.
  4. *The deciding check* is the final contact sheet of run 3: the 24 most similar cross-split
     pairs by the crop correlation and the 24 closest by the dHash.
     **Pre-registered fallback:** if a person sees same-scene pairs there, near-duplicates are
     detected again with embeddings from a small pretrained vision model, compared by cosine
     similarity, run on CPU; its threshold is calibrated the same way (distribution, contact
     sheet, random controls) before any model result. Not implemented now.
  5. *T4 x2 counts once* against the Kaggle quota (measured: a 17-minute T4 x2 session used 16
     minutes). RC-TT1 uses N = 38,000 (RC-TT0's reading for "counts once").
  6. *Our teacher's LoRA:* the published target modules, with the published rank 16, alpha 64
     and dropout 0.05 (D01 amendment 2). RC-TT0 found the published `target_modules` not equal
     to D06's seven projections as written (`lora_targets_match` 0); the value itself is
     printed by RC-TT1 before training, which stops if a name matches no layer of the language
     model.

  Recorded in D01, D02, the dataset and teacher configs, RC-D1, RC-TT0, RC-TT1, the data card,
  the pre-registration draft, the schedule and every card's session-limit line.
- **2026-10-19 (Soheil, from RC-D1 run 3), D02 amendment 3: embeddings with a rule set from
  human labels,** before any model result.
  - Run 3 (crop rule 0.65) is not accepted. Its final sheet showed same-scene pairs across
    splits that both earlier measures missed: 3 of the 24 highest crop scores (14-32 bits),
    about 5 of the 24 closest by dHash, about 9 of 12 pairs within 4 bits left apart.
  - The pre-registered fallback becomes the primary measure: the cosine of embeddings from a
    small pretrained self-supervised model. The proposal is DINOv2-small; its id, revision and
    licence are verified on the hub by RC-D1-embed. Two views: the whole image and the central
    60 %.
  - Candidates: 20 nearest neighbours per view, dHash within 12 bits, the 5 highest crop
    correlations per image, and run 3's sheet pairs.
  - About 300 candidate pairs, stratified with the border zones oversampled, are labelled by
    two raters on sheets without scores. Kappa is reported. Each of 450 rules gets its
    sensitivity and precision with Wilson intervals.
  - The rule, fixed in advance: the highest sensitivity with precision >= 0.5 within the 10 %
    guard; ties to the higher precision, the simpler rule, then the lower threshold.
  - Run 4 applies it to all pairs. The final sheet bands: highest cosine, highest crop, closest
    dHash.
  - Strict test subset: the most similar training image below the threshold minus 0.05; the
    main comparisons are reported again on it.
  - No patient IDs: a stated limitation. RC-TT1 waits for run 4.

  Recorded in D02, the dataset config, RC-D1-embed, RC-D1-label, RC-D1, RC-TT1, the data card,
  the pre-registration draft, the schedule and STATUS.
- **2026-10-19 (Soheil, from RC-D1-embed), D02 amendment 3 additions,** before any label.
  - RC-D1-embed ran on a Kaggle T4 (commit cb5de07). Run 3 reproduced (`split_sha256`
    4a595fdf...). 6,449 images, two views, in 143 s on the GPU as reported (the committed
    counts file says 156 s, `embed:seconds`). 366,028 candidate pairs; 298
    pairs to label.
  - Self-check cosines: copies 0.937-0.995 (full) and 0.988-0.999 (centre); different images
    0.843-0.937 (full) and 0.879-0.942 (centre). The cosines are compressed into the high
    range; the boundary is likely near 0.95, where the sample has 24 of 298 pairs.
  - Pinned: `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`,
    licence apache-2.0 (the hub record).
  - Top-up: if fewer than 30 labelled pairs lie within +-0.03 of the chosen threshold, 100
    more candidate pairs from that band (seed 20261013, labelled pairs left out). Both raters
    label them; the rule is chosen again once on all labels by the same criterion
    (`topup-labels`, no new embedding run).
  - Disagreements: the third team member decides every pair the raters label differently, and
    every pair of a round whose kappa is below 0.40 (agent's reading). A decided pair is
    positive only if `same_scene`. Kappa before adjudication and the number decided are
    reported.
  - Unsure: primary, not `same_scene`. Sensitivity analysis: not `same_scene` for the
    precision, `same_scene` for the sensitivity.
  - Blinding: no scores for the raters; no talk about labels until both files are committed.
    The files themselves stay private, so each rater commits the file's SHA-256
    (`commit-labels`, `docs/runs/RC-D1-label_commitments.csv`); the scoring checks the files
    against it and the run report records the commit order.

  Recorded in D02, the dataset config, RC-D1-embed (report), RC-D1-label and the
  pre-registration draft.
- **2026-10-22 (Soheil), D02 amendment 4: two raters, no third member,** before any label was
  compared with any cosine or rule, and before the scoring.
  - Facts:
    - both label files are committed by SHA-256 (298 labels each), in one commit (92b78e3);
    - rater 2 labelled pairs 11-298 blind, in letters, converted with the written legend
      T = different, F = same_scene, U = unsure. The legend was once stated the other way
      round; raw agreement was looked at under both readings (0.17 and 0.78), and the team
      confirmed the first written legend;
    - rater 2's labels for pairs 1-10 came after seeing rater 1's (all 10 agree);
    - rater 1 labelled pairs 1-10 and 291-298 after the files were exchanged;
    - the `rater_2` row replaced an earlier one (288 labels, pairs 1-10 empty). This is the
      decision `commit-labels` asks for.
  - Decision:
    - no third member, so no adjudication and no kappa rule;
    - kappa on pairs 11-298 only, reported;
    - primary truth: both raters `same_scene`, `unsure` not; the rule is chosen as in
      amendment 3;
    - reported only: the either-rater truth (the chosen rule's sensitivity and precision, and
      the rule it would choose) and the unsure analysis;
    - top-up unchanged, with both raters and no third member.
  - Code: `score-labels --no-third-member --kappa-pairs 11-298`. Without the flag the
    scoring is amendment 3's.
  - Also recorded: `results/data/near_duplicate_embed.csv` was committed (69799ff). Its
    `embed:seconds` is 156.28, not the 143 s reported from the printout; its other counts
    agree with the report.

  Recorded in D02, the pre-registration draft, the data card, RC-D1-embed (report) and
  RC-D1-label.
- **2026-10-28 (Soheil), D02 amendment 5: RC-D1 run 4 not accepted; a rule sweep,** decided
  after the deciding check and recorded as post hoc.
  - Run 4 (commit 5e08c51, rule `full:0.93` from the both-raters truth) passed every automatic
    check: val/test confirmed duplicates 0; largest group 370 (5.7 %); `split_sha256`
    4bc426fe....
  - Its deciding check failed: many cross-split pairs on the final contact sheet are the same
    scene (consecutive frames of one video). The reviewer counted, from a screenshot: band 1
    about 17 of 24, band 2 about 4 of 24, band 3 about 13 of 24. **Run 4 is not accepted.**
  - Likely cause (recorded as such): the primary truth needed both raters to say
    `same_scene`; rater 2's first written legend called the code "identical", narrower than
    `same_scene`, so near-identical frames counted as negatives and pushed the threshold up.
  - Decision:
    - truth = `same_scene` for either rater; its rule by amendment 3's criterion, read from
      the committed labels file: `full:0.92` OR crop >= 0.80;
    - sweep of `full:T` OR crop >= 0.80, T in {0.92, 0.90, 0.88, 0.86, 0.84}, each split as
      `prepare` builds it, one private three-band sheet per rule within the 10 % guard;
    - selection: the highest T within the guard whose three bands show no same-scene pair,
      judged by the team, with the counts per band per T recorded. If none is clean, stop;
    - then run 4b with `prepare --rule`, its sheet checked again;
    - the paper reports the label-chosen rule, its failure, the sweep and the selected rule;
      the strict test subset stays a sensitivity analysis.
  - Code: `prepare_data.py rule-sweep`, `prepare --rule` and `counts --rule` (only a rule of
    the committed sweep within the guard).
  - Also recorded: the committed labels file has the kappa on pairs 11-290, which settles
    amendment 4's open question.

  Recorded in D02, the pre-registration draft, the data card, the paper's data TODO and the
  RC-D1 card.
- **2026-10-28 (Soheil), D02 amendment 6: the sweep's only rule within the guard is not clean;
  a held-out filter,** recorded as post hoc and fixed before any new sheet is seen.
  - The sweep (commit 8268bef), as reported:
    - `full:0.92:or_crop:0.80` within the guard: largest group 447 (6.9 %), `split_sha256`
      becacdae..., val/test confirmed duplicates 0, images 4,513 / 968 / 968;
    - 0.90, 0.88, 0.86 and 0.84 over the guard (largest group 23.4 %, 41.6 %, 54.0 %,
      70.8 %).
  - The 0.92 sheet is not clean: band 1 about 12 of 24 same-scene pairs, band 2 about 3,
    band 3 about 1. Per amendment 5, the sweep stops.
  - Decision:
    - groups and split of `full:0.92:or_crop:0.80`, unchanged;
    - after the split, remove from val and test every image whose nearest training image has
      a full-view cosine >= F, or a crop >= 0.70, or lies within 4 dHash bits with a crop
      >= 0.50. Train is never touched, no image moves, and val/test pairs are only counted;
    - filter sweep: F in {0.90, 0.88, 0.86, 0.84, 0.82, 0.80}, with the kept counts and one
      private sheet of the closest remaining train/val and train/test pairs per F;
    - selection: the highest F with no same-scene pair in its three bands (judged by the
      team, counts per F and band recorded), if the filtered test keeps >= 50 % of the test
      images. Else stop;
    - primary test = the filtered test; the unfiltered test and the strict subset are
      sensitivity analyses.
  - Code: `prepare_data.py filter-sweep`, and `prepare` and `counts` with
    `--heldout-filter F` (only an F of the committed filter sweep, with its rule, at or above
    the floor).
  - Agent's readings, to confirm:
    - the dHash condition needs one training image within 4 bits and with crop >= 0.50;
    - the strict subset is taken within the filtered test.

  Recorded in D02, the pre-registration draft, the data card and the RC-D1 card.
- **2026-10-28 (Soheil): the two readings of amendment 6 confirmed; the unfiltered test
  built.**
  - Confirmed:
    - the dHash condition means one training image meeting both (<= 4 bits and crop >= 0.50);
    - the strict subset is taken within the filtered test.
  - Built (the open item):
    - `prepare` with the filter writes a second private test file, `test_unfiltered.json`:
      the filtered test with the removed test images added back;
    - the evaluation scores it as `test_unfiltered` when asked
      (`-o evaluation.split=test_unfiltered`), under the same lock, into its own files;
    - every test evaluation of the run cards scores both tests;
    - `09_compare_arms.py --unfiltered` repeats the comparisons on it;
    - the paper keeps both sensitivity analyses out of the primary tables, with their own
      macro names. This also fixes a macro-name collision that a metric on two splits would
      have caused.
  - The rule sweep's counts file is committed (9d5e2e3); its counts agree with the values
    reported for amendment 6.

  Recorded in D02, the pre-registration draft, the data card and the RC-D1 card.

- **2026-10-28 (Soheil), D02 amendment 7: F = 0.86, a reduced-leakage internal test, an
  external test as co-primary,** recorded post hoc, before any model result.
  - The filter sweep's sheets (team's counts, band 1 / 2 / 3):
    - F 0.90 about 3 / 0 / 0;
    - F 0.88 about 5-7 / 0 / 0;
    - F 0.86 about 4-6 / 0 / 0;
    - 0.84, 0.82 and 0.80 are below the 50 % floor and not judged.

    None is clean. The remaining pairs are frames of the same examination with the camera
    moved, not copies.
  - Decision:
    - F = 0.86, the lowest F that meets the floor. The test keeps 512 of 968 images and
      12,067 of 21,728 QA rows; validation keeps 390 of 968 images;
    - the internal test is reported as a "reduced-leakage" test, with these residual counts;
    - limitation: no image-level method removes same-examination pairs without patient or
      video ids;
    - co-primary: an external test set from another centre, PolypGen and/or GastroVision
      (licences to check). Run card RC-X2 is planned; nothing is downloaded;
    - every model is scored on four tests (filtered, unfiltered, strict subset, external), and
      the conclusions are reported for each.
  - Agent's notes:
    - at F = 0.86 the strict subset is the whole filtered test (512 of 512), so it repeats the
      filtered-test results;
    - a co-primary test needs lesion masks for the two primary endpoints;
    - GastroVision may share a hospital with HyperKvasir (the agent's recollection,
      unverified);
    - EndoBench is set aside;
    - code still to build: the external test, and the verdicts per test.

  Recorded in D02, the pre-registration draft, the data card, the RC-D1 card, RC-X2 (new),
  RC-X1 and the run-card README.

- **2026-10-29 (Soheil): RC-D1 run 4b done; the strict subset dropped; RC-TT1 ready.**
  - Run 4b (saved version `notebookd7588ea09b`): rule `full:0.92:or_crop:0.80`, F 0.86.
    - groups 4,122, the largest 447 (6.9 %); 0 val/test confirmed duplicates;
    - val 390 images (11,598 QA rows), test 512 (12,067), masked test kept 58.5 %;
    - `split_sha256` 54dfa70a..., `test_unfiltered_sha256` adf50c68...;
    - bundle 8,511 files, verify OK.

    Shared data: private Kaggle dataset `medcompress-data`, shared with Amir and Farzan. The
    committed `summary.csv` (38a20dc) agrees with the report. RC-D1 is done.
  - The strict subset is 100 % of the filtered test at F = 0.86, so it is no longer a separate
    analysis. Three tests remain: filtered, unfiltered and external.
  - The split hash is pinned (`configs/data/kvasir_vqa_x1.yaml`) and checked whenever shared
    data are attached.
  - RC-TT1 is made ready on `medcompress-data`:
    - it verifies the shared data and checks the pinned split before training;
    - it prints the published target modules before the model loads;
    - it saves its state every 81 updates (about 30 minutes, an estimate) to the version's
      output;
    - `--stop-after-hours` ends a session cleanly on a saved step, for the one resume.

  Recorded in D02, the pre-registration draft, the data card, STATUS and the RC-D1, RC-C1 and
  RC-TT1 cards.

- **2026-10-31 (Soheil): RC-TT1 done; the evaluation settings frozen (pre-registration
  amendment, before any model result); PolypGen first; a flip analysis after Hooker et al.**
  - RC-TT1 (T4 x2, two sessions): session 1 stopped on a saved step at 1,698 of 2,375 after
    11.0 h; session 2 resumed from 1,698 and finished. 38,000 examples, final loss 0.457, 15.17
    h of training in all (`teacher_train_seconds`). Adapter: 4 files in
    `handoff/E0_teacher_adapter`, the private Kaggle dataset `e0-teacher-adapter`, shared with
    Amir and Farzan. Validation and Gate 1 waited on the open settings. RC-TT1 is done.
  - Frozen (from the committed `summary.csv`; `docs/preregistration.md` section 11):
    - findings `polyp`, `ulcerative_colitis`, `oesophagitis` (train / test 810 / 106,
      691 / 98, 546 / 25); barretts, short-segment barretts and hemorrhoids excluded (test 0,
      1, 0; a limitation); display names from the dataset's wording;
    - all three common (17.9 %, 15.3 %, 12.1 %): no rare group, the rare-finding analysis
      dropped from the internal data (a limitation), per-finding results kept; no cut point
      set;
    - small-finding thresholds: the training quartiles of the polyp mask area (0.05676,
      0.11336, 0.19932); about 21 of the 86 masked test images are small (an estimate); the
      count and the interval are reported;
    - `max_new_tokens` by the D04 rule, computed in code from the summary (46), for the
      evaluation, E0's cache and the budget protocol;
    - mask findings `[polyp]`.
  - External test: PolypGen first (masks, many polyps); RC-X2 plans its licence check, probes,
    strata with our quartiles and the overlap check. Nothing downloaded.
  - Secondary analysis (Hooker, Courville, Clark, Dauphin, Frome, arXiv:1911.05248): per
    method and budget, the test probes whose answer flips between E0 and the compressed model,
    over findings and size strata, with image-bootstrap intervals; from the saved outcomes.
  - RC-E0a and RC-E0b are ready.

  Recorded in the pre-registration, D02, D04, the data card, STATUS, the configs, the
  related-work section and the RC-TT1, RC-X2, RC-E0a, RC-E0b, RC-R1 and RC-C1 cards.

- **2026-10-31 (Soheil): probe wording fix, before any model result** (RC-E0a had not run).
  Template "Is there {finding} in this image? Answer yes or no."; display names with their own
  article: "a polyp", "ulcerative colitis", "oesophagitis". Free-text answers are still
  searched for the dataset's wording (`domain.finding_names`). Recorded in the freeze
  amendment (pre-registration section 11, item 8), D02, the paper's method section and RC-X2.

- **2026-11-01 (Soheil): RC-E0a and RC-E0b done.**
  - Gate 1 passes (`gate1_pass` 1): balanced accuracy 0.803 (>= 0.65), blank-image drop 0.303
    (>= 0.10), lesion-mask effect 0.360 (CI 0.267-0.465, above 0). The verdict proposal is in
    `docs/results/E0.md`; **[team]** confirm it here.
  - E0 validation: 681 probes (227 labelled images x 3), macro balanced accuracy 0.803
    (0.768-0.838); the negatives are other abnormalities only (no "none" answer exists).
  - Teacher cache: one session (first 104 answers in 86.7 s, 2.3 h projected), manifest
    **`d6ba0cb9d166cb2ae9a5dd8af5a1bbd2eebba904df0ccd795aded0d65b4f15d8`**, the private dataset
    `e0-teacher-cache` shared with Amir and Farzan. The hash is pinned in
    `configs/training.yaml` and `configs/recovery.yaml`; every reader refuses another cache.
  - Rule 3's quality part is read when E1 seed 0 is committed; E2's training checks it.
- **2026-11-02 (Soheil): per-finding AUROC and "yes" rate, a pre-registered secondary metric
  for every model** (before any student or compressed result; pre-registration section 11
  item 9). The AUROC of the probe scores with an image-bootstrap interval, the "yes" rate and
  the class counts per finding, next to sensitivity and specificity, so a method that only
  shifts the yes/no bias can be told from one that loses discrimination. E0's from its saved
  validation outcomes on CPU (`scripts/rescore_probes.py`, RC-E0a section 3b).
- **2026-11-08 (Soheil and Amir): the seed plan, a pre-freeze amendment** (pre-registration
  section 11 item 12; replaces item 11; written before any compression result, seed 0
  included; nothing is frozen by it). (1) Main-claim tier, 3 seeds: E2, E7 at B2, the
  comparator at B2. (2) Every other arm seed 0 only, exploratory: the other E5 / E6 arms, all E3
  variants, E4, E8, E9, every arm at B1 and B3, A1-A6, the training-length check; open [team]
  item: E3 `w4` at seeds 1 and 2 (about +5.4 GPU h) only if 4-bit enters a headline claim. (3)
  Never 2 seeds (the majority answer counts a 1-1 split as wrong). (4) The comparator on
  validation, seed 0, at B2, among E5:magnitude, E5:activation, E5:taylor and E6: (i)
  `op_small_finding_recall`, (ii) the operating-point recall over all mask-positive polyps,
  (iii) the small polyps' mean margin above the arm's own `op_threshold:polyp`, (iv) the name;
  only that arm runs seeds 1 and 2 (replaces "averaged over its 3 seeds"; E1 and E2 both score
  13 of 13 on validation, so (i) alone can tie). (5) Futility-only stopping: seeds 1 and 2 of
  the main tier run unless E7 at B2 ranks below the comparator by (4); a futility-only rule does
  not raise the false-positive rate. (6) A single-seed difference smaller than E7's spread over
  its 3 seeds at B2 is reported as within run-to-run noise. (7) H1-H3 confirmatory on the
  3-seed tier, H4 exploratory on seed 0.
  - Rationale (Soheil's analysis and study doc): E1's seed spread at the default cutoff is about
    the sampling noise of 13 polyps (small-polyp recall SD 0.118 against about 0.126; AUROC SDs
    polyp 0.0004, oesophagitis 0.007, ulcerative colitis 0.049 against SEs 0.0048, 0.030,
    0.039); the yes/no rate varies more (SD 0.104 against 0.055) and the operating point re-fits
    it; the power simulation favours more small polyps (the external test) over more seeds (1
    seed at 100 small polyps detects about 18-22 points, 3 seeds at 21 polyps about 22-23).
    Sources: the NeurIPS paper checklist, Bouthillier et al. 2021, Jordan (ICLR 2024), Varoquaux
    and Cheplygina 2022, Hooker et al. 2020. GPU (estimates, recoveries at the measured 27.2 s
    per update): the main tier's extra seeds about 38 h; the whole plan about 190 h (Soheil)
    against about 360 h or more with 3 seeds everywhere at B2; the itemised schedule gives about
    236-243 h, the difference being mainly the ablations' evaluation passes and RC-L1.
  - Agent's check of the numbers against the committed E1 rows: the SDs reproduce; the SEs from
    the committed bootstrap intervals are 0.131-0.145 (small-polyp recall), 0.0023-0.0029
    (polyp AUROC), 0.021-0.024 (oesophagitis), 0.037-0.038 (ulcerative colitis); which yes/no
    rate the SD 0.104 refers to is not stated (`yes_rate:polyp` 0.048, `original_yes_rate`
    0.115). Recorded as given; the conclusion holds either way.
  - Consequences found while implementing (for the team): H2 compares the best generic pruning
    arm with E3 `w8`, which is seed 0 only, so H2 is exploratory as written unless `w8` and the
    matched arm get seeds 1 and 2 ([team], before the freeze); H1 now covers the main tier's
    pruning arms only (the other families are exploratory); under (5) E2 seeds 1 and 2 also wait
    for seed 0's futility check (they are main-tier seeds), so they no longer run in week 3.
  - Code: `src/medcompress/seed_plan.py` (tiers, the ordering, the futility rule, the 2-seed
    refusal), `scripts/choose_comparator.py` (writes `results/seed_plan/comparator_B2.json`),
    new evaluation rows `op_mask_positive_recall` and `op_small_margin_mean`
    (`probe_suites.tie_break_metrics`), the comparison step (main tier on 3 seeds, every other
    arm at seed 0 as `_seed0` rows, 2 seeds refused), the verdicts (`exploratory`), the run
    cards (seeds 1 and 2 wait for the committed choice and stop on futility), the configs' seed
    lists (E3, E4, E8, A1-A6: seed 0), `by_arm.csv` gains a `tier` column, the paper tables a
    Seeds column. Farzan's assignments removed from the schedule; seed 1 of the main tier, RC-E9
    and the E7 / E8 trend have no owner yet ([team]).
- **2026-11-12 (Soheil; the rule of pre-registration section 11 item 13): E10 uses the E2
  recipe.** RC-E10 step 1 passed (0 of 5,238 cache ids differ; 99.9 % of the answers
  re-encoded exactly, 99 % needed), so `training.mode: offline_kd`. The 2B is pinned at
  revision 89644892e4d85e24eaac8bacfd4f463576704203 (the 2026-10-12 `sha_seen`). Applying the
  pre-registered rule, not an amendment.
- **2026-11-11 (Soheil): cards reassigned; RC-G2 in four sessions.** RC-E3 seed 0 runs on
  Amir's account; RC-E9, then RC-E4 seed 0, on account 3; Soheil runs RC-E10 step 1 and RC-G2.
  RC-G2 (about 25 GPU h, estimate) is split into four sessions under Kaggle's 12-hour session
  limit with `run_card.py --only` (RC-G2 section 3); each pruned run is its own hand-off. No
  frozen setting changed: the runs, settings and order of operations are the same.
- **2026-11-11 (agent, Soheil's request): RC-B0 done; budget targets in bytes; RC-G2's
  30-group check coded.** No frozen setting changed. RC-B0 measured E2 seed 0 twice on the T4:
  7,618,471,936 bytes both times, decoder layers 5,549,543,424. The frozen fractions give B1
  6,786,040,422.4, B2 5,953,608,908.8 and B3 5,121,177,395.2 bytes (`budgets.target_value`,
  from the committed `efficiency.csv`; nothing chosen). The 30-group check that D07 and
  correction 1 of item 13 require is now code and RC-G2's last step (30 seeded random channel
  groups scored exactly; Spearman; below 0.7 the agent stops and asks). H2: RC-Q0's `w8` peak
  (5,197,114,880 bytes, smoke test, an estimate) is nearest B3; the frozen rule is applied to
  E3 `w8` seed 0's measured peak once committed; if B3, that arm's seeds 1 and 2 are needed
  (conditional, about 6 GPU h, `docs/schedule.md` section 7).
- **2026-11-11 (agent, Soheil's request): one float format for the CSVs; no committed number
  changed.** Every results and generated CSV is written with Python's shortest round-trip
  `repr` (`utils/io.format_cell`) and read with the correctly rounded `float()`, so pandas 2.3.3
  (Kaggle) and 3.x write byte-identical files; pandas stays unpinned (`pandas>=2.0`). The two
  derived summaries (`results/summary/all_rows.csv`, `by_arm.csv`) were regenerated: the old ones
  had been read through the pandas C parser, which put 1,415 cells one ulp away from the
  committed run files (at most 1.7e-14 relative); the new ones equal the run files exactly. The
  run files themselves, the paper tables and the verdicts are unchanged. An implementation fix,
  not an amendment of the pre-registration.
- **2026-11-08 (Soheil and Amir): THE PRE-REGISTRATION IS FROZEN** (section 9), at commit
  `18f1b47658eb188975dbbf3caa4b76e2e4d611fb`; Amir's written approval received; recorded on
  2026-11-09. `frozen_on` 2026-11-08, `frozen_by` Soheil and Amir and `commit` are set in
  `configs/preregistration/finding_weights.yaml` (the only pre-registered config with a freeze
  record) and in `docs/preregistration.md`. Accepted at the freeze: the three items the agent
  had decided (the C4 sensitivity analysis of the external test; the estimate of about 239-246
  GPU h; Soheil starting account 3's sessions). From now on every change to the document or to
  a pre-registered setting needs a dated amendment (pre-registration section 10) approved by
  both authors and an entry here. The compression stages and the test split are open from the
  freeze on (the guard reads `frozen_on`). RC-B0 is `ready` (attach `e2-seed0-adapter`).
- **2026-11-08 (Soheil and Amir; Soheil delegated the choices to the research lead; Farzan is
  not an author): the pre-freeze decisions** (pre-registration section 11 item 14; nothing
  frozen; `frozen_on` stays TODO). Each with its reason:
  1. Finding weights: `w_c = min(1 / prevalence_c, cap) * tier`, cap 20, normalisation none;
     tiers by rule from the names (action 2.0, follow_up 1.5, other 1.0); polyp action,
     oesophagitis and ulcerative colitis follow_up; prevalence from the RC-D1 run 4b counts
     (810, 691, 546 of 4,513), final weights 11.1432, 9.7967, 12.3983
     (`finding_weights.yaml`). Reason: an explainable priority fixed before any compression
     result; A1 and A2 are its check.
  2. Scoring method: exact for layers and KV groups, first-order for channel groups (E7 config);
     the 30-group check a report (below 0.7 stop and ask). Reason: about 7.9 h against about
     236 h per seed; Molchanov et al. 2019.
  3. Recovery: 157 updates on the first 2,500 examples of the fixed pool, identical for every
     pruning arm (`recovery.yaml`). Reason: the identical recovery is the fairness rule.
  4. A3 without the per-unit division; E7 reused for `A4:n128` and `A5:logprob_drop`.
  5. Headline: macro balanced accuracy at the operating point; McNemar on the 3-seed majority.
  6. H2 confirmatory: E3 `w8` in the 3-seed tier (about 5.4 GPU h more); its matched arm the
     best pruning arm at the budget nearest `w8`'s peak; flagged if that budget is not B2 (it
     then needs seeds 1 and 2). E3 `w4` seed 0. Code: `seed_plan` (E3 `w8` and the H2 arm in
     the main tier), the comparison step (the flag), RC-E3 seeds 1 and 2 (`w8` only, behind
     the futility gate); H1's verdict reads the pruning arms only.
  7. Co-primary tests: supported only if supported on both (intersection-union, no alpha
     split), partly supported on one, reported per test (`verdicts.combine_tests`).
  8. Budgets frozen as 15 / 30 / 45 % of E2's decoder-layer bytes; hardware "Kaggle Tesla T4";
     RC-B0 only converts (code: `budgets.decoder_bytes_removed`, the target = peak minus that
     share of `decoder_layers_bytes`). The timing GPU stays open and does not block.
  9. Secondary analyses: (a) the operating-point recall over all mask-positive polyps (row
     `op_mask_positive_recall`); (b) the headline without ulcerative colitis (new row
     `op_macro_balanced_accuracy_without:ulcerative_colitis`); (c) E2 against E0 (new rows
     `E2_vs_E0`: the small-finding recall difference, next to the flips).
  10. PolypGen: CC BY 4.0 per Ali et al., Sci Data 2023 (research use allowed; citation and
      acknowledgement required); the Synapse terms are Amir's step at the download. Loader and
      overlap check written (`data/polypgen.py`, `scripts/polypgen.py`): clusters by sequence;
      our q1 unchanged; C4 (Oslo) reported apart and the test without C4 a sensitivity
      analysis (agent's addition, before any result).
  11. E1 seed 1 final: no operating-point rows; E1's test (RC-R1) uses seed 0 only (code:
      RC-R1's plan). `e1-seed2-adapter` recorded (Amir).
  12. Team: Soheil and Amir approve the freeze; Farzan left the project; three accounts x 30 h
      = 90 h per week (one lent by Farzan); every hand-off shared to Soheil's account; the
      schedule rebalanced (about 78-88 h per account) with owners for every card.
  13. Freeze-list item 19 confirmed (A1 at B2 only; A6 `heads_channels` without layers); the
      settled items marked handled in the inbox with where they were decided.
  14. E11 (clinical mixed-precision quantization) and E12 (its generic control) documented
      only, conditional on a feasibility check of per-layer bit widths with real kernels.
- **2026-11-08 (Soheil and Amir, delegated to the research lead): pre-freeze additions**
  (pre-registration section 11 item 13; before any compression result; nothing frozen).
  (1) Correction to the freeze list, item 2: "if the check fails, channel groups are scored
  exactly" is dropped (about 236 h per seed). D07's rule stays: RC-G2's 30-group check is a
  report; below a rank correlation of 0.7 the agent stops and asks the team. Basis of the
  first-order scoring of channel groups: Molchanov et al., CVPR 2019 (a correlation above 93 %
  of the Taylor estimate with the true importance). The scoring method itself (D07 section 1,
  `pruning.clinical.ablation_method`) is still open in the config: the proposal is written next
  to it. (2) A1 (uniform weights) joins the 3-seed tier at B2 as a pre-specified secondary
  comparison, E7 against A1, outside the confirmatory verdicts; it re-weights E7's per-finding
  effects (recovery and evaluation only, about 6 GPU h for seeds 1 and 2) and runs under the
  same futility rule. (3) E10, an exploratory baseline: Qwen3-VL-2B (the D05 fallback),
  trained once with the E2 recipe at seed 0 (the E1 recipe if its tokenizer does not match the
  cache's top-20 ids), compared at matched peak memory with the nearest pruned 3B arm and at
  4 bit against E8; run card RC-E10, about 4 GPU h. (4) `docs/related_work.md` and the README
  positioning paragraph.
  - Code: the clinical criterion now keeps its per-finding effects (`unit_effects.csv`) and
    A1 / A2 re-weight E7's of the same seed without loading a model (`pruning/stage.py`; this
    closes the open item "A1 and A2 reuse of E7's effects", about 15.8 h of scoring saved at
    seed 0). The first-order part of E7's scoring now runs one backward pass per finding (each
    probe is still forwarded once); the score is the same weighted sum, by linearity.
    `seed_plan` has a secondary tier (A1 at B2), the comparison step writes `E7_vs_A1` (no
    verdict reads it), RC-A1 runs seeds 1 and 2 behind the futility gate. E10: registry,
    config (`training.mode` open until the check), `scripts/check_tokenizer_match.py`, the
    RC-E10 plan. Still to come for E10 (agent): its 4-bit run and its two comparisons.
  - The hub is not reachable from the agent's session, so the tokenizer check runs on Kaggle
    (RC-E10 step 1).
- **2026-11-08 (Soheil): E1 seed 2's operating-point rows** (RC-E1 section 3c), rescored by the
  agent locally on CPU from Amir's saved outcomes (commit 18c29e5; Python 3.13.16, numpy 2.5.3,
  no torch): the 7 check rows identical, no committed row changed, 21 `op_*` rows added. At the
  operating point (validation, in-sample): macro balanced accuracy 0.750; sensitivity polyp
  0.971, ulcerative colitis 0.000, oesophagitis 0.676; small-polyp recall 13 of 13. E1 seed 1's
  notebook output and adapter were not kept (Farzan's account); Soheil has asked Farzan. The
  amendment text for the case that they are not recovered is held until that answer.
- **2026-11-08 (Soheil): E1 seed 0's operating-point rows** (RC-E1 section 3c, CPU, Kaggle,
  commit b5f6197, torch 2.11.0+cpu, transformers 5.20.2): `metrics.csv` committed unchanged;
  the 7 committed check rows were identical to the saved outcomes. At the operating point
  (validation, in-sample): macro balanced accuracy 0.758; sensitivity polyp 0.990, ulcerative
  colitis 0.082, oesophagitis 0.622; small-polyp recall 13 of 13. Seeds 1 and 2 open (their
  notebook outputs are Farzan's and Amir's).
- **2026-11-08 (Soheil): E2 seed 0 done; the hour estimates use E2's measured step; no design
  change.** E2 seed 0 trained 625 updates in 16,975 s (27.2 s per update, 1.9 times E1's
  mean) on a commit after 8e0b3d7 (the hash is read from the `e2-seed0-adapter` manifest),
  peak 12,656,672,768 bytes; 16 of 625 updates skipped on a non-finite micro-batch loss, the
  adapter finite. Validation: plain macro balanced accuracy 0.553, at the operating point 0.836
  (thresholds fitted on these validation probes). The 13.6 s per update of RC-S0b held for E1
  but not for E2's split step, so every estimate of a distillation step was recomputed with
  27.2 s: E2 about 5.2 h per seed with the validation, a recovery 1.2 h (was 0.6), E4 7.1-14.1 h
  (was 3.5-7.1), RC-L1 14.2 / 3.5 h (was 7.1 / 1.8), section 7 of the schedule 173-194 h in all
  (was 148-159). Cards changed: RC-E1 (measured note), RC-E2, RC-E4, RC-L1, the card index;
  the pruning, Gate 2 and ablation cards hold no hour number and are unchanged. Seeds 1 and 2
  run on the fix commit; seed 1's owner is open.
- **2026-11-07 (Soheil): E2 seed 0's out-of-memory fixed: an implementation change, not a design
  change.** E2 seed 0 ran out of memory on one T4 (14.56 GiB) in the KD step: when the
  teacher's cached answer differs from the ground truth, the step made two forward passes and
  kept both graphs until one backward of `(1 - w) * ce + w * kl`. The loss is linear, so the
  step now backpropagates the CE term first, frees it, then forwards the teacher batch and
  backpropagates the KL term, through the same loss scaler and the same 1 / accumulation
  scaling (`offline_kd._step_backward`). The same loss, the same gradients up to rounding (a test
  checks them), the same logged numbers; no config value, hyper-parameter, seed, data order or
  hash changes, and gradient checkpointing stays off. The one-pass path (teacher answer equal to
  the ground truth) and E1's (`kd_weight` 0) are unchanged. E2 seeds 0, 1 and 2 all run on the
  fix commit; recovery and E4 use the same step.
- **2026-11-07 (Soheil): E2 seed 0's false stop fixed.** E2 seed 0 stopped at once (exit
  code 5: "E0 teacher_cache was made with other settings: student: 'TODO' there,
  'models/qwen2_5_vl_3b_student.yaml' here"; run on 73efbc4). The hand-off check compared the
  student for every kind, but the teacher cache and the teacher adapter are written by E0,
  whose config names no student, and their content does not depend on it. Now the compared
  settings depend on the kind: the teacher's hand-offs compare the teacher, its revision, the
  image side, the data seed and the pool size; every student hand-off (adapter, resume, pruned,
  quantized, outcomes) still compares the student too; a `TODO` is never compared as a value.
  The committed cache (`e0-teacher-cache`, manifest d6ba0cb9...) stays valid and is not
  rebuilt: its files and manifest are unchanged, and the E2 cell is unchanged.
- **2026-11-06 (Soheil): E1 done, rule 3 passes; seed 0's NaN loss; the Kaggle image changed;
  the probe operating point; seeds per budget.**
  - E1 seeds 0, 1, 2 trained (625 steps, 2.34-2.55 h each) and scored on validation: macro
    balanced accuracy 0.592 / 0.546 / 0.557. **Rule 3 passes:** E0's 0.803 >= E1 seed 0's
    0.592 + 0.02 = 0.612. No teacher extension; E2 may start.
  - Seed 0 logged `final loss nan` (seeds 1, 2: 0.548, 0.523); its validation is normal. The
    final loss is the mean of the last 10 step losses, so one non-finite step gives NaN; the
    fp16 loss scaler skips such an update. The run kept no per-step history. A CPU check of its
    adapter (RC-E1 section 3b) comes first; then the team decides whether seed 0 is kept or
    rerun (agent's proposal: keep it if every tensor is finite). The training now counts
    non-finite losses and skipped updates (`metrics.csv`), keeps `train_history.json` with the
    adapter, and stops if the adapter becomes non-finite. The `nan` row was removed from seed
    0's file (the schema needs numbers).
  - **Environment drift:** the teacher (RC-TT1, RC-E0a, RC-E0b) ran on Torch 2.11.0+cu128, E1 on
    2.11.0+cu130 (TensorFlow 2.20 -> 2.21): the Kaggle image changed. Every stage now writes
    the torch build, CUDA, cuDNN, the driver, the GPUs and the transformers, peft and other
    versions to `results/<run dir>/environment.json`.
  - **Probe operating point** (pre-registration section 11 item 10; after seeing E0 and E1 on
    validation, before any compression result and before the freeze): per model and finding,
    the score threshold at specificity 0.95 on validation, applied unchanged to every test.
    Sensitivity, small-polyp recall, the headline accuracy, the flips and McNemar use it; AUROC
    next to it; plain yes/no secondary. E0 needs its validation pass again (about 0.8 h); E1's
    rows come from its saved outcomes on CPU if kept, else 0.5 h of validation per seed.
  - **Seeds per budget** (item 11, before the freeze): 3 seeds at B2 for every arm, seed 0 only
    at B1 and B3 (the trend, stated in the paper); E1 and E2 keep 3 seeds. The compression
    split is recomputed (`docs/schedule.md` section 7).
- **2026-11-04 (Soheil): path roots outside the clone; E0's AUROC with its next evaluation.**
  RC-E0a's cell wrote `outputs/` inside the clone (the configs' relative default) and the
  session ended with `rm -rf medvlm-compression-audit`, so `outcomes.npz` was lost and
  `rescore_probes.py` could not run (`metrics.csv` was copied before and is committed). E0's
  `auroc:`, `yes_rate:`, `n_positives:`, `n_negatives:` rows come with the next E0 evaluation.
  Every card's cell now sets `MEDCOMPRESS_OUTPUT_ROOT`, `MEDCOMPRESS_CHECKPOINT_ROOT` and
  `MEDCOMPRESS_CACHE_ROOT` under `/kaggle/working/` before any script runs, and on Kaggle the
  scripts warn when one of them resolves inside the clone.
- **2026-11-02 (Soheil): inputs found by their manifests.** No cell uses a fixed
  `/kaggle/input` path any more: the shared data are the folder of a `bundle_manifest.json` with
  `private/` next to it, a hand-off the folder of a `bundle_manifest.json` whose path contains
  its name (`scripts/find_inputs.py`; `run_card.py` does the same when `--data-root` and `--from`
  are not given). A missing input stops the cell with a clear message.

- **2026-11-25 (agent; implementation and results, no pre-registered setting changed).** Committed
  unchanged: RC-R1 part 1 (E0 validation with its operating point, E0 test and unfiltered test;
  `teacher_train_peak_gpu_bytes` is RC-TT1's training peak carried over), E5R's and E7U's B2 tests
  (no test number read); E1's `environment.json` from the failed step not committed. RC-R1's E1
  failure: the 2-GPU card showed both T4s to the tied 3B student, whose planner split
  `embed_tokens` and `lm_head`; fixed by per-step GPUs (`Step.gpus`: students on one T4), a
  planner refusal for split tied weights, a smoke check that lanes match the card's GPUs, and a
  resume that skips committed-and-attached test steps. Phase-B smoke S0 passed at a304754: E8,
  A1-A6, E4, E10Q in `smoke_passed.yaml` (A2 / A3 / A4 / A6 ran stage 04; their 05-07 are E7's
  code, run in full by the E7, A1 and A5 jobs). The exploratory analysis writes the audit's
  non-finite counts and the sensitivity version (rule above).
- **2026-11-25 (agent, at the research lead's request; recorded before any effect size of it is
  computed): non-finite masked images in the lesion-mask audit.** E5R's outcomes show non-finite
  masked margins (1 / 86, 3 / 83, 15 / 144 audit probes on validation, test, unfiltered test),
  which make the margin effect undefined for every 3B-student arm; a non-finite masked pass is
  scored "no", so it counts as a flip exactly when the original answer was "yes" (checked on every
  such probe). Rule: a labelled sensitivity analysis drops, per split, every audit probe with a
  non-finite masked or original margin in any arm (union over arms, same probes for every arm,
  both components) and recomputes the flip and margin effects with the same bootstrap; the
  pre-registered version stays. Arms without audit margins are not evaluable in it until their
  audit-only rerun. FBR: a non-finite E2 margin makes the update non-finite (skipped and logged)
  instead of stopping the run. `docs/decisions/audit_nonfinite_2026-11-25.md`.
- **2026-11-24 (agent): RC-R1's smoke passed at ac6310e** (E2 test and unfiltered test, E1 test,
  E0 validation and test; every test stage dropped exactly the listed non-finite images; the
  earlier E0 stop was the unattached teacher adapter). RC-R1 added back to `smoke_passed.yaml`
  and set `ready`; nothing in RC-R1's plan or the evaluation changed between ac6310e and the head.
- **2026-11-24 (agent; implementation and results, no pre-registered setting changed).**
  Committed unchanged: E5R's resume, RC-E7's B2 test, E3 `awq_w4` (no test number read). AWQ's
  peak (+0.580e9 over `w4`) is not an untied embedding (`state_bytes` equal, `lm_head` not
  quantized) but AwqTorchLinear's per-call dequantization through int64 temporaries (0.643e9
  bytes live for an MLP projection, measured on CPU with GPTQModel 7.6.0); kept as measured,
  kernel for kernel. FBR code follows the clarification below (canonical targets, pooled margin,
  the pre-registered parser, the diagnosis printed by cell (ii)). RC-R1's smoke: the teacher
  adapter is selected when attached (tested); the smoke prints each job's hand-offs and any
  missing one. Phase B: the A-series export their outcomes; E10 and E10Q run both test
  evaluations; E4 pauses on `MEDCOMPRESS_STOP_AFTER_HOURS` (exit code 8, card "paused", resume
  hand-off exported; the resume path of E4 fixed to where its training keeps it); sessions of at
  most about 6 h on the three accounts, the smoke first for E8, A1-A6, E4, E10Q.
- **2026-11-24 (research lead; clarification to the amendment of 2026-11-22, before any FBR
  training): FBR's margin, answer target and parser; E2's answer format.** Cell (ii) stopped: E2
  seed 0's greedy first token on the 942 schedule probes was "No" 307 times, another token 635
  times, a yes spelling never. Finding about E2: the plain answer is the sign of the pooled
  first-position margin (no text is parsed), which explains its low plain-answer metrics
  (validation macro balanced accuracy 0.553, audit yes-rate 0.209) beside its good ranking.
  Changes: m is the evaluation's pooled margin with no chosen token (the choice and its stop
  removed); the CE target is the template's canonical "yes" / "no" (single-token check and stop
  kept); decoded answers use the pre-registered parser `clinical.parse_yes_no`; plain and
  decoded changes are read as format and readout results, the cross-fitted rare-finding
  sensitivity stays the primary gate quantity. Decision file section 12; pre-registration
  section 10.
- **2026-11-23 (agent; implementation of the amendment of 2026-11-22, nothing tuned): FBR
  built.** `compression/fbr.py`: the fixed values (`configs/fbr.yaml`, no key of the registered
  recovery repeated; the registered recovery section is read unchanged), the probe schedule
  (training split, one positive and one negative per finding and update, unique images per
  stratum before repeats, seed 0, reuse logged), the registered example order 10 per update, the
  margin (the evaluation's, one function for the student, E2 and the diagnostics), the update's
  loss with the paired term's exact gradient (each member's margin gradient kept, one graph alive;
  checked against a joint graph in both Huber regimes), the sampling-only control, the rebuild of
  each arm's pruned model before recovery from E2 and `removal.json` (checked against the
  committed parameter count and `removal_by_layer.csv`). `compression/fbr_stage.py` and four
  scripts: the probe-answer pass (decoded answers, token ranks), E2's margin cache with the answer
  spelling rule and report, the arm run (smoke, timed trial, pilot; its environment recorded in
  its own `training.json`, not in the registered run's). `evaluation/fbr_analysis.py`: folds from
  labels only, the cross-fitted threshold rule, the two recalibrations, the diagnostic
  (validation only), the gate, image-cluster paired bootstrap with thresholds refitted per
  resample, the mask audit in both versions. Card RC-FBR. After review: the FBR rows file is
  `fbr_metrics.csv` (the aggregator reads `metrics.csv` as a registered run's); cell (ii) uses a
  pruning arm's config (E2's has no recovery section); a gate with missing probe answers is "not
  decidable", not failed; the probe answers are checked against the outcomes; an audit rerun is
  merged only when it reproduces the saved flips; the adapter is checked finite before the merge.
  A training probe's negative is an image with another of the three abnormalities (only images
  with an abnormality value are labelled; the limitation of 2026-11-01), so the amendment's rule
  holds as written.
- **2026-11-22 (research lead; pre-registration section 10, amendment of 2026-11-22, recorded and
  committed 2026-11-23 before any FBR code): secondary study, finding-balanced recovery (FBR).**
  Developed after the futility stop from seed-0 validation results only (Gate 2; the plain-answer
  collapse of E5 magnitude, E5 activation and E6; E5R's oesophagitis drop, sensitivity 0.89 to
  0.24 and AUROC 0.961 to 0.880, E2's values committed, E5R's as reported by the research lead);
  no test or PolypGen result informed it. Exploratory, outside H1-H4. A CPU diagnostic of the
  margins, token ranks, decoded answers and two cross-fitted recalibrations (validation only); the
  recipe (the registered recovery with each update split into 10 ordinary examples and 6 labelled
  training probes, probe cross-entropy plus 0.25 times a paired Huber margin term against E2's
  cached margins; coefficients fixed) and its sampling-only control; a pilot on E5 magnitude, E5
  activation and E7 at B2 with a fixed gate; 30 T4 hours at most. Values the request left open were
  set by the agent and are listed in the decision file (section 11), among them: the ordinary
  examples' loss is the registered 0.5 CE + 0.5 KL (the request wrote CE + 0.5 KL), 1,570 ordinary
  examples instead of 2,512, the fold minimum of 5 and 5, the decoding rule, p = sigmoid(m).
  Specification: `docs/decisions/fbr_study_2026-11-22.md`.
- **2026-11-23 (agent; results committed unchanged, no test number read).** RC-E5 seed 0's B2 test
  evaluations (magnitude and activation, Soheil, 2ecf302; Taylor, account 3, ba77598) and RC-E6's
  (account 3, ba77598): `metrics_test.csv` and `metrics_test_unfiltered.csv` per run, SHA-256
  checked, verified by file list and row counts only (the rows missing against magnitude are
  empty reliability bins). RC-G2 E7U in full (2e795a9): stages 05-07, k = 29 (+2.55 %, within the
  band), its removal set equal to the 29 lowest E7 scores per layer (checked). Run reports renamed
  per session. Summaries regenerated (identical on pandas 2.3.3 and 3.1.1).
- **2026-11-22 (research lead; recorded 2026-11-23): the AWQ grid stays symmetric.** E3 `awq_w4`
  keeps the symmetric 4-bit grid without stored zero points, as built. AWQ's reference setting
  stores zero points; we matched GPTQ's grid (E3 `w4`) on purpose, so the two 4-bit arms differ
  only in the scaling and rounding. Recorded in `docs/decisions/exploratory_additions_2026-11-20.md`
  and the methods text, with the `o_proj` note (no AWQ scale under grouped-query attention; the
  manifest lists it). No setting changed.
- **2026-11-22 (research lead; implemented 2026-11-23): outcome bundles keyed by their runs.**
  Two RC-E5 sessions wrote bundles both named `E5_seed0_outcomes` (magnitude and activation;
  Taylor). The exploratory analysis keys outcome files by their run path inside a bundle
  (`<ID>[/<variant>]/<budget>/seed<k>/<file>`), never by the bundle name, so both are read when
  both are attached; a run found twice is kept once when the copies are byte-identical and stops
  the analysis, naming both, when they differ. A session that runs a chosen subset now names its
  bundles by it (`cards.runs_tag`): RC-E5 / RC-E6 / RC-E7 (variants and budgets), RC-E3 (a subset
  of its pre-registered variants), RC-E8 (budgets), e.g. `E5_seed0_magnitude_activation_B2_outcomes`;
  a full set keeps the plain name. RC-E4 and RC-R1 write one fixed set per card (no subsets);
  RC-G2's names already carry the part. The cards that need hand-offs find them by manifest
  (kind and run), never by name, so nothing reads the names.
- **2026-11-23 (agent; implementation, no setting changed): RC-R1 unblocked up to its smoke.**
  RC-R1 starts with E0's validation pass when E0's committed rows have no operating point (they
  have none: RC-E0a section 3c never ran); its test steps read it. The RC-R1 smoke now fails a test
  stage unless the evaluation reports dropping exactly the listed non-finite images of its split
  (5 filtered, 8 unfiltered), and E2's job also runs the unfiltered test. Cells: RC-R1 section 3b
  (smoke, then run), RC-EX1 section 3c (E5R after its resume, E7U); RC-EX1 lists where each B2
  validation `outcomes.npz` can be (only the RC-G2 notebook versions; no bundle holds one) and
  notes that the B2 test outcomes evaluated at 2ecf302 or later already hold the audit margins.
- **2026-11-20 (agent; implementation of the amendment of 2026-11-20, no pre-registered setting
  changed).** A and B: `evaluation/exploratory.py` and `scripts/exploratory_analysis.py` (CPU, from
  the attached outcome files, writing `results/exploratory/`); card RC-EX1. The evaluation now
  saves the mask audit's margins (`audit_original_margin`, `audit_lesion_margin`,
  `audit_random_margin`); `scripts/save_outcomes.py` scores a split again or reruns the audit
  alone without writing any results file (it compares with the committed rows and the saved
  flips). RC-R1 also exports E1's test outcomes. C: E3 `awq_w4` through GPTQModel 7.6.0 (already
  pinned): `METHOD.AWQ` on a saved copy of the merged E2 with our calibration inputs (GPTQModel's
  own Qwen2.5-VL preparation drops the answer), kernel `AwqTorchLinear` (pure PyTorch, runs on a
  T4; AWQ Marlin needs compute capability 8.0, Machete newer; the GEMM / GEMV / ExLlamaV2 AWQ
  kernels need compiled extensions); checked end to end on CPU (`tests/test_awq.py`); smoke job
  "E3 awq_w4" and scale check `awq` (AWQ on the first 4 full-size layers) before the run; RC-E3
  runs it only with `--variant awq_w4`, with its own hand-offs. The AWQ grid is symmetric (no zero
  points), as `w4`'s GPTQ grid; the AWQ paper's reference setting stores zero points (one recipe
  line; open for the research lead); `o_proj` gets no AWQ scale under grouped-query attention. The
  AWQ wrapper's processor is loaded as `load_model` loads it and the parent's generation config is
  restored (GPTQModel rewrites it). E5R's `removal.json` committed unchanged; session 5's resume
  can run.
- **2026-11-20 (research lead; pre-registration section 10, amendment of 2026-11-20): exploratory
  additions after the futility stop.** Fixed and committed before any test-split result of a
  pruned arm (the seed-0 B2 test evaluations were running). A: plain answers against the
  operating point per finding and macro on each test, their gap, ECE; the re-calibration curve
  (threshold refitted on n = 8 ... 128 and all validation images, stratified, 1,000 draws, seed 0;
  n = 0 the plain answer); the lesion-mask audit at the operating threshold next to the
  pre-registered plain-answer version. B: flips against E2 (forgotten, gained) by finding, size
  stratum and mask; Jaccard overlaps between arms and families; compression-identified exemplars
  (forgotten by at least half of the compressed student arms); image-bootstrap intervals (10,000,
  seed 0). C: E3 `awq_w4` (AWQ, 4 bit, groups of 128, the same calibration and targets as `w4`),
  with GPTQModel 7.6.0's AWQ path and its pure-PyTorch kernel `AwqTorchLinear` (runs on a T4; AWQ
  Marlin needs compute capability 8.0); a smoke and a 4-layer scale check first. All exploratory:
  no verdicts, no multiplicity correction; no pre-registered metric, the comparator or the futility
  outcome changes. The mask audit at the operating threshold needs the audit margins, saved from
  now on; earlier runs need a mask-audit-only rerun. Specification:
  `docs/decisions/exploratory_additions_2026-11-20.md`.
- **2026-11-20 (agent; the pre-registered rules applied, nothing chosen): Gate 2 read, comparator
  chosen, futility stop.** RC-G2's five B2 runs (seed 0, validation) and the 30-group check are
  committed. `scripts/choose_comparator.py` (pre-registration section 5 / section 11 item 12):
  all four candidates and E7 reach small-finding recall 1.000 (13 small polyps); E5 magnitude and
  E5 activation tie on mask-positive recall (1.000); E5 magnitude wins on the small margin (10.72
  against 7.75): **the comparator is E5 magnitude**. E7 ties it on (i) and ranks below it on (ii)
  (0.977 against 1.000, 2 of 86 probes): **the futility rule stops the study after seed 0**. No
  seed-1/2 run takes place (E2, E7 and the comparator at B2, E3 `w8` for H2, A1); seed 0 is
  reported as exploratory, with no confirmatory claim (H3 as `E7_vs_best_seed0`); by Gate 2 the
  article shape is B (`docs/FINAL_GOAL.md`). `results/seed_plan/comparator_B2.json` is identical on
  pandas 2.3.3 and 3.1.1; the cards refuse seeds 1 and 2. Observations reported in
  `docs/results/GATE2.md`, used in no decision: small-finding recall is at the ceiling for every
  arm; the lesion-mask effect is measured on the plain yes/no answer, and E5 magnitude, E5
  activation and E6 answer "no" to every mask-positive probe there, so their effect is 0 by
  construction (E7 0.395, E2 0.058). Any change to the endpoints or the rule would be a post hoc
  amendment for the team, not done here. RC-E5, RC-E6 and RC-E7 set `ready` for seed 0 (their B2
  runs are RC-G2's; only the test evaluations run).
- **2026-11-18 (agent; implementation, no setting changed): the pruned-model load no longer runs
  out of GPU memory, and RC-G2 resumes from saved pruned models.** RC-G2 session 5 (account 3,
  5aa8c69) saved E5R (k = 29, footprint 6.105e9, level check -0.08 %) and then ran out of memory in
  stage 06 loading it (14.01 GiB allocated on a 14.56 GiB T4); the smoke at 5aa8c69 had stage-06
  peaks of 14.09 GiB and E7's died at 14.40. Cause: the projections the surgery replaced and the
  layers it dropped stayed allocated (most likely held by an uncollected wrapper of the merged E2
  LoRA, as in RC-E3's benchmark on 2026-11-15; a CPU test here frees them at once, so this is
  inferred from the GPU numbers, not reproduced), and `safetensors.load_model(..., device=cuda)` built the
  whole state dict on the GPU before copying it. Why session 1 passed at 4f54655 with the same
  code (an estimate from the committed removal sets, all replaced tensors kept plus the new ones
  plus a full state dict): 13.80 GiB (magnitude: MLP of 11 layers sliced, 7 layers dropped) and
  13.12 GiB (activation: 6 and 6), under 14.56; E5R slices the MLP of all 33 unprotected layers:
  15.34 GiB. Fix: the surgery frees every tensor it replaces or drops; `load_pruned` releases the
  GPU cache and streams the weights one tensor at a time (`stream_load`: same keys, shapes and
  dtypes, tied tensors checked, parameter count against the manifest), logging allocated and
  reserved memory before and after. Other whole-checkpoint loads checked: stage 07, E8's parent
  and the A-series load through `load_pruned` (fixed with it); the quantized checkpoints (E3, E8,
  E9, E10Q) are read into host memory and swapped in module by module; the recovery resume loads
  with `map_location="cpu"`; none builds a whole state dict on the GPU. The budget footprints are
  unaffected (the meter excludes memory that is not the model's since 2026-11-15; E5R's measured
  6.105e9 agrees with the parameter estimate 6.113e9); recovery peaks of later runs can be lower.
  Also: the hand-off link tolerates a second lane creating the same link at the same instant
  (RC-SMOKE's `FileExistsError`). Resume: with a part's pruned hand-off attached and its stage-05
  results committed (the hand-off holds the model only), `run_experiment.py` runs only 06 and 07;
  a model without its results or of another removal set stops the run. A grid card skips a run
  only when stages 05, 06 (validation rows) and 07 (`efficiency.csv`) are all committed. Not yet
  run on a GPU: stage 05 with the changed surgery (search, recovery, save); the full-size check
  exercises the surgery itself. Next: the full-size check on E5R, then the resume sessions (RC-G2
  section 3).
- **2026-11-17 (research lead; agreed by Soheil and Amir; pre-registration section 10, amendment
  of 2026-11-17, post hoc and disclosed): the pruning design is locked.** Protection of layers 0,
  1 and 35 for every unit type and every arm stays. E5 (magnitude, activation, Taylor) removes
  only 128-channel MLP groups, the same number k from every unprotected layer (2-34), the k
  lowest-scoring within each layer (LLM-Pruner's same-ratio allocation); a search over k (0-85)
  keeps the smallest k at or below target x 1.03. New exploratory arms at B2, seed 0: E5R
  (seeded random scores, E5's design) and E7U (E7's committed scores, E5's design). E7, E6, A1-A6,
  the recovery, the budgets and the comparator rule are unchanged; the comparator pool stays E5 x
  3 and E6. Session 1's results stay committed as the original pre-registered E5 (appendix,
  outside H1-H4); session 2 (stopped at d9039dc) produced nothing to keep. Disclosed: made after
  session 1's validation results (op macro BA 0.598 and 0.572, E2 0.836; small-finding recall
  0), before any pruned E7 model and any test result; it applies to every generic arm and makes
  them stronger. Reasons and sources: `docs/decisions/pruning_design_2026-11-17.md`. Code:
  `pruning.removal.allocation` (`global` | `uniform_mlp`), `budgets.search_uniform_budget`,
  `recovery.search_budget`, `pruning.scores_from: E7` for E7U (stage 04 never scores for it), RC-G2
  session 5, smoke jobs `E5R` and `E7U`; a recovery resume state saved for another removal set is
  refused; E7's check adds the fit of the exact effect on the estimate and the B2 substitution.
  Offline replay (`scripts/replay_removal.py`; an estimate, E2's measured peak minus 2 bytes per
  removed parameter, within 0.5 % of session 1's measured drops): every arm reaches B1-B3 with no
  protected unit; the uniform arms at k = 13, 29 and 46 of 86 (B2: 33.7 % of each unprotected
  layer's MLP width). The search keeps the smallest k in the band, so k lands at the upper edge
  (+2.7 % at B2); k = 33 would be the first at or below the target. Rule unchanged.
- **2026-11-16 (research lead; agreed by Soheil and Amir; pre-registration section 10, amendment
  of 2026-11-16, post hoc and disclosed): every unit of the protected layers is protected.** No
  unit inside decoder layers 0, 1 and 35 is removed (whole layer, KV group or channel group), for
  every arm (E5 all criteria, E6, E7, E8, A1-A6); it supersedes the clarification of 2026-11-15.
  Prompted by RC-G2 session 1 (171 of 220 and 172 of 281 removed units inside those layers;
  validation op macro BA 0.598 and 0.572, E2 0.836; small-finding recall 0), after the baselines'
  validation results, before any E7 pruned model and any pruned model's test result. It changes
  no score, weight, budget or recovery setting and makes the comparison arms stronger. Code:
  `removal_order` leaves the units out, `validate_removal_set` refuses them, stage 05 stops on a
  removal set that touches them, every pruning run writes `removal_by_layer.csv` (36 rows: the
  share of its removed parameters per decoder layer) and keeps `removed_units_in_protected_layers`
  (now 0). Found while checking the capacity: `plan_removal` counted the decoder's layers from the
  whole-layer units only, so an arm without them (A6 `heads_channels`) protected no layer at all,
  even under the old rule; it now counts layers from every unit type (test). E7's committed
  scores stay (per-parameter normalisation is per unit, so the rest keeps its order; test); E6
  (whole layers only) never removed layers 0, 1 and 35, so its order is unchanged. Capacity of
  layers 2-34 (E7's unit list, the real removal rules; bytes of removed parameters, an estimate of
  the footprint change): with whole layers (E5, E6, E7, A1-A5, A6 `layers_only`) at most 5.087e9
  bytes, 91.7 % of the decoder; A6 `heads_channels` (keeps one KV group and one channel group per
  layer) at most 4.72e9, 85.1 %. Every arm reaches B1 (0.832e9), B2 (1.665e9) and B3 (2.497e9).
  Session 1's B2 files moved unchanged to `results/superseded/inner_units_removable/` (the
  aggregator skips it; `appendix_without_full_protection.tex` reports them); its unit scores stay
  (scores do not depend on the rule). Reruns: session 1 (Soheil), session 2 (account 3), session 4
  (Amir), after a stage-05 smoke of E5 magnitude at the new code.
- **2026-11-16 (Soheil): GPTQ kernel kept.** `TorchLinear` stays (chosen by RC-Q0 under the frozen
  plan); no change. The paper states that the latency of E3 `w4` / `w8`, E8, E9 and E10Q is with
  this pure-PyTorch kernel and that a fused kernel would be faster; memory and accuracy do not
  depend on it (`paper/sections/05_results.tex`). The open row is closed.
- **2026-11-16 (Soheil): the budget selection rule, stated.** The search keeps the smallest
  removal whose measured footprint is at or below target x 1.03
  (`budgets.smallest_prefix_meeting_target`), so arms land from 0 to +3 % above the target (below
  it when the last unit crosses it). No change to the rule. The methods text states it, and the
  results report each arm's measured footprint and its gap to the target
  (`paper/tables/budget_match.tex`, from every budget-matched run's rows).
- **2026-11-16 (Soheil): one hand-off dataset per RC-G2 session.** Kaggle makes one dataset per
  notebook output: `g2-b2-seed0-pruned-s1` (E5 magnitude and activation; saved as
  `g2-b2-seed0-pruned-e5`), `-s2` (E5 taylor and E6), `-s4` (E7). Each model folder is found by
  its own manifest when the dataset is attached and a card reads only the folders it needs
  (`tests/test_inputs.py`); RC-G2's table and the later cards' inputs follow this pattern.
- **2026-11-16 (agent): the recoveries' non-finite steps (open, a CPU check to run).** Both
  session 1 recoveries had 5 steps with a non-finite loss, at the same points (one by step 25,
  two by 50, three by 75, five by 125): both runs visit the pool in the same order (seed 0) while
  their models differ, which points to the data. Whether those are exactly the steps holding one
  of the 24 training images with a non-finite fp16 vision output (RC-CAL0) cannot be decided
  here: the training split's order is in the private shared data. `scripts/check_nonfinite_steps.py`
  (CPU, about 2 min on Kaggle) rebuilds each run's batch order and lists the steps that hold a
  listed image, with the cumulative counts the log prints. From now on each recovery also writes
  the indices of its non-finite and skipped steps (`recovery_steps.json`, indices only), so later
  runs can be compared step by step. If they match, nothing changes (the frozen recipe skips the
  step; the paper reports the count). Expected outcome (an estimate): a **subset**, not an exact
  match. Each run had exactly 5 non-finite micro-batches (one example per step), while 24 listed
  images of 4,513 predict about 13 listed rows in a 2,500-example pool (E2 showed the same ratio:
  16 non-finite steps against about 50 listed rows). Decision rule: the steps are explained by
  the listed images if every 25-step window in which the log shows a new non-finite step holds at
  least one step with a listed image; `rows_in_pool` against 5 tells a subset from an exact match.
  Session 2's recoveries use the same pool and order (seed 0), so their `recovery_steps.json`
  gives the exact step indices for session 1 as well. The scan file is not in the repository
  (it holds image ids); it is attached as a private dataset for the check.
- **2026-11-16 (agent): the footprint jump of the scale checks is the check's own selection, not
  the model.** The check measured slices of one finished removal set, and a removal set lists
  whole layers after the inner units they absorb: units 86-103 of the 275-unit set were six
  whole layers (2-7), 1.02e9 bytes, matching the measured drop (7.477e9 to 6.450e9 at 4f54655).
  The real search measures prefixes of the removal order, one accepted unit at a time; with E5
  magnitude's committed order the largest single step removes 0.116e9 bytes (a layer absorbing
  its last inner units), and every benchmark peak sits about 0.1 GB above the state bytes (no
  transient that a unit's removal shrinks). B1's band is 6.583e9 to 6.990e9; the search keeps the
  first prefix at or below 6.990e9, at most 0.116e9 lower, so at or above 6.87e9, inside the band:
  no B1 run is at risk on this evidence. A run outside the band would still be reported and
  marked (`budget_within_tolerance` 0, D07). The check now measures prefixes of the order.
- **2026-11-15 (agent; implementation, no setting changed): the efficiency measurement and the
  budget search's level.** (1) RC-E3 seed 0 (account 3, 14f69c9) committed unchanged; its accuracy
  stands (`w4` op macro BA 0.808 vs E2's 0.836), its peaks do not: each was about 5.67 GB above
  `state_bytes`, E2's fp16 decoder, because the replaced fp16 projections stayed allocated (they
  were still referenced, likely by uncollected wrappers of the merged E2 LoRA, which covers exactly
  the GPTQ targets) when the peak counter was reset. E9's 19.85 GiB is the same effect. Fixed: the
  swap frees every replaced module's tensors (`ptq.free_replaced`), and the timing sweep's meter
  collects garbage, synchronizes and empties the cache, records the live bytes and subtracts what
  is not the model's own GPU tensors from the peak (`MemoryMeter`; new rows
  `benchmark_live_before_bytes`, `benchmark_excluded_bytes`). E2's reference (7,618,471,936) came
  from a fresh load without a swap: its allocated memory after loading equalled its state (6.99
  GiB, RC-SCALE), so it was measured cleanly. `num_parameters` is now the logical count for every
  quantized class (packed codes as the matrices they stand for). RC-E3 and RC-E9 re-measure the
  benchmark alone from their hand-offs (`--only benchmark`); the new `efficiency.csv` replaces the
  old one as a dated re-measurement. (2) RC-SCALE at 74104b4: the budget search's footprints were
  lifted by the 0.59 GiB left by earlier loads (unpruned 8.246e9 vs the reference 7.618e9); a
  search would have removed about 0.63e9 bytes too many. The meter fix puts every footprint on a
  fresh process's footing; stage 05 now measures the unpruned model first and stops unless it
  matches the reference within 1 % (skipped, and printed, under a shortened protocol such as
  RC-SMOKE's); RC-SCALE's `budget_search` checks the same with the full protocol. The holder of
  the leftover (an embedding-sized float16 tensor and some LoRA tensors) was not found.
- **2026-11-15 (agent): the GPTQ kernel for timing, options for Soheil (no change made).** The
  pre-registration takes the kernel class from RC-Q0 (section 7); `quantization.kernel_class` is
  `gptqmodel.nn_modules.qlinear.torch.TorchLinear`, the fastest class at w4 among those that passed
  both w4 and w8 on a T4 (RC-Q0, per-layer 0.238 / 0.201 ms). It dequantizes in PyTorch, so whole-
  model latency is slower than a fused low-bit kernel. GPTQModel 7.6.0's source on a T4 (sm 7.5):
  `TorchLinear` 2-8 bits, any device (current); `TritonV2Linear` 2/4/8 bits on CUDA with Triton
  (passed both widths in RC-Q0, 0.240 / 0.244 ms); `MarlinLinear` 4/8 bits, symmetric only, groups
  of 128 allowed, minimum capability 7.5 with an fp16-only Turing path in the 7.6.0 source, while
  RC-Q0 recorded "Marlin/Machete need a newer GPU than T4" (to verify on a T4: the compiled Marlin
  extension may not include sm 7.5); `ExllamaV2Linear` 4-bit only, CUDA (it failed in RC-Q0 because
  our builder calls `post_init()` without the scratch space it needs: our wiring, not the T4);
  `MacheteLinear` needs sm 9.0, `QQQLinear` sm 8.0 (not on a T4); the BitBLAS classes need the
  `bitblas` package (not installed); TorchFused / TorchInt8 / TorchAten are CPU or XPU kernels.
  Options: keep TorchLinear (stated, slow); TritonV2 for w4 and w8; Marlin for w4 and w8 if it
  loads and passes the code check on a T4; ExllamaV2 for w4 only after wiring its scratch space. A
  change of the kernel class changes a setting taken from RC-Q0, so it needs Soheil's decision and
  a section-10 note; the accuracy results do not depend on the kernel (the codes are the same).
- **2026-11-15 (research lead; agreed by Soheil and Amir): protected layers, the current reading
  stands.** "First 2 and last 1 layers protected" means those whole layers are never removed;
  units inside them stay removable, as the code, `configs/pruning.yaml` and D07 already do, for
  every arm. Reasons: it matches the configuration and D07, written before any scores existed; it
  treats every arm the same; changing it after seeing that a layer-35 unit leads E7's order would
  be a choice made on the results. Recorded as the clarification of 2026-11-15 (section 10, no
  setting changes); the paper's methods text says protection is of whole layers; every pruning
  run reports its removed units inside layers 0, 1 and 35 (`removed_units_in_protected_layers`,
  and an appendix table).
- **2026-11-15 (agent): RC-G2 session 3 rerun committed; a question on the protected layers
  before session 4.** E7 seed 0's scores (Amir, 9069603; 3,204 units, 5,908 s, peak 8.28 GB) are
  committed unchanged. The code protects layers 0, 1 and 35 from **whole-layer** removal only
  (`removal_order` and `validate_removal_set`); their KV groups and channel groups stay removable,
  the same for every arm (E6 and A6 `layers_only` remove whole layers only, so they are not
  affected). D07 ("the first 2 and last 1 decoder layers are never removed, for every arm") and
  `configs/pruning.yaml` ("whole decoder layers that are never removed") match this reading; the
  pre-registration's line "first 2 and last 1 layers protected" can also be read as protecting
  every unit inside them. With E7's scores a B2-sized removal (by parameter count, an estimate)
  takes 41 of layer 35's 86 MLP groups, 21 of layer 0's and 2 of layer 1's. **Decided 2026-11-15 (research lead; agreed by Soheil and Amir): the current reading
  stands** (pre-registration section 10, clarification of 2026-11-15): whole layers 0, 1 and 35
  are never removed; their inner units stay removable, for every arm. Every pruning run now
  records `removed_units_in_protected_layers`.
- **2026-11-15 (agent; RC-G2 session 1 at 2846e81, implementation, no setting changed): the GPU
  guard and the release.** Stage 05's first budget-search load was refused: 7.12 GiB "in use", of
  which 0.59 GiB were live tensors and the rest this process's allocator cache (expandable
  segments on: GPTQModel sets `PYTORCH_ALLOC_CONF`). The same failure as RC-SCALE's
  `budget_search`, so a production bug. The guard now refuses only live tensors of this process
  or memory outside its allocator (each above 1.5 GiB); the post-load check against any
  cpu / meta / disk placement stays. The release removes accelerate hooks, clears tensors kept
  outside parameters and buffers, synchronizes, empties the cache, prints allocated and reserved,
  and prints the largest survivors once. The 0.59 GiB were not reproduced on the CPU; the 15-load
  test now also requires the live tensors to return to their level after every release. RC-SCALE
  `budget_search` (14 measurements) runs before session 1 again.
- **2026-11-14 (agent, Soheil's request): RC-E3 `ready`; RC-SCALE second run recorded.** RC-E3
  seed 0 is `ready` (the seed-0 calibration check committed, RC-SMOKE passed at a237c5c, the test
  drop list committed). RC-SCALE at 2846e81: recovery (host RSS 2.7 GiB during the updates, 8.0
  at the save, was 23.6 / 27.53), E4 training (7 of 10 updates skipped at the start, recorded)
  and E9's evaluation passed with flat memory; `budget_search` failed again in the check's
  measurement loop (7.12 GiB on GPU 0 before a load). Soheil: not blocking (production stage 05
  did its 6 loads in both smokes). Diagnostics added (`MEDCOMPRESS_LEFTOVER` after every
  measurement's release; a refused load names this process's share).
- **2026-11-14 (Soheil and Amir): skipped updates kept and reported.** RC-SCALE's recovery
  check (20 real updates on a pruned 3B) skipped 8 updates, at updates 1-7 and 11: the fp16 loss
  scaler settling. Decision: keep the frozen scaler; record the skipped updates of every recovery
  (`recovery_skipped_steps`) and every E4 run (`train_skipped_steps`, as E1, E2 and E10 already
  do) in metrics, and report them per run in the paper (appendix table `appendix_updates`).
- **2026-11-14 (agent; RC-SCALE at 48097bd, implementation, no setting changed): host RAM of the
  recovery.** 23.6 GiB during the updates and 27.53 GiB at the save. Cause in the cache reader:
  each teacher record kept its own full copy of its shard's arrays (`NpzFile` re-reads an array
  on every access); now one read per shard and each record owns its rows (same values). The
  pruned model is saved one tensor at a time from the GPU (same file format) and loaded straight
  to the GPU. Scale-check harness fixed: the first load is released and verified before measuring
  (`budget_search`), CUDA is initialised before resetting peaks (`e4_training`), each check gets
  only its card's hand-offs (`e9_eval`). The test drop list (`results/data/nonfinite_drop.csv`,
  9f04c4f) was rebuilt from the re-sent scan files: identical, so it stays as committed.
- **2026-11-14 (Soheil and Amir; pre-registration section 10, clarification of 2026-11-14): E10's
  comparisons confirmed as proposed.** "The pruned 3B arm" is E5, E6 or E7 only (not the
  ablations), the one nearest E10's measured peak memory; E10Q is compared with E8 at the budget
  nearest E10Q's measured peak memory. Both rows are descriptive, outside H1-H4. No code change.
- **2026-11-14 (agent; code the frozen plan names, section 12 note): E10 at 4 bit and E10's two
  comparisons.** New registry entry `E10Q` (config `configs/experiments/E10Q.yaml`, card
  `RC-E10Q_e10_at_4bit.md`): the merged E10 quantized with the E3 `w4` recipe (the frozen GPTQ
  settings of E8 and E9), seed 0, validation and benchmark. Its 128 calibration images of seed 0
  are cleaned by the 2B's own fp16 check (model `e10` in `results/calibration/seed0_replaced.csv`;
  the amendment of 2026-11-13 names "the model being calibrated"); the card runs that check first.
  Comparison step, seed 0, test split, descriptive (differences with intervals for the headline
  accuracy, the small-finding recall, the rare-group recall and the ECE; no p-value, no verdict;
  not part of H1 or H4): `E10_vs_matched_pruned_seed0`, E10 minus the pruned 3B arm (E5, E6 or E7,
  any budget) whose measured peak memory is nearest to E10's; `E10Q_vs_E8_seed0`, E10Q minus the
  E8 budget whose measured peak memory is nearest to E10Q's. **Confirmed by Soheil and Amir the same
  day (entry above); proposed as: to confirm by Soheil and Amir before
  the comparison step:** which arms count as "the pruned 3B arm" (the agent took E5, E6 and E7,
  not the ablations), and the nearest-peak-memory rule for E8's budget (item 14.3 says only "at 4
  bit against E8"). RC-E10Q runs after RC-SMOKE passes for it (its smoke job quantizes the smoke's
  own 2-step E10).
- **2026-11-14 (Soheil's request; implementation, no frozen setting changed): memory at full
  size.** (1) RC-G2 session 3 (Amir, 9f04c4f) ran out of GPU memory in E7's first-order pass
  after the exact pass had finished (14.18 GiB allocated). The backward pass to the gates kept
  every decoder layer's intermediate tensors at once; now each layer is recomputed in the
  backward pass (`scoring.checkpointed_decoder_layers`; the same gradients, bit for bit on the
  tiny model), the weights are checked to collect no gradient, each step's graph is freed, and
  memory is logged every 25 steps. Probe batch 4 stays. Expected peak about 8.5-9.5 GiB
  (estimate). The exact effects are saved when the exact pass ends and a rerun with the same
  settings, units and probes resumes from them (`exact_effects.resume.json`, also found in an
  attached input). (2) RC-SMOKE at a237c5c: E7's stage 05 was killed with exit -9 (host RAM).
  Every load and release prints GPU and host memory; the release drops the processor, empties
  the pinned-memory cache and returns freed heap pages (`malloc_trim`); later loads of a pinned
  model pass `local_files_only`; 15 loads in one process keep the RSS flat (test). The smoke
  samples each stage's RSS from outside and has `--sequential`. Stage 05 keeps loading a fresh
  copy for every footprint measurement (the measured path is unchanged, so there is nothing to
  prove equal); building copies from one CPU copy is not needed unless RC-SCALE shows growth.
  (3) Every recovery now records `recovery_skipped_steps` (updates the loss scaler skipped),
  `recovery_nonfinite_steps` and `recovery_nonfinite_microbatches` in its metrics, the same for
  every arm (a recorded number, not a setting). The smoke's E7 recovery skipped 2 of its 2
  updates with finite losses: the scaler starts at 65,536 and halves after each overflow, so the
  first updates of a fp16 run are where skips are expected. RC-SCALE counts them over 20 real
  updates; above 5 % the agent asks before anything changes (the scaler is part of the frozen
  recipe). (4) RC-SCALE (`scripts/scale_check.py`, card `RC-SCALE_scale_check.md`): each heavy
  step at its real settings for a few steps (E7 first-order pass, budget search, recovery, E4
  training, GPTQ, E9's evaluation), failing when memory grows step over step or a step peaks
  within 1 GiB of a T4. (5) RC-E10 committed (account 3, c6c7cbd); RC-E3, RC-E4, RC-A4 and RC-A6
  passed the reduced smoke at a237c5c. Before any E7 stage 05 production run (RC-G2 session 4),
  E7's stage 05 smokes again.
- **2026-11-14 (agent; applying the second amendment of 2026-11-13): RC-CAL0 part A.** Teacher
  encoder: no non-finite image in any split. Student encoder: 2 / 390 validation, 5 / 512 filtered
  test, 8 / 968 unfiltered test, 24 / 4,513 training images, all first at vision block 31. The
  test drop (union of both encoders) leaves 507 and 960 test images; the code is in
  (`data/nonfinite.py`, test evaluations refuse without `results/data/nonfinite_drop.csv`); the
  list is committed when the scan files arrive. Validation: only the free-text set could hold the
  2 images (the probe-based suites draw from the 227 probe images, which do not contain them);
  check and proposal in RC-CAL0 section 6.
- **2026-11-13 (Soheil and Amir; pre-registration section 10, clarification): A5 logprob_drop
  is its own run on layers and KV groups.** As section 7 and D07 state for both A5 variants (the
  KL effect is scored exactly, and channel groups are scored by the first-order estimate), A5
  `logprob_drop` runs on layers and KV groups only, as its own B2 run on seed 0; the reuse of E7
  for it (freeze item 12) assumed identical settings, which is false (E7 also removes first-order
  channel groups). `A4:n128` stays E7. Code: `configs/experiments/ablations/A5.yaml` (both
  variants on layers and KV groups), `registry.SAME_AS_E7` = `A4:n128` only, RC-A5 runs both
  variants, and A5 `logprob_drop` takes E7's exact scores of those units
  (`pruning/stage.py: subset_source`), so it adds about 3 GPU h plus a budget search.
- **2026-11-13 (Soheil and Amir): RC-CAL0 part B.** `results/calibration/seed0_replaced.csv`
  committed unchanged (SHA-256 73a999df...b747d). Student (E2 seed 0, 256 images): 260 checked, 4
  replaced, the first 256 finite. Teacher (128 images): 128 checked, none replaced, so E9's
  existing quantized hand-off (`e9-seed0-quantized`, made at 54b5b6d from the original 128
  images) stands under the amendment.
- **2026-11-13 (Soheil and Amir; pre-registration section 10, second amendment of 2026-11-13;
  before any test evaluation and before the RC-CAL0 scan results): test images that are not
  finite in fp16 are dropped.** Any test image (filtered or unfiltered internal test, later
  PolypGen) whose fp16 vision tower output is non-finite for the student or the teacher is dropped
  from that test for every arm before any test evaluation; the list is committed and its count is
  reported next to the test sizes. Reason: a non-finite probe has no defined answer; the student
  arms share the vision tower, the union with the teacher's keeps one test set for every arm, and
  the rule reads no model output. The list comes from RC-CAL0's scan; the code that leaves the
  images out comes before RC-R1.
- **2026-11-13 (Soheil and Amir; pre-registration section 10, amendment of 2026-11-13; before any
  compression result): calibration images that are not finite in fp16 are replaced.** Calibration
  images whose fp16 forward through the model gives any non-finite value are replaced by the next
  eligible image in the same seeded order, keeping the nested subsets (A4) and the finding
  stratification; applied once per seed, before any criterion or quantization, identically for
  every arm that uses the calibration set (pruning criteria, GPTQ, E4, E8, E11/E12); the replaced
  ids are committed (`results/calibration/seed<k>_replaced.csv`). Reason: RC-E3 seed 0's NaN
  Hessian; 2 of 128 calibration images (their ids go into the committed check file) give a
  non-finite vision tower output in fp16 (Amir,
  `check_gptq_inputs.py`, 9113632). The vision tower stays fp16; nothing else changes. Code:
  `data/calibration.py`, `scripts/check_calibration.py`, `make_calibration_subset(exclude=...)`
  (unchanged without exclusions), `calibration_check.required` in `configs/base.yaml`; the fp16
  scan of every split: `scripts/scan_vision_finite.py` (RC-CAL0).
- **2026-11-13 (Soheil's request; implementation, no frozen setting changed): RC-SMOKE before
  every remaining run.** A tiny end-to-end run of each card's real stages (one T4 x2 session)
  must pass before the card runs (`docs/runs/smoke_passed.yaml`, `run_card.py` refuses
  otherwise). The smoke-only settings (`smoke.*`) are never in a real run and never write to
  `results/`. Also fixed: a stage frees each student copy before loading the next (RC-G2); the
  hand-offs of E0 and E9 are compared on the teacher's settings and E9's quantized decoder is
  verified against the teacher before use (RC-E9).
- **2026-11-13 (Soheil):** rerun RC-E3 seed 0 in full after validating FP32 GPTQ Hessians.
  For a finite Hessian that fails its first Cholesky factorization, log retries in 0.0025
  increments up to 0.1 as a run deviation and keep the configured base damping.
  Resume RC-E9 from attempt 3's saved quantized hand-off in fresh evaluation processes.

## Open decisions

| Topic | Options / notes | Owner | Due |
|---|---|---|---|
| Kaggle session time limit | still not verified here; RC-TT1's 11.0 h session ran and its saved state carried into session 2 (2026-10-31), so the 11-h stop works in practice | team (check on Kaggle) | before the next long run |
| The two raters' names (RC-D1-label) | who rater 1 and rater 2 are, for the run report (D02 amendment 4: two raters, no third member) | team | at RC-D1-label's scoring |
| Co-primary tests: how they combine (D02 amendment 7) | **decided 2026-11-08**: supported only if supported on both (intersection-union), partly supported on one | — | — |
| The external test (D02 amendment 7) | **PolypGen**, CC BY 4.0 (Ali et al., Sci Data 2023); loader and overlap check written 2026-11-08; the download and the Synapse terms: Amir | Amir | at the download |
| Code for amendment 7 | done: the PolypGen loader, the overlap check, the verdict combination; still to write: the `external_test` split in the evaluation and its comparisons | agent | before RC-X2's evaluation |
| Published target modules | settled by RC-TT1 (2026-10-31): `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, 28 decoder layers each (196); the printed line itself was not pasted | — | — |
| RC-X1 and the two stages | RC-X1's overlap count uses the dHash alone (4 bits); whether it uses the same two stages as RC-D1, so external images that are adjacent frames of our images are counted too | team | before RC-X1 part A |
| Label values (RC-D1 pass 2) | frozen 2026-10-31 (polyp, ulcerative colitis, oesophagitis). Confirmed by E0's validation (2026-11-01): 681 probes = 227 of the 390 validation images x 3, so only images with an abnormality value are probed and a finding's negatives are other abnormalities, never normal mucosa. Keep it and state it as a limitation, or add normal images as negatives (that would be a change after a model result, E0's) | team | before the freeze |
| Recovery steps | **decided 2026-11-08**: 157 updates on the first 2,500 examples | — | — |
| A1 and A2 reuse of E7's effects | **done 2026-11-08**: the clinical criterion keeps per-finding effects; A1 and A2 re-weight E7's of the same seed (`pruning/stage.py`) | — | — |
| Budget hardware for stage 07 | **decided 2026-11-08**: Kaggle Tesla T4 | — | — |
| Student amendment | moot (Farzan left; Soheil and Amir know it) | — | — |
| Fixed training pool | **decided** (iteration 36; confirmed 2026-11-08) | — | — |
| EndoBench terms | set aside by D02 amendment 7 (the external test is from another centre: PolypGen and/or GastroVision); reopen only if the team brings EndoBench back | Farzan | — |
| Finding weights | **decided 2026-11-08**: cap 20; tiers action 2.0 / follow_up 1.5 / other 1.0; polyp action, oesophagitis and ulcerative colitis follow_up | — | — |
| Compression split | **decided 2026-11-08**: rebalanced over three accounts (Soheil, Amir, account 3 lent by Farzan; `docs/schedule.md` section 7) | — | — |
| Comparator "best of E5/E6" | **decided 2026-11-08** (Soheil and Amir, seed plan): validation, seed 0, B2, ordered by (i) `op_small_finding_recall`, (ii) `op_mask_positive_recall`, (iii) `op_small_margin_mean`, (iv) name; only the chosen arm runs seeds 1 and 2 (pre-registration section 5 and section 11 item 12) | — | — |
| Headline accuracy for H1 and H4 | **decided 2026-11-08**: macro balanced accuracy at the operating point | — | — |
| A3 ranking | **decided 2026-11-08**: random scores not divided by the unit size | — | — |
| Reuse of E7 for identical ablation arms | **decided 2026-11-08**: E7 stands for `A4:n128` and `A5:logprob_drop` | — | — |
| H2 matching and McNemar | **decided 2026-11-08** (H2 confirmatory with E3 `w8` on 3 seeds; McNemar on the 3-seed majority) | — | — |
| H2 under the seed plan | **decided 2026-11-08**: E3 `w8` joins the 3-seed tier | — | — |
| E10 code and comparisons | **done and decided 2026-11-14**: `E10Q` (GPTQ `w4` of the merged 2B, card RC-E10Q); "the pruned 3B arm" = E5, E6, E7 nearest E10's peak memory; E8 at the budget nearest E10Q's peak memory; descriptive, outside H1-H4 (pre-registration section 10, clarification of 2026-11-14) | — | — |
| E10 owner | **Soheil** (2026-11-08) | — | — |
| E3 `w4` at seeds 1 and 2 | **decided 2026-11-08**: seed 0 only | — | — |
| Budgets and timing hardware | budgets **decided 2026-11-08** (15 / 30 / 45 % of the decoder bytes; RC-B0 converts); the rented timing GPU (RC-T1) still open, does not block the freeze | team | before RC-T1 |
| GPTQ kernel for timing | **decided 2026-11-16**: TorchLinear kept (RC-Q0's choice); the paper states the latency is with it | — | — |
| Venue | e.g. MICCAI (Springer LNCS template) | team | TODO |
