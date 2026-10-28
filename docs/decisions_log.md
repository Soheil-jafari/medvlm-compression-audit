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

## Open decisions

| Topic | Options / notes | Owner | Due |
|---|---|---|---|
| Kaggle session time limit | the session time limit, and whether a stopped version keeps its output (unknown here); the T4 x2 quota rule is settled (counts once, 2026-10-19). RC-TT1 needs about 14.7 h (estimate) | team (check on Kaggle) | before RC-TT1 |
| The two raters' names (RC-D1-label) | who rater 1 and rater 2 are, for the run report (D02 amendment 4: two raters, no third member) | team | at RC-D1-label's scoring |
| The filter sweep's sheets (D02 amendment 6) | per F and band, how many pairs show the same scene, judged by the team on the private sheets; then F (the highest with none whose test keeps >= 50 %) or, if none qualifies, a stop | team | after the filter sweep |
| `results/data/summary.csv` | holds run 4 (not accepted) until run 4b replaces it; the paper's generated data table shows it until then | Soheil | at run 4b |
| Published target modules | which modules the published `adapter_config.json` names (RC-TT0's printout is not committed; RC-TT1 prints it). If a name is outside the language model, RC-TT1 stops and the team decides | Soheil | at RC-TT1's start |
| RC-X1 and the two stages | RC-X1's overlap count uses the dHash alone (4 bits); whether it uses the same two stages as RC-D1, so external images that are adjacent frames of our images are counted too | team | before RC-X1 part A |
| Label values (RC-D1 pass 2) | the answer values of the abnormality question that become finding ids (`configs/data/kvasir_vqa_x1.yaml`), from the first pass's `answer_count:*`; then the counts pass | agent proposes, team confirms | after RC-D1 pass 1 |
| Recovery steps | 25 % of 625 = 156.25; proposal 157 = `ceil(2,500 / 16)` (D06 memo) | team | before RC-E5 |
| A1 and A2 reuse of E7's effects | D07 says they reuse E7's per-probe effects; the code scores them again (about 7.9 h per run at 672, an estimate). Either the agent adds the reuse (the per-probe effects must then be kept) or the schedule keeps the cost | team | before the freeze |
| Budget hardware for stage 07 | `budgets.measurement.hardware` is open, so the benchmark step of RC-E1 and RC-E2 waits (the runner skips it and goes on) | Amir | before RC-B0 |
| Student amendment | tell Amir and Farzan that one student is used (D05 amendment) | Soheil | before RC-E1 |
| Fixed training pool (agent proposal, iteration 36) | E1, E2, E4 and every recovery draw their examples from one pool fixed by `seeds.data_split` (20261012, the RC-D1 seed), not from the run seed. The teacher cache holds only that pool (`teacher_cache.pool_size` = `training.data_subset`) instead of all 143,594 training rows; the three seeds vary the order and the adapter's start. Alternative: a different pool per seed (the cache then needs the union of three pools) | Soheil | before RC-E0b |
| EndoBench terms | Confirm licence and download before it is used as the external test | Farzan | before S11 |
| Finding list and weights | Pre-registered table, frozen before E7. Draft rules (iteration 40): a finding enters C with at least 20 positive training images and 10 in each of validation and test; groups common >= 10 %, rare < 2 %; cap 20; three tiers with multipliers 2.0 / 1.5 / 1.0 assigned from the finding names | team | after RC-D1 |
| Comparator "best of E5/E6" (draft proposal) | the generic arm with the highest validation small-finding recall at B2 (mean of 3 seeds), the same for both primary endpoints | team | before the freeze |
| Headline accuracy for H1 and H4 (draft proposal) | macro balanced accuracy instead of the overall probe accuracy, which mostly measures specificity; overall accuracy still reported | team | before the freeze |
| A3 ranking (agent proposal, iteration 42) | A3 orders its random scores without dividing by the unit size (`score_normalisation: none`). Divided by size, a random score sends whole layers first (estimate: about 98 % of the removed parameters at a 30 % cut), so A3 would test random layer removal, not random choice of units. D07's per-parameter rule stays for every other arm | team | before the freeze |
| Reuse of E7 for identical ablation arms (agent proposal) | `A4:n128` and `A5:logprob_drop` have exactly E7's settings (test), so E7's B2 runs are reported for them (27 ablation runs instead of 33). If D07 section 1 uses the first-order estimate for some units, A5 moves to the exactly scored units and `A5:logprob_drop` is run | team | before the freeze |
| H2 matching and McNemar (agent proposal, iteration 43) | H2: E3 `w8` against the generic pruning arm with the highest validation small-finding recall at the budget whose target is nearest to `w8`'s measured peak memory. McNemar per finding on each probe's majority answer over the 3 seeds | team | before the freeze |
| Budgets and timing hardware | B1-B3 fractions (rule after RC-B0) and the rented GPU type; metric, tolerance and protocol are decided | Amir | TODO |
| Venue | e.g. MICCAI (Springer LNCS template) | team | TODO |
