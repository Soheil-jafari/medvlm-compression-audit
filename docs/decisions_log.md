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
| E10 code still to come | the 4-bit E10 (GPTQ `w4` of the merged 2B) and its two exploratory comparisons (matched memory against the nearest pruned 3B arm; 4 bit against E8) in the comparison step | agent | before RC-E10 is `ready` |
| E10 owner | **Soheil** (2026-11-08) | — | — |
| E3 `w4` at seeds 1 and 2 | **decided 2026-11-08**: seed 0 only | — | — |
| Budgets and timing hardware | budgets **decided 2026-11-08** (15 / 30 / 45 % of the decoder bytes; RC-B0 converts); the rented timing GPU (RC-T1) still open, does not block the freeze | team | before RC-T1 |
| Venue | e.g. MICCAI (Springer LNCS template) | team | TODO |
