# Status

**Last updated:** 2026-10-19, iteration 54 (interactive run mode)
**Overall:** IN PROGRESS (not `DONE`)
**Current stage:** S03 + S05: D02 amendment 3 (near-duplicates by embeddings, the rule chosen on human labels): RC-D1-embed, then two raters label about 300 pairs, RC-D1-label's scoring, RC-D1 run 4; then RC-TT1 (N = 38,000, the published LoRA targets). Run 3 was not accepted; RC-TT0 is done
**Article shape:** not yet known (decided by Gate 2 and H3; see `docs/FINAL_GOAL.md`)

## Next agent action

0. Bug hunt requested by Soheil (iteration 35 onwards): evaluation and statistics fixed (split
   files, image-level bootstrap, bootstrap memory); training pool and teacher cache fixed
   (iteration 36). Pruning, budgets and efficiency next. Then quantization (S08).
1. Student: Qwen2.5-VL-3B (revision pinned), image side 672, micro-batch 1 x 16 without
   gradient checkpointing, G2 0.10, 90 GPU h per week (decisions of 2026-10-17, iteration 50).
   Shared data (RC-D1 bundle, `--data-root`), hand-offs between accounts (`--from`), the
   unattended runner (`scripts/run_card.py`, `notebooks/run_card.ipynb`), training resume and the
   timing of the first 100 teacher answers are in place. The work split and `docs/schedule.md` and
   `docs/team_setup.md` are written; the freeze list is in the inbox.
   2026-10-18 (iteration 51): the official split is by question, so all images are re-split
   by image (D02 amendment); the teacher gets our own LoRA (D01 amendment; rule 3; RC-TT0,
   RC-TT1); Gate 1 and the cache wait for RC-TT1.
   2026-10-18 (iteration 52): RC-D1 run 2's 4-bit groups missed adjacent frames, so the
   grouping has two stages (D02 amendment 2): dHash candidates within 12 bits, confirmed by a
   crop correlation whose threshold comes from the calibration step; a chaining guard; a final
   contact sheet.
   2026-10-19 (iteration 53): final values from the calibration (threshold 0.65, no join on
   the dHash alone, guard 10 %, the final sheet as the deciding check, an embedding fallback
   pre-registered); a T4 x2 hour counts once, so RC-TT1 trains on 38,000 examples; RC-TT1 uses
   the published adapter's target modules (which ones is printed at its start: the hub cannot
   be reached from here).
   2026-10-19 (iteration 54): run 3 not accepted (both measures missed copies); D02 amendment
   3: embeddings (proposal DINOv2-small, checked on the hub by RC-D1-embed), candidates, a
   stratified sample of about 300 pairs on score-free sheets, two raters, kappa and every
   rule's sensitivity and precision, the rule chosen by a rule fixed in advance, run 4 over all
   pairs, the strict test subset.
   2026-10-19 (iteration 55): RC-D1-embed ran (T4, 156 s of embedding per its counts file; 366,028 candidates; 298 pairs to
   label); DINOv2-small pinned at ed25f3a3..., apache-2.0. Recorded before any label: the
   top-up near the chosen threshold, the third member for disagreements (all pairs of a round
   below kappa 0.40), the unsure analysis, and rater blinding with committed SHA-256s.
   `results/data/near_duplicate_embed.csv` is not on the branch yet.
   2026-10-22 (iteration 56): both label files committed by SHA-256; D02 amendment 4 (two
   raters, no third member; kappa on pairs 11-298; primary truth both `same_scene`;
   either-rater and unsure readings reported only); `score-labels --no-third-member`; the
   counts file is now committed (69799ff; 156 s of embedding, not 143 s).
   2026-10-28 (iteration 57): RC-D1-label scored (kappa on pairs 11-290); run 4 (full:0.93)
   passed the automatic checks but failed the deciding check (same-scene pairs across splits:
   about 17, 4, 13 of 24 per band), so it is not accepted. D02 amendment 5 (post hoc):
   either-rater truth, a sweep of full:T OR crop >= 0.80 (T 0.92 to 0.84), `rule-sweep` and
   `prepare --rule`.
   2026-10-28 (iteration 58): the sweep stopped (only full:0.92:or_crop:0.80 within the guard,
   447 images or 6.9 %; its sheet about 12, 3, 1 of 24 same-scene pairs). D02 amendment 6
   (post hoc): keep its groups and split, then a held-out filter on val and test (cosine >= F,
   crop >= 0.70, or crop >= 0.50 within 4 bits), F swept from 0.90 to 0.80;
   `filter-sweep` and `prepare --heldout-filter`.
   2026-10-28 (iteration 59): both readings confirmed; the unfiltered test built (a second
   private test file, `evaluation.split=test_unfiltered`, scored by every test evaluation,
   `09_compare_arms.py --unfiltered`); the paper keeps the sensitivity analyses out of the
   primary tables, with their own macro names. The sweep's counts file is committed. Next: the
   filter sweep (RC-D1 section 3d), the team's counts on its sheets, run 4b (section 3e); then
   pass 2 (label values,
   finding list, `max_new_tokens`); then RC-TT1's target printout and progress. RC-L1 needs code
   first (run identities, parent from config, cache extension, comparison rows); the A1/A2 reuse
   of E7's effects waits on the team (freeze list item 15).
2. Quantization stage (S08): code and run cards done (iterations 37-39). Pre-registration
   draft and freeze guard done (iteration 40). Pruning grid cards RC-E5 to RC-E8 done
   (iteration 41); ablation cards RC-A1 to RC-A6 done, A3 ranking and score reuse fixed
   (iteration 42); comparison step, RC-C1 and RC-R1 done (iteration 43); RC-T1, RC-X1 part A
   and the A4 curve done (iteration 44); appendix outputs done (iteration 45). Next: the paper
   text (S12) while the run cards wait; the EndoBench loader after RC-X1 part A.
3. D07 section 1 (how a unit is switched off while scored), the timing hardware and B1-B3 are
   asked later; none is needed before E2 exists.

Nothing heavy runs in the agent session: training, scoring and measurements are written as run
cards for the team (decision of 2026-10-12).

## Waiting on the team

| # | What | Who | Since |
|---|---|---|---|
| 1 | Review `research-loop` and open the pull request into `main` | Soheil | 2026-10-12 |
| 2 | Optional: full names for the copyright line in `LICENSE` and `CITATION.cff` | team | 2026-10-12 |
| 2b | CI on `main` is red (import error fixed on `research-loop` in a7ec704); merging `research-loop` into `main` turns it green | Soheil | 2026-10-12 |
| 4 | Each member confirms own access to Kvasir-VQA-x1 and Kvasir-SEG (inbox) | all | 2026-10-12 |
| 5 | Run **RC-D1-embed** (paste its printout; upload its output as ONE private dataset); label the about 300 pairs with a second rater (card RC-D1-label); run its scoring and commit the two results files; then RC-D1 **run 4**, share its data, commit `results/data/summary.csv`, attach the final contact sheet with what you see in each band; then RC-TT1 with run 4's shared data | Soheil, a second rater | 2026-10-19 |
| 7 | Tell Amir and Farzan that the staged selection was replaced by one student with checks (D05 amendment) | Soheil | 2026-10-12 |
| 8 | Confirm the fixed training pool (one pool from `seeds.data_split` for E1, E2, E4 and the recoveries; the teacher cache holds only that pool) | Soheil | 2026-10-12 |
| 10 | Answer the [team] items of the pre-registration draft (comparator rule, headline accuracy for H1/H4, finding thresholds, prevalence cut points, cap, tiers, training pool, A3 ranking, reuse of E7 for `A4:n128` and `A5:logprob_drop`, H2 matching, McNemar on the majority answer) | all | 2026-10-12 |
| 11 | Choose the rented timing GPU and the precision (`hardware.final_timing`, `hardware.precision`) | team | 2026-10-12 |
| 12 | Run RC-X1 part A: EndoBench terms, format and overlap (no GPU) | Farzan | 2026-10-12 |
| 13 | Approve the freeze list in the inbox (19 items with recommendations; the RC-D1 and RC-B0 items follow their runs) | Soheil, then Amir and Farzan | 2026-10-17 |
| 14 | Check on Kaggle: the session time limit; whether a stopped version keeps its output (a T4 x2 session counts once: measured 2026-10-19) | team | 2026-10-17 |
| 15 | Add Amir and Farzan as collaborators on GitHub (`research-loop`) and on each private Kaggle dataset (`docs/team_setup.md`) | Soheil | 2026-10-17 |

## Stages

| Stage | Title | Status | Blocker |
|---|---|---|---|
| S00 | Repository health and loop setup | done, pending PR merge | team review |
| S01 | Decisions | done (dataset-access confirmations pending) | - |
| S02 | CPU infrastructure | done | - |
| S03 | Data pipeline | waiting on RC-D1 | team runs RC-D1 |
| S04 | Model wrappers and unit surgery | done (RC-M1 passed 2026-10-15) | - |
| S05 | Teacher: Gate 1, cache | code and run cards done (RC-E0a, RC-E0b blocked) | RC-D1, open settings |
| S05b | Student checks (D05 amendment: one student) | RC-S0 done 2026-10-16: both pass G1; training size decided; student revision pinned; RC-S0b run 2 ready (image side) | RC-S0b, team GPU hours, G2; then the selection freeze |
| S06 | Students E1, E2, budget reference | trainer done; recipe memo and run cards next | S05, S05b |
| S07 | Pre-registration freeze | draft and freeze guard done; values after RC-D1, RC-S0, RC-B0; team items open | RC-D1, team answers |
| S08 | Quantization E3, E4, E9 | RC-Q0 done 2026-10-15 (TorchLinear, versions pinned); RC-E3, RC-E4, RC-E9 blocked | S07 freeze, E2, RC-D1 |
| S09 | Pruning E5-E8, Gate 2 | code and run cards done (RC-G2, RC-E5 to RC-E8, all blocked) | D07 (scoring method, budgets), S07 freeze, E2 |
| S10 | Ablations A1-A6 | run cards done (RC-A1 to RC-A6, blocked); team: A3 ranking, E7 reuse | S09, S07 freeze |
| S11 | External test, timing, aggregation | code and cards done (comparison step, appendix; RC-R1, RC-C1, RC-T1, RC-X1 part A); waiting for results | S08-S10; RC-X1 part A for the EndoBench loader |
| S12 | Paper and release | not started | S11 (method/related work can start early) |

## Gates

| Gate | Status |
|---|---|
| Gate 1: teacher uses the image, damage visible | not run |
| Gate 2: E7 vs best of E5/E6 at B2 (validation) | not run |
| Pre-registration freeze | not frozen |
