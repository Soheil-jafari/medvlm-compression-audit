# Iteration log

Append-only, newest at the bottom. One entry per iteration:

```markdown
## Iteration N: YYYY-MM-DD
- Stage / item: S0x, <task>
- Did: <what changed, files>
- Verified: ruff check, ruff format --check, pytest (<n> passed); CI <status>
- Commit: <short sha>
- Next: <next agent action>
- Needs the team: <items added to the queue, or "nothing new">
```

## Iteration 1: 2026-10-12
- Stage / item: S00, repository health and loop setup.
- Did: found that the unanchored `data/` rule in `.gitignore` had kept
  `src/medcompress/data/` and `configs/data/` out of the scaffold commit, so CI
  failed on import. Anchored the rule, restored the data package stubs (`base`,
  `manifests`, `splits`, `masks`, `dedup`, `probes`, with `format_probe_question`
  implemented) and the seven candidate dataset configs, and added
  `tests/test_data.py` (probe formatting and ignore-rule guards). CI now also runs on
  pushes to `research-loop`. Wrote the control documents: `docs/FINAL_GOAL.md`,
  `docs/agent/`, `docs/stages/S00-S12`, `docs/runs/`, `docs/results/`,
  `docs/decisions/`, `docs/README.md`.
- Verified: ruff check, ruff format --check, pytest (all passed locally).
- Commit: see `git log` on `research-loop`.
- Next: S01 D01 memo (model pair), then D02, D03; S02 statistics in parallel.
- Needs the team: review and open the pull request into `main`; licence (the
  repository is already public).

## Iteration 2: 2026-10-12
- Stage / item: S01, D03 licence memo (D01 was first in line but needs Hugging Face
  pages, which were unreachable from this session).
- Did: wrote `docs/decisions/D03_licence.md` (MIT vs Apache-2.0 vs BSD-3-Clause,
  evidence from opensource.org and apache.org, recommendation MIT unless a patent
  grant or institutional policy calls for Apache-2.0); linked it in
  `docs/decisions/README.md`; added `tests/test_docs.py` (memo index links resolve,
  every memo is indexed and has Question / Options / Recommendation / Risks).
  Noted that CI on `main` is red because the import fix lives only on
  `research-loop`.
- Verified: ruff check, ruff format --check, pytest (306 passed).
- Commit: see `git log` on `research-loop`.
- Next: D01 memo (or a fact-collection script and run card if hub pages stay
  unreachable), then D02; S02 statistics in parallel.
- Needs the team: answer D03; merge `research-loop` into `main` to fix CI there.


## Iteration 3: 2026-10-12 (interactive session, requested by Soheil)
- Stage / item: S01 and new stage S05b, student selection protocol.
- Did: researched current small open vision-language models and verified 10
  candidates plus 5 reserves from their Hugging Face API records and configs
  (size, vision encoder, connector, language model, tokenizer, licence). Wrote
  `docs/decisions/D05_student_selection.md` (10 -> 5 -> 3 -> 1 successive halving
  on the validation split, gates, score, fairness controls, tokenizer rule options),
  `docs/stages/S05b_student_selection.md`, `configs/student_selection.yaml`, verified
  configs for the 10 candidates in `configs/models/`, and
  `tests/test_student_selection.py`. Candidate configs may now carry a source and
  licence when `verified_on` and `verified_from` are set (test updated). D01 is now
  teacher only; S04, S05, S06, the run queue, FINAL_GOAL and the experiment plan
  reference the selection.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: D01 teacher memo, then D02; S02 statistics in parallel.
- Needs the team: answer D05 (Q2, Q3 now; Q1 after D01).

## Iteration 4: 2026-10-12 (interactive)
- Stage / item: inbox, D05 answers and run mode.
- Did: recorded the D05 answers (students only from the teacher's family, keep
  size, weights 0.4 / 0.3 / 0.3) in D05, `configs/student_selection.yaml`, the
  inbox and the decisions log. The scheduled weekday run was removed at the team's
  request; `docs/agent/PROTOCOL.md` now describes back-to-back interactive
  iterations.
- Verified: pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: D01 teacher memo; then fix the student pool from the teacher's family.
- Needs the team: nothing new.

## Iteration 5: 2026-10-12 (interactive)
- Stage / item: S01, D01 teacher memo.
- Did: wrote `docs/decisions/D01_teacher.md` comparing five teachers. Verified that the
  SimulaMet Kvasir-VQA-x1 LoRA was trained on Qwen2.5-VL-7B-Instruct with the train
  file only (`args.json`), and compared tokenizer file hashes of the teacher and the
  Qwen-family student candidates. Filled verified facts into the two Qwen teacher
  configs.
- Verified: pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: team decides D01; then fix the student pool from the teacher's family.
- Needs the team: D01.

## Iteration 6: 2026-10-12 (interactive)
- Stage / item: S01, D01 decision and the final student pool.
- Did: recorded D01 = T1 (Qwen2.5-VL-7B + Kvasir-VQA-x1 LoRA, revisions pinned; the
  teacher config is `DECIDED`; optional Qwen3-VL-8B teacher check added to
  FINAL_GOAL as not required). Applied the teacher-family rule: four candidates
  dropped (other tokenizers), four Qwen-family architectures added after checking
  their hub records and tokenizer hashes (LLaVA-OneVision-1.5-4B, MedVLM-R1,
  SAIL-VL2-2B, LLaVA-OneVision-0.5B); reserves replaced with Qwen-family models.
  Tests now allow `DECIDED` configs with provenance and a pinned revision, and
  check that every candidate shares the teacher's tokenizer family.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: D02 datasets memo; then S02.
- Needs the team: D03 (licence), D02 once written.

## Iteration 7: 2026-10-12 (interactive)
- Stage / item: S01, D02 datasets memo.
- Did: wrote `docs/decisions/D02_datasets.md` (Kvasir-VQA-x1 split sizes and licence,
  Kvasir-VQA, Kvasir-SEG terms, Colon-X, EndoBench, and the retinal sets IDRiD,
  EyeVQA, FundusGround as a later second domain). Key points: the teacher saw the
  whole train file, so validation should come from the official test split; no
  patient IDs anywhere, so fingerprint deduplication is required; Kvasir-SEG masks
  must be matched to VQA images by fingerprint. Filled verified facts into the data
  configs.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: team answers D02; then S02 statistics.
- Needs the team: D02 (five questions), D03.

## Iteration 8: 2026-10-12 (interactive)
- Stage / item: S01, applying D02 and D03.
- Did: recorded D02 (endoscopy; Kvasir-VQA-x1 main QA with validation and test from
  the official test split; Kvasir-SEG masks; EndoBench external, terms to verify;
  retinal later) and D03 (MIT). Added `LICENSE`, the licence in `pyproject.toml`,
  `CITATION.cff` and `README.md`; set the endoscopy domain, Kvasir-VQA-x1 and
  Kvasir-SEG configs to `DECIDED` and pointed `configs/base.yaml` at them.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: S02, starting with `evaluation/statistics.py`.
- Needs the team: dataset-access confirmations; optional full names for the
  copyright line.

## Iteration 9: 2026-10-12 (interactive)
- Stage / item: S02, `evaluation/statistics.py`.
- Did: implemented paired and single-model bootstrap CIs (images resampled jointly,
  undefined resamples dropped), McNemar (exact binomial below 25 discordant pairs,
  chi-square with continuity correction above), Holm step-down, seed mean/SD, and
  Kendall's tau with a joint-image bootstrap for H4. `aggregate_results` stays a
  stub. New `tests/test_statistics.py` (16 tests).
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: S02 metric modules (`clinical`, `small_findings`, `rare_findings`).
- Needs the team: nothing new.

## Iteration 10: 2026-10-12 (interactive)
- Stage / item: S02, metric modules.
- Did: implemented `evaluation/clinical.py` (yes/no parser: answer must start with the
  word yes or no; per-finding sensitivity and specificity; macro-F1 leaving out
  undefined findings; accuracy; unparseable answers count as errors),
  `small_findings.py` (recall per lesion-size stratum, stratum 0 = primary
  endpoint), `rare_findings.py` (pooled or mean-per-finding recall per prevalence
  group), `calibration.py` (equal-width ECE, Brier, risk-coverage, discrete AURC) and
  the pure parts of `image_reliance.py` (flip rate, lesion-mask effect). The parsing
  and pooling rules are proposals for the pre-registration. New
  `tests/test_metrics.py` (18 tests).
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: `evaluation/language.py` and `hallucination.py`.
- Needs the team: nothing new.

## Iteration 11: 2026-10-12 (interactive)
- Stage / item: S02, `evaluation/language.py` and `hallucination.py`.
- Did: answer normalisation (multi-value answers compared as sorted lists), exact
  match, token F1, answer length; rule-based finding polarity (whole-word mention,
  negation cue within four words in the same clause) and hallucination / omission /
  negation classification with per-type rates. New `tests/test_text_metrics.py`.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: data pure functions (`probes.build_probes`, `splits`, `masks`, `dedup`).
- Needs the team: nothing new.

## Iteration 12: 2026-10-12 (interactive)
- Stage / item: S02, `data/probes.build_probes` and `data/splits.py`.
- Did: probes for every labelled (sample, finding) pair; splits grouped by patient,
  falling back to the image (Kvasir has no patient IDs), deterministic for a seed
  and independent of input order (used to halve the official test split into
  validation and test); disjointness check; nested, finding-stratified calibration
  subsets of distinct training images; split fingerprint. New
  `tests/test_splits_probes.py`.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: `data/masks.py`, `data/dedup.py`, `data/manifests.py`.
- Needs the team: nothing new.

## Iteration 13: 2026-10-12 (interactive)
- Stage / item: S02, `data/masks.py`, `data/dedup.py`, `data/manifests.py`.
- Did: mask loading, lesion area, frozen quartile strata, area-matched control region
  (the lesion's own shape moved to a non-overlapping place, least-overlap fallback);
  64-bit dHash fingerprints with vectorised Hamming search for near-duplicates
  (also used to link Kvasir-SEG masks to Kvasir-VQA-x1 images); SHA-256 manifests
  that refuse to be written inside the repository and verify files later. Pillow
  added to the light requirements (it was already installed through matplotlib).
  New `tests/test_masks_dedup_manifests.py` (synthetic images in a temp dir).
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: clinical-criterion pure parts (`load_finding_weights`, `compute_weights`,
  `delta`, `aggregate_importance`).
- Needs the team: nothing new.

## Iteration 14: 2026-10-12 (interactive)
- Stage / item: S02, clinical-criterion arithmetic.
- Did: `load_finding_weights` (refuses an unfrozen or incomplete table),
  `load_tier_multipliers`, `compute_weights` (clinical / uniform / rarity-only,
  recomputes and checks `final_weight` against the formula, optional `sum_to_n`
  normalisation), `delta` (correct-answer log-probability drop; KL over the
  renormalised yes/no pair), `aggregate_importance` (optional clipping of negative
  Delta), and deterministic ascending `rank` with ties broken by unit ID. The three
  open points of the criterion are now parameters to fix in the pre-registration.
  New `tests/test_clinical_criterion.py`.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: `budgets.py` budget search.
- Needs the team: nothing new.

## Iteration 15: 2026-10-12 (interactive)
- Stage / item: S02, `budgets.py`.
- Did: budget metric lookup, absolute targets (placeholders refused), two-sided
  tolerance check (an arm far below the target is not matched), and a bisection over
  the removal prefix that needs about log2(n) + 2 real measurements. The model
  plumbing of `search_matched_budget` stays for S04/S09. New `tests/test_budgets.py`
  with mocked footprints.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: results-tree validation and `aggregate_results` (S02's last block).
- Needs the team: nothing new.

## Iteration 16: 2026-10-12 (interactive)
- Stage / item: S02, results-tree reader and validation.
- Did: new `evaluation/aggregate.py`: classifies every CSV under `results/` by its
  path (runs, comparisons, selection rounds, data summary, smoke tests), checks the
  schema, known experiment IDs and variants, budgets, seeds, CI order and that rows
  match their folder; loads all rows and summarises them over seeds (mean, sample
  SD, min, max; single-seed CIs carried over). `tests/test_results.py` validates the
  committed tree (the check the run cards ask for) and synthetic good and bad trees.
  `results/README.md` documents the extra folders and where paired comparisons are
  computed. Budget-tolerance checking over results waits for frozen budgets.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: comparison step (paired differences and p-values from per-image outcomes)
  and the LaTeX table writers.
- Needs the team: nothing new.

## Iteration 17: 2026-10-12 (interactive)
- Stage / item: S02, paired comparisons.
- Did: `paired_bootstrap_test` (difference, CI and a two-sided bootstrap p-value) in
  `statistics.py`; new `evaluation/comparisons.py` with seed averaging of per-image
  outcomes, the "best of E5/E6" pick on validation scores (ties by name), comparison
  rows (`diff:<metric>`, `p:<metric>`) and Holm rows for the two primary endpoints.
  The rows pass the results validator. New `tests/test_comparisons.py`.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: LaTeX tables, `numbers.tex` and the `aggregate_results` entry point.
- Needs the team: nothing new.

## Iteration 18: 2026-10-12 (interactive)
- Stage / item: S02, paper tables and the aggregation entry point.
- Did: new `evaluation/report.py` writing T1 (data), T2 (main), T3 (primary), T4
  (ablations), T5 (efficiency) and T6 (external) as booktabs tables with a
  "generated, do not edit" header, `--` where results are missing, and
  `numbers.tex` with a letters-only macro for every summarised number and CI bound.
  `aggregate_results` now validates and loads `results/`, writes
  `results/summary/{all_rows,by_arm}.csv` and the tables; `scripts/08_aggregate_results.py`
  gained `--tables-dir` (default `paper/tables`, relative to the working directory).
  New `tests/test_report.py`; the script test now checks real outputs.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: figures (F1-F5), the verdicts file, and wiring the paper to the generated
  tables with a test against hand-typed numbers.
- Needs the team: nothing new.

## Iteration 19: 2026-10-12 (interactive)
- Stage / item: S02, figures F1-F5, verdicts, and the full aggregation step.
- Did: new `evaluation/figures.py` (F1 headline vs hidden damage, F2 change vs E2
  heatmap, F3 budget curves, F4 rank order, F5 mask audit; three family colours
  checked with the palette validator for colour-blind separation, direct labels on
  every point, diverging blue/red for F2, placeholders when results are missing) and
  `evaluation/verdicts.py` (H1-H4 rules over comparison rows, written to
  `docs/results/VERDICTS.md`; the rules are a proposal for the pre-registration).
  `aggregate_results` and `scripts/08_aggregate_results.py` now write summaries,
  tables, figures and verdicts. Rendered the figures from a synthetic summary and
  checked them by eye (rank axes now use whole numbers). New
  `tests/test_figures_verdicts.py`; fixed a test that wrote into the repository.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: wire `paper/sections/05_results.tex` to the generated files and add the
  test against hand-typed numbers (last S02 item).
- Needs the team: nothing new.

## Iteration 20: 2026-10-12 (interactive)
- Stage / item: S02, paper wiring (last S02 item).
- Did: `paper/sections/05_results.tex` now inputs the generated tables and figures;
  `paper/main.tex` inputs `tables/numbers.tex`; the data table in the setup section
  is generated. Ran `scripts/08_aggregate_results.py` from the repository root and
  committed its placeholder outputs (tables, figures, `results/summary/`,
  `docs/results/VERDICTS.md`). The paper compiles with latexmk (9 pages). New
  `tests/test_paper.py`: no hand-typed decimals or percentages in the abstract,
  results, discussion or conclusion; every input and figure exists; committed
  generated files match a fresh aggregation run.
- Verified: ruff check, ruff format --check, pytest (all passed); latexmk -pdf OK.
- Commit: see `git log` on `research-loop`.
- Next: S03 data pipeline (Kvasir-VQA-x1 and Kvasir-SEG readers, preparation
  script, run card RC-D1).
- Needs the team: nothing new.

## Iteration 21: 2026-10-12 (interactive)
- Stage / item: S03, dataset readers.
- Did: new `data/kvasir.py`: the local layout (images by `img_id` plus train/test
  JSONL, as the dataset card describes the join with Kvasir-VQA), a Kvasir-VQA-x1
  reader producing `VQASample` records with stable sample IDs and per-image finding
  labels derived from the source questions by configurable rules, a Kvasir-SEG
  reader, and fingerprint linking of Kvasir-SEG masks to VQA images (ties reported
  as ambiguous). `load_dataset` dispatches to the reader. The label-rule values in
  `configs/data/kvasir_vqa_x1.yaml` stay empty until RC-D1 reports the answer counts.
  New `tests/test_kvasir.py` on a synthetic layout.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: `scripts/prepare_data.py` and run card RC-D1.
- Needs the team: nothing new.

## Iteration 22: 2026-10-12 (interactive)
- Stage / item: S03, data preparation and RC-D1.
- Did: new `data/prepare.py` and `scripts/prepare_data.py` (`export` downloads the
  Kvasir-VQA images and Kvasir-VQA-x1 rows as the dataset card describes; `prepare`
  halves the official test split by image, drops near-duplicates of training images,
  links Kvasir-SEG masks, computes the size-stratum thresholds on training images
  and writes splits, exclusions, mask links and manifests outside the repository,
  plus only aggregate counts to `results/data/summary.csv`, including answer-value
  counts to fill the label rules). Run card `docs/runs/RC-D1_data_preparation.md`
  (CPU Kaggle session; proposed split seed and thresholds). End-to-end test on a
  synthetic layout.
- Verified: ruff check, ruff format --check, pytest (all passed).
- Commit: see `git log` on `research-loop`.
- Next: S04 model loading for the teacher and the ten students (CPU tests on tiny
  random models), and the RC-M1 smoke-test card; S12 method and related-work drafts.
- Needs the team: run RC-D1 (Farzan) and commit `results/data/summary.csv`.

## Iteration 23: 2026-10-12 (interactive)
- Stage / item: S04, unit enumeration and removal surgery; S12, method draft.
- Did: new `compression/pruning/surgery.py` (find the language decoder stack,
  remove layers with re-indexing, remove GQA groups, remove MLP channels, and the
  matching ablation context managers for skipping a layer, zeroing a group and
  zeroing channels). `units.py` now enumerates layers, GQA groups and MLP channel
  groups with channel ranges, and `validate_removal_set` checks the removal rules.
  `apply.py` removes units physically (groups and channels first, then layers) and
  verifies the parameter drop. New `tests/test_surgery.py` on tiny random Qwen2,
  Qwen3 and Qwen2.5-VL models: ablation gives the same logits as physical removal,
  the parameter drop equals the sum of the unit sizes, and generation still works
  after mixed removal. `heavy` pytest marker registered. `paper/sections/03_method.tex`
  rewritten; appendix section on student selection added; bibliography entries for
  successive halving, Hyperband and Wanda added from the arXiv abstract pages
  (venue and page fields still marked TODO).
- Verified: ruff check, ruff format --check, pytest (503 passed), latexmk -pdf with
  no undefined references.
- Commit: see `git log` on `research-loop`.
- Next: S04 model loading (`models/loading.py`), the wrapper with yes/no scoring,
  and the RC-M1 smoke-test card.
- Needs the team: nothing new.

## Iteration 24: 2026-10-12 (interactive)
- Stage / item: S04, model loading, wrapper, smoke test, pruned checkpoints.
- Did: `models/loading.py` (ModelSpec from the configs, refusal of unpinned or
  still-candidate models, opt-in for families that ship their own code, bitsandbytes
  option, adapter loading), `models/hf.py` (`HFVLMWrapper`: chat-template inputs with
  left padding, one-forward-pass yes/no scoring, greedy generation with top-k
  log-probabilities and the stop token kept, unit listing, parameter count),
  `models/tokens.py` (yes/no answer tokens, tokenizer fingerprint), `models/smoke.py`
  and `scripts/smoke_models.py` (RC-M1 measurements), and
  `compression/pruning/checkpoint.py` (a pruned model is saved as weights plus a
  manifest of removal steps and reloaded by replaying them on the base model, because
  per-layer shapes cannot be described by one config). Tests build a tiny offline
  Qwen2.5-VL with the real processor classes and a word-level tokenizer.
  Two bugs found by the tests and fixed: the tokenizer fingerprint changed after the
  first padded call (padding settings are now excluded), and an adapter saved with
  older module names left every language-model LoRA module untrained with the
  installed peft and transformers (loading now remaps the keys and fails if any
  language-model adapter module stays at zero). Library versions the code was tested
  with are pinned in `requirements/train.txt` and `pyproject.toml`.
  Run card `docs/runs/RC-M1_model_smoke_test.md` written with pass criteria fixed
  before the run.
- Verified: ruff check, ruff format, pytest (542 passed before the checkpoint test,
  all passed after).
- Commit: see `git log` on `research-loop`.
- Next: S05: teacher-output cache (`teacher_cache.py`), image-reliance check script
  for Gate 1, D04 memo; then the student-selection runner (S05b).
- Needs the team: run RC-M1 (Soheil) and commit `results/smoke/RC-M1.csv`.

## Iteration 25: 2026-10-12 (interactive)
- Stage / item: S05, teacher cache and Gate 1 tooling.
- Did: `models/teacher_cache.py` implemented (sharded `.npz` files written atomically,
  resumable, settings recorded in `metadata.json` and checked on resume, `manifest.json`
  with shard checksums and a manifest hash for the run card, loader that verifies the
  checksums); `data/prepared.py` (rebuilds the validation / training sample lists from
  the private RC-D1 outputs); `evaluation/image_reliance.py` implemented (blank image,
  lesion mask and area-matched random-region control, opposite-label image, balanced
  accuracy, lesion-mask effect with a paired bootstrap CI, Gate 1 verdict, stage entry
  point); `utils/io.upsert_results` so a re-run replaces its rows; stages now stop with
  exit code 4 and a message while a needed setting is still `TODO`. RC-D1 now also
  records answer-length percentiles (to set `max_new_tokens`). New config keys:
  `paths.private_root`, `inputs.image_longest_side`, `image_reliance_check.gate1/fill/
  mask_findings`. Decision memo D04 (Gate 1 thresholds, grey fill, cache `top_k`) and run
  card RC-E0a (blocked until D04, RC-D1, RC-M1 and the image size are done).
  Tests use a deterministic image-reliant fake model and a blind one, so the stress
  tests and the verdict are checked on known answers.
- Verified: ruff check, ruff format, pytest (571 passed).
- Commit: see `git log` on `research-loop`.
- Next: ask the team for D04; evaluation runner for E0 validation metrics (RC-E0b);
  student-selection runner (S05b).
- Needs the team: D04 answers.

## Iteration 26: 2026-10-12 (interactive)
- Stage / item: S06 prerequisites, training core; D04 recorded.
- Did: D04 answered in the session (moderate Gate 1 rule, flat mid-grey fill 128,
  cache `top_k` 20) and written into `configs/experiments/E0.yaml`, the D04 memo, the
  decisions log and run card RC-E0a. New: `compression/lora.py` (LoRA on decoder
  attention and MLP projections only, save, merge), `compression/distillation/batching.py`
  (prompt once, left-padded batches with right-aligned answer targets and top-k),
  `losses.py` (answer cross-entropy, KL on the teacher's cached top-k with temperature,
  weighted sum) and `offline_kd.py` (one trainer for E1, E2 and recovery: seeded order,
  gradient accumulation, fp16 base with fp32 adapters and a gradient scaler on CUDA,
  one forward pass when teacher and ground-truth answers are token-identical, stage
  entry point that saves the adapter and upserts training rows). `HFVLMWrapper.stop_token_id()`.
  Shared tiny offline model moved to `tests/tiny_vlm.py`. Method section now states the
  loss (Eq. `eq:loss`). Training hyper-parameters stay `TODO` and the stage exits with
  code 4 naming them.
  Two issues found by the tests: with `mode: sft` an unset `kd_temperature` crashed
  the parser (now ignored without a teacher), and a randomly initialised model has
  almost flat logits, so the tests widen the output layer of the tiny model; this
  limit comes from the stand-in model, not the trainer.
- Verified: ruff check, ruff format, pytest (619 passed), paper builds.
- Commit: see `git log` on `research-loop`.
- Next: evaluation runner and run card RC-E0b (S05), then the student-selection runner
  (S05b).
- Needs the team: nothing new.

## Iteration 27: 2026-10-12 (interactive)
- Stage / item: S05, evaluation runner and run card RC-E0b.
- Did: `evaluation/probe_suites.py` (overall accuracy, macro-F1 and per-finding
  sensitivity / specificity with bootstrap CIs, recall by lesion-size stratum, rare-finding
  recall by prevalence group, ECE / Brier / AURC from the yes/no scores),
  `evaluation/free_text.py` (seeded QA-row sample, language metrics, hallucination /
  omission / negation rates) and `evaluation/runner.py` (`run_evaluation`: settings
  check that names every open value per enabled suite, model loading for E0 and E1/E2,
  per-probe outcomes saved under `output_root` for paired comparisons, rows upserted
  into `metrics.csv`; the test split is refused until the pre-registration is frozen).
  `scripts/06_evaluate.py` now stops with exit code 4 while settings are open. Shared
  Gate 1 / audit settings moved from `E0.yaml` to `base.yaml`; new evaluation keys
  (`split`, `max_new_tokens`, `generation_samples`, `rare_finding_pooling`). The metric
  definitions as implemented are listed in `docs/preregistration.md` section 6 as
  proposals. Run card RC-E0b (teacher cache on the training split, resumable over
  several sessions, plus E0 validation metrics) written and blocked. The caching stage
  now prints the cache manifest hash.
  Problems found by the tests: the clinical suite and the mask audit both wrote a metric
  called `n_probes` (renamed, and writing the same name twice now raises an error), and
  my first hand-computed expectations for recall by group were wrong (the code was right).
- Verified: ruff check, ruff format, pytest (643 passed).
- Commit: see `git log` on `research-loop`.
- Next: S05b student-selection runner and ranking (`scripts/select_student.py`), run
  cards RC-S0 to RC-S3.
- Needs the team: nothing new.

## Iteration 28: 2026-10-12 (interactive)
- Stage / item: S05b, student-selection runner, ranking and run cards.
- Did: `selection/ranking.py` (score S of D05: macro-F1, small-finding recall and image
  reliance, min-max scaled within the round across candidates that passed their gates;
  paired bootstrap over validation images with the same draws for every candidate, so the
  interval of the gap between neighbouring ranks reflects the paired design; advance by
  point estimate; one or several seeds per candidate), `selection/rounds.py` (R0
  feasibility: fp16 load, 20 synthetic probes, visual tokens, one LoRA step, gate G1 that
  cannot pass without CUDA; R1 zero-shot then the same short LoRA run on the same rows,
  blank-image gate G2; R2/R3 distillation from the cached teacher answers with a
  tokenizer-match check of at least 99 % that is recorded as a gate failure rather than an
  error; a candidate that cannot load in R0 is recorded as a G1 failure) and
  `selection/driver.py` (gates from the metrics files, tie-break on Round 0 fp16 peak
  memory, `ranking.csv`, `docs/results/SELECTION.md` from committed CSVs only).
  `scripts/select_student.py` (`--round`, `--rank`, `--report`, `--list`; exit code 4
  while settings are open). The evaluation runner now returns per-probe answers
  (`evaluate_model`, blank-image and mask-audit suites) so selection rounds and E0 share
  one scoring path. Run cards RC-S0 to RC-S3 written and blocked. Design note: the
  per-probe answers are never committed, so one person collects them to rank; the report
  step is separate because it needs only the committed CSVs.
  Problems found while writing it: the derived training length of R1/R2 was rejected as
  an open `training.steps` (now pre-filled and replaced from `compute_unit_b`); an
  undefined value (a failed LoRA step has no loss) would have broken the results schema
  (left out); the first test of the waiting rule hit the missing-gate-row error before the
  missing-file one, which is the better order, so the test now checks both.
- Verified: ruff check, ruff format, pytest.
- Commit: see `git log` on `research-loop`.
- Next: D06 recipe memo (training hyper-parameters, calibration bins, bootstrap
  resamples, image size, token budget), then S06 run cards and recovery code.
- Needs the team: nothing new (RC-S0 follows RC-M1).

## Iteration 29: 2026-10-12 (interactive)
- Stage / item: S09, pruning criteria and the scoring stage (stage 04).
- Did: `pruning/scoring.py` (exact unit ablation by hooks, which equals physical removal;
  first-order estimates from one gate per unit, so one backward pass scores every unit;
  loss scaling for fp16; probe objectives with coefficients w_c / n_c), `criteria/common.py`
  (weights of a unit, removal order with the tie rule, `feasible_sequence`, `removal_set`,
  score files) and all six criteria: magnitude (L1 of the unit's weights), activation
  (Wanda-style, sum of |W| times the input norm), Taylor (answer cross-entropy, one backward
  pass per row), output similarity (Khan et al.: one minus the mean cosine of the answer
  embeddings with the layer ablated), random (seeded by unit id) and clinical (exact, first
  order or mixed per unit type). `pruning/stage.py` runs stage 04: it resolves the
  `*_by_variant` tables (`config.apply_variant`), checks the settings, loads the E2 student
  (adapter merged), draws the nested calibration images from the training split, writes
  `unit_scores.csv` without a budget (the scores are shared by B1 to B3) and a settings file,
  and reuses the file when the settings are unchanged. `HFVLMWrapper.yes_no_logprobs` returns
  the yes/no log-probabilities with gradients; `forward_answer_logits` is public.
  Config: `mask_audit.fill` is 128 (D04); new open keys `pruning.removal.score_normalisation`
  and `pruning.removal.protected_layers.{first,last}` (for D07); E6 names the sentence encoder
  (`sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, revision read from the hub today).
  Checks on the tiny model: a zero gate equals ablation equals physical removal; the gate
  gradient equals a finite difference and, for a channel group, the Taylor term of its
  down-projection weights; the first-order estimate follows exact ablation (correlation above
  0.9 for channel groups).
  Cost note for D07: exact ablation of about 3200 units on about 700 probes would take about
  80 h on a T4 (estimate, from 0.12 s per probe), so the memo proposes exact for layers and
  KV groups and first-order for channel groups (about 3 h); the numbers are estimates until
  RC-M1 reports the scoring time per probe.
  Problems found by the tests: the settings file stored unit types as a tuple and compared
  unequal to its JSON form, so the reuse check never matched; the config helper of the tests
  rewrote the weight table that a test had just written; an unneeded channel-group size
  (`TODO`) was validated for the layers-only variant of A6.
- Verified: ruff check, ruff format, pytest (755 passed).
- Commit: see `git log` on `research-loop`.
- Next: recovery (`run_prune_and_recover`), matched-budget search, efficiency measurement,
  then the D06 and D07 memos.
- Needs the team: nothing new.

## Iteration 30: 2026-10-12 (interactive)
- Stage / item: S09, matched-budget search, shared recovery and efficiency measurement.
- Did: `evaluation/efficiency.py` (measurement protocol from `budgets.measurement`, timing
  samples as named ids or a seeded draw of validation images, one sweep that gives the cold
  latency, the warm latency, the throughput, the peak memory and, where the GPU reports power,
  the energy; parameters, state bytes and file size; stage 07 writes `efficiency.csv` and
  `efficiency_setup.json`). `budgets.py` (`search_matched_budget`: the feasible prefix of the
  removal order is bisected with the **measured** footprint of a freshly loaded model after
  physical removal; `BudgetSearch` records the trace and whether the result is inside the
  tolerance band; `budget_spec_for_run` refuses placeholders; the reference is read from the
  efficiency file of E2). `compression/recovery.py` (`RemovalSettings`, `plan_removal` with the
  protected layers, `run_recovery` = the shared LoRA + offline KD recipe with the data subset
  drawn from the seed alone and the adapter merged afterwards, `run_prune_and_recover` which
  saves the pruned model through the manifest format and writes `removal.json`,
  `budget_search.csv` and the numbers of the run). `load_run_model` now reloads the pruned
  arms (E5-E7, A1-A6) by loading E2 of the same seed and replaying the manifest.
  Config: `budgets.measurement` lost `image_resolution` and `prompt_length_tokens` (the image
  side is the global input and the prompts are the real questions of the timing samples); the
  outputs of E5-E7 list what the stages write (the mask audit is part of stage 06, so the
  `mask_audit.csv` entry of E7 was wrong and is gone).
  Checks on the tiny model: the stage removes units until the measured parameter count meets a
  fraction of the full model, recovers, saves, and the runner reloads it with the same
  parameter count; a budget the units cannot reach is an error; a unit step that overshoots
  the band is reported as outside the tolerance (the memo must size the tolerance to the
  largest unit step); settings that are open are listed together.
- Verified: ruff check, ruff format, pytest (780 passed).
- Commit: see `git log` on `research-loop`.
- Next: D06 and D07 memos (ask the team), then run card RC-G2.
- Needs the team: nothing new yet (the memos will ask).


## Iteration 31: 2026-10-12 (interactive)
- Stage / item: S09, D06 and D07 memos; applying the answered settings.
- Did: D06 (training recipe, recovery recipe, evaluation numbers) answered by Soheil with every
  recommendation accepted and written to `configs/training.yaml`, `configs/recovery.yaml` and
  `configs/base.yaml` (`steps` and `data_subset` stay open until the selection freeze, because
  they follow from the speed that RC-S0 measures). D07 (pruning settings) is partly answered:
  score per parameter, protected layers (first 2, last 1, every arm), channel groups of 128,
  128 calibration images, one-shot removal then recovery, one negative per positive, signed
  effects, probe batch 4 are in `configs/pruning.yaml` and `configs/experiments/E7.yaml`. Still
  open: how a unit is switched off while it is scored (waits for the T4 speed from RC-M1) and
  the budget section. The user asked what is run where and what it costs; the answer (nothing
  heavy in the agent session, all GPU work as run cards for the team) is in
  `docs/decisions_log.md`.
  Code: `config.collect_open_settings` merges the "settings still open" messages of several
  readers; criteria have a `check_settings` hook (clinical probe settings, the embedder and the
  token budget of output similarity); stage 04 now checks settings, criterion and image side
  together before any model is loaded (the image side is not needed when the scores of the same
  settings exist and are reused); recovery uses the shared helper.
  Tests: the stage and script tests expect the keys that are really still open
  (`pruning.clinical.ablation_method`, `inputs.image_longest_side`, `evaluation.max_new_tokens`
  for E6, the budget and training-length keys for stage 05).
- Verified: ruff check, ruff format, pytest (785 passed).
- Commit: see `git log` on `research-loop`.
- Next: ask the scoring-method question of D07 again in plain words, then the budget section;
  run cards RC-B0 and RC-G2.
- Needs the team: D07 section 1 and section 4 (asked in the session).

## Iteration 32: 2026-10-12 (interactive)
- Stage / item: S09 budgets, S05b student choice.
- Did: wrote the accepted budget settings to `configs/budgets.yaml` (metric `peak_memory`,
  tolerance 0.03, batch 1, 3 warm-up and 5 timed passes, 32 validation images); the timing
  hardware, `max_new_tokens` and B1 to B3 stay open. Answered Soheil's questions: the repo
  has only been run on a tiny fake model on CPU, no real data or model was downloaded here;
  the order 1 to 10 in D05 is a list order, not a ranking. After his question about whether the
  ten-model selection is needed for a paper about compression, wrote a proposed amendment in
  D05 (one student, Qwen2.5-VL-3B, Round 0 and image checks, Qwen3-VL-2B as the named fallback,
  no ranking). It is a proposal until Soheil confirms it; no selection code was removed.
  The stage and script tests expect the keys that are still open.
- Verified: ruff check, ruff format, pytest (785 passed).
- Commit: see `git log` on `research-loop`.
- Next: apply the D05 amendment after the yes; run cards RC-B0 and RC-G2; quantization (S08).
- Needs the team: confirm or reject the D05 amendment (Soheil); tell Amir and Farzan.

## Iteration 33: 2026-10-12 (interactive)
- Stage / item: S06, run cards for the student baselines and the budget reference.
- Did: `evaluation/efficiency.py` gained `state_bytes_by_part` (decoder layers, vision side,
  everything else; the parts add up to the state bytes) and the benchmark stage writes the
  three numbers next to the others, so that the E2 reference shows how much of the footprint
  pruning can reach (tested on the tiny model: removing a layer lowers only the decoder part,
  by exactly that layer's bytes). Run cards **RC-E1**, **RC-E2** and **RC-B0** written and
  queued as blocked (student choice, training length, RC-D1, open settings, RC-E0b). RC-E2
  tells the person to compare the cache manifest hash with the logged one, because training
  checks the shards against the cache's own manifest but not against the logged hash. RC-B0
  measures the E2 peak memory twice in fresh processes, because the budget metric is only
  usable if it repeats within the 3 % tolerance. D05 amendment: added that the image-use
  checks also run on the trained E1/E2 students, and that training several students would not
  show that the choice of student matters (that is a separate question for the team).
  S06 tasks ticked: recipe memo, efficiency stage, run cards.
  Run card **RC-G2** (Gate 2 pilot: E5 with three criteria, E6 and E7 at B2, seed 0, validation
  only) written and queued as blocked (RC-E2, RC-B0, the scoring method of D07 section 1, the
  finding weights).
- Verified: ruff check, ruff format, pytest (786 passed).
- Commit: see `git log` on `research-loop`.
- Next: the quantization stage (S08); the pre-registration draft (S07); the D05 amendment once
  Soheil answers ("pick one" or "tournament").
- Needs the team: unchanged (D05 amendment; RC-D1; RC-M1).
