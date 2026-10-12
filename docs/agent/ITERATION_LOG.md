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
