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
