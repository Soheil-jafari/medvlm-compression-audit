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
