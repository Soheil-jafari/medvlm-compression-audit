# S00: Repository health and loop setup

**Goal.** A clean scaffold with green CI, and the documents the research loop runs on.

**Depends on.** Nothing.

## Tasks

- [x] [agent] Anchor the `data/` ignore rule. The unanchored rule had also hidden
  `src/medcompress/data/` and `configs/data/` from git, which broke every test.
- [x] [agent] Restore the data package stubs and the seven candidate dataset configs.
- [x] [agent] Add a test that source folders named `data` are never ignored, and that
  data, checkpoints and secrets always are.
- [x] [agent] Run CI on pushes to `research-loop` as well as `main`.
- [x] [agent] Write the control documents: `docs/FINAL_GOAL.md`,
  `docs/agent/{PROTOCOL,STATUS,ITERATION_LOG,INBOX}.md`, `docs/stages/`,
  `docs/runs/`, `docs/results/`, `docs/decisions/`.
- [ ] [team:Soheil] Review `research-loop` and open the pull request into `main`.

## Outputs

Green CI on `research-loop`; the documents above.

## Exit gate

CI green on `research-loop`, and the first pull request from `research-loop` merged
into `main`.
