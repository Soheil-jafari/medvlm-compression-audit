# Contributing

## Branches

| Prefix | Use | Example |
|---|---|---|
| `exp/<ID>-<short-name>` | Work on one experiment | `exp/E7-clinical-scoring` |
| `feat/<short-name>` | New infrastructure or method code | `feat/teacher-cache` |
| `fix/<short-name>` | Bug fixes | `fix/gqa-head-groups` |
| `docs/<short-name>` | Docs and paper | `docs/related-work-update` |

Branch from `main`, keep branches short-lived, and open a pull request back to `main`.

## Commits

- Imperative, present tense, at most about 72 characters in the subject:
  `Add GQA head grouping to unit enumeration`.
- Mention experiment IDs when relevant: `E7: write unit scores to CSV`.
- One logical change per commit; explain the "why" in the body when it is not obvious.
- Plain messages only, no trailers.

## Pull requests

- **One teammate reviews every PR** before it is merged. Nobody merges their own PR
  without a review.
- Fill in `.github/pull_request_template.md`.
- CI (ruff + pytest on CPU, Python 3.10 and 3.11) must be green.
- Changes to pre-registered items (`configs/preregistration/`, `docs/preregistration.md`,
  budgets after freezing) need approval from all three team members and a dated entry
  in `docs/decisions_log.md`.

## Never commit data

- No datasets, images, masks, manifests with restricted file names, cached teacher
  outputs, checkpoints, LoRA adapters, API keys or tokens.
- `.gitignore` and the pre-commit large-file check (> 5 MB) are a safety net, not a
  policy. Read `docs/data_policy.md`.
- Clear notebook outputs before committing.

## Before pushing

```bash
make lint test
```

`make format` fixes most lint issues. Install hooks once with `make install-dev`.

## Code rules

- Python >= 3.10, src layout, package `medcompress`.
- Never import torch, transformers, peft, accelerate or datasets at module top level.
  Use `from __future__ import annotations`, `typing.TYPE_CHECKING` imports for type
  hints, and import heavy libraries inside the functions that need them. CI runs with
  light dependencies only and `tests/test_imports.py` enforces this.
- Every public function and class has a docstring and type hints.
- Stubs raise `NotImplementedError("TODO: ...")` naming what they must do and which
  experiment IDs use them.
- Configs are plain YAML with `_base_` inheritance. Undecided values are `TODO`
  strings, never guesses.

## Adding a new experiment config

1. Register the ID in `src/medcompress/registry.py` (`ExperimentSpec`: title,
   description, kind, parent, variant_of, requires, variants, budgets, stages,
   scripts, outputs, trains).
2. Create `configs/experiments/<ID>.yaml` (or `configs/experiments/ablations/<ID>.yaml`
   for `A*`) that inherits via `_base_` from `../base.yaml` (plus `../training.yaml`,
   `../pruning.yaml`, `../recovery.yaml` as needed) or from the experiment it varies.
3. Fill the `experiment:` section so that it matches the registry entry exactly.
4. Add the experiment to `docs/experiment_plan.md` and, if it feeds the paper, to the
   relevant placeholder table in `paper/sections/`.
5. Run `make test`: `tests/test_configs.py` fails if the config and registry disagree.
6. Log the decision in `docs/decisions_log.md`.
