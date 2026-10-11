## Summary

<!-- What does this PR change and why? -->

## Experiment IDs affected

<!-- e.g. E7, A1-A6, or "none (infrastructure)" -->

## Type of change

- [ ] Infrastructure / tooling
- [ ] New or changed experiment config
- [ ] Method code (compression, evaluation, data)
- [ ] Docs / paper

## Checklist

- [ ] `make lint test` passes locally
- [ ] No data, images, checkpoints, cached outputs or secrets are committed
- [ ] New experiment IDs are registered in `src/medcompress/registry.py` and have a config
- [ ] Pre-registered items (`configs/preregistration/`, `docs/preregistration.md`) are unchanged, or the change is logged in `docs/decisions_log.md`
- [ ] One teammate has been requested as reviewer
