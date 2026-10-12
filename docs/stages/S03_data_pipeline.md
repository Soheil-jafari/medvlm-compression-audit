# S03: Data pipeline

**Goal.** The chosen datasets can be read, mapped to the finding list, split at
patient level, stratified by lesion size and deduplicated, on the team's own
machines, with only aggregate statistics coming back to the repository.

**Depends on.** S01 (datasets decided), S02 (data pure functions).

## Tasks

- [ ] [agent] One `VQADataset` reader per chosen dataset, mapping its labels to the
  domain finding list. Tests use tiny synthetic directory fixtures that copy the
  documented layout (file names and fields only; generated blank images).
- [ ] [agent] Manifest builder and verifier (`data/manifests.py`).
- [ ] [agent] A data-preparation entry point (for example `scripts/prepare_data.py`)
  that writes manifests and splits **outside** the repository and writes only
  aggregate statistics to `results/data/summary.csv`: counts per split, patients
  per split, prevalence per finding, lesion-area quantiles, duplicate counts,
  manifest and split hashes. No per-image rows.
- [ ] [agent] Run card **RC-D1** in `docs/runs/` for the preparation run.
- [ ] [team:Farzan] Execute RC-D1 on own dataset access; commit
  `results/data/summary.csv` and the completed run card.
- [ ] [agent] Data cards in `docs/data_cards/` from the verified documentation and
  the RC-D1 aggregates.
- [ ] [agent] Proposal for the finding list, prevalence groups and size thresholds,
  computed from RC-D1 training-split statistics (input to S07).

## Outputs

Readers, preparation script, `results/data/summary.csv`, data cards, finding-list
proposal.

## Exit gate

RC-D1 done; splits verified patient-disjoint; external test deduplicated with the
removed count recorded; data cards complete; table T1 can be generated.
