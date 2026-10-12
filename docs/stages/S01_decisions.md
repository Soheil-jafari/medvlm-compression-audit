# S01: Decisions (domain, model pair, datasets, licence)

**Goal.** The open decisions that block real experiments are made by the team,
on verified evidence, and written into the configs.

**Depends on.** S00.

## Tasks

- [ ] [agent] **D01 memo** `docs/decisions/D01_model_pair.md`. For each candidate pair
  in `configs/models/candidate_pairs.yaml`: verified hub IDs and revisions, licence
  terms, whether teacher and student tokenizers are identical (compare the
  tokenizer files in both model repositories), parameter counts, fp16/bf16 and 4-bit
  memory estimates against one T4 (16 GB), whether the published Kvasir-VQA-x1
  adapter exists and what it was trained on. End with a recommendation and risks.
- [ ] [agent] **D02 memo** `docs/decisions/D02_datasets.md`. For each candidate in
  `configs/data/`: verified source, licence, access type, size, label set, masks,
  patient/procedure IDs, official splits, image-overlap risks between datasets, and
  which role it can fill (main QA, masks, rare findings, stress test, external test).
  End with a proposed role assignment and a first draft of the finding list.
- [ ] [agent] **D03 memo** `docs/decisions/D03_licence.md`: short comparison of
  code licences (MIT, Apache-2.0, BSD-3-Clause). The repository is already public,
  so this is urgent.
- [ ] [team] Answer D01, D02 and D03 in `docs/agent/INBOX.md`.
- [ ] [team:Amir] [team:Soheil] [team:Farzan] Each confirms in the inbox that they
  have their own access to the chosen datasets.
- [ ] [agent] After the answers: fill the chosen configs (verified `source`,
  `revision`, `licence`), set their `status` to `DECIDED`, point
  `configs/base.yaml: components` at them, keep the others as candidates, and
  update `tests/test_configs.py` so decided configs must have non-TODO source and
  licence. Log the decisions in `docs/decisions_log.md`. Add the `LICENSE` file.

## Outputs

`docs/decisions/D01-D03`, filled configs, decisions-log entries, `LICENSE`.

## Exit gate

Domain, model pair, datasets and licence decided and logged; chosen configs carry
verified sources; CI green.
