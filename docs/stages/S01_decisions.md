# S01: Decisions (domain, model pair, datasets, licence)

**Goal.** The open decisions that block real experiments are made by the team,
on verified evidence, and written into the configs.

**Depends on.** S00.

## Tasks

- [x] [agent] **D01 memo** `docs/decisions/D01_teacher.md` (teacher only; the student
  is chosen by the selection in S05b, memo D05). For each candidate teacher in
  `configs/models/*_teacher.yaml`: verified hub IDs and revisions, licence terms,
  parameter counts, fp16 and 4-bit memory against Kaggle's 2x T4 (16 GB each),
  whether the published Kvasir-VQA-x1 adapter exists and what it was trained on,
  and which student candidates share its tokenizer (table in D05). End with a
  recommendation and risks.
- [x] [agent] **D02 memo** `docs/decisions/D02_datasets.md`. For each candidate in
  `configs/data/`: verified source, licence, access type, size, label set, masks,
  patient/procedure IDs, official splits, image-overlap risks between datasets, and
  which role it can fill (main QA, masks, rare findings, stress test, external test).
  End with a proposed role assignment and a first draft of the finding list.
- [x] [agent] **D05 memo** `docs/decisions/D05_student_selection.md`: the 10 student
  candidates (verified facts), reserves, and the 10 -> 5 -> 3 -> 1 protocol.
- [x] [agent] **D03 memo** `docs/decisions/D03_licence.md`: short comparison of
  code licences (MIT, Apache-2.0, BSD-3-Clause). The repository is already public,
  so this is urgent.
- [x] [team] Answer D01 (T1) and D05 (teacher family only, keep size, weights
  accepted), 2026-10-12.
- [x] [team] Answer D02 and D03 (2026-10-12).
- [ ] [team:Amir] [team:Soheil] [team:Farzan] Each confirms in the inbox that they
  have their own access to the chosen datasets.
- [x] [agent] After the answers: fill the chosen configs (verified `source`,
  `revision`, `licence`), set their `status` to `DECIDED`, point
  `configs/base.yaml: components` at them, keep the others as candidates, and
  update `tests/test_configs.py` so decided configs must have non-TODO source and
  licence. Log the decisions in `docs/decisions_log.md`. Add the `LICENSE` file.

## Outputs

`docs/decisions/D01-D03` and `D05`, filled configs, decisions-log entries, `LICENSE`.

## Exit gate

Domain, teacher, datasets and licence decided and logged; the D05 selection
protocol answered; chosen configs carry verified sources; CI green. (The student
is decided at the end of S05b.)
