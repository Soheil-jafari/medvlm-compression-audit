# S12: Paper and release

**Goal.** A complete, compiling manuscript built only from generated results, and a
release that lets others reproduce it with their own data access.

**Depends on.** S11 for results. Method, related work and setup can be drafted at
any time (work-ahead).

## Tasks

- [ ] [agent] Draft `03_method.tex` from the implemented code and
  `docs/project_definition.md`.
- [ ] [agent] Draft `02_related_work.tex` using only references verified against
  the primary source; record each check in `docs/related_work.md`.
- [ ] [agent] Draft `04_experimental_setup.tex` once S01 and S07 are done.
- [ ] [agent] Draft `05_results.tex`, `06_discussion.tex`, `00_abstract.tex` and
  `07_conclusion.tex` in the shape (A or B) the verdicts support, using only
  `\input` tables and `numbers.tex` macros.
- [ ] [agent] Appendix: per-finding tables, A4 curve, calibration plots,
  hyper-parameters, pre-registration and deviations, compute used.
- [ ] [agent] Novelty re-check (search terms, date, closest papers, how we differ)
  in `docs/related_work.md`, within two weeks of submission.
- [ ] [agent] Model cards and final data cards; reproduction section in `README.md`.
- [ ] [agent] `make paper` compiles; no `\todo{}` left.
- [ ] [team] Choose the venue and apply its template; internal review by all three
  members; sign `docs/FINAL_GOAL.md`.

## Outputs

`paper/main.pdf` (built locally, not committed), complete cards, README.

## Exit gate

`docs/FINAL_GOAL.md` fully ticked and signed. The loop sets STATUS to `DONE`.
