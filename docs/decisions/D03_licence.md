# D03: Code licence

**Status:** decided 2026-10-12 (Soheil): **MIT**. `LICENSE` added with the copyright
line "Amir, Soheil Jafari, Farzan" (update with full names if wanted).
**Stage:** S01. **Prepared:** 2026-10-12 (iteration 2).
**Urgency:** high. The repository is public but carries no licence, so by default
nobody else may legally reuse, modify or redistribute the code.

## Question

Which licence covers the **code** in this repository (`src/`, `scripts/`, `tests/`,
`configs/`, `notebooks/`)?

Out of scope for this memo (each has its own terms, set by its owner, and is never
relicensed by us):

- datasets, images, masks and annotations (see `docs/data_policy.md` and D02);
- model weights and adapters (see D01);
- the paper text and figures (publisher or preprint-server terms; can be decided
  later, for example a Creative Commons licence for the preprint).

## Options

All three are permissive, OSI-approved licences. Each allows use, modification and
redistribution, including commercial use, and disclaims warranty and liability.

| | MIT | Apache-2.0 | BSD-3-Clause |
|---|---|---|---|
| Main condition | Keep the copyright and permission notice in all copies or substantial portions | Ship a copy of the licence; mark modified files; keep existing notices; carry forward a `NOTICE` file if there is one | Keep the copyright notice, conditions and disclaimer in source and binary redistributions |
| Explicit patent grant | No (text does not mention patents) | Yes (section 3), ends for anyone who sues claiming the work infringes a patent | No (text does not mention patents) |
| Name / trademark clause | No | Trademarks not granted (section 6) | Contributors' names may not be used to endorse derived products without permission (clause 3) |
| Length | Very short | Long | Short |

### Evidence

- MIT: <https://opensource.org/license/mit> (accessed 2026-10-12). Permission to use,
  copy, modify, merge, publish, distribute, sublicense and sell, on condition that
  the copyright and permission notice is included; no mention of patents.
- Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0> (accessed 2026-10-12).
  Section 3 patent licence with termination on patent litigation; section 4
  redistribution conditions (licence copy, change notices, retained notices,
  `NOTICE` file); section 6 trademarks.
- BSD-3-Clause: <https://opensource.org/license/bsd-3-clause> (accessed 2026-10-12).
  Three conditions: retain the notice in source, reproduce it in binary
  distributions, no endorsement using the holders' or contributors' names; no
  mention of patents.

## Points that matter for this project

1. **Research code, academic reuse.** All three let other groups reuse the
   pruning criterion and evaluation harness. None of them restricts use on medical
   data or for medical products; the code is research software and the disclaimer of
   warranty in each licence applies.
2. **Patents.** The clinically weighted pruning criterion is a method. Apache-2.0 is
   the only option that gives users an explicit patent licence from contributors.
   If no team member or institution intends to patent the method, this difference
   is small in practice; if anyone might, check with the institution before
   choosing (TODO: team to confirm whether any institutional IP policy applies to
   the three members).
3. **Upstream code.** The repository does not vendor third-party source today. If
   code from another project is copied in later (for example a pruning or
   quantization reference implementation), its licence must be compatible and its
   notices kept. Permissive upstream code can be included under any of the three;
   Apache-2.0 upstream code brings its `NOTICE` obligations with it.
4. **Overhead.** MIT and BSD-3-Clause need one `LICENSE` file. Apache-2.0 needs the
   `LICENSE` file and, by convention, a short `NOTICE` file; per-file headers are
   optional.
5. **Institutions.** Some universities require or prefer a specific licence for
   staff code. TODO: each member checks their institution's policy.

## Recommendation

**MIT** if no member has an institutional requirement or patent interest: it is the
shortest, is the most common choice for research code, and imposes the least on
users. **Apache-2.0** if anyone wants the explicit patent grant or an institution
asks for it. BSD-3-Clause adds only the non-endorsement clause over MIT and is not
needed here.

This is a team decision; the agent will not add a licence until the answer is in the
inbox.

## Risks

- Delay: until a licence is added, the public code is "all rights reserved" by
  default and others cannot reuse it.
- Changing later is possible for the authors (all three members hold the
  copyright), but copies already taken keep the licence they were taken under.
- External contributions accepted before the choice would complicate a later change.
  Do not merge outside contributions until D03 is decided.

## What changes in the repository once decided

- Add `LICENSE` with the full licence text, copyright line
  "Copyright (c) 2026 Amir, Soheil Jafari, Farzan" (TODO: full names and order).
  For Apache-2.0 also add a short `NOTICE`.
- `pyproject.toml`: replace the licence TODO with the SPDX identifier and add the
  matching classifier.
- `CITATION.cff`: add `license: <SPDX id>`.
- `README.md` "Policies": link the licence and state that data and model weights
  keep their own licences.
- `docs/decisions_log.md`: log the decision with its date; tick the box in
  `docs/FINAL_GOAL.md` and `docs/stages/S01_decisions.md`.

## Answer format for the inbox

`- YYYY-MM-DD (name): D03 decided: <MIT | Apache-2.0 | BSD-3-Clause>; copyright line "<...>".`
