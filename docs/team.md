# Team

**Authors:** Soheil and Amir. **Farzan left the project on 2026-11-08** and is not an author;
he lends one Kaggle account (GPU quota only); he ran RC-E1 seed 1. Decisions of 2026-11-08: pre-registration section 11 item 14.

| Member | Role | Owns |
|---|---|---|
| Soheil | Lead: teacher/student setup, distillation, quantization, recovery training; the data, the evaluation harness and the statistics (Farzan's former modules; the research lead's assignment, 2026-11-08) | `models/`, `compression/distillation/`, `compression/quantization/`, `compression/recovery.py`, `data/`, `evaluation/` (except efficiency); E0-E4, E9, E10; data cards; aggregation |
| Amir | Pruning, clinical criterion, efficiency measurements; the PolypGen download | `compression/pruning/`, `budgets.py`, `evaluation/efficiency.py`; E5-E8, A1-A6; RC-X2's download |

**GPU quota:** three Kaggle accounts x 30 h = **90 h per week**: Soheil's, Amir's, and one lent
by Farzan (account 3, its sessions started by Soheil). **Every hand-off dataset is shared to
Soheil's account**, so any session can attach it.

## Review rules

- Every pull request is reviewed by the other author before merging.
- The owner of a module should not be its only reviewer for changes to the primary
  endpoints or the clinical criterion: the other author reviews them.
- Changes to pre-registered items need the approval of both authors (Soheil and Amir) and an
  entry in `docs/decisions_log.md`.
- Evaluation and statistics code used for the primary endpoints is written or reviewed
  by someone other than the person running E7.
- Results enter the paper only through `scripts/08_aggregate_results.py` outputs, never
  hand-copied numbers.
