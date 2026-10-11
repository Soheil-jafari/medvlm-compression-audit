# Team

| Member | Provisional role | Owns |
|---|---|---|
| Amir | Pruning, clinical criterion, efficiency measurements | `compression/pruning/`, `budgets.py`, `evaluation/efficiency.py`; E5-E8, A1-A6 |
| Soheil | Teacher/student setup, distillation, quantization, recovery training | `models/`, `compression/distillation/`, `compression/quantization/`, `compression/recovery.py`; E0-E4, E9 |
| Farzan | Datasets, masks, evaluation harness, statistics | `data/`, `evaluation/` (except efficiency); data cards; aggregation |

Roles are provisional and will be revisited once the domain and model pair are decided.

## Review rules

- Every pull request is reviewed by one teammate before merging.
- The owner of a module should not be its only reviewer for changes to the primary
  endpoints or the clinical criterion: ask a second teammate.
- Changes to pre-registered items need approval from all three members and an entry in
  `docs/decisions_log.md`.
- Evaluation and statistics code used for the primary endpoints is written or reviewed
  by someone other than the person running E7.
- Results enter the paper only through `scripts/08_aggregate_results.py` outputs, never
  hand-copied numbers.
