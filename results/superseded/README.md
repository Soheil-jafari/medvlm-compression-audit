# Superseded runs

Runs kept unchanged for the record but replaced under a later rule. The aggregator never reads
them as current rows; the paper reports them only where it says so.

- `inner_units_removable/`: RC-G2 session 1 (E5 magnitude and activation at B2, seed 0; Soheil,
  commit 4f54655): **the original pre-registered E5 (global per-parameter ranking, inner units
  of protected layers removable); appendix; outside H1-H4** (label of the amendment of
  2026-11-17). Made under the clarification of 2026-11-15, when the KV groups and channel groups
  inside the protected layers 0, 1 and 35 were removable. Superseded by the amendments of
  2026-11-16 (no unit of those layers is removed) and 2026-11-17 (E5 removes the same number of
  MLP channel groups from every unprotected layer). Moved here unchanged on 2026-11-16 (git history keeps the
  original paths). Their unit scores stay in `results/E5/<criterion>/nobudget/seed0/`: scores do
  not depend on the protection rule.
