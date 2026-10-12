# Run cards

GPU jobs are run by team members on their own dataset access (Kaggle, Colab or a
rented GPU). The agent prepares each job as a run card (`TEMPLATE.md`); the person
who runs it commits the result CSVs and the completed card. The agent picks up the
results in its next iteration.

## Queue

| Card | Stage | What | Who | Status |
|---|---|---|---|---|
| RC-D1 | S03 | Data preparation: manifests, patient-level splits, aggregate statistics | Farzan | not written yet |
| RC-M1 | S04 | GPU smoke test: load pair, score probes, remove one layer | Soheil | not written yet |
| RC-E0a | S05 | Gate 1: teacher image-reliance checks (validation) | Soheil | not written yet |
| RC-E0b | S05 | Teacher cache (train) and E0 validation metrics | Soheil | not written yet |
| RC-S0 | S05b | Student candidates: feasibility of all 10 on one T4 | Soheil | not written yet |
| RC-S1 | S05b | Round 1 screen, 10 candidates | all three | not written yet |
| RC-S2 | S05b | Round 2, top 5 rebuilt with the full recipe | all three | not written yet |
| RC-S3 | S05b | Round 3, top 3 with the E2 recipe, 2 seeds | all three | not written yet |
| RC-E1 / RC-E2 | S06 | Student training, 3 seeds each | Soheil | not written yet |
| RC-B0 | S06 | E2 reference footprint for the budgets | Amir | not written yet |
| RC-E3 / RC-E4 / RC-E9 | S08 | Quantization arms | Soheil | not written yet |
| RC-G2 | S09 | Gate 2 pilot at B2 (validation) | Amir | not written yet |
| RC-E5 ... RC-E8 | S09 | Pruning grid | Amir | not written yet |
| RC-A1 ... RC-A6 | S10 | Ablations at B2 | Amir | not written yet |
| RC-X1 | S11 | External test | Farzan | not written yet |
| RC-T1 | S11 | Final timing on the rented GPU | Amir | not written yet |

Statuses: `not written yet` -> `ready` -> `running` -> `done` (or `failed`, with the
reason in the card).
