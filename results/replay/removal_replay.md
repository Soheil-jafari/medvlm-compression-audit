# Offline replay of the removal sets (estimate)

Written by `scripts/replay_removal.py` (amendment of 2026-11-17). Footprints are
**estimates**: E2's measured peak (7618471936 bytes) minus 2 bytes per removed
parameter; E2's decoder layers hold 5549543424 bytes. The search is stage 05's
(smallest prefix or k at or below target x 1.03). Seed 0 scores.

k is the number of channel groups removed from every unprotected layer (layers 2-34) of
the `uniform_mlp` allocation, out of the groups each layer has; its share of the MLP
width is k divided by that number. "at target": the smallest k at or below the target
itself; "in band": every k within +-3 % of the target. The search takes the smallest k
at or below target x 1.03, the top of the band.

| arm | allocation | budget | k (MLP width) | layers | KV groups | channel groups | removed params | est. gap to target | in band |
|---|---|---|---|---|---|---|---|---|---|
| E5:magnitude | uniform_mlp | B1 | 13 of 86 (15.1%); at target 17; in band 13-19 | 0 | 0 | 429 | 337,379,328 | +2.32% | yes |
| E5:magnitude | uniform_mlp | B2 | 29 of 86 (33.7%); at target 33; in band 29-35 | 0 | 0 | 957 | 752,615,424 | +2.68% | yes |
| E5:magnitude | uniform_mlp | B3 | 46 of 86 (53.5%); at target 49; in band 46-51 | 0 | 0 | 1518 | 1,193,803,776 | +2.14% | yes |
| E5:activation | uniform_mlp | B1 | 13 of 86 (15.1%); at target 17; in band 13-19 | 0 | 0 | 429 | 337,379,328 | +2.32% | yes |
| E5:activation | uniform_mlp | B2 | 29 of 86 (33.7%); at target 33; in band 29-35 | 0 | 0 | 957 | 752,615,424 | +2.68% | yes |
| E5:activation | uniform_mlp | B3 | 46 of 86 (53.5%); at target 49; in band 46-51 | 0 | 0 | 1518 | 1,193,803,776 | +2.14% | yes |
| E5R | uniform_mlp | B1 | 13 of 86 (15.1%); at target 17; in band 13-19 | 0 | 0 | 429 | 337,379,328 | +2.32% | yes |
| E5R | uniform_mlp | B2 | 29 of 86 (33.7%); at target 33; in band 29-35 | 0 | 0 | 957 | 752,615,424 | +2.68% | yes |
| E5R | uniform_mlp | B3 | 46 of 86 (53.5%); at target 49; in band 46-51 | 0 | 0 | 1518 | 1,193,803,776 | +2.14% | yes |
| E7 | global | B1 |  | 0 | 5 | 370 | 314,579,200 | +3.00% | yes |
| E7 | global | B2 |  | 0 | 11 | 879 | 743,192,320 | +3.00% | yes |
| E7 | global | B3 |  | 4 | 13 | 1050 | 1,195,419,904 | +2.08% | yes |
| E7U | uniform_mlp | B1 | 13 of 86 (15.1%); at target 17; in band 13-19 | 0 | 0 | 429 | 337,379,328 | +2.32% | yes |
| E7U | uniform_mlp | B2 | 29 of 86 (33.7%); at target 33; in band 29-35 | 0 | 0 | 957 | 752,615,424 | +2.68% | yes |
| E7U | uniform_mlp | B3 | 46 of 86 (53.5%); at target 49; in band 46-51 | 0 | 0 | 1518 | 1,193,803,776 | +2.14% | yes |

## Share of each decoder layer removed at B2 (share of the layer's parameters)

| layer | E5:magnitude | E5:activation | E5R | E7 | E7U |
|---|---|---|---|---|---|
| 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | 29.6% | 29.6% | 29.6% | 3.1% | 29.6% |
| 3 | 29.6% | 29.6% | 29.6% | 29.6% | 29.6% |
| 4 | 29.6% | 29.6% | 29.6% | 11.2% | 29.6% |
| 5 | 29.6% | 29.6% | 29.6% | 30.6% | 29.6% |
| 6 | 29.6% | 29.6% | 29.6% | 20.4% | 29.6% |
| 7 | 29.6% | 29.6% | 29.6% | 26.5% | 29.6% |
| 8 | 29.6% | 29.6% | 29.6% | 13.3% | 29.6% |
| 9 | 29.6% | 29.6% | 29.6% | 23.5% | 29.6% |
| 10 | 29.6% | 29.6% | 29.6% | 22.4% | 29.6% |
| 11 | 29.6% | 29.6% | 29.6% | 29.6% | 29.6% |
| 12 | 29.6% | 29.6% | 29.6% | 24.5% | 29.6% |
| 13 | 29.6% | 29.6% | 29.6% | 30.6% | 29.6% |
| 14 | 29.6% | 29.6% | 29.6% | 30.6% | 29.6% |
| 15 | 29.6% | 29.6% | 29.6% | 35.7% | 29.6% |
| 16 | 29.6% | 29.6% | 29.6% | 37.8% | 29.6% |
| 17 | 29.6% | 29.6% | 29.6% | 42.9% | 29.6% |
| 18 | 29.6% | 29.6% | 29.6% | 34.7% | 29.6% |
| 19 | 29.6% | 29.6% | 29.6% | 33.7% | 29.6% |
| 20 | 29.6% | 29.6% | 29.6% | 29.6% | 29.6% |
| 21 | 29.6% | 29.6% | 29.6% | 36.7% | 29.6% |
| 22 | 29.6% | 29.6% | 29.6% | 34.7% | 29.6% |
| 23 | 29.6% | 29.6% | 29.6% | 33.7% | 29.6% |
| 24 | 29.6% | 29.6% | 29.6% | 26.5% | 29.6% |
| 25 | 29.6% | 29.6% | 29.6% | 41.8% | 29.6% |
| 26 | 29.6% | 29.6% | 29.6% | 37.8% | 29.6% |
| 27 | 29.6% | 29.6% | 29.6% | 40.8% | 29.6% |
| 28 | 29.6% | 29.6% | 29.6% | 37.8% | 29.6% |
| 29 | 29.6% | 29.6% | 29.6% | 37.8% | 29.6% |
| 30 | 29.6% | 29.6% | 29.6% | 24.5% | 29.6% |
| 31 | 29.6% | 29.6% | 29.6% | 20.4% | 29.6% |
| 32 | 29.6% | 29.6% | 29.6% | 22.4% | 29.6% |
| 33 | 29.6% | 29.6% | 29.6% | 28.6% | 29.6% |
| 34 | 29.6% | 29.6% | 29.6% | 30.6% | 29.6% |
| 35 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Overlap of the B2 removal sets

Share of the row arm's removed units that the column arm also removes.

| arm | E5:magnitude | E5:activation | E5R | E7 | E7U |
|---|---|---|---|---|---|
| E5:magnitude | 100.0% | 68.3% | 35.1% | 30.4% | 33.4% |
| E5:activation | 68.3% | 100.0% | 34.9% | 29.9% | 33.3% |
| E5R | 35.1% | 34.9% | 100.0% | 29.9% | 33.5% |
| E7 | 32.7% | 32.1% | 32.1% | 100.0% | 90.4% |
| E7U | 33.4% | 33.3% | 33.5% | 84.1% | 100.0% |

## The estimate against measured footprints

Session 1 (superseded design) measured its search's footprints. Measured drop: the unpruned footprint minus the accepted one. Estimate: 2 bytes per removed parameter.

| run | removed params | measured drop (bytes) | estimate (bytes) | ratio |
|---|---|---|---|---|
| superseded/inner_units_removable/E5/activation/B2/seed0 | 737,732,352 | 1481538560 | 1475464704 | 1.0041 |
| superseded/inner_units_removable/E5/magnitude/B2/seed0 | 781,784,320 | 1571415040 | 1563568640 | 1.0050 |
