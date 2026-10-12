# Run card RC-S3: Round 3, top 3 with the E2 recipe, two seeds

## 1. What and why

- Stage: S05b (decision D05, Round 3)
- Experiments: none (selection runs on the validation split; no budget)
- Purpose: train the three candidates that advanced from Round 2 with exactly the E2
  recipe (the training section of the config, unchanged), two seeds each, score them on
  the whole validation split with the lesion-mask audit and pick the student that
  every compression arm will start from.
- Status: `blocked` (see section 2)

## 2. Requirements

- RC-S2 ranked and committed (`results/selection/R2/ranking.csv`); the candidates are
  the ones with `advance` = 1 there.
- The E2 recipe is final: `training.steps`, `training.data_subset` and the other training
  values are set (they define E2 and are not touched by this round).
- Platform, dataset access, teacher cache and the remote-code rule: as in RC-S2.
- Expected runtime: three candidates x two seeds, one full E2 recipe each; the agent
  fills in the estimate when it sets the card to `ready`.

## 3. Commands

Environment as in RC-S2. For each of the three candidates and each seed:

```bash
for c in <candidate>; do
  for s in 0 1; do
    CUDA_VISIBLE_DEVICES=0 python scripts/select_student.py --round R3 --candidate $c --seed $s
  done
  rm -rf $HF_HOME/hub/models--*
done
python -m pytest -q tests/test_results.py                            # must pass before you commit
```

Upload `outputs/selection` as a private dataset (`<yourname>-selection-r3`).

**Ranking (one person).** As in RC-S1 with `--rank R3` and the `*-selection-r3` datasets.
The score of a candidate is the mean of its two seeds; the ranking stops if a seed is missing.

## 4. What to commit back

- `results/selection/R3/<candidate>/metrics.csv` (both seeds are in the same file)
- `results/selection/R3/ranking.csv` and the updated `docs/results/SELECTION.md`
- This card with section 6 filled.

## 5. How the result is read

The candidate with the highest score S is the proposed student. The team confirms the
choice in `docs/decisions_log.md` (including whether the gap to the runner-up is
separated by the interval, and the tie-breaks if it is not); the agent then sets that
candidate's config to `DECIDED` with a pinned revision, points
`configs/base.yaml: components.student` at it and updates the method section of the paper.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions:
- Per candidate and seed: wall time, peak GPU memory, final training loss, errors or retries:
- Name of the private dataset with your `outcomes.npz` files:
- Anything unusual:
