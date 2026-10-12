# S04: Model wrappers and unit surgery

**Goal.** The chosen teacher and student can be loaded, asked questions, scored on
yes/no probes in one forward pass, and have units ablated (for scoring) or
physically removed (for the final model).

**Depends on.** S01 (model pair decided).

## Tasks

- [ ] [agent] `models/loading.py` and a `VLMWrapper` for the chosen family:
  `generate`, `score_yes_no` (log-probabilities of the answer tokens at the answer
  position, no free generation), `list_prunable_units` (decoder layers, attention
  head groups that respect grouped key-value heads, MLP channel groups; projector
  always kept).
- [ ] [agent] `compression/pruning/units.py` (`validate_removal_set`) and
  `compression/pruning/apply.py`: reversible ablation for scoring; physical removal
  of layers, head groups and channel groups that yields a smaller saved model.
- [ ] [agent] Tests with a tiny randomly initialised model of the same architecture
  class (built from a config, no weight download), marked `heavy` and skipped in CI
  when torch is missing. The agent runs them on CPU in its own session.
- [ ] [agent] Run card **RC-M1**: GPU smoke test on Kaggle. Load teacher and student
  in the planned precision, check tokenizer identity at runtime, score 8 probes,
  remove one layer physically, record peak memory and timings to
  `results/smoke/RC-M1.csv`.
- [ ] [team:Soheil] Execute RC-M1 and commit the CSV and the completed run card.

## Outputs

Wrapper, surgery code, heavy tests, RC-M1 results.

## Exit gate

RC-M1 done; the tokenizers are identical; physical removal reduces measured memory;
the teacher fits the Kaggle plan.
