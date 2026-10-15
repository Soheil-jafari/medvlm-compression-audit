# S04: Model wrappers and unit surgery

**Goal.** The chosen teacher and student can be loaded, asked questions, scored on
yes/no probes in one forward pass, and have units ablated (for scoring) or
physically removed (for the final model).

**Depends on.** S01 (teacher decided). The student is chosen in S05b, so the
loading and yes/no scoring parts must work for all 10 student candidates
(`configs/student_selection.yaml`); pruning surgery is only needed for the
Round 2 models and the final student.

## Tasks

- [x] [agent] `models/loading.py` and a `VLMWrapper` (`models/hf.py`) for the teacher family and a
  generic path for the 10 student candidates:
  `generate`, `score_yes_no` (log-probabilities of the answer tokens at the answer
  position, no free generation), `list_prunable_units` (decoder layers, attention
  head groups that respect grouped key-value heads, MLP channel groups; projector
  always kept).
- [x] [agent] `compression/pruning/units.py` (`validate_removal_set`) and
  `compression/pruning/apply.py`: reversible ablation for scoring; physical removal
  of layers, head groups and channel groups that yields a smaller saved model.
- [x] [agent] Tests with a tiny randomly initialised model of the same architecture
  class (built from a config, no weight download), marked `heavy` and skipped in CI
  when torch is missing. The agent runs them on CPU in its own session.
- [x] [agent] Run card **RC-M1** (`docs/runs/RC-M1_model_smoke_test.md`, script `scripts/smoke_models.py`): GPU smoke test on Kaggle. Load teacher and student
  in the planned precision, check tokenizer identity at runtime, score 8 probes,
  remove one layer physically, record peak memory and timings to
  `results/smoke/RC-M1.csv`.
- [x] [team:Soheil] Execute RC-M1 and commit the CSV and the completed run card (done
  2026-10-15: all 8 checks met; the exit gate below is met).

## Outputs

Wrapper, surgery code, heavy tests, RC-M1 results.

## Exit gate

RC-M1 done; the tokenizers are identical; physical removal reduces measured memory;
the teacher fits the Kaggle plan.
