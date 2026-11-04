# Run card RC-E0b: teacher cache (training pool), after Gate 1

## 1. What and why

- Stage: S05
- Experiments: E0, no budget, seed 0
- Purpose: generate the teacher's answers and top-20 token log-probabilities once for the
  training pool (10,000 examples, `teacher_cache.pool_size`), for every distillation and
  recovery run (E2, E4, the recoveries). The E0 validation moved to RC-E0a (2026-10-31): it
  needs no cache.
- Status: `done` (2026-11-01): one session; the cache is the private dataset
  `e0-teacher-cache` (manifest `d6ba0cb9...`, section 6).

## 2. Requirements

- RC-E0a done, `gate1_pass` = 1 committed in `results/E0/nobudget/seed0/metrics.csv`, the
  verdict confirmed by the team. If Gate 1 fails, this card is not run (rule 3's extension of
  the teacher training comes first).
- Settings, all set: `teacher_cache.pool_size` 10,000 (= `training.data_subset`), `top_k` 20,
  `shard_size` 1,000, `max_new_tokens` by the D04 rule (46, computed from the committed
  summary), image side 672.
- RC-TT1 done: the adapter `e0-teacher-adapter`.

Platform: Kaggle notebook, **2 x T4**, Internet **on**. Inputs, all private: `medcompress-data`,
`e0-teacher-adapter`, and from session 2 on the output of the previous session.

Hours (**C**, not measured yet):
- The run times its first 100 answers at once and prints `the 10000 answers would take about
  <h> h (an estimate from these answers)`; the two numbers are also written to
  `results/E0/nobudget/seed0/metrics.csv` (`teacher_cache_first_answers*`).
- Bounds before that, from the measured one-at-a-time speed (2.66 tokens per second, RC-S0b
  run 2): at about 12 tokens per answer (an assumption from the median answer of 9 words)
  10,000 answers take **about 12.5 h** one at a time; the cache generates in batches of 8,
  whose speed is not measured, so probably less. If every answer ran to the 46-token cap,
  48 h (upper bound).
- So plan **two sessions or more**. `--stop-after-hours 9` stops cleanly between two shards of
  1,000 answers when the next would not fit; the next session continues from the saved
  shards. The total time is summed over the sessions (`teacher_cache_seconds`).

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil.
- **Kaggle settings:** GPU T4 x2 (both T4s); Internet on.
- **Attach (Add Input), all private datasets:** `medcompress-data` (the shared data of RC-D1); `e0-teacher-adapter` (the teacher adapter); from session 2 on, the previous session's output.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** the cells of section 3 (preferred: they stop cleanly and resume), or `notebooks/run_card.ipynb` with `CARD = "RC-E0b"`; it runs `python scripts/run_card.py --card RC-E0b` (inputs found by their manifests) without a time stop. Steps:
  - teacher cache
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E0_teacher_cache`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19). The session time limit is not verified here; RC-TT1's 11-h session ran.

## 3. Commands

Kaggle settings for every session: **GPU T4 x2**, Internet **on**, **Save Version -> Save &
Run All**, so that `/kaggle/working` (and the saved shards in it) becomes the version's
output.

**Every session, the same cell** (Add Input: `medcompress-data`, `e0-teacher-adapter`; from
session 2 on also the previous version's output: Your Work -> the notebook -> that version ->
Output). Nothing to edit:

```bash
%%bash
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone,
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # so removing the clone keeps
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # every output
python - <<'PY' || pip uninstall -y -q torchao   # peft needs torchao 0.16.0 or newer, or none
import sys
from importlib import metadata
try:
    version = tuple(int(x) for x in metadata.version("torchao").split(".")[:2])
except metadata.PackageNotFoundError:
    sys.exit(0)
sys.exit(0 if version >= (0, 16) else 1)
PY
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
DATA=$(python scripts/find_inputs.py data) || exit 1         # found by its manifest
ADAPTER=$(python scripts/find_inputs.py handoff E0_teacher_adapter) || exit 1
python scripts/prepare_data.py verify --data-root $DATA        # must print OK (8,511 files)
python scripts/handoff.py verify $ADAPTER                       # must print OK
grep gate1_pass results/E0/nobudget/seed0/metrics.csv          # its value must be 1.0 (RC-E0a)
PREV=$(find /kaggle/input -path '*cache/teacher/metadata.json' -printf '%h\n' 2>/dev/null | head -1)
echo "shards of a previous session: ${PREV:-none}"
python scripts/01_cache_teacher_outputs.py --exp E0 --seed 0 --data-root $DATA --from $ADAPTER \
    --stop-after-hours 9 ${PREV:+--resume-from "$PREV"} 2>&1 | tee -a /kaggle/working/rc_e0b.txt
code=${PIPESTATUS[0]}
echo "cache exit code: $code (3 = stopped between shards, run again; 0 = finished)"
if [ "$code" = 0 ]; then
  python scripts/handoff.py export --kind teacher_cache --exp E0 \
      --out /kaggle/working/handoff/E0_teacher_cache
fi
mkdir -p /kaggle/working/results_to_commit/results
cp -r results/E0 /kaggle/working/results_to_commit/results/
```

What it prints:
- the two `OK` lines and the `gate1_pass` row;
- after the first 100 answers (session 1 only): `teacher cache: the first 100 answers took
  <s> s; the 10000 answers would take about <h> h`. **Paste this line to the agent**: it is C;
- when the next shard would not fit in 9 hours: `STOPPED for a resume: <k> of 10 shards saved
  in /kaggle/working/cache/teacher`, and exit code 3. Run the same cell in a new version with
  this version's output attached;
- at the end: `teacher cache complete: 10000 samples, manifest <hash>`, then the hand-off
  `E0_teacher_cache`. Make it ONE private dataset, shared with Amir and Farzan.

`--stop-after-hours` counts the caching time of this session (not the install or the model
load), and a shard is started only if the longest shard of this session still fits. Lower it
if less GPU time is left this week.

Keep `cache/teacher/` on team storage (the private dataset); it is never committed.

## 4. What to commit back

- `results/E0/nobudget/seed0/metrics.csv` (from `results_to_commit/`, after each session):
  `teacher_cache_first_answers`, `teacher_cache_first_answers_seconds` and
  `teacher_cache_seconds` (summed over the sessions) are added.
- This card with section 6 filled, including the manifest hash and the dataset's name.

## 5. How the result is read

The manifest hash goes into `docs/decisions_log.md`; every later cache consumer (E2, E4, the
recoveries) checks against it.

## 6. Run report (filled by the person who ran it)

Reported by Soheil (2026-11-01).

- Run by / date: Soheil, by 2026-11-01.
- Commit actually used: not reported.
- Hardware and library versions: Kaggle T4 x2; versions not reported.
- The first-100 line (C): the first **104** answers (100 rounded up to whole batches of 8)
  took **86.7 s**; projected **2.3 h** for the 10,000 answers (0.83 s per answer, batched).
  The estimate before the run (about 12.5 h, from the one-answer-at-a-time speed) was about 5
  times too high: the batched generation is much faster.
- Sessions: one; it finished without a stop (exit code 0).
- Cache manifest hash: **`d6ba0cb9d166cb2ae9a5dd8af5a1bbd2eebba904df0ccd795aded0d65b4f15d8`**
  (`teacher cache complete: 10000 samples`). Hand-off `handoff/E0_teacher_cache`, the private
  Kaggle dataset **`e0-teacher-cache`**, shared with Amir and Farzan.
- Anything unusual: the session's `metrics.csv` rows (`teacher_cache_first_answers`,
  `teacher_cache_first_answers_seconds`, `teacher_cache_seconds`) are not in the committed
  file yet: commit `results_to_commit/results/E0/nobudget/seed0/metrics.csv` of this session
  (it holds RC-E0a's rows too).
