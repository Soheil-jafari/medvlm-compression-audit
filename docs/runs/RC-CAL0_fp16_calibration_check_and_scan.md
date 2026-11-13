# Run card RC-CAL0: the fp16 calibration check and the fp16 vision scan of every split

## 1. What and why

- Stage: before any compression run (amendment of 2026-11-13, pre-registration section 10)
- Experiments: the calibration set of seed 0 (student: E2 seed 0; teacher: E9's parent) and
  the fp16 vision tower on every image of validation, the internal test (filtered and
  unfiltered) and the training split
- Purpose: (1) apply the amendment: the calibration images whose fp16 forward is not finite are
  replaced, and the list is committed (`results/calibration/seed0_replaced.csv`); every stage
  that draws calibration images refuses to run until it is; (2) learn whether any **test** image
  is affected before RC-R1 or any test evaluation (section 5).
- Status: `ready` (2026-11-13)

## 2. Requirements

- **Who:** Soheil (account 1).
- Part A, the scan: **one T4** (the vision tower alone, about 0.7 B parameters, also the
  teacher's), Internet on (the teacher's vision weights are read from its checkpoint). Attach
  `medcompress-data`. Estimate: about 10 min per encoder for about 6,450 images (not measured).
- Part B, the calibration check: **T4 x2** (the teacher's full forward needs both T4s). Attach
  `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-adapter`. Estimate: student 256 images
  and teacher 128 images, about 10-15 min with the loads (not measured).
- Seeds 1 and 2 (student, 128 images) are checked later, with their E2 adapters, before their
  runs (phase C).

## 3. Commands

Each part is one cell, after these lines (clone and install, as every card):

```bash
%%bash
set -euo pipefail
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
```

**Part A (one T4), then:**

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/scan_vision_finite.py --model student 2>&1 | tee /kaggle/working/scan_student.txt
python scripts/scan_vision_finite.py --model teacher 2>&1 | tee /kaggle/working/scan_teacher.txt
```

**Part B (T4 x2), then:**

```bash
DATA=$(python scripts/find_inputs.py data)
E2=$(python scripts/find_inputs.py handoff E2_seed0_adapter)
TA=$(python scripts/find_inputs.py handoff teacher_adapter)
CUDA_VISIBLE_DEVICES=0 python scripts/check_calibration.py --exp E3 --seed 0 --size 256 \
  --data-root "$DATA" --from "$E2" 2>&1 | tee /kaggle/working/check_calibration_student.txt
python scripts/check_calibration.py --exp E9 --seed 0 --size 128 \
  --data-root "$DATA" --from "$TA" 2>&1 | tee /kaggle/working/check_calibration_teacher.txt
mkdir -p /kaggle/working/results_to_commit/results/calibration
cp results/calibration/seed0_replaced.csv /kaggle/working/results_to_commit/results/calibration/
```

## 4. What to send back

- Part A: the two `scan_*.txt` outputs (counts per split and the sample ids of the non-finite
  images) and `/kaggle/working/vision_scan_student.csv`, `vision_scan_teacher.csv`.
- Part B: `results_to_commit/results/calibration/seed0_replaced.csv` (the agent commits it
  unchanged) and the two `check_calibration_*.txt` outputs.

## 5. How the result is read

- **Calibration.** The student's replaced ids are expected to include the two images RC-E3
  found among the first 128 (more may appear among images 129-256, A4 `n256`). The teacher's list
  decides about E9's existing quantized hand-off (made at 54b5b6d with the original 128 images,
  before the finiteness check): if the teacher replaces none of them, E9's calibration was
  already the cleaned set and the hand-off stands; if it replaces any, E9 is quantized again.
- **Test images: decided (second amendment of 2026-11-13, Soheil and Amir, before these
  results).** Any test image (filtered or unfiltered, later PolypGen) whose fp16 vision tower
  output is non-finite for the student or the teacher is dropped from that test for every arm
  before any test evaluation; the agent commits the list from part A and reports its count next
  to the test sizes. No student arm changes the vision tower, so the student list holds for every
  student arm; the union with the teacher's keeps one test set for all arms. Validation and the training images are handled by the amendment (calibration)
  and by the training loop, which already skips a non-finite step (E2: 16 of 625).
- PolypGen (the external test) is scanned the same way once it is downloaded.

## 6. Run report

- Run by / date, commit:
- Part A, per split and encoder: non-finite images / images:
- Part B: student replaced ids, teacher replaced ids:
