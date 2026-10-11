---
name: Experiment run
about: Plan or track a single experiment run
title: "[EXP] <ID> <budget> seed <n>"
labels: experiment
---

**Experiment ID:** <!-- E0-E9 or A1-A6, must exist in src/medcompress/registry.py -->

**Variant (if any):** <!-- e.g. E3:w4, E5:taylor, A4:n64 -->

**Budget:** <!-- B1 / B2 / B3 / n.a. -->

**Seed:** <!-- 0 / 1 / 2 -->

**Config:** <!-- configs/experiments/... plus any --override flags -->

**Command:**

```bash
python scripts/run_experiment.py --exp <ID> --budget <B> --seed <n>
```

**Hardware:** <!-- Kaggle 2x T4 / Colab / rented GPU (type) -->

**Expected output:** <!-- files under results/<ID>/..., metrics CSV, checkpoints (not committed) -->

**Owner:** <!-- Amir / Soheil / Farzan -->

**Notes / blockers:**
