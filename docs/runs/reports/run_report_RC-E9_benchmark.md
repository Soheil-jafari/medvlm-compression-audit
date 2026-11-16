### RC-E9 run report

- Run by / date: <your name> / 2026-11-16 17:46 (UTC)
- Commit actually used: 4f546557fc1dfa31a1a348fd926089636a06ef30
- Hardware: Tesla T4, Tesla T4
- Library versions: python 3.13.16; pyyaml 6.0.3 (needs >=6.0); numpy 2.2.6 (needs >=1.24); pandas 2.3.3 (needs >=2.0); scipy 1.16.3 (needs >=1.10); scikit-learn 1.6.1 (needs >=1.3); matplotlib 3.10.0 (needs >=3.7); pillow 13.0.0 (needs >=9.1); torch 2.11.0+cu130 (needs >=2.8); torchvision 0.26.0+cu130 (needs >=0.23); transformers 5.19.0 (needs ==5.19.0); peft 0.21.2 (needs ==0.21.2); accelerate 1.15.0 (needs >=1.0); torchao 0.18.0 (needs >=0.16.0); gptqmodel 7.6.0 (needs ==7.6.0); bitsandbytes 0.50.2 (needs ==0.50.2); optimum 2.3.0 (needs >=1.24.0)
- Inputs verified: shared data /kaggle/input/datasets/soheiljafarifard/medcompress-data/medcompress_data (commit 4038a6bb4276ed9ce4dda2bbc8d542d074f544af); E0 seed 0 teacher_adapter /kaggle/input/datasets/soheiljafarifard/e0-teacher-adapter/handoff/E0_teacher_adapter; E9 seed 0 quantized /kaggle/input/datasets/soheiljafarifard/e9-seed0-quantized/handoff/E9_seed0_quantized
- Per step (wall time, status):
  - benchmark E9 seed0: 24.2 min, ok
- Peak GPU memory (from the results files):
  - results/E9/nobudget/seed0/efficiency.csv: peak_memory_bytes = 7,351,677,856
- Hand-offs (each folder = ONE private Kaggle dataset; write its name here):
  - none
- Results files to commit (3; copies in /kaggle/working/results_to_commit):
  - results/E9/nobudget/seed0/efficiency.csv
  - results/E9/nobudget/seed0/efficiency_setup.json
  - results/E9/nobudget/seed0/environment.json
- Anything unusual: none
- Outcome: all steps ok
