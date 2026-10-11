# Notebooks

`kaggle_template.ipynb` is the only notebook kept in the repository. It clones the
repo, installs it, sets paths through environment variables and runs a script.
It works on Kaggle (2x T4) and Google Colab.

Rules:

- Notebooks are thin: all logic lives in `src/medcompress/` and `scripts/`.
- Clear all outputs before committing (outputs can contain images or file paths of
  restricted data).
- Never paste tokens or keys into a notebook. Use Kaggle Secrets / Colab secrets.
- Personal exploration notebooks stay out of the repo (or in a personal branch
  with outputs cleared).
