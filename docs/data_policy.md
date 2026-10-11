# Data policy

1. **No data in the repository.** Never commit datasets, images, masks, videos,
   manifests that reveal restricted file names, cached model outputs, checkpoints,
   adapters, API keys or tokens. `.gitignore` and the pre-commit large-file check are
   only a safety net.
2. **Own access for each member.** Every team member obtains dataset access
   themselves and accepts the dataset's terms individually. Do not share copies,
   download links or credentials with each other.
3. **Restricted data stays local.** Restricted or credentialed data is never uploaded
   to online AI tools, chat assistants or cloud coding sessions, and never pasted
   into issues, pull requests or notebooks. Only run it on machines and services the
   dataset terms allow.
4. **Kaggle / Colab.** Use private datasets / your own Drive. Never make a copy of a
   restricted dataset public. Clear notebook outputs before committing.
5. **Data card for every dataset.** Before a dataset is used in any experiment, its
   owner writes `docs/data_cards/<name>.md` from `docs/data_cards/TEMPLATE.md`
   (source, licence, access, splits, preprocessing, deduplication, known biases).
6. **Logs and results.** Logs, results CSVs and paper figures must not contain
   patient images or identifiers. Aggregated metrics are fine.
7. **Incidents.** If data is committed by mistake, tell the team immediately. Removing
   it requires rewriting history and rotating any exposed credentials; do not just
   delete the file in a new commit.
