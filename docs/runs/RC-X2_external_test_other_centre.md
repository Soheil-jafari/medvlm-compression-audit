# Run card RC-X2: external test set from another centre (PolypGen and/or GastroVision), part A: terms, centres, annotations and overlap

## 1. What and why

- Stage: S11 (D02 amendment 7 of 2026-10-28, before any model result)
- Experiments: none. This part checks the external candidates before any model sees them.
- Purpose:
  - the internal test is a reduced-leakage test: same-examination pairs remain across splits,
    and they cannot be removed without patient or video ids (D02 amendment 7);
  - so an **external test set from another centre is co-primary**. Every model is scored on
    four tests (the filtered test, the unfiltered test, the strict subset and this external
    test), and the conclusions are reported for each;
  - the candidates are **PolypGen and/or GastroVision**;
  - before the agent writes a loader, the team confirms for each candidate:
    - that we may use it;
    - that it really comes from another centre than our Kvasir images;
    - that its annotations can carry our endpoints;
    - how many of its images are near-copies of ours.
- Who: **[team]** (proposal: Farzan, as for EndoBench).
- Status: `blocked` (the terms are not checked; RC-D1 run 4b, which writes the splits the
  overlap is counted against, has not run). Nothing is downloaded yet.
- Part B (the loader, the probes, the evaluation of every model on this test, and the
  comparisons on it) is written by the agent from what this part reports. It is not written
  yet.

## 2. Requirements

- **The terms first** (section 3, step 1). Download nothing until they are clear.
- RC-D1 run 4b done: the overlap is counted against its splits.
- No GPU: a Kaggle CPU notebook, or any machine where you may store the data under your own
  access.
- Runtime: RC-X1 expects minutes for its overlap count (each image is fingerprinted once). Not
  measured for these sets, whose sizes are not known here.

## 3. Steps

Write every answer in section 6. Use numbers and field names only: no images, no patient
identifiers, no per-image rows.

1. **Terms, for each candidate.** Find its official release from its paper (the paper's own
   link, not a mirror). Record:
   - the exact URL or hub ID you opened, and its version or date;
   - the licence name;
   - whether research use, use together with other datasets, and publishing results are
     allowed, and any condition (registration, citation, no redistribution).

   If any of these is not allowed or not clear for a candidate, **stop for that candidate**
   and tell the agent through the inbox.
2. **The centres, for each candidate.** Record which hospitals or centres its images come
   from, as its paper or release states. Also record where our images come from:
   - Kvasir-VQA-x1 uses the images of Kvasir-VQA, which come from HyperKvasir and
     Kvasir-Instrument (D02);
   - record the hospital(s) their papers name.

   If a candidate shares a centre with them, say which part of it does. That part is not
   "another centre": either it is left out, or the candidate is.

   The agent's recollection, **not verified from here**: HyperKvasir and GastroVision both come
   at least partly from Bærum Hospital (Vestre Viken Health Trust, Norway). Check this first.
3. **Annotations, for each candidate** (counts and field names only):
   - the number of images, and whether they are single frames, video frames or sequences;
   - whether patient, video or sequence ids exist (they would let a later step group images);
   - the image-level labels: which classes, and how many images each;
   - **lesion masks**: whether they exist, for which findings, and for how many images.

     Both primary endpoints need masks: the small-finding recall (it needs the lesion-area
     strata) and the lesion-mask effect. A candidate without masks can carry the secondary
     endpoints only. A co-primary test needs at least one candidate with masks of our mask
     findings (polyp, if Kvasir-SEG's masks are the mask findings);
   - which labels match findings on our finding list (`configs/domain/endoscopy.yaml`; the
     list is frozen after RC-D1's second pass), and with which wording;
   - images without any lesion (negatives), if the set has them: the probes need absent
     findings too.
4. **Download** with your own access, outside the repository, only for a candidate that passed
   steps 1 and 2. Keep it private: share it with the team only, as one private Kaggle dataset
   per candidate.
5. **Overlap** against our splits (counts only), on CPU, with run 4b's shared data attached:

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
python scripts/prepare_data.py overlap --data-root /kaggle/input/<run-4b-shared-data> \
    --images /kaggle/input/<candidate-dataset>/<images-folder> --name <polypgen|gastrovision> \
    --dedup-threshold 4
```

   This writes `results/data/<name>_overlap.csv` (counts only). The overlap uses the dHash
   alone within 4 bits, as RC-X1's (open item "RC-X1 and the two stages"). Images of another
   centre should show little or none; a large count suggests shared sources.

## 4. What to paste back and commit

- Section 6, filled for each candidate.
- Commit `results/data/<name>_overlap.csv` for each candidate that reached step 5.
- Never commit the data, the images, file names or per-image rows.

## 5. Pass criteria

| Check | Pass if |
|---|---|
| Terms | research use, use with other datasets and publishing results allowed (step 1) |
| Another centre | no image shares a centre with Kvasir's sources, or that part can be left out (step 2) |
| Primary endpoints | at least one candidate has lesion masks of our mask findings (step 3) |
| Overlap | counted; the agent reads it with the team before part B |

## 6. Run report (filled by the person who ran it)

For each candidate (PolypGen, GastroVision):
- Checked by / date:
- Release opened (URL or hub ID, version or date):
- Licence name; research use / use with other data / publishing results (allowed or not):
- Centres (as stated) and our sources' centres; shared centre (yes / no / which part):
- Images (number; frames, videos or sequences); patient, video or sequence ids (yes / no):
- Labels (classes and counts); lesion masks (findings, number of images); negatives:
- Labels that match our findings (and their wording):
- Downloaded (yes / no; the private dataset's name):
- Overlap counts (`results/data/<name>_overlap.csv`, commit used):
- Anything unusual:
