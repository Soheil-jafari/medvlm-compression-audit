# Run card RC-X2: external test set from another centre (PolypGen first), part A: licence, centres, masks and overlap

## 1. What and why

- Stage: S11 (D02 amendment 7 of 2026-10-28; PolypGen first by the amendment of 2026-10-31,
  `docs/preregistration.md` section 11, before any model result)
- Experiments: none. This part checks PolypGen before any model sees it.
- Purpose:
  - the internal test is a reduced-leakage test: same-examination pairs remain across splits,
    and they cannot be removed without patient or video ids (D02 amendment 7);
  - so an **external test set from another centre is co-primary**. Every model is scored on
    three tests (the filtered test, the unfiltered test and this external test), and the
    conclusions are reported for each;
  - **PolypGen first**: it has polyp masks and many polyps, so it carries both primary
    endpoints, and its small-polyp stratum also gets enough cases. GastroVision is not
    planned now (it has no masks of our mask findings as far as the agent knows, and may share
    a hospital with HyperKvasir; both unverified);
  - before the agent writes a loader, the team confirms that we may use PolypGen, that its
    images come from other centres than our Kvasir images, what its masks hold, and how many
    of its images are near-copies of ours.
- Who: **[team]** (proposal: Farzan, as for EndoBench).
- Status: `blocked` (the licence is not checked). Nothing is downloaded.
- Part B (the loader, the probes, the evaluation of every model on this test and the
  comparisons on it) is written by the agent from what this part reports (section 7 below is
  its plan). It is not written yet.

## 2. Requirements

- **The licence first** (section 3, step 1). Download nothing until it is recorded.
- RC-D1 run 4b done (2026-10-29): the overlap is counted against its splits
  (`medcompress-data`).
- No GPU: a Kaggle CPU notebook, or any machine where you may store the data under your own
  access.
- Runtime: minutes for the overlap count (RC-X1's estimate; each image is fingerprinted once).
  Not measured for PolypGen.

## 3. Steps

Write every answer in section 6. Use numbers and field names only: no images, no patient
identifiers, no per-image rows, no file names.

1. **Licence.** Find PolypGen's official release from its paper (the paper's own link, not a
   mirror). Record:
   - the exact URL you opened, and its version or date;
   - the **exact licence name** as the release states it (for example the full name of a
     Creative Commons licence and its version), and where it is stated;
   - whether research use, use together with other datasets, and publishing results are
     allowed, and any condition (registration, a data-use agreement, citation, no
     redistribution, non-commercial use).

   The agent does not know the licence and cannot check it from here. **If the licence is not
   stated, or any of these is not allowed or not clear: stop**, and tell the agent through the
   inbox.
2. **The centres.** Record the hospitals or centres PolypGen's images come from, as its paper
   or release states, with the number of images from each. Also record where our images come
   from: Kvasir-VQA-x1 uses the images of Kvasir-VQA, from HyperKvasir and Kvasir-Instrument
   (D02); record the hospital(s) their papers name.

   The agent's recollection, **not verified from here**: one PolypGen centre is in Norway
   (Oslo University Hospital), and HyperKvasir comes from Bærum Hospital (Vestre Viken Health
   Trust, Norway). Check whether any PolypGen centre is a source of Kvasir. If one is, record
   which; its images are left out of the external test (or the team decides otherwise, before
   part B).
3. **Annotations** (counts and field names only):
   - the number of images: single frames, and frames from video sequences (separately);
   - whether patient, video or sequence ids exist, and per centre;
   - **polyp masks**: how many images have a mask with a polyp, and how many have an empty
     mask or are released as frames without a polyp (the **negatives**: the probes need absent
     polyps too);
   - whether one mask can hold several polyps, and the mask format (one binary image per
     frame, or something else);
   - any other labels (classes and counts).
4. **Download** with your own access, outside the repository, only if steps 1 and 2 passed.
   Keep it private: ONE private Kaggle dataset, shared with the team only. Write its name in
   section 6.
5. **Overlap** against our splits (counts only), on CPU, with run 4b's shared data attached:

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
pip install -q -r requirements/base.txt && pip install -q -e . --no-deps
python scripts/prepare_data.py overlap --data-root /kaggle/input/medcompress-data \
    --images /kaggle/input/<polypgen-dataset>/<images-folder> --name polypgen \
    --dedup-threshold 4
```

   This writes `results/data/polypgen_overlap.csv` (counts only): how many PolypGen images lie
   within 4 dHash bits of an image of each of our splits. Images of another centre should show
   none; a large count suggests shared sources. This is the first, cheap count; the full check
   of section 7 is part B.

## 4. What to paste back and commit

- Section 6, filled.
- Commit `results/data/polypgen_overlap.csv` if step 5 ran.
- Never commit the data, the images, file names or per-image rows.

## 5. Pass criteria

| Check | Pass if |
|---|---|
| Licence | the exact licence is recorded, and research use, use with other datasets and publishing results are allowed (step 1); otherwise stop |
| Another centre | no PolypGen centre is a source of Kvasir, or that centre's images can be left out (step 2) |
| Primary endpoints | polyp masks exist for enough images to give a small-polyp stratum (step 3; read with the agent, no number fixed in advance) |
| Negatives | frames without a polyp exist, so the probes have absent cases (step 3) |
| Overlap | counted; the agent reads it with the team before part B |

## 6. Run report (filled by the person who ran it)

- Checked by / date:
- Release opened (URL, version or date):
- Licence (exact name, where stated); research use / use with other data / publishing results
  (allowed or not); conditions:
- Centres (as stated, images per centre) and our sources' centres; shared centre (yes / no /
  which):
- Images (single frames; sequence frames); patient, video or sequence ids (yes / no):
- Polyp masks (images with a polyp; empty masks or frames without a polyp); several polyps per
  mask (yes / no); mask format:
- Other labels (classes and counts):
- Downloaded (yes / no; the private dataset's name):
- Overlap counts (`results/data/polypgen_overlap.csv`, commit used):
- Anything unusual:

## 7. Part B plan (agent, after part A; recorded 2026-10-31, before any model result)

Written so that the external test asks the same questions as the internal one:

- **Probes:** one image-level probe per PolypGen image, polyp present or absent, with the same
  template and display name: "Is there a polyp in this image? Answer yes or no."
  (`domain.probe.template` "Is there {finding} in this image? Answer yes or no.", display name
  "a polyp"; wording fix of 2026-10-31). Present = a mask with at least one polyp
  pixel; absent = an empty mask or a frame released without a polyp. One forward pass, log
  P(yes) against log P(no), as for the internal test.
- **Size strata from its masks, with OUR training quartiles:** the lesion area is the mask's
  polyp pixels over all image pixels (`data/masks.lesion_area_fraction`, as for Kvasir-SEG),
  and the strata use the frozen cut points q1 0.05676, q2 0.11336, q3 0.19932
  (`domain.small_findings.thresholds`), never quartiles of PolypGen itself. Small = below q1.
  The count of small polyps is reported with the small-finding recall and its interval.
- **The same near-duplicate check against our training images:** the held-out filter of D02
  amendment 6 applied to PolypGen against our training split, with the same settings as for
  our test (full-view cosine >= 0.86 with the pinned DINOv2-small embeddings, or a crop
  correlation >= 0.70, or within 4 dHash bits with a crop correlation >= 0.50). Images it
  flags are removed and counted, as for our test. (Step 5's dHash count comes first; the
  embedding check needs code in part B.)
- **Endpoints on PolypGen:** the small-finding recall and the lesion-mask effect (polyp is the
  mask finding), the per-finding sensitivity and specificity of polyp, and the flip analysis
  (section 11 of the pre-registration). The other two findings (ulcerative colitis,
  oesophagitis) have no labels there, so the multi-finding metrics are internal only.
- **Code:** a PolypGen loader (aggregate counts only leave the machine), the split
  `external_test` in the evaluation, its outcomes file, the comparisons on it, and the
  verdicts for each test.
