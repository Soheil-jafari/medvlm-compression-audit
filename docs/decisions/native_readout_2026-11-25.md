# The yes/no readout and E2's answer format: a native-format readout (2026-11-25)

**Status:** exploratory note, at the research lead's request, committed **before** anything is
computed with the readout below. The pre-registered metrics, the pre-registered parser
(`clinical.parse_yes_no`) and the plain answer stay as they are; nothing here changes a
pre-registered result.

## What FBR cell (ii) showed (training probes, E2 seed 0)

On the schedule's 942 training probes (471 positive, 471 negative), E2's greedy first answer
token is never a yes spelling:

| Label | First token: yes spelling | "No" | "e" | other |
|---|---|---|---|---|
| Positive | 0 | 29 | 442 | 0 |
| Negative | 0 | 278 | 188 | 5 ("!" 4, "pol" 1) |

E2 answers in the dataset's descriptive style: "evidence of polypoid lesion" / "No evidence of
polypoid lesion" (and "evidence of ulcerative colitis" also for the ulcerative-colitis negatives
shown). The pre-registered parser leaves 443 of the positives and 193 of the negatives unparsed
(the rest parse as "no"). The four "!" first tokens are the four probes whose margin is
non-finite (argmax of NaN logits returns token 0; all four negatives): at most 4 of the 157 FBR
updates are skipped by the rule of 2026-11-25.

## What this means for the readouts (the research lead's reading, checked)

- **The plain answer** (log p(yes) >= log p(no), pooled over the single-token spellings at the
  first answer position) compares a token E2 never produces ("yes") with one it produces only
  when its answer begins "No evidence". It answers "yes" only when p("No") falls below p(yes),
  which almost never happens: this explains E2's low plain yes-rates (0.106 / 0.013 / 0.009) and
  plain macro balanced accuracy (0.553). Its high AUROC remains, because p("No") falls on
  positives: **the margin is a valid ranking signal; its sign is not a valid yes/no answer for a
  model trained on this answer format.**
- **The operating point** (a per-finding threshold on the margin at specificity 0.95, fitted per
  arm) reads the ranking, not the sign, and is not affected in this way.
- **Gate 2's "silent collapse"** (plain yes-rate 0 for E5 magnitude, E5 activation, E6) is
  therefore at least partly a readout artifact over a real change. Which part cannot be told
  without the arms' decoded answers: do they answer "No evidence of ..." to every positive (a real
  collapse of the answer) or "evidence of ..." that the yes/no readout cannot see?
- **E7's plain advantage** (yes-rate 0.709, plain macro BA 0.688, the best ECE at Gate 2) may come
  from its criterion, which scores units by the yes/no answer tokens' log-probability on the same
  probe template, i.e. by exactly this readout; it may not reflect better-preserved knowledge.
  This is settled before exploratory addition A's plain-answer results are interpreted and before
  the paper uses the words "silent collapse".

## The native-format readout (exploratory)

**(a) The parser** (`evaluation/native_readout.py`), threshold-free, on the decoded free answer to
a probe: lower-cased, leading non-letters dropped;

- starts with the word "no" ("no evidence ...", "no ...", "no" alone) -> **no**;
- starts with "evidence of" or with the word "yes" -> **yes**;
- anything else -> **unparsed**, counted and reported (counted wrong in sensitivity and
  specificity, as the pre-registered parser's unparsed answers).

It is fixed here, from the research lead's specification. Its basis is the training split's
answer strings: a CPU step (`scripts/native_answer_patterns.py`, training split only, no model,
no evaluation output) lists the leading-word patterns of the training answers to finding
questions with their counts and the parser's coverage of them, and is committed (phrases and
counts only). If the parser leaves more than 5 % of those training answers unparsed, the list is
reported to the research lead before the readout is used; the parser is not extended silently.

**(b) Where it runs:** on the decoded probe answers saved by `fbr_probe_answers.py`
(`probe_answers_<split>.npz`): on validation for every B2 arm and E2 (FBR cell (i)), and on any arm
and test split whose decoded answers exist or are run with `--split test` / `test_unfiltered`
(the same pass; the non-finite drop list applies). No arm has decoded answers saved yet.

**(c) Reported** next to the pre-registered plain readout and the operating-point readout, per arm,
split and finding (and macro): yes-rate, sensitivity, specificity, balanced accuracy, unparsed
count. Plus a **first-token table** per arm and split: the greedy first answer token as a yes
spelling, a no spelling, "e", or other (the probe-answer pass now saves the first token).

**(d) The E7 question:** E7's first-token table and native readout next to the E5 arms', on
validation.

## FBR

Its cross-entropy targets ("yes" / "no") teach a format E2 never uses. So any change in the
pre-registered plain readout, and in the decoded-answer gate item read with the pre-registered
parser, after FBR is a **format effect** (decision file section 12). The native-format readout is
the relevant plain check; the primary gate quantity (the cross-fitted rare-finding sensitivity,
ranking-based) is unaffected. The recipe stays locked and unchanged (no FBR training has run).
The gate's decoded-answer item is reported both with the pre-registered parser (as locked) and
with the native parser, the latter labelled exploratory; the gate verdict uses the locked one.
