# Result reports

One short report per experiment (`E0.md` ... `E9.md`, `A1.md` ... `A6.md`), plus
`GATE2.md` and `VERDICTS.md`. Reports are written from the committed CSVs under
`results/` only, and every number names the CSV it came from. The tables and
figures in the paper are generated separately by `scripts/08_aggregate_results.py`;
these reports are for the team, not for copying numbers into the paper.

Template:

```markdown
# <ID>: <title>

- Runs included: <paths under results/>, run cards <RC-xx>
- Split(s): validation / test
- Status: partial (k of n runs) | complete

## Numbers
| Metric | Mean | SD (seeds) | 95% CI | Source CSV |

## Reading
Two to five sentences: what changed vs E2 (or the relevant parent), anything odd,
and whether the stage exit gate is met. No claims beyond the numbers above.
```

`VERDICTS.md` is generated, not hand-written.
