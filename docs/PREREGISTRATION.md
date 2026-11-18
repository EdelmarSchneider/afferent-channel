# Pre-registration

The experiments of *State Beside the Text* were judged against criteria fixed before they ran. The published record of those criteria is **Appendix A of the paper** ([appendix PDF](../paper/draft2-appendix.pdf)). This page says what was fixed, how it was judged, and where to find each part.

## What was fixed in advance

- **Pre-registered experiments.** Every experiment from the comparison with text notices onward was pre-registered. Before its first run, its success line, its predictions and its failure or stop rule were written down and committed to version control.
- **Amendments.** A criterion changed only through a dated amendment, written before the run it concerned. Thirteen amendments exist; twelve stand behind results reported in the paper, and one describes a diagnostic that was not run.
- **Earlier lines.** For the first experiment (the agent at its own limits), the scoring rule and a success reference were set in a design document dated before any evaluation. That document is not part of the pre-registration, so the paper labels this row a design reference. The background-job experiment set its entry bar before its model had any result.

## How results were judged

- Each line is judged against its own words, on the measured value. A line missed by a small margin is reported as not met.
- Verdicts are *met* or *not met* for a success line, *held* or *failed* for a prediction, and *triggered* or *not triggered* for a stop or failure rule. A run that never happened is marked *not run*.
- Readings made after the data were seen are labeled as such. They stand beside a line and never replace it.
- Choices made after test results were seen (for example, the main configuration of the channel, and the slot twins in the background-job training data) are listed in Appendix A.1 and disclosed where each result is reported.

The ledger has 92 rows: 54 lines met or predictions held, 33 not met, failed or triggered, and 5 not run.

## Where to find it in the paper

| Part of Appendix A | What it holds |
|---|---|
| Table A0 | Every result with its control or comparison and its verdicts; the unmet ones in bold |
| A.1 | What was fixed in advance, how each line was judged, and the choices made after test results were seen |
| A.2 | A key from the plain names of the paper's Table 2 to the labels of the ledger |
| A.3 | The ledger: all 92 lines, predictions and stop rules, each with its measured value and verdict |
| A.4 | The amendments, copied line by line with their dates. Four kinds of edit are marked there: clauses on runs that were never made are omitted, one model name is replaced by a bracketed description, the author's instructions quoted in Portuguese are omitted, and, for this public version, internal role names and pointers to unpublished working files are replaced by bracketed descriptions |

The pre-registration was kept in the author's working repository, which is not public. Appendix A is the version published with the paper.
