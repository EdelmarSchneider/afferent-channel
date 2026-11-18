# Analyses behind the paper's numbers

Each script answers one question and reads saved evaluation folders (`results/<folder>/`), datasets (`data/<folder>/`) or trained runs (`runs/<run>/`). Run them from the repository root with `uv run python scripts/analysis/<name>.py ...`. The usage of each script is in its docstring, and the exact commands used for the paper are in [docs/REPRODUCE.md](../../docs/REPRODUCE.md).

Most of them run on the CPU in seconds; `mechanism.py` and `sampled_eval.py` need a GPU.

| Script | What it computes | Section of docs/REPRODUCE.md |
|---|---|---|
| `audit_recompute.py` | Recomputes the headline numbers of the operating-state experiment from the evaluation records alone, including the subset free of episode selection | 1 |
| `decision_floor.py` | The lowest decision loss a model can reach per signal (the training check) | 1 |
| `boundary_delay.py` | How late a text harness that can only add text at the next tool call or the end of the turn would deliver a state change, and how often the six signals change value | 1 |
| `batch_equivalence.py` | Agreement between two evaluations of the same runs at different batch sizes | Conventions |
| `bg_identification.py` | Background jobs: which job the model named, against chance, with Wilson intervals | 3 |
| `msg_reading.py` | Messenger: readings made after the evaluation (why runs miss the strict rule, what forged text did, written examples), beside the pre-registered lines | 4 |
| `second_signal.py` | Game character: first commands of runs that differ only in the second signal, and the distance between the channel's level vectors | 5 |
| `paired_detection.py` | Paired detection between two evaluations of the same episodes (exact McNemar test) | 6 |
| `preservation_compare.py` | Ordinary answers: two trained models on the same items | 2 |
| `spoof_compare.py` | Forged status: channel model against text-notice model, per cell (Fisher exact test) | 7 |
| `spoof_seeds.py` | Forged status: one model across its three seeds | 7 |
| `spoof_hits.py` | Forged status: every counted reaction of chosen cells, decoded for reading by hand | 7 |
| `recovery_reading.py` | Wrong refusals after the state recovered, compared across models | 8 |
| `flips_compare.py` | Several changes before a decision: one model against another on the same episodes | 9 |
| `evict_reading.py` | Context eviction: correct decisions with the turn cut to its last tokens, against the whole turn | 10 |
| `r12_read.py` | Confirmatory evaluation: event runs decoded for reading by hand | 11 |
| `r12_named_close.py` | Confirmatory evaluation: runs whose thinking closes right after text that names the deadline (a keyword count beside the scored result) | 11 |
| `level_geometry.py` | Geometry of the learned level tables, readers and writers | 13 |
| `mechanism.py`, `mech_utils.py` | Ablations, patching and probes: where the channel is read (GPU) | 13 |
| `wording_reading.py` | How a model words its reactions: share of trained sentences, distinct sentences, written summaries | 14 |
| `think_close.py` | Where the thinking closes in thinking-budget runs (a reading made after the evaluation) | 14 |
| `sampled_eval.py` | A quick evaluation decoded with sampling (GPU) | 14 |
| `sampled_reading.py` | The sampled draws of one model, pooled | 14 |

Readings made after an evaluation are labeled as such in their output. They stand beside the pre-registered lines in Appendix A of the paper, never in their place.
