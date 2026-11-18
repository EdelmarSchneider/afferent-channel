# Reproducing the paper

This page lists the commands behind every experiment of *State Beside the Text: An Afferent Channel for Agents and Game Characters*: building the data, training, evaluating and running the analyses. Run times were measured on one RTX 4070 Ti SUPER (16 GB) under Windows 11, the machine all reported runs used. Treat them as rough guides.

The sections follow the paper. Each one names the paper section and the appendix with the full results.

## Contents

- [Hardware](#hardware)
- [Environment](#environment)
- [Using the published weights and data](#using-the-published-weights-and-data)
- [Conventions](#conventions)
- [Folder names](#folder-names)
- [1. The agent at its own limits](#1-the-agent-at-its-own-limits)
- [2. Ordinary answers](#2-ordinary-answers)
- [3. Background jobs](#3-background-jobs)
- [4. The messenger](#4-the-messenger)
- [5. The game character](#5-the-game-character)
- [6. The text-notice model and detection](#6-the-text-notice-model-and-detection)
- [7. Forged status](#7-forged-status)
- [8. Role-marked text and one inserted vector](#8-role-marked-text-and-one-inserted-vector)
- [9. Several changes before a decision](#9-several-changes-before-a-decision)
- [10. Context eviction](#10-context-eviction)
- [11. The confirmatory evaluation](#11-the-confirmatory-evaluation)
- [12. Qwen3-0.6B](#12-qwen3-06b)
- [13. Where the channel is read](#13-where-the-channel-is-read)
- [14. Varied wording, and runs that stopped at a check](#14-varied-wording-and-runs-that-stopped-at-a-check)
- [Indexing all results](#indexing-all-results)

## Hardware

- **GPU:** one GPU with 16 GB of memory and bfloat16 support. Every training and evaluation in the paper ran on one such card. Training uses batch size 1 with gradient accumulation and activation checkpointing; longer episodes do not fit at batch size 2.
- **RAM:** 32 GB is advisable. A training or evaluation needs about 10 GB of free system memory.
- **Disk:** Qwen3-1.7B takes about 3.8 GB in the Hugging Face cache. Each dataset takes 0.2-0.35 GB. A trained channel run takes about 240 MB (adapter and channel); a text-notice run about 70 MB. While it trains, a run also keeps a resume snapshot of about 0.6 GB.
- **CPU:** the data variants, the readings and most analyses run on the CPU in seconds to minutes. A heavy CPU job running next to a training slows the training down.

## Environment

Install [uv](https://docs.astral.sh/uv/), then, from the repository root:

```bash
uv sync                    # Python 3.12, PyTorch 2.11 with CUDA 12.8, transformers 5.x (pinned in uv.lock)
uv run pytest -q           # unit tests: tiny random GPT-2 on the CPU, no download of Qwen3
uv run pytest -q -m slow   # real-model tests: GPU, downloads Qwen3
```

Every command below runs from the repository root. The first data build downloads Qwen3-1.7B and the GSM8K problems (through the `datasets` library).

**CPU-only installs.** `pyproject.toml` takes PyTorch from the CUDA 12.8 index of download.pytorch.org, and `uv.lock` pins its Linux (x86-64, ARM64) and Windows wheels. That build also runs on a machine without a GPU, so the unit tests and the examples (which run on the CPU by default) work as installed; only the download is large. To install the smaller CPU build instead, or on macOS (the CUDA index has no macOS wheels), change the `url` of `[[tool.uv.index]]` in `pyproject.toml` to `https://download.pytorch.org/whl/cpu`, then run `uv lock` and `uv sync`. The paper's numbers come from the CUDA build in bfloat16 on a GPU; another build or precision can change a rare near-tie token.

## Using the published weights and data

You do not need to train before evaluating. The [model repository](https://huggingface.co/DominuZ/afferent-channel-qwen3-1.7b) holds the trained adapters and channels, and the [dataset repository](https://huggingface.co/datasets/DominuZ/afferent-channel-data) holds the datasets and the evaluation outputs reported in the paper. Download them into the folders the commands below expect (while the repositories are private, log in first with `uv run hf auth login`):

```bash
uv run python scripts/fetch_release.py runs      # trained runs -> runs/<run name>/ (about 1.6 GB)
uv run python scripts/fetch_release.py data      # data/ and results/ (about 1.7 GB; the dataset card is not downloaded)
uv run python scripts/fetch_release.py data data/n6 results/phase1    # only some folders
```

The model repository names its subfolders by setting; the script places each one under the run name the commands use:

| Model-repo subfolder | Run folder |
|---|---|
| `operating-state` | `runs/n6-wide-levels-s0` |
| `operating-state-seed1`, `operating-state-seed2` | `runs/n6-wide-levels-s1`, `runs/n6-wide-levels-s2` |
| `background-jobs` | `runs/bg3s-wide-levels-s0` |
| `messenger` | `runs/n7msg-wide-levels-s0` |
| `game-character` | `runs/npc6-wide-levels-s0` |
| `text-notice-baseline` | `runs/n6-notice-none-s0` |

The other runs named below (earlier designs, the role-marked and inserted-vector models, the retrained controls, Qwen3-0.6B, varied wording) are not published: train them with the commands of their section. Likewise, data folders that are not in the dataset repository (for example `data/bg3`, `data/n6-insert` and its several-changes files, `data/n6-notice-m2`, `data/n6v`) are built by the commands of their section; most of them take seconds on the CPU from `data/n6`. Each model and dataset card lists what it contains.

## Conventions

- **Resuming.** Data builds, trainings and evaluations can be resumed: after an interruption, run the same command again. A training writes `resume.pt` every 10 minutes and after every epoch, and removes it when the run is saved; a resumed run must be given the same arguments.
- **Run contents.** A run folder holds `lora/` (the adapter), the channel's weights (`channel.pt` when written by training, `channel.safetensors` in a downloaded run; every command loads either), `channel.json` (the channel's configuration) and `train_log.json` (the losses and every command option).
- **Training check.** The training log prints a `decision react / hold` loss for the decision tokens. If the `react` loss of a limit signal is still near the base rate at the end of epoch 2 (about 6.7 on the operating-state data), the channel was not learned: stop and investigate rather than evaluate.
- **Text-based models.** The text-notice, role-marked and inserted-vector models are always evaluated with `--mode harness`. In the standard mode their controls are meaningless, because the standard pairings change only the channel, which these models do not have.
- **Batch size.** `--batch-size 16` is much faster than batch size 1, but batched bfloat16 generation can flip rare near-tie tokens. Never mix batch sizes in one results folder. The channel model's full evaluation of the operating-state experiment ran at batch size 1; most later evaluations at 16. To compare two folders of the same runs: `uv run python scripts/analysis/batch_equivalence.py <reference folder> <candidate folder>`.
- **Fresh folders.** An evaluation reuses `records.partial.jsonl` in its output folder. After any change to the code or the data, evaluate into a new folder.
- **Reading outputs.** `uv run python scripts/read_generations.py results/<folder> 3` prints decoded generations of an evaluation.

## Folder names

Module, run and results names carry the project's short codes. This key maps them to the names used in the paper.

| Code in a name | Meaning |
|---|---|
| `n6` | Operating-state data, six signals (Experiment 1) |
| `phase1` | Evaluations of the operating-state channel models |
| `wide-levels` | The main configuration: widened stream (lanes) with the level table |
| `-s0`, `-s1`, `-s2` | Training seed |
| `-b16` | Evaluated at batch size 16 |
| `bg3`, `bg3s` | Background jobs, three slots; `s` = with slot twins |
| `n7msg`, `r23-msg` | The messenger: data (seven signals), evaluation |
| `npc6`, `r22-npc` | The game character: data, evaluation |
| `notice-none`, `b3-` | The text-notice model |
| `notice-segment`, `b3s-` | The role-marked text model |
| `insert-levels`, `b6b-` | One inserted vector per change |
| `spoof-`, `spoof-p5-` | Forged status; `p5` = inside the model's own turn |
| `flips`, `flips4` | Several changes before a decision (two or three; four) |
| `m2` | Retrained with two-change examples |
| `r11-` | Ordinary answers |
| `r12p`, `r12c`, `r12-` | The confirmatory evaluation: pilot set, confirmatory set, its runs |
| `r21-`, `w16`, `w64` | Context eviction; tokens of the turn kept |
| `q06`, `r5-` | Qwen3-0.6B |
| `n6v`, `r24-`, `r24b-` | Varied wording; sampled decoding |
| `n6-limits`, `r10` | Limits given in words |

Forged-status cells are named `<placement>/<content>`. Placements: `P1` user message, `P2` quoted document in the user message, `P3` tool response, `P4` another agent's turn, `P5` inside the model's own turn. Contents: `F1` plain sentence, `F2` API-tag style, `F3` harness-format notice (`[status x=v]`), `F4` chat-template forgery (`F4s`: the same with special-token text split into plain text), `S2-F<k>` an "all fine" claim in format k, `S1` an instruction to ignore limits. Each cell is paired with a control: the same wrapper without the claim.

## 1. The agent at its own limits

Paper: Section 3.1. Appendix: D (results), B (method), C (scoring).

**Data** (GPU, about 10 minutes; the untrained model's runs are cached in `data/n6/base_runs.json`):

```bash
uv run python -m prio.datagen.build --n-signals 6 --model Qwen/Qwen3-1.7B --train 3000 --test 600 --seed 0 --batch-size 32 --out data/n6
uv run python scripts/inspect_reactions.py data/n6 2      # read a few training episodes
```

**Train** the main configuration (about 2.5 hours; seeds 1 and 2 about 2.4 hours each with `--seed 1 --out runs/n6-wide-levels-s1`, and so on):

```bash
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6 --n-signals 6 --seed 0 --out runs/n6-wide-levels-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery wide --projector levels
```

**Evaluate.** The full evaluation is the paper's result (batch size 1, about 6 hours; it includes the untrained model's floor and an earlier, smaller check of ordinary answers on 120 problems; the paper reports the check of section 2). The quick evaluation runs every fourth episode of each kind and signal (about 1.5 hours at batch size 1, about 8 minutes at batch size 16); a later full evaluation into the same folder, at the same batch size, only adds the rest.

```bash
uv run python -m prio.eval.run --run runs/n6-wide-levels-s0 --data data/n6 --out results/phase1
uv run python -m prio.eval.run --run runs/n6-wide-levels-s0 --data data/n6 --out results/phase1-wide-levels --every 4 --preservation-n 0 --base-n 0
```

The seeds were checked with the quick evaluation: `--run runs/n6-wide-levels-s1 --out results/phase1-wide-levels-s1`, and the same for seed 2.

**Readings** (CPU):

```bash
uv run python scripts/time_misses.py results/phase1                          # the deadline: strict and looser readings, every miss decoded
uv run python scripts/throughput.py runs/n6-wide-levels-s0 results/phase1    # generation speed with the channel on and off (GPU, minutes)
uv run python scripts/cost_table.py                                          # context cost of status text against the channel
uv run python scripts/decision_probe.py runs/n6-wide-levels-s0 data/n6 8     # does the run use the channel, per signal (about 20 minutes)
```

**Deliveries and encoders compared** (Appendix B.9). The same training command with other options, each followed by the quick evaluation above:

| Configuration | Options instead of `--delivery wide --projector levels` | Run name | Training time |
|---|---|---|---|
| Embedding addition, MLP encoder (first design) | none (the defaults) | `n6-s0` | about 2.1 h |
| Read-only lanes, MLP encoder | `--delivery lanes` | `n6-lanes-s0` | about 2.1 h |
| Read-only lanes, level table | `--delivery lanes --projector levels` | `n6-lanes-levels-s0` | about 2.1 h |
| Widened stream, level table (main) | `--delivery wide --projector levels` | `n6-wide-levels-s0` | about 2.5 h |
| Embedding addition, level table, channel learning rate 1e-4 | `--projector levels --lr-proj 1e-4 --pause-after 1` | `n6-sum-levels-lr1e-4-s0` | one epoch |

## 2. Ordinary answers

Paper: Section 3.1 and Table 2. Appendix: D.7.

The untrained model and the trained model answer the same 300 GSM8K problems and 100 explanation tasks through the same batched loop. The second command needs the text-notice model of section 6.

```bash
uv run python -m prio.eval.preservation --run runs/n6-wide-levels-s0 --out results/r11-wide-levels-s0-b16 --batch-size 16    # about 40 minutes
uv run python -m prio.eval.preservation --run runs/n6-notice-none-s0 --out results/r11-notice-none-s0-b16 --batch-size 16   # about 30 minutes
uv run python scripts/analysis/preservation_compare.py results/r11-wide-levels-s0-b16 results/r11-notice-none-s0-b16 --out results/r11-notice-none-s0-b16/preservation_compare.md
```

## 3. Background jobs

Paper: Section 3.2. Appendix: E.

**Data** (CPU, under a minute each; built from the untrained model's runs in `data/n6`). The `--slot-twins` set repeats every training event with two or three jobs on another slot, with the same text, so only the channel tells which job changed. Both sets share the same test split. The dataset repository publishes `data/bg3s` only; build `data/bg3` with the first command (the evaluation and readings below read its test split).

```bash
uv run python -m prio.bg.build --base data/n6 --model Qwen/Qwen3-1.7B --out data/bg3 --seed 0 --train 4500 --test-events 400 --test-openings 100
uv run python -m prio.bg.build --base data/n6 --model Qwen/Qwen3-1.7B --out data/bg3s --seed 0 --train 4500 --test-events 400 --test-openings 100 --slot-twins
uv run python scripts/inspect_bg.py data/bg3 2
```

**Train and evaluate** the reported model (training about 4.3 hours; evaluation about 1 hour, a multi-turn loop with a simulated harness):

```bash
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/bg3s --n-signals 6 --seed 0 --out runs/bg3s-wide-levels-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery wide --projector levels
uv run python -m prio.bg.evaluate --run runs/bg3s-wide-levels-s0 --data data/bg3 --out results/bg3s-wide-levels-full --base-n 100 --batch-size 16
uv run python scripts/analysis/bg_identification.py results/bg3s-wide-levels-full data/bg3
```

The first model, without slot twins, is the same training on `data/bg3` (`--out runs/bg3-wide-levels-s0`, about 3 hours), evaluated into `results/bg-wide-levels-full`.

## 4. The messenger

Paper: Section 3.3. Appendix: F.

The build reads `data/n6`: the messenger's training set also carries the operating-state episodes. The model has seven signals. The evaluation includes the forged-status cells.

```bash
uv run python -m prio.msg.build --n6 data/n6 --model Qwen/Qwen3-1.7B --out data/n7msg --seed 0 --train 400 --test 300 --batch-size 32      # GPU, about 15 minutes
uv run python scripts/inspect_table.py data/n7msg 2 test
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n7msg --n-signals 7 --seed 0 --epochs 4 --batch-size 1 --grad-accum 4 --out runs/n7msg-wide-levels-s0 --delivery wide --projector levels   # about 3.7 hours
uv run python -m prio.msg.evaluate --run runs/n7msg-wide-levels-s0 --data data/n7msg --out results/r23-msg-wide-levels-s0-b16 --batch-size 16   # about 1.4 hours
uv run python scripts/analysis/msg_reading.py results/r23-msg-wide-levels-s0-b16 data/n7msg --out results/r23-msg-wide-levels-s0-b16/msg_reading.md
```

`msg_reading.py` gives the readings made after the evaluation (why runs miss the strict rule, what forged text did, written examples). They stand beside the pre-registered lines, never in their place.

## 5. The game character

Paper: Section 4. Appendix: G.

The build applies a stop rule fixed in advance: if fewer than 70% of the untrained model's runs are usable, it says so in `data/npc6/manifest.json` (`stop_rule`), and the model should not be trained on that build. The evaluation includes the typed-alarm cells.

```bash
uv run python -m prio.npc.build --model Qwen/Qwen3-1.7B --out data/npc6 --seed 0 --train 1050 --test 300 --batch-size 32      # GPU, about 16 minutes
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/npc6 --n-signals 6 --seed 0 --epochs 4 --batch-size 1 --grad-accum 4 --out runs/npc6-wide-levels-s0 --delivery wide --projector levels   # about 2.4 hours
uv run python -m prio.npc.evaluate --run runs/npc6-wide-levels-s0 --data data/npc6 --out results/r22-npc-wide-levels-s0-b16 --batch-size 16   # about 1 hour
```

The reading of the second signal (Appendix G.4: why the choice by water or food was not learned in this run) compares first commands of sibling runs and the channel's level vectors. It also reads the messenger run:

```bash
uv run python scripts/analysis/second_signal.py results/r22-npc-wide-levels-s0-b16 \
    --pairs thirst-water:thirst-dry:drink_water:go_to hunger-food:hunger-none:eat:go_to \
    --levels runs/npc6-wide-levels-s0:water_near:0:99 runs/npc6-wide-levels-s0:food_near:0:99 \
             runs/n7msg-wide-levels-s0:net:0:99 runs/n7msg-wide-levels-s0:cell:0:99 \
    --unvisited runs/npc6-wide-levels-s0:water_near:50 --out results/r22-npc-wide-levels-s0-b16/second_signal.md
```

## 6. The text-notice model and detection

Paper: Section 5.1. Appendix: H.1, H.2.

The text-notice model gets the same episodes, targets, twins and training as the channel model, but receives the state as a 9-token notice such as `[status ctx=0.90]`, spliced in right after the state change.

**Data** (CPU, seconds to a minute each):

```bash
uv run python -m prio.datagen.variants --src data/n6 --variant notice --out data/n6-notice
uv run python -m prio.datagen.variants --src data/n6 --variant swap --test-only-into data/n6-notice
uv run python -m prio.datagen.variants --src data/n6 --variant near_miss --test-only-into data/n6-notice
uv run python -m prio.datagen.variants --src data/n6 --variant quiet --test-only-into data/n6-notice
```

**Train** (about 1.8 hours per seed; the forgery tests also use seeds 1 and 2: `--seed 1 --out runs/n6-notice-none-s1`, and so on):

```bash
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6-notice --n-signals 6 --seed 0 --out runs/n6-notice-none-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery none
```

**Evaluate** in harness mode (full: about 16 minutes at batch size 16; quick: add `--every 4`, about 4 minutes), then pair it with the channel model's full evaluation:

```bash
uv run python -m prio.eval.run --run runs/n6-notice-none-s0 --data data/n6-notice --out results/b3-notice-none-full-b16 --mode harness --batch-size 16
uv run python -m prio.eval.run --run runs/n6-notice-none-s0 --data data/n6-notice --out results/b3-notice-none-quick-b16 --mode harness --every 4 --batch-size 16
uv run python scripts/analysis/paired_detection.py results/phase1 results/b3-notice-none-full-b16
```

The detection comparison pairs the channel model at batch size 1 with the text-notice model at batch size 16 (stated in Appendix N.5 of the paper).

## 7. Forged status

Paper: Section 5.2. Appendix: H.3-H.7.

Each command runs the default cells with their paired controls on the quick subset. The channel model's first command also runs the untrained model's floor (`--conditions on,base`, the default); the others use `--conditions on`, because the untrained model is the same.

```bash
# Channel model, seed 0: default cells (about 36 minutes) and the own-turn cells (about 5 minutes)
uv run python -m prio.eval.spoof --run runs/n6-wide-levels-s0 --data data/n6 --out results/spoof-wide-levels-s0-b16 --batch-size 16
uv run python -m prio.eval.spoof --run runs/n6-wide-levels-s0 --data data/n6 --out results/spoof-p5-wide-levels-s0-b16 --cells P5/F3,P5/F1 --suppress-cells= --conditions on --batch-size 16

# Channel model, seeds 1 and 2 (about 25 minutes each; shown for seed 1)
uv run python -m prio.eval.spoof --run runs/n6-wide-levels-s1 --data data/n6 --out results/spoof-wide-levels-s1-b16 --conditions on --batch-size 16
uv run python -m prio.eval.spoof --run runs/n6-wide-levels-s1 --data data/n6 --out results/spoof-p5-wide-levels-s1-b16 --cells P5/F3,P5/F1 --suppress-cells= --conditions on --batch-size 16

# Text-notice model (about 19 minutes and 3 minutes; seeds 1 and 2 likewise)
uv run python -m prio.eval.spoof --run runs/n6-notice-none-s0 --data data/n6-notice --out results/spoof-notice-none-s0-b16 --conditions on --batch-size 16
uv run python -m prio.eval.spoof --run runs/n6-notice-none-s0 --data data/n6-notice --out results/spoof-p5-notice-none-s0-b16 --cells P5/F3,P5/F1 --suppress-cells= --conditions on --batch-size 16
```

Cells are comma-separated. A single cell for a smoke test: `--cells P1/F3 --suppress-cells= --every 8`.

**Readings** (CPU, seconds):

```bash
# Channel against text, per cell (Fisher exact test; the per-signal part is an extra reading)
uv run python scripts/analysis/spoof_compare.py results/spoof-wide-levels-s0-b16 results/spoof-notice-none-s0-b16 --by-signal --out results/spoof-notice-none-s0-b16/compare.md

# One model across its three seeds (one argument per seed; a seed's two folders joined with +)
uv run python scripts/analysis/spoof_seeds.py \
    results/spoof-wide-levels-s0-b16+results/spoof-p5-wide-levels-s0-b16 \
    results/spoof-wide-levels-s1-b16+results/spoof-p5-wide-levels-s1-b16 \
    results/spoof-wide-levels-s2-b16+results/spoof-p5-wide-levels-s2-b16 \
    --by-signal P3/F3 P1/F1 --out results/spoof-wide-levels-s2-b16/seeds.md
```

## 8. Role-marked text and one inserted vector

Paper: Section 5.3. Appendix: H.8.

The role-marked model reads the text notices with a learned harness-mark vector added on harness tokens. The inserted-vector model gets the channel's level vectors once per change, at one input position no text can produce (token id 151669, one past the tokenizer's vocabulary).

```bash
# Inserted-vector data (CPU)
uv run python -m prio.datagen.variants --src data/n6 --variant insert --insert-id 151669 --out data/n6-insert
uv run python -m prio.datagen.variants --src data/n6 --variant swap --test-only-into data/n6-insert
uv run python -m prio.datagen.variants --src data/n6 --variant near_miss --test-only-into data/n6-insert
uv run python -m prio.datagen.variants --src data/n6 --variant quiet --test-only-into data/n6-insert

# Train (about 1.8 hours each)
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6-notice --n-signals 6 --seed 0 --out runs/n6-notice-segment-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery segment
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6-insert --n-signals 6 --seed 0 --out runs/n6-insert-levels-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery insert --projector levels

# Quick evaluations in harness mode (about 4 minutes each)
uv run python -m prio.eval.run --run runs/n6-notice-segment-s0 --data data/n6-notice --out results/b3s-notice-segment-quick-b16 --mode harness --every 4 --batch-size 16
uv run python -m prio.eval.run --run runs/n6-insert-levels-s0 --data data/n6-insert --out results/b6b-insert-levels-quick-b16 --mode harness --every 4 --batch-size 16
```

Forged status for both: the text-notice commands of section 7 with `--run runs/n6-notice-segment-s0 --data data/n6-notice` or `--run runs/n6-insert-levels-s0 --data data/n6-insert`, into `results/spoof-notice-segment-s0-b16`, `results/spoof-insert-levels-s0-b16` and their `spoof-p5-` counterparts.

Stale refusals after a recovery, compared across models (CPU):

```bash
uv run python scripts/analysis/recovery_reading.py results/b6b-insert-levels-quick-b16 data/n6-insert \
    --vs results/phase1-wide-levels-b16 results/b3-notice-none-quick-b16 results/b3s-notice-segment-quick-b16
```

`results/phase1-wide-levels-b16` is the channel model's quick evaluation at batch size 16: the quick command of section 1 with `--batch-size 16` into that folder.

## 9. Several changes before a decision

Paper: Section 5.3. Appendix: I.

The state crosses its limit two or three times before the model decides on a gated tool, one the state can rule out. All models here were trained on single changes. The episodes come from 57 gating events of the quick subset; each carrier gets the same change points.

**Data** (CPU, seconds):

```bash
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-flips
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-notice-flips --carrier notice
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-insert-flips --carrier insert --insert-id 151669
# Four changes (no line; an extra reading)
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-flips-m4 --ms 4
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-notice-flips-m4 --ms 4 --carrier notice
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-insert-flips-m4 --ms 4 --carrier insert --insert-id 151669
```

**Evaluate** every model in harness mode (about 1 minute each), then summarize against the line and pair with the channel model:

```bash
uv run python -m prio.eval.run --run runs/n6-wide-levels-s0 --data data/n6-flips --out results/flips-wide-levels-s0-b16 --mode harness --batch-size 16
uv run python -m prio.eval.run --run runs/n6-notice-none-s0 --data data/n6-notice-flips --out results/flips-notice-none-s0-b16 --mode harness --batch-size 16
uv run python -m prio.eval.run --run runs/n6-notice-segment-s0 --data data/n6-notice-flips --out results/flips-notice-segment-s0-b16 --mode harness --batch-size 16
uv run python -m prio.eval.run --run runs/n6-insert-levels-s0 --data data/n6-insert-flips --out results/flips-insert-levels-s0-b16 --mode harness --batch-size 16

uv run python -m prio.eval.flips --data data/n6-flips --results results/flips-wide-levels-s0-b16 --base results/phase1-wide-levels-b16
uv run python -m prio.eval.flips --data data/n6-notice-flips --results results/flips-notice-none-s0-b16 --base results/b3-notice-none-quick-b16
uv run python scripts/analysis/flips_compare.py results/flips-wide-levels-s0-b16 results/flips-notice-none-s0-b16 --out results/flips-notice-none-s0-b16/flips_compare.md
```

`--base` is the same model's quick evaluation at batch size 16 (the rate with at most one change): `results/b3s-notice-segment-quick-b16` for the role-marked model, `results/b6b-insert-levels-quick-b16` for the inserted vector.

**Controls retrained with two-change examples** (one recipe, one seed; Appendix I.3, I.4). The training set is the carrier's own training file plus 429 two-change pairs built from training-split gating events (CPU, about 15 seconds):

```bash
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-notice-m2 --carrier notice --train-with data/n6-notice
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-insert-m2 --carrier insert --insert-id 151669 --train-with data/n6-insert

# Train (about 2.1 hours each)
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6-notice-m2 --n-signals 6 --seed 0 --out runs/n6-notice-none-m2-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery none
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6-insert-m2 --n-signals 6 --seed 0 --out runs/n6-insert-levels-m2-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery insert --projector levels

# Text arm: quick evaluation (validity, at most one change), two-three changes, four changes
uv run python -m prio.eval.run --run runs/n6-notice-none-m2-s0 --data data/n6-notice --out results/b3-notice-none-m2-quick-b16 --mode harness --every 4 --batch-size 16
uv run python -m prio.eval.run --run runs/n6-notice-none-m2-s0 --data data/n6-notice-flips --out results/flips-notice-none-m2-s0-b16 --mode harness --batch-size 16
uv run python -m prio.eval.run --run runs/n6-notice-none-m2-s0 --data data/n6-notice-flips-m4 --out results/flips4-notice-none-m2-s0-b16 --mode harness --batch-size 16
uv run python -m prio.eval.flips --data data/n6-notice-flips --results results/flips-notice-none-m2-s0-b16 --base results/b3-notice-none-m2-quick-b16

# Channel model at four changes
uv run python -m prio.eval.run --run runs/n6-wide-levels-s0 --data data/n6-flips-m4 --out results/flips4-wide-levels-s0-b16 --mode harness --batch-size 16
```

The inserted-vector arm follows the same pattern: `runs/n6-insert-levels-m2-s0` on `data/n6-insert`, `data/n6-insert-flips` and `data/n6-insert-flips-m4`, into `results/b6b-insert-levels-m2-quick-b16`, `results/flips-insert-levels-m2-s0-b16` and `results/flips4-insert-levels-m2-s0-b16`.

## 10. Context eviction

Paper: Section 6. Appendix: J.

The prompt stays whole; of the model's own turn before the decision, only the last W tokens stay (the same cut for every carrier). Shown for the channel model; the notice and inserted-vector carriers use their own flips folders.

```bash
# Single-change reference files, then the cut files (CPU, seconds)
uv run python -m prio.datagen.flips --src data/n6 --out data/n6-flips-m1 --ms 1
uv run python -m prio.datagen.evict --src data/n6-flips-m1 --out data/n6-flips-m1-w16 --window 16
uv run python -m prio.datagen.evict --src data/n6-flips-m1 --out data/n6-flips-m1-w64 --window 64
uv run python -m prio.datagen.evict --src data/n6-flips --out data/n6-flips-w16 --window 16
uv run python -m prio.datagen.evict --src data/n6-flips --out data/n6-flips-w64 --window 64

# Evaluate (about 1 minute each): the reference and the four cut files
uv run python -m prio.eval.run --run runs/n6-wide-levels-s0 --data data/n6-flips-m1 --out results/r21-wide-levels-s0-m1-b16 --mode harness --batch-size 16
uv run python -m prio.eval.run --run runs/n6-wide-levels-s0 --data data/n6-flips-m1-w16 --out results/r21-wide-levels-s0-m1-w16-b16 --mode harness --batch-size 16
uv run python -m prio.eval.run --run runs/n6-wide-levels-s0 --data data/n6-flips-w16 --out results/r21-wide-levels-s0-m23-w16-b16 --mode harness --batch-size 16
# ... and likewise for m1-w64 and m23-w64

# Reading against the same runs without eviction
uv run python scripts/analysis/evict_reading.py data/n6-flips-w16 results/r21-wide-levels-s0-m23-w16-b16 --ref results/flips-wide-levels-s0-b16
```

## 11. The confirmatory evaluation

Paper: Section 3.1. Appendix: D.9.

This evaluation was planned in advance and run once. The code enforces that order:

- **Pilot first.** Build the pilot set, run its eight evaluations and two batch-size-1 slices, write its report, read ten event runs per arm and model by hand, then run the smoke checks with a note of what you read. Pilot numbers are never reported as results.
- **Then the confirmatory set,** built only from a clean, committed source tree on which the pilot's smoke checks came out clean. It is never rebuilt.
- **Run once.** Every start is logged in `results/r12/RUN_ONCE.json` with the commit and the hashes of the data it reads. A second start for the same arm and model is refused; `--rerun-reason "<text>"` runs it again into a new folder and leaves the first run untouched.
- **Fresh tasks.** The builder checks that no task was used before; it needs the earlier datasets (`data/n6` and the others) on disk.

The arms: `paired` (fresh tasks and a new task family, under the standard pairings), `free` (the crossing falls on a token the model wrote itself), `live` (the counters are computed from the generation), `Ls` (the same with the thinking budget and the deadline counted from zero; no line). Models: `channel` (`runs/n6-wide-levels-s0`) and `notice` (`runs/n6-notice-none-s0`).

**Pilot** (about 45 minutes in all):

```bash
uv run python -m prio.r12.build --set pilot --model Qwen/Qwen3-1.7B --batch-size 32     # about 5 minutes -> data/r12p/
uv run python -m prio.r12.run --set pilot --arm paired --model channel                  # repeat for each arm (paired, free, live, Ls) and model (channel, notice)
uv run python -m prio.r12.run --set pilot --arm free --model channel --batch-size 1 --limit 16 --run-types event
uv run python -m prio.r12.run --set pilot --arm live --model channel --batch-size 1 --limit 16 --run-types event
uv run python -m prio.r12.lines --set pilot                                             # -> results/r12p/r12_lines.md
uv run python scripts/analysis/r12_read.py results/r12p-free-wide-levels-s0-b16 10      # hand reading (arm paired: scripts/read_generations.py)
uv run python -m prio.r12.smoke --hand-read "<what you read and saw>"                    # -> results/r12p/smoke.json
```

**Confirmatory set** (about 3.4 hours; commit first):

```bash
uv run python -m prio.r12.build --set confirm --model Qwen/Qwen3-1.7B --batch-size 32   # 15-25 minutes -> data/r12c/
uv run python -m prio.r12.run --set confirm --arm paired --model channel                # about 45 minutes
uv run python -m prio.r12.run --set confirm --arm paired --model notice                 # about 25 minutes
uv run python -m prio.r12.run --set confirm --arm free --model channel                  # about 20 minutes
uv run python -m prio.r12.run --set confirm --arm free --model notice                   # about 12 minutes
uv run python -m prio.r12.run --set confirm --arm live --model channel                  # about 23 minutes
uv run python -m prio.r12.run --set confirm --arm live --model notice                   # about 14 minutes
uv run python -m prio.r12.run --set confirm --arm Ls --model channel                    # about 8 minutes
uv run python -m prio.r12.run --set confirm --arm Ls --model notice                     # about 4 minutes
uv run python -m prio.r12.lines --set confirm                                           # CPU -> results/r12/r12_lines.md
```

The confirmatory set runs whole at batch size 16. To re-score the published records without running anything, download the results (see above) and run only the last command.

## 12. Qwen3-0.6B

Paper: Section 6. Appendix: H.9, K.

Same data and recipe as the 1.7B models, with `--model Qwen/Qwen3-0.6B`:

```bash
uv run python -m prio.train --model Qwen/Qwen3-0.6B --data data/n6 --n-signals 6 --seed 0 --out runs/n6-wide-levels-q06-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery wide --projector levels   # about 2.2 hours
uv run python -m prio.train --model Qwen/Qwen3-0.6B --data data/n6-notice --n-signals 6 --seed 0 --out runs/n6-notice-none-q06-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery none          # about 1.5 hours

uv run python -m prio.eval.run --run runs/n6-wide-levels-q06-s0 --data data/n6 --out results/r5-wide-levels-q06-quick-b16 --every 4 --preservation-n 0 --base-n 0 --batch-size 16
uv run python -m prio.eval.run --run runs/n6-notice-none-q06-s0 --data data/n6-notice --out results/r5-notice-none-q06-quick-b16 --mode harness --every 4 --batch-size 16
```

Forged status: the commands of section 7 with these runs, into `results/spoof-wide-levels-q06-s0-b16` (with the untrained floor, about 33 minutes), `results/spoof-notice-none-q06-s0-b16` (`--conditions on`, about 17 minutes) and the `spoof-p5-` folders.

## 13. Where the channel is read

Paper: Section 6. Appendix: K.

```bash
# Geometry of the level tables (CPU, about 25 seconds)
uv run python scripts/analysis/level_geometry.py --runs runs/n6-wide-levels-s0 runs/n6-wide-levels-s1 runs/n6-wide-levels-s2 runs/n6-lanes-levels-s0 --data data/n6 --out results/mechanism/b1

# Ablations, patching and probes on Qwen3-1.7B (GPU, about 40 minutes; resumable)
uv run python scripts/analysis/mechanism.py --run runs/n6-wide-levels-s0 --data data/n6 --out results/mechanism/r1-wide-levels-s0 --device cuda --probes

# The ablations repeated on Qwen3-0.6B
uv run python scripts/analysis/mechanism.py --run runs/n6-wide-levels-q06-s0 --data data/n6 --out results/mechanism/r1-wide-levels-q06-s0 --device cuda --no-t3
```

The two models give different readings (Appendix K); a statement about layers or lanes holds only for the model it names.

## 14. Varied wording, and runs that stopped at a check

Paper: Section 6. Appendix: L, M.

**Varied wording and a written summary** (Appendix M). The rewrite draws the naming sentences from a large generated bank and asks the untrained model for a short summary when the thinking budget runs out (GPU, about 5 minutes; cached):

```bash
uv run python -m prio.datagen.varied --src data/n6 --out data/n6v --model Qwen/Qwen3-1.7B --seed 0 --batch-size 32 --close-within 48
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6v --n-signals 6 --seed 0 --out runs/n6v-wide-levels-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery wide --projector levels   # about 2.3 hours
uv run python -m prio.eval.run --run runs/n6v-wide-levels-s0 --data data/n6v --out results/r24-varied-quick-b16 --every 4 --preservation-n 0 --base-n 0 --batch-size 16
uv run python scripts/analysis/wording_reading.py results/r24-varied-quick-b16 data/n6v --out results/r24-varied-quick-b16/wording_reading.md --examples 30
```

Sampled decoding (temperature 0.6, top-p 0.95, three draws per model; about 15 minutes per draw), then the reading:

```bash
uv run python scripts/analysis/sampled_eval.py --run runs/n6v-wide-levels-s0 --data data/n6v --out results/r24b-varied-t06-s0-b16 --sampling-seed 0   # draws 0, 1, 2; main model: runs/n6-wide-levels-s0 on data/n6
uv run python scripts/analysis/sampled_reading.py --name varied --data data/n6v --greedy results/r24-varied-quick-b16 \
    --draws results/r24b-varied-t06-s0-b16 results/r24b-varied-t06-s1-b16 results/r24b-varied-t06-s2-b16
```

**Runs that stopped at a check fixed in advance** (Appendix L). Each training stops after epoch 2 (`--pause-after 2`) for its check. In the paper all three checks fired, so none of these models was evaluated.

```bash
# Scale control: embedding addition with an RMS-scaled gain
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6 --n-signals 6 --seed 0 --out runs/n6-sum-levels-rms-s0 --epochs 4 --batch-size 1 --grad-accum 4 --projector levels --sum-norm rms --pause-after 2

# Limits given in words (first run with value reports; second run without them)
uv run python -m prio.datagen.limits --src data/n6 --out data/n6-limits --seed 0
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6-limits --n-signals 6 --seed 0 --out runs/n6-limits-wide-levelsfourier-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery wide --projector levels+fourier --pause-after 2
uv run python -m prio.datagen.limits --src data/n6 --out data/n6-limits-a1 --seed 0 --no-reports --test-reports 0
uv run python -m prio.train --model Qwen/Qwen3-1.7B --data data/n6-limits-a1 --n-signals 6 --seed 0 --out runs/n6-limits-a1-wide-levels-s0 --epochs 4 --batch-size 1 --grad-accum 4 --delivery wide --projector levels --pause-after 2
```

A paused run continues with the same command without `--pause-after`. If a check passes, its evaluation is `prio.eval.run` (scale control, quick) or `prio.eval.limits_eval --run <run> --data <data> --out <folder> --every 4 --base-n 0 --batch-size 16` (limits in words).

## Indexing all results

```bash
uv run python scripts/index_results.py
```

This writes `results/index.md` and `results/index.json`: every run and evaluation found under `runs/` and `results/`, with its main numbers.
