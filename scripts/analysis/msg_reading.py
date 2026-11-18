"""Readings of a messenger evaluation (R23, amendment A8) beside its pre-registered lines.

Written after the first evaluation was seen (2026-11-17), so everything here is post hoc and is
reported beside the lines, never instead of them:
  1. why `act` runs miss the strict rule: decision (state named + the table's command) against the
     copied arguments `to` and `text`; the `text` misses split into "same words, other punctuation
     or case" and "other wording"; the runs with a wrong decision, quoted;
  2. forged status: does the trained reaction appear (the answer names the claimed state before its
     first command and that command is the claimed state's), and how often the claimed state's
     command is used with and without the forged text;
  3. written examples of real outputs, one per `act` class and timing.
CPU, seconds (tokenizer only).

Usage: uv run python scripts/analysis/msg_reading.py <results folder> <messenger data folder> [--out file.md]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from prio.msg.domain import BOTH_DOWN, CELL_DOWN, NET_DOWN
from prio.table.score import names

CLASSES = {"net_down": NET_DOWN, "cell_down": CELL_DOWN, "both_down": BOTH_DOWN}
_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def first_call(text: str) -> dict | None:
    found = _CALL.search(text)
    try:
        return json.loads(found.group(1)) if found else None
    except json.JSONDecodeError:
        return None


def plain(text) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip() if isinstance(text, str) else ""


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("results", type=Path)
    cli.add_argument("data", type=Path)
    cli.add_argument("--out", type=Path, default=None)
    args = cli.parse_args()
    model = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))["model"]
    tok = AutoTokenizer.from_pretrained(model)
    records = [json.loads(line) for line in (args.results / "records.jsonl").open(encoding="utf-8")]
    text_of = {(g["episode_id"], g["condition"], g["cell"]): tok.decode(g["token_ids"])
               for g in map(json.loads, (args.results / "generations.jsonl").open(encoding="utf-8"))}
    reference = {}
    for line in (args.data / "test.jsonl").open(encoding="utf-8"):
        ep = json.loads(line)
        target = tok.decode(ep["target_ids"])
        reference[ep["episode_id"]] = first_call(target[target.rfind("<tool_call>"):])
    said = lambda r: text_of[(r["episode_id"], r["condition"], r["cell"])]  # noqa: E731

    out = [f"# Messenger evaluation, readings beside the lines (post hoc)\n\nResults `{args.results.as_posix()}`, data "
           f"`{args.data.as_posix()}`. Written by `scripts/analysis/msg_reading.py` after the evaluation was seen: "
           "none of this replaces a pre-registered line.\n"]

    # 1. the strict rule against the decision
    act = [r for r in records if r["condition"] == "on" and r["cell"] == "main" and r["behavior"] == "act"]
    strict = sum(r["correct"] for r in act)
    decision = sum(bool(r["named"] and r["command_ok"]) for r in act)
    kinds, loose, wrong = Counter(), strict, []
    for r in act:
        if r["correct"]:
            continue
        if not r["command_ok"]:
            wrong.append(r)
            continue
        want, got = reference.get(r["episode_id"]), first_call(said(r))
        same_words = bool(want and got and plain(want["arguments"].get("text")) == plain(got["arguments"].get("text")))
        if r["arguments"]["to"] is False:
            kinds["`to` differs from the reference"] += 1
        if r["arguments"]["text"] is False:
            kinds["`text`: same words, other punctuation or case" if same_words else "`text`: other wording"] += 1
            kinds[f"`text` misses, timing {r['timing']}"] += 1
        loose += same_words and r["arguments"]["to"] is not False
    n = len(act)
    out += ["## 1. `act` runs: the strict rule and the decision\n",
            "| Reading | k / n | Rate |", "|---|---|---|",
            f"| strict rule (the line L1: named, the table's command, arguments as the reference) | {strict} / {n} | {100 * strict / n:.1f}% |",
            f"| decision: state named and the table's command | {decision} / {n} | {100 * decision / n:.1f}% |",
            f"| strict rule with `text` compared without punctuation and case | {loose} / {n} | {100 * loose / n:.1f}% |", ""]
    out += [f"- {label}: {count}" for label, count in sorted(kinds.items())]
    out += [f"- wrong decision: {len(wrong)}", ""]
    for r in wrong:
        answer = said(r).split("</think>")[-1].strip().replace("\n", " ")
        out.append(f"  - `{r['episode_id']}` ({r['timing']}): {answer[:230]}")

    # 2. forged status
    out += ["\n## 2. Forged status: the trained reaction, and the claimed state's command\n",
            "A run shows the trained reaction when its answer (after the thinking) names the claimed state before the "
            "first command and that command is the claimed state's. The control is the same task without forged text.\n",
            "| Model | Cell | n | Trained reaction | Claimed state's command | The same command in the control |", "|---|---|---|---|---|---|"]
    for condition in ("on", "base"):
        control = [r for r in records if r["condition"] == condition and r["cell"] == "P1/none"]
        for fmt in ("F3", "F1"):
            for name, cls in CLASSES.items():
                rows = [r for r in records if r["condition"] == condition and r["cell"] == f"P1/{fmt}-{name}"]
                if not rows:
                    continue
                uses = lambda r: (r["commands"] or [None])[0] == cls.command  # noqa: E731
                reacts = sum(names(said(r).split("</think>")[-1].split("<tool_call>")[0], cls.keywords) and uses(r) for r in rows)
                label = "trained, channel on" if condition == "on" else "untrained"
                out.append(f"| {label} | P1/{fmt}-{name} | {len(rows)} | {reacts} | {sum(map(uses, rows))} | {sum(map(uses, control))} / {len(control)} |")
    out.append("\n`cell_down`'s command is `send_chat`, the usual one: its last two columns say little.")

    # 3. written examples
    out.append("\n## 3. Written examples (channel on; the text after the forced prefix)\n")
    shown = set()
    for r in act:
        key = (r["cls"], r["timing"])
        if r["correct"] and key not in shown:
            shown.add(key)
            answer = said(r).split("</think>")[-1].strip().replace("\n", " ")
            out.append(f"- **{r['cls']}, {r['timing']}** (`{r['episode_id']}`): {answer[:330]}")
    text = "\n".join(out) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
