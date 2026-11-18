"""R24 (amendment A9), an extra reading labeled post hoc: where the thinking closes in thinking-budget runs.

The classifier counts a thinking-budget reaction when the thinking closes within `REACTION_WINDOW`
(48) tokens of the first token the model could react in, and the text before the close names the
budget (`reactions.find_reaction`). A model trained to write a summary before it closes its
thinking (R24) can close later than that: naming sentence + summary + close.

For every channel-on run in which a thinking-budget reaction is due, per kind of episode:
- counted: the evaluation's own count (unchanged);
- late: not counted, the thinking closes at or after the window and the `_NAMING_SPAN` tokens
  before the close name the budget (the classifier's rule with only the window lifted);
- other: not counted for another reason (no close in the generation, or the budget is not named).

The pre-registered rate is the counted one. The rate with the window lifted is printed beside it,
never instead. CPU: the tokenizer only.

uv run python scripts/analysis/think_close.py results/r24-varied-quick-b16 [--out file.md]
uv run python scripts/analysis/think_close.py results/phase1-wide-levels-b16
"""

import argparse
import json
from pathlib import Path

WINDOW = 48  # reactions.REACTION_WINDOW
SPAN = 60  # reactions._NAMING_SPAN
KINDS = ("event", "swap", "starts_past")


def close_of(ids: list[int], think_close: int, names, span: int = SPAN) -> tuple[int | None, bool]:
    """(index of the first thinking close or None, the `span` tokens before it name the budget)."""
    if think_close not in ids:
        return None, False
    at = ids.index(think_close)
    return at, bool(names(ids[max(0, at - span):at]))


def reading(records: list[dict], generations: list[dict], think_close: int, names, window: int = WINDOW,
            span: int = SPAN) -> dict:
    """Per kind and "all": n, counted, late [(episode, paired, close index)], other [(episode, paired, close index)]."""
    key = lambda row: (row["episode_id"], row["signal"], row["condition"], row["paired"])
    written = {key(g): g["token_ids"] for g in generations}
    table: dict[str, dict] = {}
    for r in records:
        if r["condition"] != "on" or r["signal"] != "think" or not r["event_occurred"] or key(r) not in written:
            continue
        for name in (r["kind"], "all"):
            row = table.setdefault(name, {"n": 0, "counted": 0, "late": [], "other": []})
            row["n"] += 1
            if r["reaction_signal"] == "think":
                row["counted"] += 1
                continue
            at, named = close_of(written[key(r)], think_close, names, span)
            late = at is not None and at >= window and named
            row["late" if late else "other"].append((r["episode_id"], r["paired"], at))
    return table


def _share(k: int, n: int) -> str:
    return f"{k} / {n} = {100 * k / n:.1f}%" if n else "-"


def report(table: dict, results, window: int = WINDOW) -> str:
    lines = [f"# Where the thinking closes in thinking-budget runs: {results}", "",
             "Extra reading, labeled post hoc (not in amendment A9). The pre-registered count is the column "
             f"`Counted` (thinking closed within {window} tokens, budget named). `Late` = not counted, the budget is "
             f"named and the thinking closes at or after token {window}. `With the window lifted` = counted + late.",
             "", "| Kind | n | Counted | Late | With the window lifted | Other |", "|---|---|---|---|---|---|"]
    for name in (*KINDS, "all"):
        row = table.get(name)
        if row is None:
            continue
        lines.append(f"| {name} | {row['n']} | {_share(row['counted'], row['n'])} | {len(row['late'])} | "
                     f"{_share(row['counted'] + len(row['late']), row['n'])} | {len(row['other'])} |")
    everything = table.get("all", {"late": [], "other": []})
    if everything["late"]:
        lines += ["", "Late closes:", ""]
        lines += [f"- `{e}` ({p}): thinking closes at token {at}" for e, p, at in everything["late"]]
    if everything["other"]:
        lines += ["", "Not counted for another reason:", ""]
        lines += [f"- `{e}` ({p}): " + ("the thinking never closes in the generation" if at is None
                                         else f"thinking closes at token {at}, budget not named before it")
                  for e, p, at in everything["other"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec
    from prio.reactions import names_signal

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="an evaluation folder (records.jsonl + generations.jsonl)")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="tokenizer")
    parser.add_argument("--out", type=Path, default=None, help="write the reading here (.md) and its numbers beside it (.json)")
    args = parser.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.model)
    ms = resolve_spec(args.model, tok)
    load = lambda name: [json.loads(line) for line in (args.results / name).read_text(encoding="utf-8").splitlines() if line]
    table = reading(load("records.jsonl"), load("generations.jsonl"), ms.think_close,
                    lambda ids: names_signal(tok.decode(ids), "think"))
    text = report(table, args.results.as_posix())
    print(text)
    if args.out is not None:
        args.out.write_bytes((text + "\n").encode("utf-8"))
        args.out.with_suffix(".json").write_bytes((json.dumps(table, indent=1) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
