"""R12 (third note on amendment A12): runs whose thinking closes right after text that names the deadline.

The frozen classifier counts a deadline reaction only when the turn ends within 128 tokens of the
position it is given. In the arms without a forced prefix a model can name the deadline early, close
its thinking and go on answering: no counted reaction, so neither the premature count nor the false
alarms show it. This count stands beside them, reported without a line: a keyword match, not a
counted reaction.

A run is counted when a deadline keyword of the classifier (`reactions._KEYWORDS["time"]`) occurs in
the 47 tokens before its first `</think>`:
- event runs: only when the thinking closes before token k (after token k the close is the reaction
  to the crossing itself); with how many of them are hits;
- twin and off runs: anywhere; with how many of them are already counted as false alarms.

Outside `src/prio`: the code tree of R12's smoke checks does not change. CPU: the tokenizer only.

uv run python scripts/analysis/r12_named_close.py results/r12-free-wide-levels-s0-b16 \
    results/r12-free-notice-none-s0-b16 [more folders of arms free, live, Ls] [--out file.md]
"""

import argparse
import json
from pathlib import Path

WINDOW = 47  # tokens before the close that are read (the classifier's window for a thinking-budget reaction, less one)
RUN_TYPES = ("event", "twin", "off")


def named_close(tokens: list[int], think_close: int, names, window: int = WINDOW) -> int | None:
    """Index of the first thinking close when the `window` tokens before it name the deadline, else None."""
    if think_close not in tokens:
        return None
    at = tokens.index(think_close)
    return at if names(tokens[max(0, at - window):at]) else None


def table(records: list[dict], rows: list[dict], think_close: int, names, window: int = WINDOW) -> dict:
    """Per run type: runs, named_close, of_them_hits (event) or of_them_false_alarms (twin, off), tasks."""
    generations = {(g["task_id"], g["run_type"]): g for g in rows}
    out = {}
    for run_type in RUN_TYPES:
        part = [r for r in records if r["run_type"] == run_type and (r["task_id"], run_type) in generations]
        beside = "of_them_hits" if run_type == "event" else "of_them_false_alarms"
        row = {"runs": len(part), "named_close": 0, beside: 0, "tasks": []}
        for r in part:
            g = generations[(r["task_id"], run_type)]
            at = named_close(list(g["pre"]) + list(g["post"]), think_close, names, window)
            if at is None or (run_type == "event" and at >= r["k"]):
                continue
            row["named_close"] += 1
            row[beside] += bool(r["hit"] if run_type == "event" else r["false_alarm"])
            row["tasks"].append(r["task_id"])
        if part or run_type in ("twin", "off"):
            out[run_type] = row
    return out


def report(tables: dict[str, dict]) -> str:
    lines = ["# R12: the thinking closes right after text that names the deadline", "",
             "Third note on amendment A12. A keyword match only, not a counted reaction: a deadline keyword of the "
             f"classifier in the {WINDOW} tokens before the first `</think>`. Event runs: only when the thinking closes "
             "before token k. Twin and off runs: anywhere. Reported without a line, beside the premature count and the "
             "false alarms.", "",
             "| Folder | Event runs, close before token k | of them hits | Twin runs | of them false alarms | Off runs | of them false alarms |",
             "|---|---|---|---|---|---|---|"]
    empty = {"runs": 0, "named_close": 0, "of_them_hits": 0, "of_them_false_alarms": 0, "tasks": []}
    for folder, t in tables.items():
        e, tw, off = (t.get(k, empty) for k in RUN_TYPES)
        lines.append(f"| {folder} | {e['named_close']} / {e['runs']} | {e['of_them_hits']} | {tw['named_close']} / {tw['runs']} | "
                     f"{tw['of_them_false_alarms']} | {off['named_close']} / {off['runs']} | {off['of_them_false_alarms']} |")
    for folder, t in tables.items():
        named = {k: v["tasks"] for k, v in t.items() if v["tasks"]}
        if named:
            lines += ["", f"`{folder}`: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in named.items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec
    from prio.reactions import names_signal

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", type=Path, nargs="+", help="results folders of arms free, live or Ls")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="tokenizer")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.model)
    ms = resolve_spec(args.model, tok)
    names = lambda ids: names_signal(tok.decode(ids), "time")
    load = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tables = {f.as_posix(): table(load(f / "records.jsonl"), load(f / "generations.jsonl"), ms.think_close, names)
              for f in args.folders}
    text = report(tables)
    print(text)
    if args.out is not None:
        args.out.write_bytes((text + "\n").encode("utf-8"))
        args.out.with_suffix(".json").write_bytes((json.dumps(tables, indent=1) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
