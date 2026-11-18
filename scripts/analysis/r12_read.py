"""R12: event runs of an arm without a forced prefix (free, live, Ls), decoded for reading by hand.

The pilot's smoke check S9 asks for 10 event runs per arm and model read by a person: is the text
after token k a reaction, does the notice sit after token k (text-notice model), does the record say
what the text shows. Arm paired is read with `scripts/read_generations.py`.

For n event runs at even spacing over the records file: one header line (task, family, signal, k,
whether the crossing occurred, hit or miss, onset, premature), the last `--tail` tokens before the
crossing, the mark `<<< token k >>>`, the notice when there is one, and the first `--head` tokens of
the judged part. CPU: the tokenizer only.

uv run python scripts/analysis/r12_read.py results/r12p-free-wide-levels-s0-b16 [10] [--tail 60 --head 120]
"""

import argparse
import json
from pathlib import Path

MARK = "<<< token k >>>"


def pick(records: list[dict], n: int) -> list[dict]:
    """Up to n event runs at even spacing over the file (all of them when there are fewer)."""
    events = [r for r in records if r["run_type"] == "event"]
    if n >= len(events):
        return events
    step = len(events) / n
    return [events[int(i * step)] for i in range(n)]


def render(record: dict, row: dict, decode, tail: int = 60, head: int = 120) -> str:
    """One event run as text: header, the end of what came before token k, the mark, the judged part."""
    crossing = "crossing occurred" if record["occurred"] else f"no crossing ({record['why_not']})"
    verdict = "HIT" if record["hit"] else "miss"
    onset = f" | onset {record['onset']}" if record.get("onset") is not None else ""
    premature = " | PREMATURE" if record.get("premature") else ""
    reacted = f" | reacted to: {record['reaction_signal']}" if record.get("reaction_signal") and not record["hit"] else ""
    header = (f"{record['task_id']} | {record['family']} | {record['signal']} | k = {record['k']} | {crossing} | "
              f"{verdict}{onset}{premature}{reacted} | stop: {row.get('stop_reason')}")
    pre, post = list(row["pre"]), list(row["post"])
    before = ("... " if len(pre) > tail else "") + decode(pre[-tail:] if tail else [])
    notice = f"{row['notice_text']}\n" if row.get("notice_text") else ""
    after = decode(post[:head]) + (" ..." if len(post) > head else "")
    return f"{header}\n{before}\n{MARK}\n{notice}{after}\n"


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="a results folder of arm free, live or Ls")
    parser.add_argument("n", type=int, nargs="?", default=10)
    parser.add_argument("--tail", type=int, default=60, help="tokens shown before the crossing")
    parser.add_argument("--head", type=int, default=120, help="tokens shown of the judged part")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="tokenizer")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.model)
    load = lambda name: [json.loads(line) for line in (args.results / name).read_text(encoding="utf-8").splitlines() if line]
    records_file = "records.jsonl" if (args.results / "records.jsonl").exists() else "records.partial.jsonl"
    rows = {(g["task_id"], g["run_type"]): g for g in load("generations.jsonl")}
    blocks = [render(r, rows[(r["task_id"], "event")], tok.decode, args.tail, args.head)
              for r in pick(load(records_file), args.n) if (r["task_id"], "event") in rows]
    text = f"# Event runs for reading by hand: {args.results.as_posix()} ({len(blocks)} runs)\n\n" + "\n".join(
        f"## {i + 1}\n\n```\n{b}```\n" for i, b in enumerate(blocks))
    print(text.encode("utf-8", "replace").decode("utf-8"))
    if args.out is not None:
        args.out.write_bytes((text + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
