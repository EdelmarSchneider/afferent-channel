"""R24 (preregistration amendment A9): how a model words its reactions, read from an evaluation.

Counted reactions = channel on, an event, the reaction is to the run's own signal (the runs of the
verbatim-wording audit). For them, per signal and pooled:
- V2 (reported; prediction for the varied model <= 70%): the share whose naming sentence is word
  for word a naming sentence of the training file. Two readings: the reaction OPENS with it (first
  line of the generation: the paper's reading, 938 of 954 on the main model's full evaluation), and its
  naming sentence is one (the first line that names the signal: also a run that first finishes the
  word the cut interrupted);
- the number of distinct naming sentences, beside the training file's;
- V3 (line >= 80%): among counted thinking-budget reactions, a summary is present: at least 8
  tokens of text between the naming sentence and the end of the thinking; beside it, the same rate
  per kind of episode (declared in the note on A9 of 2026-11-17 01:47: kind `event` = reactions
  that come after some thinking; `starts_past` reactions were trained without a summary);
- `--examples N`: N thinking-budget reactions spread over the evaluation, in full, for reading by
  hand (the thinking before, the sentence, the summary, the answer). A9 asks for 30 read against
  the reasoning above them.

A naming sentence is a whole line (the generator's sentence, with its lead-in for the thinking
budget). Where text follows a lead-in colon on the same line, the sentence ends at that colon and
the rest of the line belongs to the summary. CPU: the tokenizer only.

uv run python scripts/analysis/wording_reading.py results/r24-varied-quick-b16 data/n6v \
    --out results/r24-varied-quick-b16/wording_reading.md --examples 30
uv run python scripts/analysis/wording_reading.py results/phase1-wide-levels-b16 data/n6
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prio.datagen.episodes import reaction_start, read_jsonl
from prio.datagen.variants import manifest_model
from prio.modelspec import resolve_spec
from prio.reactions import _KEYWORDS, names_signal

SIGNALS = ("ctx", "think", "energy", "time", "net", "ram")
MIN_SUMMARY = 8  # tokens (A9, V3)
LINE = 0.80  # V3
READ = 80  # tokens of a reaction read for its naming sentence (as the verbatim-wording audit)
_AT_DECISION = ("gating_event", "recovery")  # forced prefix up to the tool decision (`eval.run`)


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def _cut(line: str, signal: str) -> str:
    """The line up to a lead-in colon that comes after the signal's keyword and has text behind it."""
    found = [m for k in _KEYWORDS.get(signal, ()) if (m := re.search(rf"\b{re.escape(k)}\b", line, re.IGNORECASE))]
    if found:
        colon = line.find(": ", min(m.end() for m in found))
        if colon >= 0 and line[colon + 2:].strip():
            return line[:colon + 1]
    return line


def naming_sentence(text: str, signal: str) -> tuple[str, str]:
    """(the first line of the reaction, its naming sentence: the first line that names the signal,
    else the first line), each ended at a lead-in colon (`_cut`)."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return "", ""
    naming = next((line for line in lines if names_signal(line, signal)), lines[0])
    return _cut(lines[0], signal), _cut(naming, signal)


def reaction_text(ids: list[int], tok, ms) -> str:
    """What is read for the naming sentence: up to the end of the thinking when it closes within
    the first tokens, else the first `READ` tokens."""
    head = ids[:2 * READ]
    return tok.decode(head[:head.index(ms.think_close)] if ms.think_close in head else ids[:READ])


def summary_of(ids: list[int], naming: str, tok, ms) -> tuple[str | None, int]:
    """(text, tokens) between the naming sentence and the end of the thinking; tokens that hold
    only white space do not count. (None, 0) when the thinking never closes."""
    if ms.think_close not in ids:
        return None, 0
    close = ids.index(ms.think_close)
    full = tok.decode(ids[:close])
    at = full.find(naming) if naming else -1
    end = at + len(naming) if at >= 0 else 0
    count, before = 0, 0
    for k in range(1, close + 1):
        text = tok.decode(ids[:k])
        count += bool(text[max(before, end):].strip())
        before = len(text)
    return full[end:].strip() or None, count


def reading(records: list[dict], generations: list[dict], train_eps, tok, ms) -> dict:
    """The table (rows per signal and "all"), the V3 reading and one item per counted reaction."""
    key = lambda row: (row["episode_id"], row["condition"], row["paired"])
    written = {key(g): g for g in generations}
    seen: dict[str, set] = defaultdict(set)
    trained = Counter()
    for ep in train_eps:
        start = reaction_start(ep)
        if start is not None and ep.signal:
            seen[ep.signal].add(naming_sentence(reaction_text(ep.target_ids[start:], tok, ms), ep.signal)[1])
            trained[ep.signal] += 1
    items = []
    for r in records:
        if r["condition"] != "on" or not r["event_occurred"] or r["reaction_signal"] != r["signal"] or key(r) not in written:
            continue
        ids, signal = written[key(r)]["token_ids"], r["signal"]
        first, naming = naming_sentence(reaction_text(ids, tok, ms), signal)
        item = {"episode_id": r["episode_id"], "paired": r["paired"], "kind": r["kind"], "signal": signal, "first": first,
                "naming": naming, "opens_seen": first in seen[signal], "naming_seen": naming in seen[signal]}
        if signal == "think":
            item["summary"], item["summary_tokens"] = summary_of(ids, naming, tok, ms)
        items.append(item)

    def row(name: str, part: list[dict], signals) -> dict:
        return {"signal": name, "n": len(part), "opens_seen": sum(i["opens_seen"] for i in part),
                "naming_seen": sum(i["naming_seen"] for i in part), "distinct": len({(i["signal"], i["naming"]) for i in part}),
                "training_distinct": sum(len(seen[s]) for s in signals), "training_reactions": sum(trained[s] for s in signals)}

    names = [s for s in SIGNALS if any(i["signal"] == s for i in items)]
    names += sorted({i["signal"] for i in items} - set(names))
    rows = [row(s, [i for i in items if i["signal"] == s], [s]) for s in names]
    rows.append(row("all", items, sorted(seen)))
    think = [i for i in items if i["signal"] == "think"]
    with_summary = sum(i["summary_tokens"] >= MIN_SUMMARY for i in think)
    rate = with_summary / len(think) if think else None
    by_kind = {}
    for kind in sorted({i["kind"] for i in think}):
        part = [i for i in think if i["kind"] == kind]
        got = sum(i["summary_tokens"] >= MIN_SUMMARY for i in part)
        by_kind[kind] = {"n": len(part), "with_summary": got, "rate": got / len(part)}
    return {"rows": rows, "items": items,
            "summary": {"n": len(think), "with_summary": with_summary, "rate": rate, "line": LINE,
                        "passes": None if rate is None else rate >= LINE, "min_tokens": MIN_SUMMARY, "by_kind": by_kind}}


def _share(k: int, n: int) -> str:
    return f"{k} / {n} = {k / n:.1%}" if n else "-"


def report(table: dict, results: Path, data: Path) -> str:
    lines = [f"# Wording of the reactions: {results.as_posix()} against {data.as_posix()}/train.jsonl", "",
             "Counted reactions (channel on, an event, the run's own signal). A naming sentence is \"seen\" when it is",
             "word for word a naming sentence of the training file (V2 of A9: reported, no line).", "",
             "| Signal | Counted reactions | Opens with a training sentence | Naming sentence seen in training | "
             "Distinct naming sentences | Distinct in the training file (of reactions) |", "|---|---|---|---|---|---|"]
    for r in table["rows"]:
        lines.append(f"| {r['signal']} | {r['n']} | {_share(r['opens_seen'], r['n'])} | {_share(r['naming_seen'], r['n'])} | "
                     f"{r['distinct']} | {r['training_distinct']} of {r['training_reactions']} |")
    s = table["summary"]
    lines += ["", "## Thinking budget: a summary before the thinking closes (V3 of A9)", ""]
    if not s["n"]:
        lines.append("No counted thinking-budget reaction.")
    else:
        sizes = [i["summary_tokens"] for i in table["items"] if i["signal"] == "think" and i["summary_tokens"] >= MIN_SUMMARY]
        lines.append(f"A summary of at least {MIN_SUMMARY} tokens between the naming sentence and the end of the thinking: "
                     f"{_share(s['with_summary'], s['n'])} of the counted reactions; line >= {LINE:.0%}: "
                     f"{'met' if s['passes'] else 'not met'}.")
        if sizes:
            lines.append(f"Summary length in tokens: median {statistics.median(sizes):g}, {min(sizes)}-{max(sizes)}.")
        kinds = s["by_kind"]
        if "event" in kinds:
            lines += ["", "Beside the line (note on A9 of 2026-11-17 01:47; a reaction at the very start of the turn has nothing to "
                      "sum up and was trained without a summary): among the reactions that come after some thinking (kind "
                      f"`event`): {_share(kinds['event']['with_summary'], kinds['event']['n'])}."]
        others = [f"{k} {_share(v['with_summary'], v['n'])}" for k, v in kinds.items() if k != "event"]
        if others:
            lines.append("Other kinds: " + ", ".join(others) + ".")
    return "\n".join(lines)


def examples(table: dict, generations: list[dict], tests: dict, tok, ms, n: int) -> str:
    """`n` counted thinking-budget reactions at even spacing, in full."""
    think = [i for i in table["items"] if i["signal"] == "think"]
    if n <= 0 or not think:
        return ""
    picked = think if n >= len(think) else [think[k] for k in np.linspace(0, len(think) - 1, n).round().astype(int)]
    written = {(g["episode_id"], g["condition"], g["paired"]): g for g in generations}
    lines = ["", f"## Thinking-budget reactions for reading by hand ({len(picked)} of {len(think)})", "",
             "Read each summary against the thinking above it and count those that describe it (A9). The answer is cut",
             "by the evaluation's horizon."]
    for item in picked:
        ids = written[(item["episode_id"], "on", item["paired"])]["token_ids"]
        lines += ["", f"### {item['episode_id']} ({item['kind']}, pairing {item['paired']})", ""]
        ep = tests.get(item["episode_id"])
        if ep is not None:
            asked = re.search(r"(?s)<\|im_start\|>user\n(.*?)<\|im_end\|>", tok.decode(ep.prompt_ids))
            if asked:
                lines += ["Request: " + asked.group(1).strip(), ""]
            prefix = ep.target_ids[:ep.t_decision if ep.kind in _AT_DECISION else ep.t_cross + 1]
            lines += ["Thinking before:", "", "```", tok.decode(prefix).strip(), "```", ""]
        close = ids.index(ms.think_close) if ms.think_close in ids else len(ids)
        lines += ["Naming sentence: " + item["naming"] + ("" if item["naming_seen"] else "  (not in the training file)"), "",
                  f"Summary ({item['summary_tokens']} tokens): " + (item["summary"] or "none"), "",
                  "Answer: " + tok.decode(ids[close + 1:]).strip()]
    return "\n".join(lines)


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    return tok, resolve_spec(model, tok)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="an evaluation folder (records + generations.jsonl)")
    parser.add_argument("data", type=Path, help="the folder of the model's training file (train.jsonl, test.jsonl, manifest.json)")
    parser.add_argument("--out", type=Path, default=None, help="write the reading here (.md) and its numbers beside it (.json)")
    parser.add_argument("--examples", type=int, default=0, help="thinking-budget reactions printed in full")
    parser.add_argument("--model", default=None, help="tokenizer; defaults to the model in <data>/manifest.json")
    args = parser.parse_args(argv)
    model = args.model or manifest_model(json.loads((args.data / "manifest.json").read_text(encoding="utf-8")))
    tok, ms = _load_tokenizer(model)
    records = args.results / "records.jsonl"
    records = _load(records if records.exists() else args.results / "records.partial.jsonl")
    generations = _load(args.results / "generations.jsonl")
    table = reading(records, generations, read_jsonl(args.data / "train.jsonl"), tok, ms)
    text = report(table, args.results, args.data)
    if args.examples > 0 and (args.data / "test.jsonl").exists():
        tests = {ep.episode_id: ep for ep in read_jsonl(args.data / "test.jsonl")}
        text += "\n" + examples(table, generations, tests, tok, ms, args.examples)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes((text + "\n").encode("utf-8"))
        args.out.with_suffix(".json").write_bytes((json.dumps(table, indent=1) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
