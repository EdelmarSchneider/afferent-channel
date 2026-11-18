"""Sort every missed reaction of an evaluation, channel on, and give the practical reading for `time`.

`time` misses: named the deadline and ended the turn after the 128-token window (practical success),
named but still writing at the horizon, or not named (with what the model did instead).
Other signals: every miss and every false alarm, with the start of what the model wrote.
CPU only (tokenizer). Source of the `time` tables in `results/phase1/report.md` section 3.
Usage: uv run python scripts/time_misses.py results/phase1 [model_id]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from prio.modelspec import resolve_spec
from prio.reactions import _NAMING_SPAN, names_signal, parse_tool_calls


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    out = Path(sys.argv[1])
    model_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(model_id)
    ms = resolve_spec(model_id, tok)
    key = lambda row: (row["episode_id"], row["condition"], row["paired"])
    gens = {key(g): g for g in load(out / "generations.jsonl")}
    records_file = out / "records.jsonl" if (out / "records.jsonl").exists() else out / "records.partial.jsonl"
    recs = [r for r in load(records_file) if r["condition"] == "on" and key(r) in gens]

    tally: dict[str, Counter] = defaultdict(Counter)
    print("--- time misses ---")
    for r in recs:
        if r["signal"] != "time" or not r["event_occurred"]:
            continue
        kind, ids = r["kind"], gens[key(r)]["token_ids"]
        tally[kind]["n"] += 1
        if r["reaction_signal"] == "time":
            tally[kind]["strict"] += 1
            tally[kind]["practical"] += 1
            continue
        stop = gens[key(r)]["stop_reason"]
        named = names_signal(tok.decode(ids[:_NAMING_SPAN]), "time")
        calls = [c.name for c in parse_tool_calls(ids, tok, ms)]
        if named and stop == "eos" and not calls:
            tally[kind]["practical"] += 1
            tally[kind]["named, ended late"] += 1
            print(f"{kind:12s} {r['phase']:8s} {r['episode_id']}: named, ended after {len(ids)} tokens")
        elif named:
            tally[kind]["named, not ended"] += 1
            print(f"{kind:12s} {r['phase']:8s} {r['episode_id']}: named, not ended ({stop}, {len(ids)} tokens, calls {calls})")
        else:
            tally[kind]["not named"] += 1
            print(f"{kind:12s} {r['phase']:8s} {r['episode_id']}: NOT NAMED (reaction {r['reaction_signal']}, "
                  f"calls {calls}, {stop}, {len(ids)} tokens)")
            print("    ", repr(tok.decode(ids[:120])[:300]))

    print("\n--- time tally (strict = pre-set 128-token rule; practical = named + turn ended within the horizon) ---")
    for kind, counts in tally.items():
        n = counts["n"]
        print(f"{kind:12s} n {n:3d}  strict {counts['strict']:3d} ({counts['strict'] / n:.1%})  "
              f"practical {counts['practical']:3d} ({counts['practical'] / n:.1%})  "
              f"ended late {counts['named, ended late']}  not ended {counts['named, not ended']}  not named {counts['not named']}")

    print("\n--- other signals: misses and false alarms ---")
    for r in recs:
        if r["signal"] == "time":
            continue
        miss = r["event_occurred"] and r["reaction_signal"] != r["signal"]
        alarm = not r["event_occurred"] and r["reaction_signal"] is not None
        if miss or alarm:
            g = gens[key(r)]
            print(f"{'MISS' if miss else 'ALARM'} {r['signal']} {r['kind']} {r['phase']} {r['paired']} {r['episode_id']} "
                  f"reaction {r['reaction_signal']} {g['stop_reason']} {len(g['token_ids'])} tokens")
            print("    ", repr(tok.decode(g["token_ids"][:120])[:300]))


if __name__ == "__main__":
    main()
