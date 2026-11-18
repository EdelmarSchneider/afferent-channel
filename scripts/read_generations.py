"""Read what the model actually wrote during an evaluation, next to how each run was classified.

Channel on, per signal: reactions, misses (event, no reaction), false alarms (nominal or
near-miss trajectory, a reaction anyway), unnamed reaction-tool calls, swap-control outcomes.
CPU only: needs just the tokenizer.
Usage: uv run python scripts/read_generations.py results/phase1 [per_group] [model_id]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def outcome(r: dict) -> str | None:
    own = r["reaction_signal"] == r["signal"]
    if r["kind"] == "swap":
        return "swap: followed the injected signal" if own else "SWAP: DID NOT FOLLOW"
    if r["event_occurred"]:
        return "reaction" if own else "MISS"
    if r["reaction_signal"] is not None:
        return f"FALSE ALARM ({r['paired']} trajectory, kind {r['kind']})"
    if r.get("unnamed_call"):
        return f"unnamed tool call ({r['paired']} trajectory)"
    return None


def main() -> None:
    out_dir = Path(sys.argv[1])
    per_group = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    model_id = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(model_id)

    key = lambda row: (row["episode_id"], row["condition"], row["paired"])
    written = {key(g): g for g in load(out_dir / "generations.jsonl")}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in load(out_dir / "records.partial.jsonl"):
        label = outcome(r) if r["condition"] == "on" and key(r) in written else None
        if label:
            groups[(r["signal"], label)].append(r)

    for (signal, label), rows in sorted(groups.items()):
        print("=" * 100)
        print(f"{signal} | {label} | {len(rows)} runs, showing {min(per_group, len(rows))}")
        for r in rows[:: max(1, len(rows) // per_group)][:per_group]:
            g = written[key(r)]
            text = tok.decode(g["token_ids"], skip_special_tokens=False)
            print("-" * 100)
            print(f"{r['episode_id']} kind={r['kind']} phase={r['phase']} paired={r['paired']} "
                  f"reaction_signal={r['reaction_signal']} delay={r['delay']} named={r['named']} "
                  f"gated_call={r['called_gated_tool']} alternative={r['used_alternative']} stop={g['stop_reason']}")
            print(repr(text[:600]))


if __name__ == "__main__":
    main()
