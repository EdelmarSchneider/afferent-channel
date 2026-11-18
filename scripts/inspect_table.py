"""Read sibling families of a state-table dataset by hand (prio.table; e.g. data/n7msg, data/npc6):
the task's own system prompt when it has one (a character card), the user message, the tail of the
shared prefix, and for every sibling its state class, timing kind, the signals it sets (value at
the decision; also those it holds over the whole episode) and what follows the cut. CPU only
(tokenizer).
Usage: uv run python scripts/inspect_table.py data/n7msg [n families] [test|train]
"""

import json
import sys
from pathlib import Path

from prio.datagen.episodes import read_jsonl
from prio.table.evaluate import families


def _value(v: float) -> str:
    """Two decimals; three for the center of a 1% step (0.495 must not read as 0.50, past a band edge)."""
    return f"{v:.2f}" if abs(v * 100 - round(v * 100)) < 1e-3 else f"{v:.3f}"


def show(data: Path, n: int, split: str, tok) -> None:
    groups = families(read_jsonl(Path(data) / f"{split}.jsonl"))
    step = max(1, len(groups) // n)
    for name in list(groups)[::step][:n]:
        siblings = groups[name]
        first = siblings[0].meta["table"]
        cut = first["cut"]
        print("=" * 100)
        print(f"family {name}: {len(siblings)} siblings, cut at target token {cut}, own command {first['own_command']}")
        if first["system"]:
            print("system:", first["system"])
        print("user:", first["user"])
        print("--- shared prefix, last 40 tokens:", repr(tok.decode(siblings[0].target_ids[max(0, cut - 40):cut])))
        for ep in siblings:
            t = ep.meta["table"]
            when = {"onset": f" t_on={t['t_on']}", "back": f" on [{t['t_on']}, {t['t_back']})"}.get(t["timing"], "")
            state = " ".join(f"{s}={_value(t['state'][s])}" for s in (t["signals"] + t.get("whole", []) or first["state"]))
            print(f"- {t['variant']}: class {t['cls']} ({t['behavior']}), timing {t['timing'] or 'none'}{when}, "
                  f"at the decision {state}; weighted {ep.focus_spans or '-'} + token {ep.t_diverge}")
            print("    after the cut:", repr(tok.decode(ep.target_ids[cut:])))


def main() -> None:
    from transformers import AutoTokenizer

    data = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for key in ("tasks_requested", "base_runs_finished", "base_runs_kept", "stop_rule", "dropped", "own_command",
                "families", "episodes", "by_class_timing"):
        print(f"{key}: {manifest.get(key)}")
    for key in ("false_naming_floor", "families_dropped", "families_without_mention", "episodes_per_task", "by_role", "by_phase"):
        if key in manifest:  # R22 (prio.npc)
            print(f"{key}: {manifest[key]}")
    show(data, n, split, AutoTokenizer.from_pretrained(manifest["model"]))


if __name__ == "__main__":
    main()
