"""What the `decision react` loss can reach: entropy of the token at t_diverge, per signal.
Usage: uv run python decision_floor.py data/n6
State-table data too (data/n7msg, data/npc6): there the entropy is the wording floor of each class
(varied sentence banks), not a sign of confusion between classes."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from prio.datagen.episodes import reaction_start, read_jsonl

data = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(json.loads((data / "manifest.json").read_text(encoding="utf-8"))["model"])
train = read_jsonl(data / "train.jsonl")
by_signal = defaultdict(Counter)
offsets = Counter()
n_target = 0
for ep in train:
    n_target += len(ep.target_ids)
    start = reaction_start(ep)
    # same rule as train.collate: bg events and state-table reactions have no reaction_start
    reacts = start is not None or ep.kind in ("bg_event", "table_act", "table_mention")
    if not reacts or ep.t_diverge is None:
        continue
    by_signal[(ep.signal, ep.kind)][ep.target_ids[ep.t_diverge]] += 1
    if start is not None:
        offsets[ep.t_diverge - start] += 1
print("target tokens:", n_target, "| reactions:", sum(sum(c.values()) for c in by_signal.values()),
      "| blind per-token rate -> loss", round(-math.log(sum(sum(c.values()) for c in by_signal.values()) / n_target), 2))
print("t_diverge - reaction start:", dict(sorted(offsets.items())))
for key, counts in sorted(by_signal.items()):
    total = sum(counts.values())
    entropy = -sum(c / total * math.log(c / total) for c in counts.values())
    top = [(repr(tok.decode([t])), c) for t, c in counts.most_common(6)]
    print(f"{key}: n={total} distinct={len(counts)} entropy={entropy:.2f} top={top}")
