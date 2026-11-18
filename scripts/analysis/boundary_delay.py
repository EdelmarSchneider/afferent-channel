"""Boundary-only delivery delay and state-change rate on the operating-state test file.

Boundary-only harness: can add text only at the next tool call or at the end of the turn (API pattern:
an update after each tool call). Delay = tokens from the first token that could react (t_cross + 1) to that
boundary, in the base run. Also counts how often the stored trajectories change value (1% steps), i.e. how
many updates an event-driven text harness would need to keep the six signals current.

    uv run python scripts/analysis/boundary_delay.py [data/n6/test.jsonl]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from prio.datagen.episodes import read_jsonl  # noqa: E402

THINK_CLOSE, TOOL_OPEN = 151668, 151657
eps = read_jsonl(sys.argv[1] if len(sys.argv) > 1 else "data/n6/test.jsonl")
rows = defaultdict(list)
for ep in eps:
    if ep.kind != "event":
        continue
    t = ep.target_ids
    c = ep.t_cross + 1
    tool = next((i for i in range(c, len(t)) if t[i] == TOOL_OPEN), None)
    boundary = tool if tool is not None else len(t)
    close = t.index(THINK_CLOSE) if THINK_CLOSE in t else None
    thinking_over = ep.phase == "thinking" and close is not None and close < boundary
    rows[ep.signal].append((boundary - c, tool is not None, thinking_over, ep.phase))

print("signal n median_delay p25 p75 share_tool_boundary share_thinking_over_before_update")
allrows = []
for s, r in sorted(rows.items()):
    d = np.array([x[0] for x in r])
    allrows += r
    print(s, len(r), int(np.median(d)), int(np.percentile(d, 25)), int(np.percentile(d, 75)),
          f"{np.mean([x[1] for x in r]):.2f}", f"{np.mean([x[2] for x in r if x[3] == 'thinking'] or [0]):.2f}")
d = np.array([x[0] for x in allrows])
print("all", len(d), int(np.median(d)), int(np.percentile(d, 25)), int(np.percentile(d, 75)),
      f"{np.mean([x[1] for x in allrows]):.2f}",
      f"{np.mean([x[2] for x in allrows if x[3] == 'thinking']):.2f}")

# State-change rate over generated positions (target part) of every test episode.
changes, tokens = 0, 0
per_signal = np.zeros(6)
for ep in eps:
    v = ep.signals[len(ep.prompt_ids):]
    diff = np.any(v[1:] != v[:-1], axis=1)
    changes += int(diff.sum())
    per_signal += (v[1:] != v[:-1]).sum(axis=0)
    tokens += len(v) - 1
print(f"rows with any change: {changes} / {tokens} = {changes / tokens:.3f} per token "
      f"(one change every {tokens / changes:.1f} tokens)")
print("per signal changes per 100 tokens:", np.round(100 * per_signal / tokens, 2))
