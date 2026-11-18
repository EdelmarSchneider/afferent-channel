"""The manual check of a built dataset before training: counts, a few reactions per signal, and
what the episodes WITHOUT a reaction look like (the first build taught reactions there too).

Usage: uv run python scripts/inspect_reactions.py data/n6 [n_per_signal]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from prio.datagen.episodes import reaction_start, read_jsonl
from prio.modelspec import resolve_spec
from prio.reactions import calls_reaction_tool, own_tools, parse_tool_calls

data = Path(sys.argv[1])
per_signal = int(sys.argv[2]) if len(sys.argv) > 2 else 3
manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(manifest["model"])
ms = resolve_spec(manifest["model"], tok)
train, test = read_jsonl(data / "train.jsonl"), read_jsonl(data / "test.jsonl")

print("episodes:", manifest["episodes"])
print("train kinds:", manifest["kinds"]["train"])
print("test kinds :", manifest["kinds"]["test"])
for key in ("base_runs_finished", "base_runs_kept", "base_runs_kept_by_family", "event_phases"):
    print(f"{key}:", manifest.get(key))
print("acceptance (first try):", {k: round(v, 2) for k, v in manifest["acceptance_rate"].items()})
print("fallback:", manifest["fallback"], "| teacher failed:", manifest["teacher_failed"], "| fully templated:", manifest["templated"])

# Every check below must print 0.
quiet = [ep for ep in train if reaction_start(ep) is None]
print("train episodes without a reaction that call a reaction tool:",
      sum(calls_reaction_tool(ep.target_ids, tok, ms) for ep in quiet), "of", len(quiet))
reacting = [ep for ep in train if reaction_start(ep) is not None]
print("reactions that call another signal's tool:",
      sum(calls_reaction_tool(ep.target_ids, tok, ms, allowed=own_tools(ep.signal)) for ep in reacting), "of", len(reacting))
gated = {"net": "web_search", "ram": "run_heavy"}
unused = [ep for ep in train + test if ep.kind == "gating_unused"]
print("gating_unused episodes that call the gated tool:",
      sum(any(c.name == gated[ep.signal] for c in parse_tool_calls(ep.target_ids, tok, ms)) for ep in unused), "of", len(unused))
print("reaction episodes and twins without the decision token:",
      sum(ep.t_diverge is None for ep in reacting), "of", len(reacting), "|",
      sum(ep.t_diverge is not None for ep in quiet), "twins marked")
lengths = sorted(len(ep.prompt_ids) + len(ep.target_ids) for ep in train)
print("train length median / p90 / max:", lengths[len(lengths) // 2], lengths[int(len(lengths) * 0.9)], lengths[-1])
print("answer forms in episodes without a reaction:",
      dict(Counter("tool call" if ms.tool_open in ep.target_ids else "text" for ep in quiet)))

shown = defaultdict(int)
for ep in train:
    start = reaction_start(ep)
    if start is None or shown[(ep.signal, ep.phase)] >= per_signal:
        continue
    shown[(ep.signal, ep.phase)] += 1
    crossing = ep.signals[len(ep.prompt_ids) + (ep.t_cross or 0)]
    print("=" * 100)
    print(f"{ep.episode_id} kind={ep.kind} signal={ep.signal} phase={ep.phase} templated={ep.templated} "
          f"t_diverge-start={None if ep.t_diverge is None else ep.t_diverge - start} "
          f"signals at crossing={[round(float(v), 3) for v in crossing]}")
    print("BEFORE :", repr(tok.decode(ep.target_ids[max(0, start - 25):start])))
    print("REACTION:", repr(tok.decode(ep.target_ids[start:start + 140])))

families = defaultdict(int)
for ep in quiet:
    if families[ep.family] >= 1:
        continue
    families[ep.family] += 1
    print("=" * 100)
    print(f"{ep.episode_id} kind={ep.kind} family={ep.family} (no reaction) ENDS WITH:")
    print(repr(tok.decode(ep.target_ids[-90:])))
