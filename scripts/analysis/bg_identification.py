"""Does a background-task model read WHICH slot changed? (The reading that led to slot twins.)

For every multi-task event evaluated under "on" and "swap" (event moved to the next slot): the
named task per event slot, and how often the model names the same task in both conditions (same
task = it ignores the slot). Usage: uv run python scripts/analysis/bg_identification.py
results/bg-wide-levels data/bg3 [records-quick.jsonl]   (CPU, seconds; the
first numbers came from the quick evaluation, kept as records-quick.jsonl).
"""

import json
import sys
from collections import Counter

results, data = sys.argv[1], sys.argv[2]
records = sys.argv[3] if len(sys.argv) > 3 else "records.jsonl"
eps = {d["episode_id"]: d for d in map(json.loads, open(f"{data}/test.jsonl")) if d["kind"] == "bg_event"}
recs = [json.loads(l) for l in open(f"{results}/{records}")]
by = {(r["episode_id"], r["condition"]): r for r in recs
      if r["kind"] == "event" and (r["n_watched"] or 0) > 1 and r["condition"] in ("on", "swap")}
same, conf = Counter(), Counter()
for (eid, c), r in by.items():
    m = eps[eid]["meta"]
    n, slot = len(m["tasks"]), m["event"]["slot"]
    if c == "swap":
        slot = slot % n + 1
    conf[(c, n, slot, r.get("named_task"))] += 1
    if c == "on" and (eid, "swap") in by:
        same[r.get("named_task") == by[(eid, "swap")].get("named_task")] += 1
print(f"on vs swap, same task named: {same[True]} / {same[True] + same[False]}")
# Right task vs chance (naming one watched task at random: 1 / n), over reacted multi-task "on" events.
from prio.eval.metrics import wilson_ci  # noqa: E402

for n_set in ((2,), (3,), (2, 3)):
    on = [r for (eid, c), r in by.items() if c == "on" and r["reacted"] and r["n_watched"] in n_set]
    if on:
        right = [bool(r["right_task"]) for r in on]
        lo, hi = wilson_ci(right)
        chance = sum(1 / r["n_watched"] for r in on) / len(on)
        print(f"{'/'.join(map(str, n_set))} tasks, on, reacted: right task {sum(right)} / {len(on)} = "
              f"{sum(right) / len(on):.1%} [{lo:.1%}, {hi:.1%}], chance {chance:.1%}")
for n in (2, 3):
    for c in ("on", "swap"):
        print(f"{n} tasks, {c}: event slot -> named task")
        for s in range(1, n + 1):
            print(f"  slot {s}:", dict(sorted((k[3] or "-", v) for k, v in conf.items() if k[:3] == (c, n, s))))
