"""Paired detection test between two evaluations of the same test episodes (program A R4: "no detection
claim either way unless paired McNemar p < 0.01 on the full set").

Pairs runs by (episode_id, kind) for the event pairing of condition `on` (channel model: paired
"event"; text model in harness mode: paired "event"), counts hits (reaction_signal == signal) in
each, prints both rates, the discordant pairs and the exact McNemar p, per kind group and overall.

uv run python scripts/analysis/paired_detection.py results/phase1 results/b3-notice-none-full-b16
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from prio.eval.spoof import mcnemar_exact

GROUPS = {"event": "interrupt", "gating_event": "gating", "starts_past": "starts_past"}


def load(folder: Path) -> dict:
    out = {}
    for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["condition"] == "on" and r["paired"] == "event" and r["kind"] in GROUPS:
            out[(r["episode_id"], r["kind"])] = r["reaction_signal"] == r["signal"]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    a, b = load(args.a), load(args.b)
    by = defaultdict(list)
    for key in sorted(set(a) & set(b)):
        by[GROUPS[key[1]]].append((a[key], b[key]))
        by["all"].append((a[key], b[key]))
    print(f"A = {args.a.as_posix()}, B = {args.b.as_posix()}; pairs: {len(by['all'])} "
          f"(A only {len(set(a) - set(b))}, B only {len(set(b) - set(a))})")
    print("| Group | n | A | B | A only / B only | McNemar p |\n|---|---|---|---|---|---|")
    for group in ("interrupt", "gating", "starts_past", "all"):
        pairs = by.get(group, [])
        if not pairs:
            continue
        n, ka, kb = len(pairs), sum(x for x, _ in pairs), sum(y for _, y in pairs)
        only_a, only_b = sum(x and not y for x, y in pairs), sum(y and not x for x, y in pairs)
        print(f"| {group} | {n} | {ka} ({100 * ka / n:.1f}%) | {kb} ({100 * kb / n:.1f}%) | {only_a} / {only_b} | "
              f"{mcnemar_exact(only_a, only_b):.3g} |")


if __name__ == "__main__":
    main()
