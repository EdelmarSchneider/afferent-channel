"""Recovery episodes of one evaluation: wrongful refusals (the signal came back before the gated tool
decision, the model still refuses), per signal, beside what the delivery showed.

For a notice / insert data folder each recovery episode carries two insertions (crossing, recovery):
prints, for refused vs called runs, the distance from the recovery insertion to the tool decision
and the recovered value. `--vs`: the same count in other evaluations of the same episodes, with a
two-sided Fisher exact p. Extra reading (not pre-registered); first use: R7, the one inserted vector.

uv run python scripts/analysis/recovery_reading.py results/b6b-insert-levels-quick-b16 data/n6-insert \
    --vs results/phase1-wide-levels-b16 results/b3-notice-none-quick-b16 results/b3s-notice-segment-quick-b16
"""

import argparse
import json
import statistics
from pathlib import Path

from prio.eval.spoof import fisher_exact


def refusals(folder: Path) -> dict:
    """episode_id -> (signal, refused) for the recovery runs of condition `on`."""
    out = {}
    for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["condition"] == "on" and r["kind"] == "recovery":
            out[r["episode_id"]] = (r["signal"], not r["called_gated_tool"])
    return out


def shown(data: Path, ids: set) -> dict:
    """episode_id -> (recovery insertion to decision in tokens, recovered value, crossing to recovery)."""
    out = {}
    for line in (data / "test.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["episode_id"] not in ids:
            continue
        roles = {x["role"]: x for x in ((e.get("meta") or {}).get("variant") or {}).get("insertions", [])}
        if "recovery" in roles and "crossing" in roles:
            back = roles["recovery"]
            out[e["episode_id"]] = (e["t_decision"] - back["at"], float(back["shown"]),
                                    back["at"] - roles["crossing"]["at"])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--vs", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    own = refusals(args.results)
    seen = shown(args.data, set(own))
    k, n = sum(x for _, x in own.values()), len(own)
    print(f"{args.results.as_posix()}: wrongful refusals {k} / {n} = {100 * k / n:.1f}%")
    print("| Signal | Outcome | n | Recovery insertion -> decision, tokens (median, min-max) | "
          "Crossing -> recovery (median) | Recovered values |\n|---|---|---|---|---|---|")
    for signal in sorted({s for s, _ in own.values()}):
        for refused in (True, False):
            rows = [seen[i] for i, (s, x) in own.items() if s == signal and x == refused and i in seen]
            if rows:
                gap = [r[0] for r in rows]
                print(f"| {signal} | {'refused' if refused else 'called'} | {len(rows)} | "
                      f"{statistics.median(gap):g} ({min(gap)}-{max(gap)}) | {statistics.median(r[2] for r in rows):g} | "
                      f"{', '.join(f'{r[1]:.2f}' for r in sorted(rows, key=lambda r: r[1]))} |")
    for other in args.vs:
        theirs = refusals(other)
        common = sorted(set(own) & set(theirs))
        ka, kb = sum(own[i][1] for i in common), sum(theirs[i][1] for i in common)
        print(f"vs {other.as_posix()}: {ka} / {len(common)} vs {kb} / {len(common)}, "
              f"Fisher p {fisher_exact(ka, len(common), kb, len(common)):.3g}")


if __name__ == "__main__":
    main()
