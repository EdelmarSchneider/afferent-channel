"""R21 context eviction (preregistration amendment A7): one arm, one evicted test file.

Reads the evaluation's records and the evicted file's per-run bookkeeping (`meta["evict"]`):
- line of A7 per m: correct decisions >= 95% and not more than 3 points under the same model on the
  same runs without eviction (`--ref`: that evaluation's folder);
- correct decisions by final state and by how many state changes begin inside the window
  (0 / 1 = only the last / 2+), beside the same runs without eviction, exact McNemar;
- A7's predictions for the notice-based arms: final bad with every change outside the window
  (wrong by construction, reported only); final bad, m in {2, 3}, only the last change inside
  (prediction 2: correct >= 90%; fewer than 20 runs pooled over windows = too few to read).

python scripts/analysis/evict_reading.py data/n6-notice-flips-w16 results/r21-notice-none-s0-m23-w16-b16 \
    --ref results/flips-notice-none-s0-b16 [--out <results>/evict_reading.md]
"""

import argparse
import json
from pathlib import Path

from prio.datagen.episodes import read_jsonl
from prio.eval.flips import MAX_DROP, MIN_RATE, correct
from prio.eval.metrics import wilson_ci
from prio.eval.spoof import mcnemar_exact


def _load(folder: Path) -> dict[str, bool]:
    path = folder / "records.jsonl"
    path = path if path.exists() else folder / "records.partial.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["episode_id"]: correct(r) for r in rows if r["condition"] == "on" and r["paired"] == "event"}


def _cell(flags: list[bool]) -> str:
    if not flags:
        return "-"
    lo, hi = wilson_ci(flags)
    return f"{sum(flags)} / {len(flags)} = {sum(flags) / len(flags):.1%} [{lo:.1%}, {hi:.1%}]"


def _inside(k: int) -> str:
    return "2+" if k >= 2 else str(k)


def reading(episodes, got: dict[str, bool], ref: dict[str, bool] | None) -> str:
    runs = [(e.meta["flips"], e.meta["evict"], got[e.episode_id], None if ref is None else ref.get(e.episode_id))
            for e in episodes if e.episode_id in got]
    window = runs[0][1]["window"] if runs else None
    lines = [f"Window: last {window} tokens of the turn before the decision; {len(runs)} runs.", "",
             "## Line of A7 (per m, pooled over final states)", "",
             "| m | Evicted | Same runs, no eviction | Drop (points) | Line (>= 95%, drop <= 3) |", "|---|---|---|---|---|"]
    for m in sorted({f["m"] for f, _, _, _ in runs}):
        flags = [ok for f, _, ok, _ in runs if f["m"] == m]
        before = [r for f, _, _, r in runs if f["m"] == m and r is not None]
        rate = sum(flags) / len(flags)
        drop = None if not before else sum(before) / len(before) - rate
        met = rate >= MIN_RATE and (drop is None or drop <= MAX_DROP + 1e-9)
        lines.append(f"| {m} | {_cell(flags)} | {_cell(before)} | {'-' if drop is None else f'{100 * drop:+.1f}'} | "
                     f"{'met' if met else 'NOT met'} |")
    lines += ["", "## By final state and state changes that begin inside the window", "",
              "| m | Final state | Changes inside | Evicted | Same runs, no eviction | Evicted only / reference only correct | McNemar p |",
              "|---|---|---|---|---|---|---|"]
    for m in sorted({f["m"] for f, _, _, _ in runs}):
        for final in ("bad", "good"):
            for k in ("0", "1", "2+"):
                group = [(ok, r) for f, e, ok, r in runs if f["m"] == m and f["final"] == final and _inside(e["changes_inside"]) == k]
                if not group:
                    continue
                paired = [(ok, r) for ok, r in group if r is not None]
                only_a, only_b = sum(ok and not r for ok, r in paired), sum(r and not ok for ok, r in paired)
                lines.append(f"| {m} | {final} | {k} | {_cell([ok for ok, _ in group])} | {_cell([r for _, r in paired])} | "
                             f"{only_a} / {only_b} | {mcnemar_exact(only_a, only_b):.3g} |" if paired else
                             f"| {m} | {final} | {k} | {_cell([ok for ok, _ in group])} | - | - | - |")
    gone = [ok for f, e, ok, _ in runs if f["final"] == "bad" and e["changes_inside"] == 0]
    last = [(ok, r) for f, e, ok, r in runs if f["final"] == "bad" and f["m"] in (2, 3) and e["changes_inside"] == 1]
    lines += ["", "## A7's predictions (notice-based arms; reported for every arm)", "",
              f"- Final bad, every change outside the window: wrong {_cell([not ok for ok in gone])} "
              "(true by construction for a notice-based arm; never a finding on its own).",
              f"- Final bad, m in {{2, 3}}, only the last change inside: correct {_cell([ok for ok, _ in last])}; the same runs "
              f"without eviction: {_cell([r for _, r in last if r is not None])}. Prediction 2 (text-notice model): >= 90%; "
              "fewer than 20 such runs pooled over both windows = too few to read."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="R21: reading of one arm on one evicted file.")
    parser.add_argument("data", type=Path, help="the evicted test folder (prio.datagen.evict)")
    parser.add_argument("results", type=Path, help="the arm's evaluation of that folder")
    parser.add_argument("--ref", type=Path, default=None, help="the same arm on the same runs without eviction")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = (f"# Context eviction (R21, amendment A7): {args.results} on {args.data}"
            + (f", reference {args.ref}" if args.ref else "") + "\n\n"
            + reading(read_jsonl(args.data / "test.jsonl"), _load(args.results), _load(args.ref) if args.ref else None))
    if args.out:
        args.out.write_bytes(text.encode("utf-8"))
    print(text)


if __name__ == "__main__":
    main()
