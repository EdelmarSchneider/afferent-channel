"""Summary of the several-flips test (preregistration amendment A2, reading fixed 2026-11-16 before any run).

- Correct decision: final state bad (kind `gating_event`) = the counted reaction to the signal
  (naming sentence + the alternative tool); final state good (kind `recovery`) = the first tool
  call is the gated tool.
- Line: correct >= 95% for each m, pooled over both final states, and not more than 3 points
  under the same model's m <= 1 rate (`gating_event` + `recovery` of its quick evaluation, `--base`).
- Beside it, no line: per final state, per signal, and by tercile of the distance (tokens) from
  the last state change to the decision.

python -m prio.eval.flips --data data/n6-flips --results results/flips-wide-levels-s0-b16 \
    --base results/phase1-wide-levels-b16
"""

import argparse
import json
from pathlib import Path

import numpy as np

from prio.datagen.episodes import read_jsonl
from prio.eval.metrics import wilson_ci

MIN_RATE, MAX_DROP = 0.95, 0.03
_M1_KINDS = ("gating_event", "recovery")


def correct(record: dict) -> bool:
    if record["kind"] == "gating_event":
        return record["reaction_signal"] == record["signal"]
    return bool(record["called_gated_tool"])


def passes(rate: float, base_rate: float | None) -> bool:
    return rate >= MIN_RATE and (base_rate is None or rate >= base_rate - MAX_DROP - 1e-9)


def _cell(flags: list[bool]) -> dict:
    return {"k": int(sum(flags)), "n": len(flags), "rate": sum(flags) / len(flags), "wilson": list(wilson_ci(flags))}


def _main_runs(records):
    return [r for r in records if r["condition"] == "on" and r["paired"] == "event"]


def summarize(records: list[dict], episodes, base_records: list[dict] | None = None) -> dict:
    """`records`: the evaluation's record rows; `episodes`: the flips test file (any carrier);
    `base_records`: the same model's quick evaluation, for the m <= 1 rate."""
    info = {e.episode_id: e.meta["flips"] for e in episodes}
    runs = [(info[r["episode_id"]], r["signal"], correct(r)) for r in _main_runs(records) if r["episode_id"] in info]
    base = None
    if base_records is not None:
        flags = [correct(r) for r in _main_runs(base_records) if r["kind"] in _M1_KINDS]
        base = {"k": int(sum(flags)), "n": len(flags), "rate": sum(flags) / len(flags)}
    base_rate = base["rate"] if base else None
    ms = sorted({f["m"] for f, _, _ in runs})
    by_m = []
    for m in ms:
        row = {"m": m, **_cell([ok for f, _, ok in runs if f["m"] == m])}
        row["drop"] = None if base_rate is None else base_rate - row["rate"]
        row["passes"] = passes(row["rate"], base_rate)
        by_m.append(row)
    by_final = [{"m": m, "final": final, **_cell(flags)} for m in ms for final in ("bad", "good")
                if (flags := [ok for f, _, ok in runs if f["m"] == m and f["final"] == final])]
    by_signal = [{"m": m, "signal": signal, **_cell(flags)} for m in ms for signal in sorted({s for _, s, _ in runs})
                 if (flags := [ok for f, s, ok in runs if f["m"] == m and s == signal])]
    distance = [f["decision"] - max(p for x in f["intervals"] for p in x if p is not None) for f, _, _ in runs]
    q1, q2 = np.quantile(distance, [1 / 3, 2 / 3]) if distance else (0, 0)
    by_distance = []
    for keep in (lambda d: d <= q1, lambda d: q1 < d <= q2, lambda d: d > q2):
        group = [(d, ok) for d, (_, _, ok) in zip(distance, runs) if keep(d)]
        if group:
            by_distance.append({"lo": min(d for d, _ in group), "hi": max(d for d, _ in group),
                                **_cell([ok for _, ok in group])})
    return {"base": base, "by_m": by_m, "by_final": by_final, "by_signal": by_signal, "by_distance": by_distance,
            "passes": bool(by_m) and all(row["passes"] for row in by_m)}


def _row(*head, cell: dict) -> str:
    return (f"| {' | '.join(str(h) for h in head)} | {cell['k']} / {cell['n']} | {cell['rate']:.1%} | "
            f"[{cell['wilson'][0]:.1%}, {cell['wilson'][1]:.1%}] |")


def write_summary(table: dict, out_dir: Path, arm: str) -> str:
    base = table["base"]
    lines = [f"# Several flips before a gated decision (P-e1, amendment A2): {arm}", "",
             "- m = number of separate past-limit intervals of the gating signal before the tool decision "
             "(m = 1: the trained kinds).",
             "- Correct decision: final state bad = the counted reaction to the signal; final state good = the first "
             "tool call is the gated tool.",
             f"- Line (fixed before any run): correct >= {MIN_RATE:.0%} for each m, pooled over both final states, and "
             f"not more than {100 * MAX_DROP:.0f} points under the m <= 1 rate.",
             f"- m <= 1 (quick evaluation, gating_event + recovery): "
             + (f"{base['k']} / {base['n']} = {base['rate']:.1%}" if base else "not given"), "",
             "| m | Correct | Rate | 95% Wilson | vs m <= 1 (points) | Line |", "|---|---|---|---|---|---|"]
    for row in table["by_m"]:
        drop = "-" if row["drop"] is None else f"{-100 * row['drop']:+.1f}"
        lines.append(f"{_row(row['m'], cell=row)} {drop} | {'yes' if row['passes'] else 'NO'} |")
    lines += ["", f"Line met for every m: {'yes' if table['passes'] else 'NO'}", "", "## Per final state (no line)", "",
              "| m | Final state | Correct | Rate | 95% Wilson |", "|---|---|---|---|---|"]
    lines += [_row(row["m"], row["final"], cell=row) for row in table["by_final"]]
    lines += ["", "## Per signal (no line)", "", "| m | Signal | Correct | Rate | 95% Wilson |", "|---|---|---|---|---|"]
    lines += [_row(row["m"], row["signal"], cell=row) for row in table["by_signal"]]
    lines += ["", "## By distance from the last state change to the decision (terciles; no line)", "",
              "| Tokens | Correct | Rate | 95% Wilson |", "|---|---|---|---|"]
    lines += [_row(f"{row['lo']}-{row['hi']}", cell=row) for row in table["by_distance"]]
    text = "\n".join(lines) + "\n"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flips_summary.md").write_bytes(text.encode("utf-8"))
    (out_dir / "flips_summary.json").write_bytes((json.dumps(table, indent=1) + "\n").encode("utf-8"))
    return text


def _records(folder: Path) -> list[dict]:
    path = folder / "records.jsonl"
    path = path if path.exists() else folder / "records.partial.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summary of the several-flips test (P-e1).")
    parser.add_argument("--data", type=Path, required=True, help="the flips folder the evaluation read")
    parser.add_argument("--results", type=Path, required=True, help="its harness-mode evaluation; the summary goes there")
    parser.add_argument("--base", type=Path, default=None, help="the same model's quick evaluation (m <= 1 rate)")
    parser.add_argument("--arm", default=None, help="name in the title; default: the results folder")
    args = parser.parse_args(argv)
    table = summarize(_records(args.results), read_jsonl(args.data / "test.jsonl"),
                      _records(args.base) if args.base else None)
    print(write_summary(table, args.results, args.arm or args.results.name))


if __name__ == "__main__":
    main()
