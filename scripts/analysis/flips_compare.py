"""Several-flips test (P-e1, amendment A2): one arm against another on the same episodes.

Pairs by episode id (the change points are the same for every carrier). Per m and pooled, and
per final state: correct decisions of each arm, discordant pairs, exact McNemar p. Extra table:
what each arm did (named the signal / called the gated tool / other) per m and final state.

python scripts/analysis/flips_compare.py results/flips-wide-levels-s0-b16 \
    results/flips-notice-none-s0-b16 [--out results/flips-notice-none-s0-b16/flips_compare.md]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from prio.eval.flips import correct
from prio.eval.spoof import mcnemar_exact


def _load(folder: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["episode_id"]: r for r in rows if r["condition"] == "on" and r["paired"] == "event"}


def _key(episode_id: str) -> tuple[int, str]:
    _, m, final = episode_id.rsplit("-", 2)
    return int(m[1:]), final


def _did(r: dict) -> str:
    if r["reaction_signal"]:
        return "named"
    return "gated tool" if r["called_gated_tool"] else "other"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of two arms of the several-flips test.")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    a, b = _load(args.a), _load(args.b)
    ids = sorted(set(a) & set(b))
    lines = [f"# Several flips: A = {args.a.as_posix()}, B = {args.b.as_posix()}", "",
             f"Pairs: {len(ids)} (A only {len(set(a) - set(b))}, B only {len(set(b) - set(a))}). "
             "Correct decision as in `prio.eval.flips`; exact McNemar, two-sided.", "",
             "| m | Final state | n | A correct | B correct | A only / B only | McNemar p |", "|---|---|---|---|---|---|---|"]
    groups = [(m, final) for m in sorted({_key(i)[0] for i in ids}) for final in ("bad", "good", "both")]
    groups.append(("all", "both"))
    for m, final in groups:
        sel = [i for i in ids if (m == "all" or _key(i)[0] == m) and (final == "both" or _key(i)[1] == final)]
        if not sel:
            continue
        ca, cb = [correct(a[i]) for i in sel], [correct(b[i]) for i in sel]
        only_a = sum(x and not y for x, y in zip(ca, cb))
        only_b = sum(y and not x for x, y in zip(ca, cb))
        lines.append(f"| {m} | {final} | {len(sel)} | {sum(ca)} ({sum(ca) / len(sel):.1%}) | {sum(cb)} ({sum(cb) / len(sel):.1%}) | "
                     f"{only_a} / {only_b} | {mcnemar_exact(only_a, only_b):.3g} |")
    lines += ["", "## What each arm did (first decision)", "", "| Arm | m | Final state | Signal | Named the signal | Called the gated tool | Other |",
              "|---|---|---|---|---|---|---|"]
    for name, rows in (("A", a), ("B", b)):
        counts = Counter((*_key(i), rows[i]["signal"], _did(rows[i])) for i in ids)
        for m, final, signal in sorted({k[:3] for k in counts}):
            lines.append(f"| {name} | {m} | {final} | {signal} | {counts[(m, final, signal, 'named')]} | "
                         f"{counts[(m, final, signal, 'gated tool')]} | {counts[(m, final, signal, 'other')]} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
