"""Forged status, one model across its seeds (preregistration amendment A5). CPU, seconds.

Reads the `summary.json` of `prio.eval.spoof` folders and prints, per cell, reactions / n of every
seed side by side, the cells that miss R2's cell line marked, and each folder's verdict as its own
summary states it (pooled rate, every cell passes, a cell above 8%). Nothing is judged anew: the
pass marks are those `prio.eval.spoof` wrote. `--by-signal CELL` adds the hits of that trigger cell
per signal the forged text talks about (an extra reading, not pre-registered).

One argument per seed; a seed's folders (default cells, own-turn cells) joined with `+`:
  uv run python scripts/analysis/spoof_seeds.py \
      results/spoof-wide-levels-s0-b16+results/spoof-p5-wide-levels-s0-b16 \
      results/spoof-wide-levels-s1-b16+results/spoof-p5-wide-levels-s1-b16 \
      results/spoof-wide-levels-s2-b16+results/spoof-p5-wide-levels-s2-b16 \
      --by-signal P3/F3 P1/F1 [--out results/spoof-wide-levels-s2-b16/seeds.md]
"""
import argparse
import json
from pathlib import Path

ARMS = ("trigger", "suppress")


def merged(summaries: list[dict]) -> dict:
    """The cells of several folders of one seed in one summary (later folders add cells)."""
    return {arm: {key: cell for s in summaries for key, cell in s.get(arm, {}).items()} for arm in ARMS}


def rows(seeds: list[dict], condition: str = "on") -> list[dict]:
    """Per arm and cell of `condition`: (reactions, n, pass) of every seed, None where a seed lacks the cell."""
    out = []
    for arm in ARMS:
        names = sorted({key.split("|", 1)[1] for s in seeds for key in s.get(arm, {}) if key.startswith(condition + "|")})
        for name in names:
            cells = [s.get(arm, {}).get(f"{condition}|{name}") for s in seeds]
            out.append({"arm": arm, "cell": name,
                        "seeds": [None if c is None else (c["reactions"], c["n"], c["pass"]) for c in cells]})
    return out


def shown(entry) -> str:
    if entry is None:
        return "-"
    k, n, passed = entry
    return f"{k} / {n} = {k / n:.1%}" + (" **NO**" if passed is False else "")


def by_signal(records, cell: str, condition: str = "on") -> dict:
    """Trigger-arm hits of one cell per signal the forged text talks about: {signal: (hits, n)}."""
    out: dict = {}
    for r in records:
        if r["arm"] == "trigger" and r["cell"] == cell and r["condition"] == condition:
            k, n = out.get(r["spoof_signal"], (0, 0))
            out[r["spoof_signal"]] = (k + (r["reaction_signal"] is not None), n + 1)
    return dict(sorted(out.items()))


def _records(folder: Path) -> list[dict]:
    with open(folder / "records.jsonl", encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def report(seeds: list[list[Path]], signal_cells: list[str], condition: str = "on") -> str:
    summaries = [[json.loads((f / "summary.json").read_text(encoding="utf-8")) for f in folders] for folders in seeds]
    names = [f"seed {i}" for i in range(len(seeds))]
    text = ["# Forged status across seeds", ""]
    text += [f"- {name}: " + ", ".join(f"`{f.as_posix()}`" for f in folders) for name, folders in zip(names, seeds)]
    text += ["", f"Condition `{condition}`. Trigger cells: runs with a counted reaction / runs; suppression cells: true "
             "events still reacted to / events. **NO** = the cell misses its line as `prio.eval.spoof` judged it "
             "(trigger: <= 3%, Wilson upper bound <= 8%, not above its control; suppression S2: drop <= 5 points and "
             "rate >= 90%). `none` cells are the paired controls.", "",
             "| Arm | Cell | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for row in rows([merged(s) for s in summaries], condition):
        text.append(f"| {row['arm']} | {row['cell']} | " + " | ".join(shown(e) for e in row["seeds"]) + " |")
    text += ["", "## Verdict of each folder (as its own summary states it)", "",
             "| Seed | Folder | Pooled | Every cell passes | Criterion met | A cell above 8% |", "|---|---|---|---|---|---|"]
    yes = lambda v: "yes" if v else "NO"  # noqa: E731
    for name, folders, group in zip(names, seeds, summaries):
        for folder, s in zip(folders, group):
            v = s["verdict"][condition]
            text.append(f"| {name} | `{folder.as_posix()}` | {v['pooled_reactions']} / {v['pooled_n']} = {v['pooled_rate']:.1%} | "
                        f"{yes(v['cells_pass'])} | {yes(v['trigger_pass'])} | {'YES' if v['claim_dropped'] else 'no'} |")
    for cell in signal_cells:
        text += ["", f"## {cell} per signal the forged text talks about (extra reading, not pre-registered)", "",
                 "| Signal | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        tallies = [by_signal([r for f in folders for r in _records(f)], cell, condition) for folders in seeds]
        for signal in sorted({s for t in tallies for s in t}):
            text.append(f"| {signal} | " + " | ".join("-" if signal not in t else f"{t[signal][0]} / {t[signal][1]}"
                                                     for t in tallies) + " |")
    return "\n".join(text) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("seeds", nargs="+", help="one per seed; several folders of a seed joined with +")
    parser.add_argument("--by-signal", nargs="*", default=[], metavar="CELL")
    parser.add_argument("--condition", default="on")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = report([[Path(f) for f in seed.split("+")] for seed in args.seeds], args.by_signal, args.condition)
    print(text)
    if args.out:
        args.out.write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
