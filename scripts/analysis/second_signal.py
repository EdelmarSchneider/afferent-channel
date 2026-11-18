"""R22 (the game character), a reading made after the evaluation (not part of any line of amendment A8): when
the table makes a command depend on a second signal, does the command follow that signal, and did
the channel learn to tell the signal's two values apart? CPU, seconds.

1. Commands. In a state-table evaluation (`prio.npc.evaluate`, `prio.msg.evaluate`) two siblings of
   a family share the text up to the cut and the level of the felt signal and differ only in the
   second signal (water near or not). Per pair of variants `<with>:<without>:<command with>:<command
   without>`: the first commands of each variant, and the families sorted by the commands of their
   two siblings (follows the signal / the same command both times / the wrong way round), with an
   exact McNemar test on the families that take the command `with` in one sibling only. Channel
   on, main cell.
2. Level vectors. Per `<run>:<signal>:<level a>:<level b>`: distance and cosine between the two
   vectors of the `levels` projector, their norms, and how far each one moved from the table the
   run started with (rebuilt from the run's seed, as `level_geometry.py` does; a level that training
   never visits must still equal its start: checked on the levels named in `--unvisited`).

Usage (project root):
  uv run python scripts/analysis/second_signal.py results/r22-npc-wide-levels-s0-b16 \
      --pairs thirst-water:thirst-dry:drink_water:go_to hunger-food:hunger-none:eat:go_to \
      --levels runs/npc6-wide-levels-s0:water_near:0:99 runs/npc6-wide-levels-s0:food_near:0:99 \
               runs/n7msg-wide-levels-s0:net:0:99 runs/n7msg-wide-levels-s0:cell:0:99 \
      --unvisited runs/npc6-wide-levels-s0:water_near:50 --out second_signal.md
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import level_geometry as lg  # noqa: E402

from prio.eval.spoof import mcnemar_exact  # noqa: E402
from prio.release import load_channel_tensors  # noqa: E402


def _main_cell(records):
    return [r for r in records if r["condition"] == "on" and r["cell"] == "main"]


def _first(record) -> str:
    return record["commands"][0] if record["commands"] else "-"


def first_commands(records, variants) -> dict[str, dict[str, int]]:
    """Per variant: how often each command is the first one of the response ("-": no command)."""
    out = {v: collections.Counter() for v in variants}
    for r in _main_cell(records):
        if r["variant"] in out:
            out[r["variant"]][_first(r)] += 1
    return {v: dict(c) for v, c in out.items()}


def _siblings(records, with_it: str, without: str) -> list[tuple[str, str]]:
    """(first command with the resource, first command without it) of every family that has both siblings."""
    first = collections.defaultdict(dict)
    for r in _main_cell(records):
        if r["variant"] in (with_it, without):
            first[r["family"]][r["variant"]] = _first(r)
    return [(got[with_it], got[without]) for got in first.values() if len(got) == 2]


def paired(records, with_it: str, without: str, command_with: str, command_without: str) -> dict[str, int]:
    """Families that have both siblings, sorted by their two first commands: `follows` (the command
    of each sibling is its own), `both_with` / `both_without` (one command whatever the signal
    says), `inverted` (each takes the other's), `other` (some other command, or none)."""
    names = {(command_with, command_without): "follows", (command_with, command_with): "both_with",
             (command_without, command_without): "both_without", (command_without, command_with): "inverted"}
    out = {"families": 0, "follows": 0, "both_with": 0, "both_without": 0, "inverted": 0, "other": 0}
    for got in _siblings(records, with_it, without):
        out["families"] += 1
        out[names.get(got, "other")] += 1
    return out


def discordant(records, with_it: str, without: str, command_with: str) -> tuple[int, int]:
    """Families where only one sibling takes `command_with` first: (only the sibling with the
    resource, only the sibling without it). The input of the exact McNemar test."""
    got = _siblings(records, with_it, without)
    return (sum(a == command_with != b for a, b in got), sum(b == command_with != a for a, b in got))


def separation(table: np.ndarray, start: np.ndarray, signal: int, a: int, b: int) -> dict:
    """Two level vectors of one signal ([signals, levels, width] tables): distance, cosine, norms,
    the distance of the same two vectors in the starting table, and each one's drift from its start."""
    va, vb = table[signal, a].astype(np.float64), table[signal, b].astype(np.float64)
    sa, sb = start[signal, a].astype(np.float64), start[signal, b].astype(np.float64)
    norm = np.linalg.norm
    return {"distance": float(norm(va - vb)), "cosine": float(va @ vb / (norm(va) * norm(vb))),
            "norms": [float(norm(va)), float(norm(vb))], "start_distance": float(norm(sa - sb)),
            "drift": [float(norm(va - sa)), float(norm(vb - sb))]}


def tables(run: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(trained table, starting table, signal names) of a run with the `levels` projector."""
    log = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
    args = log["args"]
    names = json.loads((Path(args["data"].replace("\\", "/")) / "manifest.json").read_text(encoding="utf-8"))["signals"]
    state = load_channel_tensors(run)  # channel.pt or channel.safetensors
    trained = lg.level_vectors(state, len(names))
    start = lg.init_table(args["seed"], len(names), trained.shape[2]).T.reshape(trained.shape)
    return trained, start, names


def report(folder: Path, pairs, levels, unvisited) -> str:
    text = [f"# Second signal: commands and level vectors ({folder.as_posix()})", "",
            "A reading made after the evaluation; no line of amendment A8 rests on it.", ""]
    if pairs:
        with open(folder / "records.jsonl", encoding="utf-8") as lines:
            records = [json.loads(line) for line in lines if line.strip()]
        text += ["## Commands (channel on, main cell)", "",
                 "| Variant | n | First commands |", "|---|---|---|"]
        for pair in pairs:
            for variant, counts in first_commands(records, pair[:2]).items():
                listed = ", ".join(f"`{c}` {k}" for c, k in sorted(counts.items(), key=lambda item: -item[1]))
                text.append(f"| {variant} | {sum(counts.values())} | {listed} |")
        text += ["", "Families by the first commands of their two siblings (same text up to the cut, same level of the "
                     "felt signal; only the second signal differs):", "",
                 "| Pair | Families | Follows the signal | Command `with` both times | Command `without` both times | "
                 "Wrong way round | Other | McNemar p |", "|---|---|---|---|---|---|---|---|"]
        for with_it, without, c_with, c_without in pairs:
            p = paired(records, with_it, without, c_with, c_without)
            only_with, only_without = discordant(records, with_it, without, c_with)
            text.append(f"| {with_it} / {without} (`{c_with}` / `{c_without}`) | {p['families']} | {p['follows']} "
                        f"({p['follows'] / p['families']:.1%}) | {p['both_with']} | {p['both_without']} | {p['inverted']} | "
                        f"{p['other']} | {mcnemar_exact(only_with, only_without):.3g} ({only_with} / {only_without}) |")
        text += ["", "McNemar (exact, two-sided): families where only the sibling with the resource takes the command "
                     "`with` first / families where only the sibling without it does.", ""]
    if levels:
        text += ["## Level vectors of the `levels` projector", "",
                 "| Run | Signal | Levels | Distance | Cosine | Norms | Distance at the start | Drift from the start |",
                 "|---|---|---|---|---|---|---|---|"]
        cache = {}
        for item in levels:
            run, name, a, b = item.split(":")
            trained, start, names = cache.setdefault(run, tables(Path(run)))
            s = separation(trained, start, names.index(name), int(a), int(b))
            text.append(f"| {run} | {name} | {a} / {b} | {s['distance']:.3f} | {s['cosine']:+.3f} | "
                        f"{s['norms'][0]:.3f} / {s['norms'][1]:.3f} | {s['start_distance']:.3f} | "
                        f"{s['drift'][0]:.3f} / {s['drift'][1]:.3f} |")
        text.append("")
        for item in unvisited:
            run, name, at = item.split(":")
            trained, start, names = cache.setdefault(run, tables(Path(run)))
            moved = float(np.abs(trained[names.index(name), int(at)] - start[names.index(name), int(at)]).max())
            text.append(f"- Check of the rebuilt starting table: {run}, {name} level {at} (never visited in training) "
                        f"differs from its start by at most {moved:.2g}.")
        text.append("")
    return "\n".join(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", type=Path, help="a state-table evaluation folder (records.jsonl)")
    parser.add_argument("--pairs", nargs="*", default=[], help="<variant with>:<variant without>:<command with>:<command without>")
    parser.add_argument("--levels", nargs="*", default=[], help="<run>:<signal>:<level a>:<level b>")
    parser.add_argument("--unvisited", nargs="*", default=[], help="<run>:<signal>:<level> never visited in training")
    parser.add_argument("--out", help="file name inside the results folder")
    args = parser.parse_args(argv)
    text = report(args.results, [p.split(":") for p in args.pairs], args.levels, args.unvisited)
    print(text)
    if args.out:
        (args.results / args.out).write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
