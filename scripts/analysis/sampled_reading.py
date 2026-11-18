"""R24b (preregistration amendment A13): the draws of one model decoded with sampling, pooled.

Reads the folders written by `sampled_eval.py` for one model and prints, against A13:
- S1 (prediction for the varied model, low confidence: <= 70%): the share of counted reactions, pooled
  over the draws, whose naming sentence is word for word a naming sentence of the training file;
  distinct naming sentences per draw and pooled, per signal; each signal's most used sentence;
- S2 (prediction): in every draw the limit-crossing reaction is >= 90% and the false alarms on quiet
  twins are <= 5%, and the pooled reaction is not more than 5 points under the model's greedy rate on
  the same episodes; near-miss alarms beside;
- S3 (no prediction): thinking budget, a summary of at least 8 tokens among the counted reactions,
  pooled and by kind of episode.

Counted reactions and naming sentences are read by `wording_reading.reading` (same rules as R24).
CPU: the tokenizer only.

uv run python scripts/analysis/sampled_reading.py --name varied --data data/n6v \
    --greedy results/r24-varied-quick-b16 --draws results/r24b-varied-t06-s0-b16 \
    results/r24b-varied-t06-s1-b16 results/r24b-varied-t06-s2-b16 [--out file.md]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SIGNALS = ("ctx", "think", "energy", "time", "net", "ram")
MIN_SUMMARY = 8  # tokens (A9, V3)
REACTION_LINE, ALARM_LINE, DROP_LINE = 0.90, 0.05, 5.0  # A13, S2
GROUP = "on|all interrupt|all"


def pooled_wording(tables: list[dict]) -> dict:
    """Per signal and "all": n, seen (verbatim training sentences), distinct per draw and pooled, top (sentence, count)."""

    def row(parts: list[list[dict]]) -> dict:
        everything = [i for part in parts for i in part]
        counts = Counter((i["signal"], i["naming"]) for i in everything)
        (_, sentence), k = counts.most_common(1)[0] if counts else ((None, None), 0)
        return {"n": len(everything), "seen": sum(bool(i["naming_seen"]) for i in everything),
                "distinct_per_draw": [len({(i["signal"], i["naming"]) for i in part}) for part in parts],
                "distinct_pooled": len(counts), "top": (sentence, k)}

    draws = [t["items"] for t in tables]
    names = [s for s in SIGNALS if any(i["signal"] == s for d in draws for i in d)]
    table = {s: row([[i for i in d if i["signal"] == s] for d in draws]) for s in names}
    table["all"] = row(draws)
    return table


def _counts(summary: dict) -> dict:
    g = summary["groups"][GROUP]
    return {"n": g["n"], "hits": round(g["reaction_rate"] * g["n"]),
            "alarms": round((g["false_alarm_rate"] or 0.0) * g["n_nominal"]), "n_quiet": g["n_nominal"],
            "near": round((g["near_miss_alarm_rate"] or 0.0) * g["n_near_miss"]), "n_near": g["n_near_miss"]}


def detection(summaries: list[dict], greedy: dict) -> dict:
    """Each draw, the pool, the greedy reference and the verdict of S2."""
    draws = [_counts(s) for s in summaries]
    pooled = {k: sum(d[k] for d in draws) for k in draws[0]}
    ref = _counts(greedy)
    under = 100 * (ref["hits"] / ref["n"] - pooled["hits"] / pooled["n"])
    reaction_ok = all(d["hits"] / d["n"] >= REACTION_LINE for d in draws)
    alarms_ok = all(d["alarms"] / d["n_quiet"] <= ALARM_LINE for d in draws)
    return {"draws": draws, "pooled": pooled, "greedy": ref, "every_draw_reaction_ok": reaction_ok,
            "every_draw_alarms_ok": alarms_ok, "points_under_greedy": under,
            "held": reaction_ok and alarms_ok and under <= DROP_LINE}


def summary_presence(tables: list[dict]) -> dict:
    """Thinking-budget reactions with a summary of at least MIN_SUMMARY tokens: (with, n) for "all" and each kind."""
    think = [i for t in tables for i in t["items"] if i["signal"] == "think"]
    out = {"all": (sum(i["summary_tokens"] >= MIN_SUMMARY for i in think), len(think))}
    for kind in sorted({i["kind"] for i in think}):
        part = [i for i in think if i["kind"] == kind]
        out[kind] = (sum(i["summary_tokens"] >= MIN_SUMMARY for i in part), len(part))
    return out


def _share(k: int, n: int) -> str:
    return f"{k} / {n} = {100 * k / n:.1f}%" if n else "-"


def report(name: str, folders: list[str], wording: dict, det: dict, presence: dict, verbatim_line: float | None) -> str:
    everything = wording["all"]
    lines = [f"# Decoded with sampling: the {name} model, {len(folders)} draws (R24b, amendment A13)", "",
             "Draws: " + ", ".join(f"`{f}`" for f in folders) + ". Predictions written before the runs; no line.", "",
             "## S1. Wording", ""]
    verdict = ""
    if verbatim_line is not None:
        held = everything["seen"] / everything["n"] <= verbatim_line if everything["n"] else None
        verdict = f" (prediction <= {100 * verbatim_line:.0f}%: {'held' if held else 'failed'})"
    lines += [f"Counted reactions whose naming sentence is word for word a training sentence, pooled: "
              f"{_share(everything['seen'], everything['n'])}{verdict}.", "",
              "| Signal | Counted reactions | Verbatim training sentence | Distinct per draw | Distinct pooled | Most used sentence |",
              "|---|---|---|---|---|---|"]
    for signal in (*[s for s in SIGNALS if s in wording], "all"):
        row = wording[signal]
        top = f"\"{row['top'][0]}\" ({_share(row['top'][1], row['n'])})" if row["n"] else "-"
        lines.append(f"| {signal} | {row['n']} | {_share(row['seen'], row['n'])} | "
                     f"{', '.join(map(str, row['distinct_per_draw']))} | {row['distinct_pooled']} | {top} |")
    lines += ["", "## S2. Detection under sampling", "",
              "| Run | Limit-crossing reaction | False alarms (quiet twins) | Near-miss alarms |", "|---|---|---|---|"]
    for k, d in enumerate(det["draws"]):
        lines.append(f"| draw {k} | {_share(d['hits'], d['n'])} | {_share(d['alarms'], d['n_quiet'])} | {_share(d['near'], d['n_near'])} |")
    for label, d in (("pooled", det["pooled"]), ("greedy (reference)", det["greedy"])):
        lines.append(f"| {label} | {_share(d['hits'], d['n'])} | {_share(d['alarms'], d['n_quiet'])} | {_share(d['near'], d['n_near'])} |")
    lines += ["", f"S2, prediction (every draw: reaction >= 90% and false alarms <= 5%; pooled reaction not more than 5 points "
                  f"under greedy): reaction in every draw {'ok' if det['every_draw_reaction_ok'] else 'NOT ok'}, false alarms in "
                  f"every draw {'ok' if det['every_draw_alarms_ok'] else 'NOT ok'}, pooled {det['points_under_greedy']:.1f} "
                  f"points under greedy: **{'held' if det['held'] else 'failed'}**."]
    if presence.get("all", (0, 0))[1]:
        lines += ["", "## S3. Thinking budget: a summary before the thinking closes (no prediction)", "",
                  "; ".join(f"{kind}: {_share(k, n)}" for kind, (k, n) in presence.items()) + "."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import wording_reading as W
    from prio.datagen.episodes import read_jsonl
    from prio.datagen.variants import manifest_model

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="how the report names the model (varied, main)")
    parser.add_argument("--data", type=Path, required=True, help="the model's training data folder")
    parser.add_argument("--greedy", type=Path, required=True, help="the model's greedy quick evaluation")
    parser.add_argument("--draws", type=Path, nargs="+", required=True)
    parser.add_argument("--verbatim-line", type=float, default=None, help="S1's prediction (0.70 for the varied model)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    model = manifest_model(json.loads((args.data / "manifest.json").read_text(encoding="utf-8")))
    tok, ms = W._load_tokenizer(model)
    train = read_jsonl(args.data / "train.jsonl")
    load = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    tables = [W.reading(load(d / "records.jsonl"), load(d / "generations.jsonl"), train, tok, ms) for d in args.draws]
    summaries = [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in args.draws]
    greedy = json.loads((args.greedy / "summary.json").read_text(encoding="utf-8"))
    wording, det, presence = pooled_wording(tables), detection(summaries, greedy), summary_presence(tables)
    text = report(args.name, [d.as_posix() for d in args.draws], wording, det, presence, args.verbatim_line)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes((text + "\n").encode("utf-8"))
        args.out.with_suffix(".json").write_bytes((json.dumps(
            {"wording": wording, "detection": det, "summary_presence": presence}, indent=1) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
