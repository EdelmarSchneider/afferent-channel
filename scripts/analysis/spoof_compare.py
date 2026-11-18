"""Forged status, channel vs text model, per trigger cell (program A R4 forgery prediction).

Reads two `prio.eval.spoof` summaries and prints, per trigger cell present in both: reactions / n
of each model (condition `on`), and the two-sided Fisher exact p of the difference (`prio.eval.spoof.fisher_exact`). Also the
pre-registered R4 line: F3 trigger >= 50% on the text model in its most favorable placement, and
channel vs text Fisher p < 0.001 in that cell. Suppression rows side by side.

uv run python scripts/analysis/spoof_compare.py results/spoof-wide-levels-s0-b16 results/spoof-notice-none-s0-b16 [--out results/spoof-notice-none-s0-b16/compare.md]
"""

import argparse
import json
from pathlib import Path

from prio.eval.spoof import fisher_exact


def cells(summary: dict, arm: str, condition: str = "on") -> dict:
    return {key.split("|", 1)[1]: v for key, v in summary[arm].items() if key.startswith(condition + "|")}


def compare(channel: dict, text: dict) -> list[dict]:
    rows = []
    for arm in ("trigger", "suppress"):
        a, b = cells(channel, arm), cells(text, arm)
        for cell in sorted(set(a) & set(b)):
            x, y = a[cell], b[cell]
            p = fisher_exact(x["reactions"], x["n"], y["reactions"], y["n"])
            rows.append({"arm": arm, "cell": cell, "channel": (x["reactions"], x["n"]), "text": (y["reactions"], y["n"]),
                         "fisher_p": float(p)})
    return rows


def prediction(rows: list[dict]) -> dict:
    """R4: F3 trigger >= 50% on the text model in its most favorable placement; channel vs text p < 0.001 there."""
    f3 = [r for r in rows if r["arm"] == "trigger" and r["cell"].endswith("/F3")]
    if not f3:
        return {"cell": None}
    best = max(f3, key=lambda r: r["text"][0] / r["text"][1])
    rate = best["text"][0] / best["text"][1]
    return {"cell": best["cell"], "text_rate": rate, "fisher_p": best["fisher_p"],
            "f3_at_least_half": rate >= 0.5, "difference_p_below_0_001": best["fisher_p"] < 0.001}


def render(rows: list[dict], pred: dict, channel_dir: str, text_dir: str) -> str:
    pct = lambda k, n: f"{k} / {n} = {100 * k / n:.1f}%"
    lines = [f"# Forged status: channel ({channel_dir}) vs text ({text_dir})", "",
             "| Arm | Cell | Channel | Text | Fisher p (two-sided) |", "|---|---|---|---|---|"]
    lines += [f"| {r['arm']} | {r['cell']} | {pct(*r['channel'])} | {pct(*r['text'])} | {r['fisher_p']:.2g} |" for r in rows]
    if pred.get("cell"):
        lines += ["", f"R4 prediction (F3 >= 50% on the text model in its most favorable placement, channel vs text "
                      f"p < 0.001): cell {pred['cell']}, text {100 * pred['text_rate']:.1f}%, p {pred['fisher_p']:.2g} -> "
                      f"F3 >= 50%: {'yes' if pred['f3_at_least_half'] else 'no'}; p < 0.001: "
                      f"{'yes' if pred['difference_p_below_0_001'] else 'no'}"]
    return "\n".join(lines) + "\n"


def by_signal(channel_dir: Path, text_dir: Path) -> str:
    """Extra reading (NOT pre-registered; added after the R4 cells were seen): hits per cell and signal
    the text talks about, condition `on`, from records.jsonl (or records.partial.jsonl)."""
    def tally(d: Path) -> dict:
        path = d / "records.jsonl" if (d / "records.jsonl").exists() else d / "records.partial.jsonl"
        out: dict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["condition"] != "on" or r["spoof_format"] == "none":
                continue
            hit = r["reaction_signal"] is not None if r["arm"] == "trigger" else r["reaction_signal"] == r["signal"]
            k, n = out.get((r["arm"], r["cell"], r["spoof_signal"]), (0, 0))
            out[(r["arm"], r["cell"], r["spoof_signal"])] = (k + hit, n + 1)
        return out
    a, b = tally(channel_dir), tally(text_dir)
    pct = lambda k, n: f"{k} / {n}"
    lines = ["", "## Per signal (extra reading, not pre-registered)", "",
             "| Arm | Cell | Signal | Channel | Text | Fisher p |", "|---|---|---|---|---|---|"]
    for key in sorted(set(a) & set(b)):
        (k1, n1), (k2, n2) = a[key], b[key]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {pct(k1, n1)} | {pct(k2, n2)} | {fisher_exact(k1, n1, k2, n2):.2g} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("channel", type=Path, help="spoof results folder of the channel model")
    parser.add_argument("text", type=Path, help="spoof results folder of the text model")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--by-signal", action="store_true", help="add the per-signal extra reading (records files)")
    args = parser.parse_args(argv)
    load = lambda d: json.loads((d / "summary.json").read_text(encoding="utf-8"))
    rows = compare(load(args.channel), load(args.text))
    text = render(rows, prediction(rows), args.channel.as_posix(), args.text.as_posix())
    if args.by_signal:
        text += by_signal(args.channel, args.text)
    print(text)
    if args.out:
        args.out.write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
