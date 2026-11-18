"""Independent recompute of the phase-1 headline numbers (operating-state experiment).

Reads only results/*/records.jsonl (no project imports, own Wilson interval, CPU, seconds).
Run from the project root:  uv run python scripts/analysis/audit_recompute.py [--decode]
(--decode adds reaction onset and verbatim-bank checks: loads the tokenizer only, no model.)
Sections: (1) full evaluation of runs/n6-wide-levels-s0; (2) quick evaluations of every
configuration; (3) selection-free subset of the full evaluation (episodes outside the quick
subset that drove the choice of the main configuration); (4) paired exact McNemar tests
between configurations on the same quick episodes.
"""
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERRUPT = ("ctx", "think", "energy", "time")
QUICK = ("phase1-wide-levels", "phase1-wide-levels-s1", "phase1-wide-levels-s2",
         "phase1-lanes-levels", "phase1-lanes-mlp", "phase1-sum-mlp")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if k == 0 else max(0.0, c - h)) * 100, (100.0 if k == n else min(1.0, c + h) * 100)


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{lo:.1f}, {hi:.1f}]"


def load(folder: str) -> list[dict]:
    with open(ROOT / "results" / folder / "records.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick(R, cond="on", kind=None, paired=None, signal=None, phase=None, event=None, ids=None, inside=True):
    out = []
    for r in R:
        if r["condition"] != cond or (kind and r["kind"] != kind) or (paired and r["paired"] != paired):
            continue
        if (signal and r["signal"] != signal) or (phase and r["phase"] != phase):
            continue
        if event is not None and r["event_occurred"] != event:
            continue
        if ids is not None and (r["episode_id"] in ids) != inside:
            continue
        out.append(r)
    return out


def hits(rs) -> int:
    return sum(r["reaction_signal"] == r["signal"] for r in rs)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not xs else xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def p90(xs):  # linear interpolation, as numpy.percentile
    xs = sorted(xs)
    if not xs:
        return None
    pos = 0.9 * (len(xs) - 1)
    lo = int(pos)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)


def evaluation(folder: str) -> None:
    R = load(folder)
    print(f"\n===== {folder}: {len(R)} records, {len({r['episode_id'] for r in R})} episodes")
    ev = pick(R, kind="event", paired="event", event=True)
    d = [r["delay"] for r in ev if r["reaction_signal"] == r["signal"] and r["delay"] is not None]
    reacted = [r for r in ev if r["reaction_signal"] is not None]
    print("interrupt:", fmt(hits(ev), len(ev)), "| delay median/p90", median(d), p90(d),
          "| delay histogram", sorted(Counter(d).items()))
    print("  identification (among runs that reacted):", hits(reacted), "/", len(reacted),
          "| wrong signal:", [(r["episode_id"], r["signal"], r["reaction_signal"]) for r in reacted if r["reaction_signal"] != r["signal"]])
    for s in INTERRUPT:
        parts = [f"{ph} {hits(rs)}/{len(rs)}" for ph in ("thinking", "answer")
                 if (rs := pick(R, kind="event", paired="event", signal=s, phase=ph, event=True))]
        rs = pick(R, kind="event", paired="event", signal=s, event=True)
        print(f"  {s}: {fmt(hits(rs), len(rs))} ({', '.join(parts)})")
    g = pick(R, kind="gating_event", paired="event", event=True)
    print("gating:", fmt(hits(g), len(g)))
    for s in ("net", "ram"):
        rs = pick(R, kind="gating_event", paired="event", signal=s, event=True)
        print(f"  {s}: {fmt(hits(rs), len(rs))} | alternative tool first {sum(r['used_alternative'] for r in rs)}"
              f" | gated tool first {sum(r['called_gated_tool'] for r in rs)}")
    sp = pick(R, kind="starts_past", event=True)
    spr = [r for r in sp if r["reaction_signal"] is not None]
    print("starts_past:", fmt(hits(sp), len(sp)), {s: f"{hits(x := [r for r in sp if r['signal'] == s])}/{len(x)}" for s in INTERRUPT},
          "| identification", hits(spr), "/", len(spr))
    sw = pick(R, kind="swap", paired="swapped", event=True)
    print("swap:", fmt(hits(sw), len(sw)), {s: f"{hits(x := [r for r in sw if r['signal'] == s])}/{len(x)}" for s in INTERRUPT})
    for kind in ("event", "gating_event"):
        for paired in ("nominal", "near_miss"):
            rs = pick(R, kind=kind, paired=paired)
            fa = [r for r in rs if r["reaction_signal"] is not None]
            print(f"false alarms {kind}/{paired}:", fmt(len(fa), len(rs)), [(r["episode_id"], r["signal"], r["reaction_signal"]) for r in fa])
    for kind in ("gating_unused", "recovery"):
        rs = pick(R, kind=kind)
        fa = [r for r in rs if r["reaction_signal"] is not None]
        extra = f" | gated tool not called first {sum(not r['called_gated_tool'] for r in rs)}" if kind == "recovery" else ""
        print(f"{kind}:", fmt(len(fa), len(rs)), [(r["episode_id"], r["signal"], r["reaction_signal"]) for r in fa], extra)
    for cond in ("zeroed", "base"):
        for kind in ("event", "gating_event", "starts_past"):
            rs = pick(R, cond=cond, kind=kind, paired="event", event=True)
            if rs:
                print(f"{cond} {kind}:", fmt(hits(rs), len(rs)), "| any reaction", sum(r["reaction_signal"] is not None for r in rs),
                      "| gated tool first", sum(r["called_gated_tool"] for r in rs))
    on_ev = [r for r in R if r["condition"] == "on" and r["event_occurred"] and r["signal"] != "time"]
    miss = [(r["episode_id"], r["kind"], r["signal"], r["reaction_signal"]) for r in on_ev if r["reaction_signal"] != r["signal"]]
    print(f"non-time runs with an event: {len(on_ev)} (without swap {sum(r['kind'] != 'swap' for r in on_ev)}); "
          f"misses {len(miss)}: {miss[:12]}{' ...' if len(miss) > 12 else ''}")


def selection_free() -> None:
    quick_ids = {r["episode_id"] for r in load("phase1-wide-levels")}
    R = load("phase1")
    print("\n===== selection-free subset of the full evaluation (episodes outside the quick subset)")
    for label, inside in (("quick subset", True), ("rest", False)):
        ev = pick(R, kind="event", paired="event", event=True, ids=quick_ids, inside=inside)
        t = [r for r in ev if r["signal"] == "time"]
        g = pick(R, kind="gating_event", paired="event", event=True, ids=quick_ids, inside=inside)
        sp = pick(R, kind="starts_past", event=True, ids=quick_ids, inside=inside)
        q = pick(R, kind="event", paired="nominal", ids=quick_ids, inside=inside)
        print(f"{label}: interrupt {fmt(hits(ev), len(ev))} | time {fmt(hits(t), len(t))} | gating {fmt(hits(g), len(g))}"
              f" | starts_past {fmt(hits(sp), len(sp))} | quiet-twin alarms {sum(r['reaction_signal'] is not None for r in q)}/{len(q)}")


def mcnemar() -> None:
    def outcome(folder):
        return {(r["episode_id"], r["paired"], r["kind"]): r["reaction_signal"] == r["signal"]
                for r in load(folder) if r["condition"] == "on" and r["event_occurred"]
                and r["kind"] in ("event", "gating_event", "starts_past", "swap")}

    print("\n===== paired exact McNemar, wide + levels seed 0 vs other quick evaluations (same episodes)")
    a = outcome("phase1-wide-levels")
    for other in QUICK[1:]:
        b = outcome(other)
        for label, kinds in (("interrupt", ("event",)), ("all event kinds", ("event", "gating_event", "starts_past", "swap"))):
            keys = [k for k in a if k[2] in kinds and k in b]
            x = sum(a[k] and not b[k] for k in keys)
            y = sum(b[k] and not a[k] for k in keys)
            n = x + y
            p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
            print(f"{other} {label}: n {len(keys)}, wide-only {x}, other-only {y}, p = {p:.3f}")


def decode_checks() -> None:
    """Reaction onset and verbatim bank sentences (needs the tokenizer and prio.datagen.teacher)."""
    from transformers import AutoTokenizer

    from prio.datagen.teacher import REACTION_BANK, reaction_sentences

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    bank = {s.strip() for k in REACTION_BANK for s in reaction_sentences(k)}
    recs = {(r["episode_id"], r["condition"], r["paired"]): r for r in load("phase1")}
    onset, verbatim, used, n = Counter(), 0, Counter(), 0
    with open(ROOT / "results" / "phase1" / "generations.jsonl", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            r = recs.get((g["episode_id"], g["condition"], g["paired"]))
            if not r or r["condition"] != "on" or not r["event_occurred"] or r["reaction_signal"] != r["signal"]:
                continue
            ids = g["token_ids"]
            if r["kind"] in ("event", "gating_event"):
                onset[next(i for i, t in enumerate(ids) if tok.decode([t]).strip())] += 1
            first = tok.decode(ids[:80]).lstrip().split("\n")[0].strip()
            match = next((b for b in bank if first.startswith(b)), None)
            n += 1
            if match:
                verbatim += 1
                used[match] += 1
    print("\n===== decode checks (results/phase1)")
    print("onset index of the first non-whitespace token, interrupt + gating hits:", sorted(onset.items()))
    print(f"first line = verbatim bank sentence: {verbatim} / {n}; distinct bank sentences used: {len(used)} of {len(bank)}")


if __name__ == "__main__":
    import sys

    evaluation("phase1")
    for folder in QUICK:
        evaluation(folder)
    selection_free()
    mcnemar()
    print("\nthroughput: latency increase = 20.358 / 14.670 - 1 =", f"{20.357642078873898 / 14.670139614879817 - 1:.1%}",
          "| throughput drop =", f"{1 - 14.670139614879817 / 20.357642078873898:.1%}")
    if "--decode" in sys.argv:
        decode_checks()
