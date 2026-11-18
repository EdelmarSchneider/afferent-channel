"""The three pre-registered lines of R12 and the report (AP 7, 8, 9).

    uv run python -m prio.r12.lines --set pilot|confirm   -> results/r12p|r12/r12_lines.md + r12_lines.json

Each line is computed and judged per model (channel model, text-notice model), never pooled. A
line is compared as an exact fraction (254 / 299 does not meet 85%) and judged on its point
estimate; the 95% Wilson interval is printed beside it, with whether its bound also clears the
line. Under the minimum counts of AP 9 a line is "not evaluable": never met.

| Line | Numerator | Denominator | Met when |
| L1a | records with `reaction_signal == signal` | arm paired, family code, kind `event`, pairing `event`, condition `on`, `event_occurred` | >= 85% |
| L1b | records with any counted reaction | the same episodes' quiet twins (`nominal`; text-notice model: `quiet`) | <= 5% |
| L1c | (rate of L1a on math + explain) minus (rate of L1a) | same run, same conditions | <= 10 points |
| L2a | hits | arm free, event runs whose crossing occurs | >= 90% |
| L2b | hits with onset index 0 or 1 | hits of L2a | >= 95% |
| L3 | hits | arm live, event runs whose crossing occurs, the three counters pooled | >= 85% |

Arm Ls is reported beside arm live and never enters line 3. Everything else in the report is
labeled "not a line". In the arms without a forced prefix the false alarms of twin and off runs
and the premature reactions are read at the anchors of the second note on amendment A12
(`prio.r12.stream`: the first generated token, the crossing position, the close of the thinking);
the report prints the union and the count from the first token alone, a keyword-only count of
deadlines named at the end of a turn, and a second onset reading beside L2b. Pilot numbers are never reported as results. The confirmatory result of an
(arm, model) is its first run in the marker file; reruns are printed beside it, labeled.
"""

import argparse
import ast
import json
import math
import subprocess
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np

from prio.eval.metrics import RunRecord, summarize, wilson_ci
from prio.eval.spoof import fisher_exact, mcnemar_exact
from prio.r12 import (ARMS, BATCH_SIZE, FREE_K, FREEZE_COMMIT, LIVE, LIVE_K, MIN_COUNTS, MODEL_ID, PREFIX, RUNS, SET_CODE,
                      STREAM_ARMS, ledger, live, stream, utf8_console)
from prio.r12.run import out_folder
from prio.r12.stream import LIVE_ARMS, WHY_NOT, StreamRecord
from prio.reactions import _NAMING_SPAN, names_signal, parse_tool_calls
from prio.signals import signal_set

THRESHOLD = {"L1a": Fraction(85, 100), "L1b": Fraction(5, 100), "L1c": Fraction(10, 100), "L2a": Fraction(90, 100),
             "L2b": Fraction(95, 100), "L3": Fraction(85, 100)}
WORDS = {"L1a": "new family reaction >= 85%", "L1b": "false alarms <= 5%", "L1c": "within 10 points of seen families",
         "L2a": "free-run reaction >= 90% of crossings that occur", "L2b": "onset index <= 1 in >= 95% of hits",
         "L3": "live counters >= 85%"}
MAIN_KINDS = ("event", "gating_event")
SEEN = ("math", "explain")  # the families with the same episode kind as the new one
NEW = "code"
CLAIM_P = 0.01  # no claim of a difference in detection between the models unless p is under it (R4's rule)
SCORING_FILES = ("src/prio/reactions.py", "src/prio/eval/metrics.py")
# The full evaluations of the two models on data/n6: the seen families' known rates, printed beside
# their rates on the fresh tasks (not a line; L1c's reference is the fresh tasks of the same run).
REFERENCE = {"channel": "results/phase1", "notice": "results/b3-notice-none-full-b16"}
_NO_MINIMUM = {name: 0 for name in MIN_COUNTS}


# --- verdicts and intervals ------------------------------------------------------------------------

def verdict(k: int, n: int, *, at_least: Fraction | None = None, at_most: Fraction | None = None, minimum: int = 0) -> str:
    """ "met" | "not met" | "not evaluable" for k of n against a line, as exact fractions."""
    if n <= 0 or n < minimum:
        return "not evaluable"
    rate = Fraction(k, n)
    if (at_least is not None and rate < at_least) or (at_most is not None and rate > at_most):
        return "not met"
    return "met"


def _wilson(k: int, n: int) -> list[float] | None:
    return list(wilson_ci([True] * k + [False] * (n - k))) if n else None


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """95% interval of the difference k1 / n1 - k2 / n2 of two independent proportions (Newcombe
    1998, method 10: the square-and-add of the two Wilson intervals)."""
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = _wilson(k1, n1), _wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def _rate(k: int, n: int, *, at_least: Fraction | None = None, at_most: Fraction | None = None, minimum: int = 0) -> dict:
    ci = _wilson(k, n)
    clears = None
    if ci is not None and (at_least is not None or at_most is not None):
        clears = bool((at_least is None or ci[0] >= float(at_least)) and (at_most is None or ci[1] <= float(at_most)))
    out = {"k": k, "n": n, "rate": k / n if n else None, "ci": ci}
    if at_least is not None or at_most is not None:
        out.update(verdict=verdict(k, n, at_least=at_least, at_most=at_most, minimum=minimum), bound_clears=clears, minimum=minimum)
    return out


def _combine(evaluable: bool, parts: Sequence[str]) -> str:
    """A line from its parts: under the line's own minimum counts it is not evaluable; a part that
    is not met decides; a part that cannot be evaluated keeps the line from being met."""
    if not evaluable:
        return "not evaluable"
    if "not met" in parts:
        return "not met"
    return "not evaluable" if "not evaluable" in parts else "met"


# --- the lines -------------------------------------------------------------------------------------

def line1(records: Sequence[RunRecord], family_of: dict[str, str], minimums: dict = MIN_COUNTS) -> dict:
    """L1a, L1b, L1c on the records of one paired evaluation (`family_of`: episode id -> family)."""
    quiet = "quiet" if any(r.paired == "quiet" for r in records) else "nominal"
    on = [r for r in records if r.condition == "on" and r.kind == "event"]

    def events(families) -> list[RunRecord]:
        return [r for r in on if r.paired == "event" and r.event_occurred and family_of.get(r.episode_id) in families]

    new, seen = events((NEW,)), events(SEEN)
    twins = [r for r in on if r.paired == quiet and family_of.get(r.episode_id) == NEW]
    k_new, k_seen = sum(r.reaction_signal == r.signal for r in new), sum(r.reaction_signal == r.signal for r in seen)
    a = _rate(k_new, len(new), at_least=THRESHOLD["L1a"], minimum=minimums["L1"])
    b = _rate(sum(r.reaction_signal is not None for r in twins), len(twins), at_most=THRESHOLD["L1b"], minimum=minimums["L1"])
    c = {"seen": _rate(k_seen, len(seen)), "new": {"k": k_new, "n": len(new)}, "gap_points": None, "gap_ci_points": None,
         "verdict": "not evaluable", "bound_clears": None, "minimum": minimums["L1c"]}
    if new and seen:
        gap = Fraction(k_seen, len(seen)) - Fraction(k_new, len(new))
        low, high = newcombe(k_seen, len(seen), k_new, len(new))
        c.update(gap_points=float(gap * 100), gap_ci_points=[100 * low, 100 * high], bound_clears=bool(high <= float(THRESHOLD["L1c"])))
        if len(seen) >= minimums["L1c"] and len(new) >= minimums["L1"]:
            c["verdict"] = "met" if gap <= THRESHOLD["L1c"] else "not met"
    evaluable = bool(new and twins and len(new) >= minimums["L1"] and len(twins) >= minimums["L1"])
    return {"L1a": a, "L1b": b, "L1c": c, "quiet_pairing": quiet,
            "verdict": _combine(evaluable, [a["verdict"], b["verdict"], c["verdict"]])}


def _events(records: Sequence[StreamRecord], arm: str) -> list[StreamRecord]:
    return [r for r in records if r.arm == arm and r.run_type == "event"]


def _not_occurred(events: Sequence[StreamRecord]) -> dict[str, int]:
    return {why: sum(1 for r in events if not r.occurred and r.why_not == why) for why in WHY_NOT}


def line2(records: Sequence[StreamRecord], minimums: dict = MIN_COUNTS) -> dict:
    """L2a and L2b on the records of arm free."""
    events = _events(records, "free")
    occurred = [r for r in events if r.occurred]
    hits = [r for r in occurred if r.hit]
    a = _rate(len(hits), len(occurred), at_least=THRESHOLD["L2a"], minimum=minimums["L2a"])
    b = _rate(sum(1 for r in hits if r.onset is not None and r.onset <= 1), len(hits), at_least=THRESHOLD["L2b"],
              minimum=minimums["L2b"])
    # Beside L2b, not a line: the same share with the index taken at the last keyword before the reaction's
    # marker (a keyword in ordinary text before the reaction can only lower the index of L2b).
    last = _rate(sum(1 for r in hits if r.onset_last is not None and r.onset_last <= 1), len(hits))
    return {"planned": len(events), "not_occurred": _not_occurred(events), "L2a": a, "L2b": b, "onset_last": last,
            "verdict": _combine(a["verdict"] != "not evaluable", [a["verdict"], b["verdict"]])}


def _counters(events: Sequence[StreamRecord]) -> dict[str, dict]:
    out = {}
    for signal in LIVE:
        rows = [r for r in events if r.signal == signal and r.occurred]
        if any(r.signal == signal for r in events):
            out[signal] = _rate(sum(r.hit for r in rows), len(rows))
    return out


def line3(records: Sequence[StreamRecord], minimums: dict = MIN_COUNTS) -> dict:
    """L3 on the records of arm live: the three counters pooled; each counter beside."""
    events = _events(records, "live")
    occurred = [r for r in events if r.occurred]
    by_counter = _counters(events)
    enough = bool(occurred) and len(occurred) >= minimums["L3"] and all(
        by_counter.get(signal, {"n": 0})["n"] >= max(minimums["L3_counter"], 1) for signal in LIVE)
    pooled = _rate(sum(r.hit for r in occurred), len(occurred), at_least=THRESHOLD["L3"], minimum=minimums["L3"])
    if not enough:
        pooled["verdict"] = "not evaluable"
    return {"planned": len(events), "not_occurred": _not_occurred(events), "L3": pooled, "by_counter": by_counter,
            "minimum_per_counter": minimums["L3_counter"], "verdict": pooled["verdict"]}


def no_line(records: Sequence[StreamRecord], arm: str) -> dict:
    """The rates of an arm that has no line (arm Ls), in the shape of line 3."""
    events = _events(records, arm)
    occurred = [r for r in events if r.occurred]
    return {"arm": arm, "verdict": "no line", "planned": len(events), "not_occurred": _not_occurred(events),
            "pooled": _rate(sum(r.hit for r in occurred), len(occurred)), "by_counter": _counters(events)}


# --- tables that are not lines -----------------------------------------------------------------------

def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return [_plain(v) for v in value] if isinstance(value, (tuple, list)) else value


def _table(rows: Sequence[RunRecord], labels: Sequence[str]) -> dict:
    return {key[1]: _plain(v) for key, v in summarize(list(rows), list(labels), ["all"] * len(rows)).items()}


def paired_tables(records: Sequence[RunRecord], family_of: dict[str, str]) -> dict:
    """Arm paired, not a line: reaction, false alarms, near-miss alarms, identification, unnamed
    calls, wrongful calls and refusals, delay, by family, signal, phase and kind; channel off; floor."""
    on = [r for r in records if r.condition == "on"]
    main = [r for r in on if r.kind in MAIN_KINDS]
    events = [r for r in on if r.kind == "event"]
    out = {"by_family": _table(main, [family_of.get(r.episode_id, "?") for r in main]),
           "by_signal": _table(main, [r.signal for r in main]),
           "by_phase": _table(events, [str(r.phase) for r in events]),
           "by_kind": _table(on, [r.kind for r in on])}
    for condition, name in (("zeroed", "channel_off"), ("base", "untrained_floor")):
        rows = [r for r in records if r.condition == condition and r.kind in MAIN_KINDS]
        out[name] = _table(rows, ["all"] * len(rows)).get("all")
    return out


def _core(rows: Sequence[StreamRecord]) -> dict:
    occurred = [r for r in rows if r.occurred]
    return {"planned": len(rows), "occurred": len(occurred), "hits": sum(r.hit for r in occurred)}


def k_terciles(arm: str) -> list[tuple[int, int]]:
    """Three ranges of k of equal width, fixed by the range k is drawn from (not by outcomes)."""
    lo, hi = FREE_K if arm == "free" else LIVE_K
    width = (hi - lo + 1) // 3
    return [(lo, lo + width - 1), (lo + width, lo + 2 * width - 1), (lo + 2 * width, hi)]


STEEPNESS = ((0.0, 0.05, "under 0.05"), (0.05, 0.10, "0.05 to 0.10"), (0.10, math.inf, "0.10 and over"))  # points per token


def stream_tables(records: Sequence[StreamRecord], arm: str) -> dict:
    """Arms free, live, Ls, not a line: reaction by k (terciles fixed by k), and for the live arms
    the budgets drawn and the reaction by steepness of the target counter (points per token)."""
    events = _events(records, arm)
    out = {"by_k": {f"{a}-{b}": _core([r for r in events if a <= r.k <= b]) for a, b in k_terciles(arm)}}
    if arm != "free":
        steep = {}
        for low, high, name in STEEPNESS:
            steep[name] = _core([r for r in events if r.budget and low <= 100 / r.budget < high])
        drawn = {}
        for signal in LIVE:
            rows = [r for r in events if r.signal == signal and r.budget]
            if rows:
                budgets, starts = [r.budget for r in rows], [r.start or 0 for r in rows]
                drawn[signal] = {"n": len(rows), "budget": [min(budgets), float(np.median(budgets)), max(budgets)],
                                 "start": [min(starts), float(np.median(starts)), max(starts)]}
        out.update(by_steepness=steep, budgets_drawn=drawn)
    return out


def between_paired(channel: Sequence[RunRecord], notice: Sequence[RunRecord], family_of: dict[str, str]) -> dict:
    """Arm paired, channel model against text-notice model on the same episodes: exact McNemar on
    the `event` runs of kind `event`, all families and the new family alone."""
    def hits(records) -> dict[str, bool]:
        return {r.episode_id: r.reaction_signal == r.signal for r in records
                if r.condition == "on" and r.kind == "event" and r.paired == "event" and r.event_occurred}

    a, b = hits(channel), hits(notice)
    out = {}
    for name, keep in (("all", lambda e: True), (NEW, lambda e: family_of.get(e) == NEW)):
        common = [e for e in a if e in b and keep(e)]
        only_a, only_b = sum(a[e] and not b[e] for e in common), sum(b[e] and not a[e] for e in common)
        p = mcnemar_exact(only_a, only_b)
        out[name] = {"pairs": len(common), "channel_hits": sum(a[e] for e in common), "notice_hits": sum(b[e] for e in common),
                     "channel_only": only_a, "notice_only": only_b, "p": p, "test": "exact McNemar", "claim": bool(p < CLAIM_P)}
    return out


def between_stream(channel: Sequence[StreamRecord], notice: Sequence[StreamRecord], arm: str) -> dict:
    """Arms without a forced prefix: each model wrote its own prefix, so the comparison is unpaired
    (Fisher exact on hits of the crossings that occurred)."""
    def counts(records) -> tuple[int, int]:
        occurred = [r for r in _events(records, arm) if r.occurred]
        return sum(r.hit for r in occurred), len(occurred)

    (k1, n1), (k2, n2) = counts(channel), counts(notice)
    p = fisher_exact(k1, n1, k2, n2) if n1 and n2 else None
    return {"channel": [k1, n1], "notice": [k2, n2], "p": p, "test": "Fisher exact", "claim": bool(p is not None and p < CLAIM_P)}


# --- files -------------------------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def episode_rows(path: Path) -> list[dict]:
    """Id, family, kind and signal of every episode of a test file (the signals are dropped)."""
    return [{key: row.get(key) for key in ("episode_id", "family", "kind", "signal")} for row in _read_jsonl(path)]


def reference_rates(model: str, root: Path) -> dict | None:
    """Interrupt events of the model's full evaluation on `data/n6` (kind `event`, all signals)."""
    summary = _read_json(Path(root) / REFERENCE[model] / "summary.json")
    group = (summary or {}).get("groups", {}).get("on|all interrupt|all")
    if group is None:
        return None
    return {"folder": REFERENCE[model], "n": group["n"], "reaction_rate": group["reaction_rate"],
            "n_quiet": group["n_nominal"], "false_alarm_rate": group["false_alarm_rate"]}


def expected_paired_runs(episodes: Sequence[dict], model: str) -> int:
    """Runs a paired evaluation makes for these episodes: every episode once with the channel or
    the notice on; the main kinds also with the quiet and the near-miss pairing, kind `event` also
    with the swap; channel model: every episode once with the channel off, and the untrained floor
    on at most 150 main episodes."""
    main = sum(1 for e in episodes if e["kind"] in MAIN_KINDS)
    on = len(episodes) + 2 * main + sum(1 for e in episodes if e["kind"] == "event")
    return on + (len(episodes) + min(150, main) if model == "channel" else 0)


def run_folders(set_name: str, root: Path) -> dict[tuple[str, str], dict]:
    """Where the result of each (arm, model) is: the confirmatory set's first run in the marker file
    (reruns beside it); the pilot's default folder at batch 16."""
    rows = ledger.entries(set_name, root)
    out = {}
    for arm in ARMS:
        for model in RUNS:
            if set_name == "confirm":
                first = ledger.first_run(rows, arm, model)
                out[(arm, model)] = {"entry": first, "folder": Path(root) / first["out"] if first else None,
                                     "reruns": ledger.reruns(rows, arm, model)}
            else:
                folder = Path(root) / out_folder("pilot", arm, model)
                out[(arm, model)] = {"entry": None, "folder": folder if (folder / "records.jsonl").exists() else None, "reruns": []}
    return out


def _paired_records(folder: Path) -> list[RunRecord]:
    return [RunRecord(**row) for row in _read_jsonl(Path(folder) / "records.jsonl")]


def _stream_records(folder: Path) -> list[StreamRecord]:
    return [StreamRecord(**row) for row in _read_jsonl(Path(folder) / "records.jsonl")]


# --- readings that need the tokenizer ----------------------------------------------------------------

def paired_reading(folder: Path, records: Sequence[RunRecord], family_of: dict[str, str], tok, ms) -> dict:
    """Arm paired, from the saved generations: `time` under the strict rule and the practical
    reading (deadline named, turn ended inside the horizon, no call), and every miss and every
    false alarm of the new family, decoded."""
    key = lambda row: (row["episode_id"], row["condition"], row["paired"])  # noqa: E731
    saved = {key(g): g for g in _read_jsonl(Path(folder) / "generations.jsonl")}
    quiet = "quiet" if any(r.paired == "quiet" for r in records) else "nominal"
    time_rows, listed = Counter(), []
    for r in records:
        g = saved.get((r.episode_id, r.condition, r.paired))
        if g is None or r.condition != "on" or r.kind != "event":
            continue
        ids = g["token_ids"]
        if r.signal == "time" and r.paired == "event" and r.event_occurred:
            strict = r.reaction_signal == "time"
            late = (g["stop_reason"] == "eos" and names_signal(tok.decode(ids[:_NAMING_SPAN]), "time")
                    and not parse_tool_calls(ids, tok, ms))
            time_rows["n"] += 1
            time_rows["strict"] += strict
            time_rows["practical"] += bool(strict or late)
        if family_of.get(r.episode_id) != NEW:
            continue
        miss = r.paired == "event" and r.event_occurred and r.reaction_signal != r.signal
        alarm = r.paired == quiet and r.reaction_signal is not None
        if miss or alarm:
            listed.append({"what": "miss" if miss else "false alarm", "episode_id": r.episode_id, "signal": r.signal,
                           "phase": r.phase, "reaction": r.reaction_signal, "stop_reason": g["stop_reason"],
                           "tokens": len(ids), "text": tok.decode(ids[:120])[:300]})
    return {"time": {"n": time_rows["n"], "strict": time_rows["strict"], "practical": time_rows["practical"]},
            "new_family": listed}


def _last_rows(folder: Path) -> dict[tuple[str, str], dict]:
    """The generation rows of a folder of the arms without a forced prefix; the later row of a key counts."""
    return {(g["task_id"], g["run_type"]): g for g in _read_jsonl(Path(folder) / "generations.jsonl")}


def stream_reading(folder: Path, records: Sequence[StreamRecord], tok, ms) -> dict:
    """Arms free, live, Ls, from the saved generations: every miss (an event run whose crossing
    occurred and that is not a hit) and every false alarm (a twin or off run with a counted
    reaction) of the new family, decoded. A miss shows the start of its judged part; a false alarm
    is shown around its reaction, wherever in the run it is. The same cap on the text as in arm paired."""
    saved, specs, listed = _last_rows(folder), signal_set(6), []
    for r in records:
        g = saved.get((r.task_id, r.run_type))
        if g is None or r.family != NEW:
            continue
        miss = r.run_type == "event" and r.occurred and not r.hit
        alarm = r.run_type != "event" and bool(r.false_alarm)
        if not (miss or alarm):
            continue
        ids, start, found = list(g["post"]), 0, None
        if alarm:  # shown around the reaction's marker, at the anchor of the classifier that found it
            ids = list(g["pre"]) + list(g["post"])
            marker = stream.alarm_token(r.k, g["pre"], g["post"], g["stop_reason"], tok, ms, specs)
            start = max(0, marker - 60) if marker is not None else 0
            found = ("from the first token" if r.alarm_start else "at the crossing position" if r.alarm_at_k
                     else "at the close of the thinking")
        listed.append({"what": "miss" if miss else "false alarm", "task_id": r.task_id, "run_type": r.run_type, "signal": r.signal,
                       "k": r.k, "phase": r.phase, "premature": r.premature, "reaction": r.reaction_signal, "found": found,
                       "stop_reason": r.stop_reason, "tokens": r.n_pre + r.n_post, "from_token": start,
                       "text": tok.decode(ids[start:start + 120])[:300]})
    return {"new_family": listed}


def _level_of(signal: str, milli: int) -> int | None:
    """The 1% level behind a saved value of a live counter (thousandths); None for `think` at its nominal 1.0."""
    if signal == "think":
        return None if milli >= 1000 else (995 - milli) // 10
    return (milli - 5) // 10


def twin_end_levels(folder: Path, plans: dict[str, stream.Plan]) -> dict | None:
    """Live arms: the level at which the target counter of each twin ends. Channel model: the level
    its enlarged budget gives at the largest count of the run (`fed_max`: the highest level that was
    fed). Text-notice model: the level its below-limit notice shows. The plan aims at 85-89 (0.5 to
    4.5 points short of the limit); above 89 is a defect, below 85 is reported."""
    ends, fed = [], []
    for (task_id, run_type), row in _last_rows(folder).items():
        plan = plans.get(task_id)
        if plan is None or run_type != "twin":
            continue
        if row.get("rows_milli") is not None and row.get("budgets"):
            budgets = live.Budgets(**row["budgets"])
            top = live.max_count(plan.signal, row["n_prompt"], plan.k, plan.horizon, budgets)
            ends.append(live.level(top, getattr(budgets, plan.signal)))
            fed += [v for v in (_level_of(plan.signal, m) for m in row["rows_milli"][plan.signal]) if v is not None]
        elif row.get("notice_text"):
            shown = round(100 * float(row["notice_text"].rsplit("=", 1)[1].rstrip("]")))
            ends.append(100 - shown if plan.signal == "think" else shown)
    if not ends:
        return None
    return {"n": len(ends), "min": min(ends), "max": max(ends), "below_85": sum(v < 85 for v in ends),
            "above_89": sum(v > 89 for v in ends), "fed_max": max(fed) if fed else None}


def rows_recomputed(folder: Path, plans: dict[str, stream.Plan], set_code: int, ms) -> dict:
    """Live arms: the counters recomputed from the saved tokens against the rows saved with each run."""
    specs = signal_set(6)
    checked, wrong = 0, []
    for row in _read_jsonl(Path(folder) / "generations.jsonl"):
        plan = plans.get(row["task_id"])
        if plan is None or row.get("rows_milli") is None:
            continue
        checked += 1
        if not stream.recomputed_rows_match(row, row["n_prompt"], plan, set_code, specs, ms):
            wrong.append(f"{row['task_id']} / {row['run_type']}")
    return {"checked": checked, "mismatches": wrong}


# --- integrity ---------------------------------------------------------------------------------------

def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--no-optional-locks", *args], capture_output=True, text=True)


def _function_source(text: str, name: str) -> str | None:
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


def scoring_unchanged(root: Path = Path(".")) -> dict:
    """Classifier and scoring as they stood at the freeze commit: no difference in `reactions.py` and
    `eval/metrics.py`, and the source text of `classify` in `eval/run.py` identical. Git is only read.
    A repository without the freeze commit (a copy without the project's history) checks nothing:
    the three verdicts are None."""
    if _git("-C", str(root), "cat-file", "-e", f"{FREEZE_COMMIT}^{{commit}}").returncode != 0:
        return {"freeze_commit": FREEZE_COMMIT, "files_unchanged": None, "classify_unchanged": None, "passed": None}
    diff = _git("-C", str(root), "diff", "--quiet", FREEZE_COMMIT, "--", *SCORING_FILES)
    frozen = _git("-C", str(root), "show", f"{FREEZE_COMMIT}:src/prio/eval/run.py")
    now = (Path(root) / "src" / "prio" / "eval" / "run.py").read_text(encoding="utf-8")
    same = frozen.returncode == 0 and _function_source(frozen.stdout.replace("\r\n", "\n"), "classify") == \
        _function_source(now.replace("\r\n", "\n"), "classify") is not None
    return {"freeze_commit": FREEZE_COMMIT, "files_unchanged": diff.returncode == 0, "classify_unchanged": bool(same),
            "passed": diff.returncode == 0 and bool(same)}


def _check(name: str, passed: bool | None, detail: str, shown: str | None = None) -> dict:
    """One row of the integrity table. `passed` None: nothing was checked (`shown` says why not)."""
    return {"check": name, "passed": passed, "detail": detail,
            "shown": shown or ("not checked" if passed is None else "yes" if passed else "no")}


# --- the report --------------------------------------------------------------------------------------

def report(set_name: str, root: Path = Path("."), tok=None, ms=None) -> dict:
    """Everything of one set: the lines per model, the tables that are not lines, the comparison of
    the two models, the accounting table, the integrity checks, every entry of the marker file.
    `tok`, `ms`: for the readings of the saved generations (without them those parts are left out)."""
    root = Path(root)
    data = root / "data" / PREFIX[set_name]
    minimums = MIN_COUNTS if set_name == "confirm" else _NO_MINIMUM
    folders = run_folders(set_name, root)
    episodes = {"channel": episode_rows(data / "paired" / "test.jsonl"), "notice": episode_rows(data / "paired-notice" / "test.jsonl")}
    family_of = {e["episode_id"]: e["family"] for e in episodes["channel"] + episodes["notice"]}
    plans = {arm: {p.task_id: p for p in stream.load_plans(data / "stream.json", arm)} if (data / "stream.json").exists() else {}
             for arm in STREAM_ARMS}
    manifest = _read_json(data / "paired" / "manifest.json") or {}
    marker = ledger.entries(set_name, root)
    checks, records, models = [], {}, {}

    def section(arm: str, model: str, folder: Path) -> dict:
        if arm == "paired":
            rows = _paired_records(folder)
            out = {"folder": folder.relative_to(root).as_posix(), "runs": len(rows), "line1": line1(rows, family_of, minimums),
                   "tables": paired_tables(rows, family_of),
                   "runs_by": dict(sorted(Counter(f"{r.condition} / {r.paired}" for r in rows).items())),
                   "reference": reference_rates(model, root),
                   "reading": paired_reading(folder, rows, family_of, tok, ms) if tok is not None else None}
            return out, rows
        rows = _stream_records(folder)
        out = {"folder": folder.relative_to(root).as_posix(), "runs": len(rows), "summary": stream.stream_summary(rows),
               "tables": stream_tables(rows, arm),
               "twin_end_levels": twin_end_levels(folder, plans[arm]) if arm in LIVE_ARMS else None,
               "reading": stream_reading(folder, rows, tok, ms) if tok is not None else None}
        out.update({"free": lambda: {"line2": line2(rows, minimums)}, "live": lambda: {"line3": line3(rows, minimums)},
                    "Ls": lambda: {"beside": no_line(rows, "Ls")}}[arm]())
        return out, rows

    for model in RUNS:
        models[model] = {}
        for arm in ARMS:
            where = folders[(arm, model)]
            if where["folder"] is None or not (where["folder"] / "records.jsonl").exists():
                models[model][arm] = None
                continue
            models[model][arm], records[(arm, model)] = section(arm, model, where["folder"])
            reruns = []
            for entry in where["reruns"]:
                again = root / entry["out"]
                if (again / "records.jsonl").exists():
                    reruns.append({"rerun": entry["rerun"], "reason": entry["rerun_reason"], **section(arm, model, again)[0]})
                else:
                    reruns.append({"rerun": entry["rerun"], "reason": entry["rerun_reason"], "folder": entry["out"], "runs": 0})
            models[model][arm]["reruns"] = reruns

    # Lines per model, and the statement: which lines are met, not met, not evaluable, not run.
    lines, statement = {}, {}
    for model in RUNS:
        got = models[model]
        lines[model] = {"line 1": got["paired"]["line1"] if got["paired"] else None,
                        "line 2": got["free"]["line2"] if got["free"] else None,
                        "line 3": got["live"]["line3"] if got["live"] else None}
        statement[model] = {name: (line["verdict"] if line else "not run") for name, line in lines[model].items()}

    between = {}
    if ("paired", "channel") in records and ("paired", "notice") in records:
        between["paired"] = between_paired(records[("paired", "channel")], records[("paired", "notice")], family_of)
    for arm in STREAM_ARMS:
        if (arm, "channel") in records and (arm, "notice") in records:
            between[arm] = between_stream(records[(arm, "channel")], records[(arm, "notice")], arm)

    # Accounting: every planned run in exactly one cell.
    accounting = {"paired_tasks": {family: {"requested": n, **(manifest.get("dropped", {}).get(family, {})),
                                            "episode": manifest.get("main_episodes", {}).get(family)}
                                   for family, n in manifest.get("tasks_requested", {}).items()}, "runs": {}}
    for model in RUNS:
        for arm in ARMS:
            got = models[model][arm]
            if arm == "paired":
                planned = expected_paired_runs(episodes[model], model) if episodes[model] else None
                cell = {"planned": planned, "produced": got["runs"] if got else 0, "judged": got["runs"] if got else 0}
            else:
                planned = 3 * len(plans[arm]) if plans[arm] else None
                cell = {"planned": planned, "produced": got["runs"] if got else 0}
                if got:
                    e = got["summary"]["event"]
                    cell.update(event_judged=e["occurred"], event_not_occurred=e["not_occurred"],
                                twin_judged=got["summary"]["twin"]["n"], off_judged=got["summary"]["off"]["n"])
            accounting["runs"][f"{arm} / {model}"] = cell
            if got:
                checks.append(_check(f"planned and produced runs, {arm} / {model}", planned is not None and planned == got["runs"],
                                     f"planned {planned}, produced {got['runs']}"))

    # Integrity checks of AP 9.
    scoring = scoring_unchanged(root)
    checks.append(_check("classifier and scoring unchanged since the freeze commit", scoring["passed"],
                         f"reactions.py and eval/metrics.py unchanged: {scoring['files_unchanged']}; source of classify identical: "
                         f"{scoring['classify_unchanged']} (commit {FREEZE_COMMIT[:7]})" if scoring["passed"] is not None else
                         f"not checked: commit {FREEZE_COMMIT[:7]} is not in this repository's history"))
    sizes = set()
    for (arm, model), where in folders.items():
        if where["folder"] is not None:
            meta = _read_json(where["folder"] / "eval_meta.json") or {"invocations": []}
            sizes |= {inv.get("args", {}).get("batch_size", 1) for inv in meta["invocations"]}
    sizes |= {e["batch_size"] for e in marker} if set_name == "confirm" else set()
    checks.append(_check("one batch size", sizes <= {BATCH_SIZE}, f"batch sizes found: {sorted(sizes)}"))
    for (arm, model), rows in records.items():
        if arm == "paired":
            continue
        if arm == "free":  # the rows plan the crossing: a run that occurred with its crossing elsewhere is a defect of the rows
            off = models[model][arm]["summary"]["event"]["crossing_off_k"]
            checks.append(_check(f"crossing on token k, {arm} / {model}", off == 0, f"{off} crossing(s) that occurred off token k"))
        name = f"saved rows equal the rows recomputed from the tokens, {arm} / {model}"
        if model == "notice":  # nothing here could fail: said so, instead of a row that looks like a check
            checks.append(_check(name, None, "true by construction, not a check: the harness computed these counters from the "
                                 "same tokens with the same function" if arm in LIVE_ARMS else
                                 "not a check: no rows are saved for this model in arm free (it reads none)", shown="not a check"))
        elif ms is None:
            checks.append(_check(name, None, "not checked: no tokenizer"))
        else:
            again = rows_recomputed(folders[(arm, model)]["folder"], plans[arm], SET_CODE[set_name], ms)
            checks.append(_check(name, not again["mismatches"],
                                 f"{again['checked']} runs checked, {len(again['mismatches'])} differ {again['mismatches'][:5]}"))
    # The data files named in the marker entries, hashed again: are they still the files the runs read? (The
    # pilot may be built again: there only the latest entry of each arm and model at the report's batch size.)
    digests: dict[str, str | None] = {}
    hashed = marker if set_name == "confirm" else list({(e["arm"], e["model"]): e for e in marker if e["batch_size"] == BATCH_SIZE}.values())
    for entry in hashed:
        changed = []
        for file, digest in entry["data"].items():
            if file not in digests:
                digests[file] = ledger.sha256(root / file) if (root / file).exists() else None
            if digests[file] != digest:
                changed.append(file)
        name = f"data files of the marker entry {entry['arm']} / {entry['model']}" + (f" (rerun {entry['rerun']})" if entry["rerun"] else "")
        checks.append(_check(name, not changed, f"{len(entry['data']) - len(changed)} of {len(entry['data'])} files are the files the "
                             "run read" + (f"; changed or missing: {changed}" if changed else "")))
    if set_name == "confirm":
        for entry in marker:
            name = f"marker entry {entry['arm']} / {entry['model']}" + (f" (rerun {entry['rerun']})" if entry["rerun"] else "")
            file = root / entry["out"] / "records.jsonl"
            if entry["finished"] is None:
                checks.append(_check(name, False, "unfinished: the run never recorded its end"))
            elif not file.exists() or ledger.sha256(file) != entry["records_sha256"]:
                checks.append(_check(name, False, f"{entry['out']}/records.jsonl is not the file the run recorded"))
            else:
                checks.append(_check(name, True, "records.jsonl is the file the run recorded"))
        trees = sorted({e["tree"] for e in marker if e["kind"] == "run"})  # a rerun may come from a corrected tree
        checks.append(_check("one code state", len(trees) <= 1, f"src/prio trees of the first runs: {trees}"))
    return {"set": set_name, "pilot": set_name == "pilot", "generated": time.strftime("%Y-%m-%d %H:%M"),
            "lines": lines, "statement": statement, "models": models, "between_models": between, "accounting": accounting,
            "integrity": checks, "integrity_passed": all(c["passed"] is not False for c in checks), "marker": marker,
            "base_runs": {"kept_share": manifest.get("kept_share"), "under_70_percent": manifest.get("under_70_percent")}}


# --- the report as text ------------------------------------------------------------------------------

def _pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


def _ci(ci) -> str:
    return "-" if ci is None else f"[{ci[0]:.1%}, {ci[1]:.1%}]"


def _yes(value) -> str:
    return "-" if value is None else "yes" if value else "no"


def _line_rows(model: str, got: dict) -> list[str]:
    rows = []

    def row(name: str, part: dict, shown: str | None = None) -> None:
        rows.append(f"| {model} | {name} | {WORDS[name]} | {part['k']} / {part['n']} | {shown or _pct(part['rate'])} | "
                    f"{_ci(part['ci'])} | {part['minimum']} | **{part['verdict']}** | {_yes(part['bound_clears'])} |")

    one, two, three = got["line 1"], got["line 2"], got["line 3"]
    if one:
        row("L1a", one["L1a"])
        row("L1b", one["L1b"])
        c = one["L1c"]
        gap = "-" if c["gap_points"] is None else f"{c['gap_points']:+.1f} points"
        ci = "-" if c["gap_ci_points"] is None else f"[{c['gap_ci_points'][0]:+.1f}, {c['gap_ci_points'][1]:+.1f}]"
        rows.append(f"| {model} | L1c | {WORDS['L1c']} | seen {c['seen']['k']} / {c['seen']['n']} ({_pct(c['seen']['rate'])}) minus new "
                    f"{c['new']['k']} / {c['new']['n']} | {gap} | {ci} | {c['minimum']} | **{c['verdict']}** | {_yes(c['bound_clears'])} |")
    if two:
        row("L2a", two["L2a"])
        row("L2b", two["L2b"])
    if three:
        row("L3", three["L3"])
    return rows


def _summary_rows(table: dict) -> list[str]:
    rows = ["| Row | n | Reaction | 95% CI | Delay med | Delay p90 | n quiet | False alarms | 95% CI | n near | Near-miss alarms | "
            "Identification | Unnamed calls | Wrongful call | Wrongful refusal |", "|" + "---|" * 15]
    for name, v in table.items():
        delay = ["-" if v[key] is None else f"{v[key]:.0f}" for key in ("delay_median", "delay_p90")]
        rows.append(f"| {name} | {v['n']} | {_pct(v['reaction_rate'])} | {_ci(v['reaction_ci'])} | {delay[0]} | {delay[1]} | "
                    f"{v['n_nominal']} | {_pct(v['false_alarm_rate'])} | {_ci(v['false_alarm_ci'])} | {v['n_near_miss']} | "
                    f"{_pct(v['near_miss_alarm_rate'])} | {_pct(v['identification_rate'])} | {_pct(v['unnamed_call_rate'])} | "
                    f"{_pct(v['wrongful_call_rate'])} | {_pct(v['wrongful_refusal_rate'])} |")
    return rows


def _core_rows(title: str, table: dict) -> list[str]:
    rows = [f"| {title} | Runs | Crossings that occurred | Hits | Reaction |", "|---|---|---|---|---|"]
    for name, v in table.items():
        rows.append(f"| {name} | {v['planned']} | {v['occurred']} | {v['hits']} | {_pct(v['hits'] / v['occurred'] if v['occurred'] else None)} |")
    return rows


def _stream_text(arm: str, model: str, got: dict) -> list[str]:
    s, e = got["summary"], got["summary"]["event"]
    t, o = s["twin"], s["off"]
    out = [f"### Arm {arm}, {model} model (not a line): `{got['folder']}`", "",
           f"- Event runs {e['planned']}; crossings that occurred {e['occurred']}; hits {e['hits']} ({_pct(e['reaction_rate'])} "
           f"{_ci(e['reaction_ci'])}); did not occur: {', '.join(f'{why} {n}' for why, n in e['not_occurred'].items())}.",
           f"- Premature reactions (a counted reaction within the first k tokens; such a run is never a hit): {e['premature']}.",
           f"- False alarms (a counted reaction at any of the three anchors): twin runs {t['false_alarms']} / {t['n']} "
           f"({_pct(t['false_alarm_rate'])} {_ci(t['false_alarm_ci'])}), of them {t['at_start']} from the first generated token "
           f"alone, {t['at_k']} at the crossing position, {t['at_close']} at the close of the thinking; off runs "
           f"{o['false_alarms']} / {o['n']} ({_pct(o['false_alarm_rate'])} {_ci(o['false_alarm_ci'])}), of them {o['at_start']} "
           f"from the first generated token alone, {o['at_k']} at the crossing position, {o['at_close']} at the close of the "
           "thinking.",
           "- Turns that ended and name the deadline in their last 128 tokens (keyword match only, not a counted reaction): "
           f"event runs {s['deadline_near_end']['event']} (of them not hits: {s['deadline_near_end']['event_not_hit']}), twin "
           f"runs {s['deadline_near_end']['twin']}, off runs {s['deadline_near_end']['off']}.",
           f"- Twins that share their first k tokens with the event run: {t['same_prefix']} / {t['same_prefix_of']}.",
           f"- Delay of the hits (tokens until the signal is named): median {e['delay_median']}, 90th percentile {e['delay_p90']}.",
           f"- Onset index of the hits: (b), the line's definition {e['onset']['b']}; (c) first word of the naming sentence "
           f"{e['onset']['c']}; (a) first token with text {e['onset']['a']}.",
           f"- Hits whose reaction could not be placed (no keyword of the signal found in the judged part: onset none, counted "
           f"against L2b; expected 0): {e['onset']['b'].get('none', 0)}.",
           f"- `time`: crossings that occurred {e['time_practical']['occurred']}, strict rule {e['time_practical']['strict']}, "
           f"practical reading {e['time_practical']['practical']}."]
    ends = got.get("twin_end_levels")
    if ends:
        fed = "" if ends["fed_max"] is None else f"; highest level fed to a twin {ends['fed_max']}"
        out.append(f"- Twins' target counter, level at the end of the run (planned 85-89; text-notice model: the level its notice "
                   f"shows): {ends['n']} twins, lowest {ends['min']}, highest {ends['max']}; under 85: {ends['below_85']}; "
                   f"over 89 (a defect): {ends['above_89']}{fed}.")
    out.append("")
    for title, table in (("Signal", e["by_signal"]), ("Family", e["by_family"]), ("Phase at token k", e["by_phase"]),
                         ("k (terciles fixed by k)", got["tables"]["by_k"])):
        out += _core_rows(title, table) + [""]
    if "by_steepness" in got["tables"]:
        out += _core_rows("Steepness of the target counter, points per token", got["tables"]["by_steepness"]) + [""]
        out += ["| Counter | Runs | Budget, tokens (min / median / max) | Starting count (min / median / max) |", "|---|---|---|---|"]
        out += [f"| {name} | {v['n']} | {v['budget'][0]} / {v['budget'][1]:.0f} / {v['budget'][2]} | "
                f"{v['start'][0]} / {v['start'][1]:.0f} / {v['start'][2]} |" for name, v in got["tables"]["budgets_drawn"].items()] + [""]
    if got.get("reading"):
        listed = got["reading"]["new_family"]
        out += [f"Misses and false alarms of the new family in arm {arm}, decoded ({len(listed)}; a miss: the start of the judged "
                "part; a false alarm: the text around its reaction):", ""]
        out += [f"- {x['what']} `{x['task_id']}` ({x['run_type']}, {x['signal']} at k {x['k']}, {x['phase']}"
                f"{', premature reaction' if x['premature'] else ''}; reaction {x['reaction']}"
                f"{', found ' + x['found'] if x['found'] else ''}, {x['stop_reason']}, {x['tokens']} "
                f"tokens, shown from token {x['from_token']}): {x['text']!r}" for x in listed] + [""]
    return out


def render(rep: dict) -> str:
    """The report as Markdown."""
    names = {"channel": "channel", "notice": "text-notice"}
    out = [f"# R12 lines report, {'PILOT' if rep['pilot'] else 'confirmatory'} set ({rep['generated']})", ""]
    if rep["pilot"]:
        out += ["**Pilot set: smoke checks and debugging only. These numbers are never reported as results; the minimum "
                "counts of the confirmatory set are not applied.**", ""]
    if not rep["integrity_passed"]:
        out += ["**INTEGRITY FAILURE: at least one check below did not pass. Read section \"Integrity checks\" before any number.**", ""]
    out += ["## The three lines, per model", "",
            "A line is judged on its point estimate, as an exact fraction; \"not evaluable\" is never \"met\".", "",
            "| Model | Line | Pre-registered words | Count | Rate | 95% interval | Minimum n | Verdict | Interval bound also clears |",
            "|" + "---|" * 9]
    for model in RUNS:
        out += _line_rows(names[model], rep["lines"][model])
    out += ["", "| Model | Line 1 (new family) | Line 2 (free-running crossings) | Line 3 (live counters) |", "|---|---|---|---|"]
    out += [f"| {names[m]} | {s['line 1']} | {s['line 2']} | {s['line 3']} |" for m, s in rep["statement"].items()]
    out += ["", "Line 1 is met when L1a, L1b and L1c are met; line 2 when L2a and L2b are met; line 3 when L3 is met. `time` is "
            "judged by the strict rule in every line.", "",
            "Where the frozen classifier is anchored in the arms without a forced prefix (second note on amendment A12). It "
            "finds a tool-call reaction anywhere, a thinking-budget reaction only where the thinking closes within 48 tokens "
            "of the position it is given, and a deadline reaction only when the turn ends within 128 tokens of that position "
            "with the deadline named in its first 60 tokens. A false alarm of a twin or off run is therefore a counted reaction "
            "found with the classifier anchored at the first generated token (over the whole run), or at the crossing position "
            "(the part after token k, judged exactly as the event run's judged part), or at the close of the thinking wherever "
            "it falls (the budget named within the 47 tokens before `</think>`). The rate is the union of the three; the count "
            "from the first generated token alone is printed beside it. A premature reaction (never a hit) is a counted "
            "reaction within the first k tokens, anchored at the first generated token or at the close of the thinking; a tool "
            "call that opens before token k and closes after it is seen by neither side. A deadline reaction that starts "
            "elsewhere (not at the crossing position, and in a run longer than 128 tokens or with the deadline named after its "
            "first 60 tokens) is not seen by the frozen classifier: turns that ended and name the deadline in their last 128 "
            "tokens are counted beside, as a keyword match only. A hit whose naming sentence cannot be placed (no keyword of "
            "the signal in the judged part) has no onset index and counts against L2b.", "",
            "### Onset index at the last keyword before the reaction's marker (reported without a line, beside L2b)", "",
            "L2b takes the index at the first keyword of the judged part; a keyword in ordinary text before the reaction can "
            "only lower it. At the last keyword before the reaction's marker (the tool call, or `</think>`; the deadline has no "
            "marker and is unchanged):", "",
            "| Model | Hits with index 0 or 1 | Share | 95% interval | L2b, the same hits |", "|---|---|---|---|---|"]
    for model in RUNS:
        two = rep["lines"][model]["line 2"]
        if two:
            last, b = two["onset_last"], two["L2b"]
            out.append(f"| {names[model]} | {last['k']} / {last['n']} | {_pct(last['rate'])} | {_ci(last['ci'])} | "
                       f"{b['k']} / {b['n']} ({_pct(b['rate'])}) |")
    out += ["", "### Turns that ended and name the deadline in their last 128 tokens (keyword match only, not a counted reaction)", "",
            "A deadline reaction that starts elsewhere (not at the crossing position, and in a run longer than 128 tokens or "
            "with the deadline named after its first 60 tokens) is not seen by the frozen classifier. These counts stand "
            "beside the false alarms and the misses; they are a match of the classifier's deadline keywords, nothing more "
            "(an event run that is a `time` hit is among them).", "",
            "| Arm | Model | Event runs | Event runs that are not hits | Twin runs | Off runs |", "|---|---|---|---|---|---|"]
    for arm in STREAM_ARMS:
        for model in RUNS:
            got = rep["models"][model][arm]
            if got:
                d = got["summary"]["deadline_near_end"]
                out.append(f"| {arm} | {names[model]} | {d['event']} | {d['event_not_hit']} | {d['twin']} | {d['off']} |")
    out.append("")

    out += ["## Arm Ls beside arm live (no line)", "",
            "| Model | Arm | Counter | Hits / crossings that occurred | Reaction | 95% interval |", "|---|---|---|---|---|---|"]
    for model in RUNS:
        for arm, key in (("live", "line3"), ("Ls", "beside")):
            got = rep["models"][model][arm]
            if got:
                for counter, v in got[key]["by_counter"].items():
                    out.append(f"| {names[model]} | {arm} | {counter} | {v['k']} / {v['n']} | {_pct(v['rate'])} | {_ci(v['ci'])} |")
    out.append("")

    for model in RUNS:
        got = rep["models"][model]["paired"]
        if got:
            out += [f"## Arm paired, {names[model]} model (not a line): `{got['folder']}`", ""]
            for title, key in (("By family, main kinds", "by_family"), ("By signal, main kinds", "by_signal"),
                               ("By phase, kind `event`", "by_phase"), ("By kind (swap = judged against the injected signal)", "by_kind")):
                out += [f"{title} (not a line):", ""] + _summary_rows(got["tables"][key]) + [""]
            other = {name: got["tables"][key] for name, key in (("channel off", "channel_off"), ("untrained floor", "untrained_floor"))
                     if got["tables"].get(key)}
            if other:
                out += ["Channel off and untrained floor, main kinds (not a line):", ""] + _summary_rows(other) + [""]
            seen, known = rep["lines"][model]["line 1"]["L1c"]["seen"], got.get("reference")
            if known:
                out += [f"Seen families (math + explain), kind `event` (not a line): fresh tasks {seen['k']} / {seen['n']} "
                        f"({_pct(seen['rate'])}); the same model's full evaluation on `data/n6` (`{known['folder']}`, all families' "
                        f"interrupt events): reaction {_pct(known['reaction_rate'])} of {known['n']}, quiet-twin alarms "
                        f"{_pct(known['false_alarm_rate'])} of {known['n_quiet']}.", ""]
            if got["reading"]:
                t = got["reading"]["time"]
                out += [f"`time`, kind `event` (not a line): {t['n']} runs, strict rule {t['strict']}, practical reading "
                        f"(deadline named, turn ended inside the horizon) {t['practical']}.", "",
                        f"Misses and false alarms of the new family, decoded ({len(got['reading']['new_family'])}):", ""]
                out += [f"- {x['what']} `{x['episode_id']}` ({x['signal']}, {x['phase']}; reaction {x['reaction']}, {x['stop_reason']}, "
                        f"{x['tokens']} tokens): {x['text']!r}" for x in got["reading"]["new_family"]] + [""]
            for again in got.get("reruns", []):
                out += [f"RERUN {again['rerun']} of arm paired, {names[model]} model (reason: {again['reason']}; beside the first run, "
                        f"which is the confirmatory result): `{again['folder']}`, {again['runs']} runs."]
                if again.get("line1"):
                    out += [f"  L1a {again['line1']['L1a']['k']} / {again['line1']['L1a']['n']}, L1b {again['line1']['L1b']['k']} / "
                            f"{again['line1']['L1b']['n']}, line 1 {again['line1']['verdict']}."]
                out.append("")
    for arm in STREAM_ARMS:
        for model in RUNS:
            got = rep["models"][model][arm]
            if got:
                out += _stream_text(arm, names[model], got)
                for again in got.get("reruns", []):
                    out += [f"RERUN {again['rerun']} of arm {arm}, {names[model]} model (reason: {again['reason']}; beside the first "
                            f"run, which is the confirmatory result): `{again['folder']}`, {again['runs']} runs."]
                    if again.get("summary"):
                        e = again["summary"]["event"]
                        out += [f"  Hits {e['hits']} / {e['occurred']} crossings that occurred."]
                    out.append("")

    out += ["## Channel model against text-notice model (not a line)", "",
            f"No claim of a difference in detection unless p < {CLAIM_P}.", "",
            "| Arm | Rows | Channel | Text-notice | Only channel / only text-notice | Test | p | Difference may be claimed |",
            "|---|---|---|---|---|---|---|---|"]
    for arm, got in rep["between_models"].items():
        if arm == "paired":
            for name, v in got.items():
                out.append(f"| paired | {name}, {v['pairs']} pairs | {v['channel_hits']} | {v['notice_hits']} | {v['channel_only']} / "
                           f"{v['notice_only']} | {v['test']} | {v['p']:.3g} | {_yes(v['claim'])} |")
        else:
            p = "-" if got["p"] is None else f"{got['p']:.3g}"
            out.append(f"| {arm} | unpaired | {got['channel'][0]} / {got['channel'][1]} | {got['notice'][0]} / {got['notice'][1]} | - | "
                       f"{got['test']} | {p} | {_yes(got['claim'])} |")
    out.append("")

    out += ["## Accounting (not a line): every planned run in one cell", "",
            "| Family | Tasks requested | Not ended | Reaction tool | No position | Too long | Main episode built |", "|---|---|---|---|---|---|---|"]
    for family, v in rep["accounting"]["paired_tasks"].items():
        out.append(f"| {family} | {v['requested']} | {v.get('not_ended', '-')} | {v.get('reaction_tool', '-')} | {v.get('no_position', '-')} | "
                   f"{v.get('too_long', '-')} | {v.get('episode', '-')} |")
    base = rep["base_runs"]
    if base["kept_share"]:
        out += ["", "Base runs kept / requested: " + ", ".join(f"{f} {share:.1%}" for f, share in base["kept_share"].items())
                + (f". Under the 70% rule: {', '.join(base['under_70_percent'])}." if base["under_70_percent"] else ". None under 70%.")]
    out += ["", "| Arm / model | Planned runs | Produced | Event runs judged | Not occurred, by reason | Twin runs judged | Off runs judged |",
            "|---|---|---|---|---|---|---|"]
    for name, v in rep["accounting"]["runs"].items():
        reasons = "-" if "event_not_occurred" not in v else ", ".join(f"{why} {n}" for why, n in v["event_not_occurred"].items())
        out.append(f"| {name} | {'-' if v['planned'] is None else v['planned']} | {v['produced']} | {v.get('event_judged', v.get('judged', '-'))} | "
                   f"{reasons} | {v.get('twin_judged', '-')} | {v.get('off_judged', '-')} |")
    out.append("")

    out += ["## Integrity checks", "", "| Check | Passed | Detail |", "|---|---|---|"]
    out += [f"| {c['check']} | {c['shown']} | {c['detail']} |" for c in rep["integrity"]]
    out += ["", "## Marker file: every entry", ""]
    if not rep["marker"]:
        out.append("No entry.")
    for e in rep["marker"]:
        label = f"RERUN {e['rerun']} (reason: {e['rerun_reason']})" if e["kind"] == "rerun" else "first run"
        out.append(f"- {e['arm']} / {e['model']}, {label}: `{e['out']}`, started {e['started']}, finished {e['finished']}, "
                   f"resumes {len(e['resumes'])}, batch {e['batch_size']}, commit {e['commit'][:7]}, tree {e['tree'][:7]}, "
                   f"records sha256 {(e['records_sha256'] or '-')[:12]}, data {', '.join(e['data'])}")
    return "\n".join(out) + "\n"


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec

    tok = AutoTokenizer.from_pretrained(model_id)
    return tok, resolve_spec(model_id, tok)


def main(argv: list[str] | None = None) -> None:
    utf8_console()  # the report prints decoded model text
    parser = argparse.ArgumentParser(description="Compute the lines of R12 and write the report.")
    parser.add_argument("--set", choices=tuple(SET_CODE), required=True, dest="set_name")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (holds data/ and results/)")
    parser.add_argument("--model", default=MODEL_ID, help="tokenizer of the saved generations")
    parser.add_argument("--no-tokenizer", action="store_true", help="leave out the readings of the saved generations")
    args = parser.parse_args(argv)
    tok, ms = (None, None) if args.no_tokenizer else _load_tokenizer(args.model)
    rep = report(args.set_name, args.root, tok, ms)
    out = ledger.marker_file(args.set_name, args.root).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "r12_lines.json").write_bytes((json.dumps(rep, indent=1) + "\n").encode("utf-8"))
    text = render(rep)
    (out / "r12_lines.md").write_bytes(text.encode("utf-8"))
    print(text)
    if not rep["integrity_passed"]:
        print("INTEGRITY FAILURE: see the section \"Integrity checks\"", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
