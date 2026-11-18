"""Smoke checks on the pilot set of R12 (implementation plan Section 6; AP 9).

    uv run python -m prio.r12.smoke --hand-read "<what was read and seen>"   -> results/r12p/smoke.json

All gates must pass on one clean `src/prio` tree before the confirmatory set is built: the builder
and every confirmatory run read `results/r12p/smoke.json` (`passed`, `tree`). CPU only (tokenizer).

Gates (a failure is a bug to fix; no rate of a model is a gate):
- S1 task sets: the planned counts; pilot and confirmatory set disjoint; the built task list and
  plans are the planned ones; the freshness audit passed for both sets;
- S2 arm paired, data: every family keeps >= 70% of its base runs; all five kinds; the four notice
  files pair by episode id; every notice is ` [status <signal>=<d.dd>]` in 9 tokens;
- S3 arm paired, runs: the records of both models are exactly the planned runs;
- S4 arm free, channel model: the fed rows are the planned trajectory; every crossing that
  occurred is on token k; the twin shares its first k tokens with the event run in >= 90% of pairs;
- S5 arms free, live, Ls, text-notice model: every run type continues the same k tokens of pass 1;
  the event notice is there iff the crossing occurs and shows the value recomputed from the tokens;
  the twin's notice shows a value under the limit; the off run has none; no run excluded as
  `crossing_off_k` (the counter past its limit before token k). The counters saved with a run are
  not compared again (computed from the same tokens by the same function: true by construction);
- S6 arms live and Ls, channel model: the fed rows are the counters recomputed from the tokens;
  no crossing on another token than the k-th (such a run is excluded under `crossing_off_k`;
  expected 0, any one stops the checks for a reading); no other counter within 5 points of its
  limit; no twin's target counter above level 89 (planned or fed; a planned end under 85 is
  reported). The share of twins with the event run's first k tokens is reported without a gate: a
  live twin has another budget from its first token;
- S7 batch agreement: the two batch-1 slices (arms free and live, channel model, event runs)
  against the same runs at batch 16: at most 1 of 32 with another outcome (hit, occurred, reaction signal);
- S8 onset definition: the saved phase-1 evaluations give the numbers of the plan's reading 13;
- S9 hand reading: a note (10 event runs per arm and model read by hand).

Triggers are outcomes that start a bug hunt; they are computed and stored, never part of `passed`.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from prio.datagen import variants
from prio.datagen.episodes import read_jsonl
from prio.eval.run import stop_ids
from prio.r12 import (LIVE, MODEL_ID, RUNS, SET_CODE, SLICES, SLICE_ORDER, STREAM_ARMS, build, ledger, lines, live, stream,
                      utf8_console)
from prio.r12.run import out_folder
from prio.r12.stream import LIVE_ARMS, StreamRecord
from prio.r12.tasks import read_tasks, task_sets
from prio.signals import signal_set
from prio.tasks import build_prompt

NOTICE_TOKENS = 9
_NOTICE = re.compile(r" \[status (ctx|think|energy|time|net|ram)=(\d\.\d\d)\]")
KINDS = ("event", "gating_event", "starts_past", "gating_unused", "recovery")
MAIN_KINDS = ("event", "gating_event")
KEPT_MIN = 0.7
TWIN_SHARE_MIN = (9, 10)  # arm free: at least 9 of 10 twins share the event run's first k tokens
OTHER_COUNTER_MAX = 84  # a counter that is not the target stays 5 points from its limit
SLICE_ARMS = ("free", "live")
SLICE_RUNS = 32  # two slices of 16 event runs
# AP reading 13, read on the saved evaluations before any R12 code existed.
ONSET_EXPECTED = {
    "results/phase1": {"hits": 550, "a_at_most_1": 550, "b_at_most_1": 549, "c_at_most_1": 544,
                       "b_by_kind": {"event": {"0": 320, "1": 5, "3": 1}, "gating_event": {"0": 224}}},
    "results/b3-notice-none-full-b16": {"hits": 557, "b": {"0": 557}},
}
MODEL_NAME = {"channel": "channel", "notice": "text-notice"}
SET = "pilot"


class Missing(Exception):
    """A file the check needs is not there."""


def _need(path: Path) -> Path:
    if not Path(path).exists():
        raise Missing(Path(path).as_posix())
    return Path(path)


def _jsonl(path: Path) -> list[dict]:
    with open(_need(path), encoding="utf-8") as rows:
        return [json.loads(line) for line in rows if line.strip()]


def _json(path: Path):
    return json.loads(_need(path).read_text(encoding="utf-8"))


def _folder(root: Path, arm: str, model: str, batch_size: int = 16) -> Path:
    return Path(root) / out_folder(SET, arm, model, batch_size)


def _stream_inputs(root: Path, arm: str, model: str) -> tuple[dict[str, stream.Plan], list[StreamRecord], dict]:
    folder = _folder(root, arm, model)
    plans = {p.task_id: p for p in stream.load_plans(_need(Path(root) / "data" / "r12p" / "stream.json"), arm)}
    records = [StreamRecord(**row) for row in _jsonl(folder / "records.jsonl")]
    _need(folder / "generations.jsonl")
    return plans, records, lines._last_rows(folder)


# --- the checks: each returns (numbers, failures) ---------------------------------------------------

def _s1(root: Path) -> tuple[dict, list[str]]:
    data, sets, failures = Path(root) / "data" / "r12p", task_sets(), []
    counts = {name: {arm: len(tasks) for arm, tasks in arms.items()} for name, arms in sets.items()}
    planned = {name: {arm: sum(SLICES[f][i] for f in SLICES) for i, (s, arm) in enumerate(SLICE_ORDER) if s == name} for name in sets}
    if counts != planned:
        failures.append(f"the task sets hold {counts}, the plan says {planned}")
    pilot = [t for tasks in sets["pilot"].values() for t in tasks]
    confirm = [t for tasks in sets["confirm"].values() for t in tasks]
    shared = len({t.prompt for t in pilot} & {t.prompt for t in confirm}) + len({t.task_id for t in pilot} & {t.task_id for t in confirm})
    if shared:
        failures.append(f"{shared} prompts or ids in both sets")
    _need(data / "tasks.json")
    if read_tasks(data) != sets["pilot"]:
        failures.append(f"{(data / 'tasks.json').as_posix()} is not the planned pilot set")
    arms = sets["pilot"]
    plans = _json(data / "stream.json")
    if plans != build.stream_plans(arms["free"], arms["live"], arms["Ls"], SET):
        failures.append(f"{(data / 'stream.json').as_posix()} does not hold the planned plans")
    audit = _json(data / "freshness.json")
    total = {name: sum(arm.values()) for name, arm in counts.items()}
    if audit.get("passed") is not True:
        failures.append("the freshness audit did not pass")
    if audit.get("by_seed", {}).get("tasks") != total:
        failures.append(f"the freshness audit covered {audit.get('by_seed', {}).get('tasks')}, not both sets {total}")
    found = len(audit.get("by_seed", {}).get("used", [])) + len(audit.get("by_seed", {}).get("in_both_sets", [])) + \
        len(audit.get("by_file", {}).get("matches", []))
    if found:
        failures.append(f"the freshness audit lists {found} match(es)")
    return {"pilot": counts["pilot"], "confirm": counts["confirm"], "shared_prompts": shared,
            "plans": dict(Counter(p["arm"] for p in plans)),
            "freshness": {"tasks": audit.get("by_seed", {}).get("tasks"), "files": audit.get("by_file", {}).get("files"),
                          "episodes": audit.get("by_file", {}).get("episodes")}}, failures


def _s2(root: Path, tok) -> tuple[dict, list[str]]:
    data, failures = Path(root) / "data" / "r12p", []
    manifest = _json(data / "paired" / "manifest.json")
    for family, share in manifest["kept_share"].items():
        if share < KEPT_MIN:
            failures.append(f"{family} keeps {share:.1%} of its base runs (under 70%)")
    source = lines.episode_rows(_need(data / "paired" / "test.jsonl"))
    kinds = Counter(e["kind"] for e in source)
    failures += [f"no episode of kind {kind}" for kind in KINDS if not kinds[kind]]
    ids = lambda rows: [e["episode_id"] for e in rows]  # noqa: E731
    main = [e for e in source if e["kind"] in MAIN_KINDS]
    expected = {"test.jsonl": ids(source), "test_quiet.jsonl": ids(main), "test_near_miss.jsonl": ids(main),
                "test_swap.jsonl": ids([e for e in source if e["kind"] == "event"])}
    notices = 0
    for name, wanted in expected.items():
        path = _need(data / "paired-notice" / name)
        if ids(lines.episode_rows(path)) != wanted:
            failures.append(f"paired-notice/{name} does not hold the episodes of the source file ({len(wanted)} expected)")
        if name == "test_quiet.jsonl":
            continue
        for ep in read_jsonl(path):
            for a, b in ep.harness_spans or ():
                notices += 1
                text = tok.decode(ep.target_ids[a:b])
                if b - a != NOTICE_TOKENS or _NOTICE.fullmatch(text) is None:
                    failures.append(f"paired-notice/{name} {ep.episode_id}: notice {text!r} in {b - a} tokens "
                                    f"(planned: ` [status <signal>=<d.dd>]` in {NOTICE_TOKENS} tokens)")
    return {"kept_share": manifest["kept_share"], "kinds": dict(kinds), "main_episodes": len(main), "notices": notices}, failures


def _s3(root: Path) -> tuple[dict, list[str]]:
    numbers, failures = {}, []
    for model in RUNS:
        data = Path(root) / "data" / "r12p" / ("paired" if model == "channel" else "paired-notice")
        episodes = lines.episode_rows(_need(data / "test.jsonl"))
        main = sum(1 for e in episodes if e["kind"] in MAIN_KINDS)
        planned = {"on / event": len(episodes), f"on / {'nominal' if model == 'channel' else 'quiet'}": main,
                   "on / near_miss": main, "on / swapped": sum(1 for e in episodes if e["kind"] == "event")}
        if model == "channel":
            planned.update({"zeroed / event": len(episodes), "base / event": min(150, main)})
        rows = _jsonl(_folder(root, "paired", model) / "records.jsonl")
        produced = dict(Counter(f"{r['condition']} / {r['paired']}" for r in rows))
        keys = Counter((r["episode_id"], r["condition"], r["paired"]) for r in rows)
        twice = sum(1 for n in keys.values() if n > 1)
        numbers[model] = {"planned": planned, "produced": produced, "runs": len(rows), "runs_twice": twice}
        if produced != planned:
            failures.append(f"{model}: produced {produced}, planned {planned}")
        if twice:
            failures.append(f"{model}: {twice} run(s) recorded twice")
        if sum(planned.values()) != lines.expected_paired_runs(episodes, model):
            failures.append(f"{model}: the plan of this check and of the lines report differ")
    return numbers, failures


def _channel_rows(root: Path, arm: str, ms) -> tuple[dict, list[str], dict, dict]:
    """What S4 and S6 share: run count, fed rows against recomputed rows, crossings on token k, prefix share."""
    plans, records, rows = _stream_inputs(root, arm, "channel")
    specs, failures = signal_set(6), []
    if len(records) != 3 * len(plans):
        failures.append(f"{arm}: {len(records)} runs, {3 * len(plans)} planned")
    differ = [f"{key[0]} / {key[1]}" for key, row in rows.items()
              if key[0] in plans and not stream.recomputed_rows_match(row, row["n_prompt"], plans[key[0]], SET_CODE[SET], specs, ms)]
    if differ:
        failures.append(f"{arm}: the saved rows of {len(differ)} run(s) are not the rows recomputed from the tokens, first {differ[:3]}")
    occurred = [r for r in records if r.run_type == "event" and r.occurred]
    # expected 0: live arms, a run excluded because its first past-limit row is on another token than
    # the k-th; any arm, a run counted as occurred although its saved crossing is elsewhere
    off_k = [r.task_id for r in records if r.run_type == "event"
             and (r.why_not == "crossing_off_k" or (r.occurred and r.crossing_at != r.k - 1))]
    if off_k:
        failures.append(f"{arm}: {len(off_k)} crossing(s) not on token k (crossing_off_k), first {off_k[:3]}: a reading is "
                        "needed before the confirmatory set")
    twins = [r for r in records if r.run_type == "twin" and r.same_prefix is not None]
    numbers = {"runs": len(records), "planned": 3 * len(plans), "rows_checked": sum(1 for key in rows if key[0] in plans),
               "rows_differ": len(differ), "crossings_occurred": len(occurred), "crossings_off_k": len(off_k),
               "twins": len(twins), "twins_sharing_prefix": sum(bool(r.same_prefix) for r in twins)}
    return numbers, failures, plans, rows


def _s4(root: Path, ms) -> tuple[dict, list[str]]:
    numbers, failures, _, _ = _channel_rows(root, "free", ms)
    if not numbers["twins"] or TWIN_SHARE_MIN[1] * numbers["twins_sharing_prefix"] < TWIN_SHARE_MIN[0] * numbers["twins"]:
        failures.append(f"free: {numbers['twins_sharing_prefix']} of {numbers['twins']} twins share the event run's first k tokens (under 90%)")
    return numbers, failures


def _s6(root: Path, ms) -> tuple[dict, list[str]]:
    numbers, failures = {}, []
    for arm in LIVE_ARMS:
        try:
            mine, problems, plans, rows = _channel_rows(root, arm, ms)
        except Missing as missing:
            failures.append(f"{arm}: missing {missing}")
            continue
        highest = -1
        for (task_id, _), row in rows.items():
            plan = plans.get(task_id)
            if plan is None or row.get("rows_milli") is None:
                continue
            for name in LIVE:
                if name != plan.signal:
                    levels = [v for v in (lines._level_of(name, m) for m in row["rows_milli"][name]) if v is not None]
                    highest = max([highest] + levels)
        if highest > OTHER_COUNTER_MAX:
            problems.append(f"{arm}: a counter that is not the target reaches level {highest} (within 5 points of its limit)")
        ends = lines.twin_end_levels(_folder(root, arm, "channel"), plans)
        if ends is None:
            problems.append(f"{arm}: no twin run with its budgets")
        elif ends["above_89"] or (ends["fed_max"] is not None and ends["fed_max"] > 89):
            problems.append(f"{arm}: a twin's target counter is above 89 (planned end levels above 89: {ends['above_89']}; "
                            f"highest level fed: {ends['fed_max']})")
        numbers[arm] = {**mine, "other_counters_highest_level": highest, "twin_end_levels": ends}
        failures += problems
    return numbers, failures


def _s5(root: Path, tok, ms) -> tuple[dict, list[str]]:
    specs = signal_set(6)
    names = [s.name for s in specs]
    stops = stop_ids(tok, ms)
    numbers, failures = {}, []
    for arm in STREAM_ARMS:
        try:
            plans, records, rows = _stream_inputs(root, arm, "notice")
            first = {row["task_id"]: row for row in _jsonl(_folder(root, arm, "notice") / "pass1.jsonl")}
            tasks = {t.task_id: t for t in task_sets()[SET][arm]}  # the planned tasks (S1 checks that the built list is the same)
        except Missing as missing:
            failures.append(f"{arm}: missing {missing}")
            continue
        by_key = {(r.task_id, r.run_type): r for r in records}
        problems, with_notice, without, twins = [], 0, 0, 0
        if len(records) != 3 * len(plans):
            problems.append(f"{len(records)} runs, {3 * len(plans)} planned")
        for task_id, plan in plans.items():
            spec, column = specs[names.index(plan.signal)], names.index(plan.signal)
            if task_id not in tasks:
                problems.append(f"{task_id}: a plan for a task that is not in the planned set")
                continue
            n_prompt = len(build_prompt(tok, tasks[task_id]))
            pre = (first.get(task_id) or {}).get("pre")
            if pre is None:
                problems.append(f"{task_id}: no pass 1")
                continue
            open_turn = not any(t in stops for t in pre)
            if open_turn and len(pre) != plan.k:
                problems.append(f"{task_id}: pass 1 wrote {len(pre)} tokens, k is {plan.k}")
            occurs, _ = stream.occurs(pre, plan.k, plan.signal, stops, ms)
            if arm in LIVE_ARMS:  # the value the harness computes from the k tokens
                budgets = live.draw_budgets(arm, plan.signal, n_prompt, plan.k, plan.horizon, SET_CODE[SET], plan.index)["event"]
                stored = live.base_rows(arm, specs, n_prompt, plan.k, plan.horizon, SET_CODE[SET], plan.index)
                counters = live.counter_rows(n_prompt, pre, budgets, stored, ms)
                first_past = live.first_past(counters, plan.signal, specs)
                value = float(counters[plan.k - 1, column]) if len(pre) >= plan.k else None
                if occurs and first_past is not None and first_past != plan.k - 1:
                    problems.append(f"{task_id}: the counter is past its limit before token k (crossing_off_k; expected 0)")
                occurs = bool(occurs and first_past == plan.k - 1)
            else:
                value = float(stream.free_trajectories(specs, n_prompt, plan, SET_CODE[SET])["event"][n_prompt + plan.k - 1, column])
            for run_type in stream.RUN_TYPES:
                row, record = rows.get((task_id, run_type)), by_key.get((task_id, run_type))
                where = f"{task_id} / {run_type}"
                if row is None or record is None:
                    problems.append(f"{where}: not run")
                    continue
                if row["pre"] != pre:
                    problems.append(f"{where}: its first tokens are not the k tokens of pass 1")
                if row["n_prompt"] != n_prompt:
                    problems.append(f"{where}: prompt of {row['n_prompt']} tokens, the task's prompt has {n_prompt}")
                text = row.get("notice_text")
                if text is not None and row.get("notice") != variants.encode(tok, text):
                    problems.append(f"{where}: the notice ids are not the ids of {text!r}")
                if run_type == "event":
                    if occurs:
                        with_notice += 1
                        if text is None:
                            problems.append(f"{where}: the crossing occurs and the run got no notice")
                        elif text != variants.notice_text(spec, value):
                            problems.append(f"{where}: notice {text!r}, the harness computes {variants.notice_text(spec, value)!r}")
                        if not record.occurred or record.crossing_at != plan.k - 1:
                            problems.append(f"{where}: the record does not say that the crossing occurred on token k")
                    else:
                        without += 1
                        if text is not None or row["post"]:
                            problems.append(f"{where}: no crossing occurs and the run got a notice or went on")
                        if record.occurred:
                            problems.append(f"{where}: the record says a crossing occurred, the tokens say none did")
                    if record.why_not == "crossing_off_k":
                        problems.append(f"{where}: excluded as crossing_off_k (expected 0): a reading is needed before the "
                                        "confirmatory set")
                    # (The counters saved with the row are not compared again: the harness computed them from the
                    # same tokens with the same function, so that comparison is true by construction.)
                elif run_type == "twin":
                    shown = _NOTICE.fullmatch(text) if text else None
                    if open_turn:
                        twins += 1
                        if shown is None or shown.group(1) != plan.signal or spec.past_limit(float(shown.group(2))):
                            problems.append(f"{where}: the twin's notice {text!r} is not a below-limit notice of {plan.signal}")
                    elif text is not None:
                        problems.append(f"{where}: a notice after a turn that had ended")
                elif text is not None:
                    problems.append(f"{where}: the off run got a notice")
        numbers[arm] = {"plans": len(plans), "runs": len(records), "events_with_notice": with_notice,
                        "events_without_notice": without, "twins_with_notice": twins, "problems": len(problems)}
        failures += [f"{arm} {line}" for line in problems]
    return numbers, failures


def _s7(root: Path) -> tuple[dict, list[str]]:
    outcome = lambda r: (r["hit"], r["occurred"], r["reaction_signal"])  # noqa: E731
    slices, failures, compared, differing = {}, [], 0, 0
    for arm in SLICE_ARMS:
        try:
            one = {(r["task_id"], r["run_type"]): r for r in _jsonl(_folder(root, arm, "channel", 1) / "records.jsonl")}
            sixteen = {(r["task_id"], r["run_type"]): r for r in _jsonl(_folder(root, arm, "channel") / "records.jsonl")}
        except Missing as missing:
            failures.append(f"{arm}: missing {missing}")
            continue
        both = [key for key in one if key in sixteen]
        other = [f"{key[0]} / {key[1]}" for key in both if outcome(one[key]) != outcome(sixteen[key])]
        if len(both) != len(one):
            failures.append(f"{arm}: {len(one) - len(both)} run(s) of the batch-1 slice are not in the batch-16 folder")
        slices[arm] = {"compared": len(both), "differing": len(other), "runs": other}
        compared, differing = compared + len(both), differing + len(other)
    if compared < SLICE_RUNS:
        failures.append(f"{compared} runs compared, {SLICE_RUNS} planned (two slices of 16 event runs)")
    if differing * SLICE_RUNS > max(compared, SLICE_RUNS):
        failures.append(f"{differing} of {compared} runs have another outcome at batch 16 (at most 1 of {SLICE_RUNS})")
    return {"compared": compared, "differing": differing, "slices": slices}, failures


def _s8(root: Path, tok) -> tuple[dict, list[str]]:
    numbers, failures = {}, []
    for name, expected in ONSET_EXPECTED.items():
        folder = Path(root) / name
        _need(folder / "records.jsonl")
        _need(folder / "generations.jsonl")
        counts = stream.evaluation_onsets(folder, tok)
        numbers[name] = {key: counts.get(key) for key in expected}
        failures += [f"{name}: {key} is {counts.get(key)}, the plan's reading says {value}"
                     for key, value in expected.items() if counts.get(key) != value]
    return numbers, failures


def _s9(hand_read: str | None) -> tuple[dict, list[str]]:
    words = len((hand_read or "").split())
    return {"words": words}, [] if words else ["no --hand-read note: 10 event runs per arm and model are read by hand first"]


WHAT = {
    "S1": "task sets: planned counts, the two sets disjoint, the built list and plans as planned, freshness audit passed",
    "S2": "arm paired, data: >= 70% of the base runs kept per family, five kinds, four notice files paired, notices as planned",
    "S3": "arm paired, runs: the records of both models are the planned runs",
    "S4": "arm free, channel model: fed rows as planned, crossing on token k, >= 90% of twins share the first k tokens",
    "S5": "arms free, live, Ls, text-notice model: same k tokens in every run type, notice iff the crossing occurs, computed "
          "value, no crossing off token k",
    "S6": "arms live and Ls, channel model: fed rows recomputed, no crossing off token k, other counters 5 points away, twin "
          "never above 89",
    "S7": "batch agreement: the batch-1 slices against batch 16, at most 1 of 32 with another outcome",
    "S8": "onset definition: the saved evaluations give the numbers of reading 13",
    "S9": "hand reading: the note",
}


def checks(root: Path = Path("."), tok=None, ms=None, hand_read: str | None = None) -> dict:
    """Every gate S1-S9 with its numbers and pass / fail. `tok`, `ms`: the model's tokenizer and
    special tokens (loaded when not given). A missing file fails its check, never the command."""
    if tok is None:
        tok, ms = _load_tokenizer(MODEL_ID)
    steps: dict[str, Callable[[], tuple[dict, list[str]]]] = {
        "S1": lambda: _s1(root), "S2": lambda: _s2(root, tok), "S3": lambda: _s3(root), "S4": lambda: _s4(root, ms),
        "S5": lambda: _s5(root, tok, ms), "S6": lambda: _s6(root, ms), "S7": lambda: _s7(root), "S8": lambda: _s8(root, tok),
        "S9": lambda: _s9(hand_read)}
    out = {}
    for name, step in steps.items():
        try:
            numbers, failures = step()
        except Missing as missing:
            numbers, failures = {}, [f"missing {missing}"]
        except (KeyError, IndexError, TypeError, ValueError) as error:  # a file that is not what the check reads
            numbers, failures = {}, [f"unreadable input: {type(error).__name__}: {error}"]
        out[name] = {"what": WHAT[name], "passed": not failures, "numbers": numbers, "failures": failures}
    return out


# --- triggers: outcomes that start a bug hunt, never a gate ------------------------------------------

def _trigger(name: str, k: int | None, n: int | None, fired: bool | None) -> dict:
    detail = "not computed: no records" if k is None else f"{k} / {n}" + (f" = {k / n:.1%}" if n else "")
    return {"trigger": name, "fired": fired, "detail": detail}


def triggers(root: Path = Path(".")) -> list[dict]:
    """Arm paired: `event` reaction of either model under 70%, quiet-twin alarms over 10%, reaction
    with the channel off over 10%. Arm free: crossings that occur under 75% of the planned. Live
    arms: crossings that occur under 60% for a counter."""
    out = []
    for model in RUNS:
        name = MODEL_NAME[model]
        path = _folder(root, "paired", model) / "records.jsonl"
        rows = [r for r in _jsonl(path) if r["kind"] == "event"] if path.exists() else None
        quiet = "nominal" if model == "channel" else "quiet"
        groups = [("event reaction under 70%", lambda r: r["condition"] == "on" and r["paired"] == "event" and r["event_occurred"],
                   lambda r: r["reaction_signal"] == r["signal"], lambda k, n: 10 * k < 7 * n),
                  ("quiet-twin alarms over 10%", lambda r: r["condition"] == "on" and r["paired"] == quiet,
                   lambda r: r["reaction_signal"] is not None, lambda k, n: 10 * k > n)]
        if model == "channel":
            groups.append(("reaction with the channel off over 10%", lambda r: r["condition"] == "zeroed" and r["event_occurred"],
                           lambda r: r["reaction_signal"] == r["signal"], lambda k, n: 10 * k > n))
        for label, keep, count, rule in groups:
            if rows is None:
                out.append(_trigger(f"arm paired, {name} model: {label}", None, None, None))
                continue
            mine = [r for r in rows if keep(r)]
            k = sum(1 for r in mine if count(r))
            out.append(_trigger(f"arm paired, {name} model: {label}", k, len(mine), bool(mine) and rule(k, len(mine))))
    for arm in STREAM_ARMS:
        for model in RUNS:
            name = MODEL_NAME[model]
            path = _folder(root, arm, model) / "records.jsonl"
            events = [r for r in _jsonl(path) if r["run_type"] == "event"] if path.exists() else None
            if arm == "free":
                label = f"arm free, {name} model: crossings that occur under 75% of the planned"
                if events is None:
                    out.append(_trigger(label, None, None, None))
                else:
                    k = sum(1 for r in events if r["occurred"])
                    out.append(_trigger(label, k, len(events), bool(events) and 4 * k < 3 * len(events)))
                continue
            for counter in sorted({r["signal"] for r in events or []}, key=LIVE.index) if events else ():
                mine = [r for r in events if r["signal"] == counter]
                k = sum(1 for r in mine if r["occurred"])
                out.append(_trigger(f"arm {arm}, {name} model: {counter} crossings that occur under 60%", k, len(mine),
                                    5 * k < 3 * len(mine)))
            if events is None:
                out.append(_trigger(f"arm {arm}, {name} model: crossings that occur under 60% for a counter", None, None, None))
    return out


def code_states(root: Path = Path(".")) -> dict:
    """Where the pilot's data and runs were made (reported, not a gate: after a fix only the steps
    concerned are run again; the checks themselves are evaluated on the tree that is checked out)."""
    manifest = Path(root) / "data" / "r12p" / "manifest.json"
    last: dict[tuple, dict] = {}
    for entry in ledger.entries(SET, root):
        last[(entry["arm"], entry["model"], entry["batch_size"])] = entry
    return {"current": ledger.code_tree(), "data_built_on": json.loads(manifest.read_text(encoding="utf-8")).get("tree") if manifest.exists() else None,
            "runs_by_tree": dict(Counter(entry["tree"] for entry in last.values()))}


def verdict(root: Path = Path("."), tok=None, ms=None, hand_read: str | None = None) -> dict:
    """The content of smoke.json. `passed`: every gate passed on a clean `src/prio` tree, whose id is recorded."""
    done = checks(root, tok, ms, hand_read)
    clean = ledger.tree_clean()
    return {"passed": bool(clean and all(check["passed"] for check in done.values())), "tree": ledger.code_tree(),
            "commit": ledger.commit(), "clean": clean, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hand_read": hand_read.strip() if hand_read and hand_read.strip() else None, "checks": done,
            "triggers": triggers(root), "code_states": code_states(root)}


def _load_tokenizer(model_id: str):
    return lines._load_tokenizer(model_id)


def main(argv: list[str] | None = None) -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description="Smoke checks on the pilot set of R12; writes results/r12p/smoke.json.")
    parser.add_argument("--hand-read", default=None, help="what was read by hand and seen (10 event runs per arm and model); needed to pass")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (holds data/ and results/)")
    parser.add_argument("--model", default=MODEL_ID, help="tokenizer of the pilot's files")
    args = parser.parse_args(argv)
    tok, ms = _load_tokenizer(args.model)
    result = verdict(args.root, tok, ms, args.hand_read)
    path = ledger.smoke_file(args.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(result, indent=1) + "\n").encode("utf-8"))
    for name, check in result["checks"].items():
        print(f"{name} {'passed' if check['passed'] else 'FAILED'}: {check['what']}")
        for line in check["failures"][:10]:
            print(f"    {line}")
        if len(check["failures"]) > 10:
            print(f"    ... {len(check['failures']) - 10} more in {path.as_posix()}")
    fired = [t for t in result["triggers"] if t["fired"]]
    print(f"triggers fired (a bug hunt, not a gate): {len(fired)}" + "".join(f"\n    {t['trigger']}: {t['detail']}" for t in fired))
    print(f"source tree {result['tree']} ({'clean' if result['clean'] else 'NOT clean: commit first'}); "
          f"smoke checks {'PASSED' if result['passed'] else 'NOT passed'} -> {path.as_posix()}")
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
