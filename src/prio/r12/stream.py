"""Runner and judge of the R12 arms without a forced prefix (AP 5 and 6): the trained model writes
from the prompt and a signal crosses its limit on the k-th token it wrote itself.

Arm `free`: a simulated trajectory that crosses on token k. Arms `live` and `Ls`: `ctx`, `think`,
`time` counted by the harness from the generation (`prio.r12.live`). Run types per task:

- event: channel model = one pass under the event rows; text-notice model = pass 1 writes k tokens
  with no notice, pass 2 appends the notice with the value at the crossing and generates H tokens;
- twin (the false-alarm control): channel model = arm `free`: the same rows with the event signal
  held, from the crossing on, at its last value before it; live arms: the target's budget enlarged;
  text-notice model = the same k tokens, then a below-limit notice of the same signal;
- off: channel model = channel off, event rows; text-notice model = the same k tokens, no notice.

Definitions (AP 5): a crossing occurs when the run wrote at least k tokens, none of its first k
tokens ends the turn and, for `think`, none of them is `</think>`; live arms: read from the rows
(the first generated token whose target value is past its limit must be the k-th: `crossing_at`
= k - 1; on another token the run is excluded under the reason `crossing_off_k` and not judged).
The judged part = the tokens after the k-th, judged by `prio.eval.run.classify` as an `event` run.

Where the frozen classifier is anchored (second note on amendment A12). It finds a tool-call
reaction anywhere, but a thinking-budget reaction only where the thinking closes within 48 tokens
of the position it is given, and a deadline reaction only when the turn ends within 128 tokens of
that position with the deadline named in its first 60 tokens. So:
- twin and off runs: a false alarm = a counted reaction with the classifier anchored (a) at the
  first generated token, over the whole run, or (b) at the crossing position: the part after
  token k, judged by `classify` exactly as the event run's judged part, as a run with nothing to
  react to, or (c) at the close of the thinking, wherever it falls (the budget named within the 47
  tokens before `</think>`). `false_alarm` is the union; `alarm_start`, `alarm_at_k`,
  `alarm_at_close` say which anchor found one;
- premature reaction (event runs): a counted reaction within the first k tokens at anchor (a) or
  (c). Such a run is never a hit. A tool call that begins before token k and ends after it is seen
  by neither side;
- stated limit: a deadline reaction that starts elsewhere is not seen. `deadline_near_end` counts
  beside, as a keyword match only: the turn ended and its last 128 tokens name the deadline.

Files of a results folder: `records.partial.jsonl` (appended run by run; an interrupted run
resumes), `generations.jsonl` (tokens, notice, the rows fed or computed), `pass1.jsonl` (text-notice
model: the k tokens of each task), at the end `records.jsonl` and `stream_summary.md|json`. A last
line torn by a kill or a power cut is moved to `<file>.torn` before a resume (`prio.r12.torn`).
"""

import functools
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from prio.datagen import variants
from prio.datagen.episodes import Episode
from prio.eval.metrics import wilson_ci
from prio.eval.run import _Runs, _think_index, classify, follow_thinking, stop_ids
from prio.generate import GenResult
from prio.r12 import SEED, STREAM_CODE, live, torn
from prio.reactions import (REACTION_WINDOW, TIME_END_WINDOW, _KEYWORDS, _NAMING_SPAN, Reaction, counts, find_reaction,
                            first_reaction, names_signal, parse_tool_calls, unnamed_call)
from prio.signals import SignalSpec
from prio.trajectories import sample_trajectory

RUN_TYPES = ("event", "twin", "off")
LIVE_ARMS = ("live", "Ls")
WHY_NOT = ("ended", "thinking_over", "no_crossing", "crossing_off_k")
_SENTENCE_END = re.compile(r"[.!?](?=\s)|\n")


@dataclass(frozen=True)
class Plan:
    arm: str  # "free" | "live" | "Ls"
    index: int  # position of the task in its arm: the i of the generators
    task_id: str
    family: str
    signal: str  # the signal that crosses
    k: int  # on the k-th generated token
    horizon: int  # tokens generated after it


def load_plans(path: Path, arm: str) -> list[Plan]:
    """The plans of one arm from a set's stream.json (`prio.r12.build.stream_plans`)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Plan(**row) for row in rows if row["arm"] == arm]


@dataclass
class StreamRecord:
    task_id: str
    family: str
    arm: str
    model: str  # "channel" | "notice"
    signal: str
    run_type: str  # "event" | "twin" | "off"
    k: int
    occurred: bool  # event runs: the crossing occurred
    why_not: str | None  # "ended" | "thinking_over" | "no_crossing" | "crossing_off_k" (live arms: not on token k)
    phase: str | None  # "thinking" | "answer" at token k
    hit: bool  # event runs: the counted reaction to the crossing signal, and nothing premature
    reaction_signal: str | None  # event: what the judged part reacted to; twin, off: the false alarm's signal
    named: bool
    delay: int | None  # hits: tokens of the judged part until the signal is named
    onset: int | None  # hits, reading 13 (b): index of the reaction's first token (whitespace tokens before it included)
    onset_word: int | None  # (c): index of the first word of the naming sentence
    onset_first_text: int | None  # (a): index of the first token of the judged part that holds text
    premature: bool  # event: a counted reaction within the first k tokens (anchored at the first token or at the close of the thinking)
    false_alarm: bool | None  # twin, off: a counted reaction at any of the three anchors
    unnamed_call: bool
    crossing_at: int | None  # index of the generated token that first carries the past-limit value (k - 1 as planned)
    same_prefix: bool | None  # channel model's twin: its first k tokens are the event run's
    n_pre: int
    n_post: int
    stop_reason: str
    budget: int | None = None  # live arms: the target counter's budget, tokens
    start: int | None = None  # live arms: its starting count
    # `time` event runs whose crossing occurred: the practical reading beside the strict one (a hit, or
    # the deadline named and the turn ended inside the generated horizon, later than the 128-token rule)
    practical: bool | None = None
    # hits, a second reading beside `onset` (not L2b): the index taken at the last keyword before the reaction's marker
    onset_last: int | None = None
    # twin, off: which anchor of the classifier found a counted reaction (None for event runs)
    alarm_start: bool | None = None  # (a) the first generated token, over the whole run
    alarm_at_k: bool | None = None  # (b) the crossing position: the part after token k, judged as the event run's
    alarm_at_close: bool | None = None  # (c) the close of the thinking, wherever it falls
    # Keyword match only, not a counted reaction: the turn ended and its last 128 tokens name the deadline
    # (a deadline reaction that starts at none of the anchors is not seen by the frozen classifier).
    deadline_near_end: bool = False


# --- the onset index (AP reading 13) ---------------------------------------------------------------

def _first_mention(text: str, signal: str) -> int | None:
    starts = [m.start() for k in _KEYWORDS[signal] if (m := re.search(rf"\b{re.escape(k)}\b", text, re.IGNORECASE))]
    return min(starts) if starts else None


def onset_index(post: Sequence[int], signal: str, tok, lead: bool = True) -> int | None:
    """Index, in the judged part, of the first token of the reaction: the sentence in which the
    signal is first named. Sentence start = the first non-whitespace character after the last line
    break or sentence end (`.`, `!`, `?` followed by whitespace) before the first keyword match, or
    the start of the judged part. `lead` (definition (b)): the whitespace-only tokens directly
    before the sentence belong to the reaction; without it (definition (c)): the first word. None
    when the signal is never named."""
    post = list(post)
    text = tok.decode(post)
    at = _first_mention(text, signal)
    return None if at is None else _sentence_start(post, text, at, tok, lead)


def _sentence_start(post: list[int], text: str, at: int, tok, lead: bool) -> int:
    """Index of the token that starts the sentence holding character `at` of `text` (the decoded
    `post`, or a prefix of it); with `lead`, the whitespace-only tokens directly before it included."""
    cut = 0
    for match in _SENTENCE_END.finditer(text, 0, at):
        cut = match.end()
    while cut < at and text[cut].isspace():
        cut += 1
    low, high = 0, len(post) - 1
    while low < high:  # the token that holds character `cut` (decoded prefixes only grow)
        mid = (low + high) // 2
        if len(tok.decode(post[:mid + 1])) > cut:
            high = mid
        else:
            low = mid + 1
    while lead and low > 0 and not tok.decode([post[low - 1]]).strip():
        low -= 1
    return low


def onset_last_index(post: Sequence[int], signal: str, stop_reason: str, tok, ms, lead: bool = True) -> int | None:
    """A second reading of the onset index, beside the one of line L2b: the same sentence-start
    rule applied at the LAST keyword match before the reaction's marker, where the classifier
    itself finds the marker (`find_reaction`): the reaction's tool call, or `</think>` for the
    thinking budget. A keyword in ordinary text before the reaction can only lower the first
    reading; this one finds the sentence next to the marker. The deadline has no marker (the turn
    ends): there the two readings are the same."""
    post = list(post)
    reaction = find_reaction(post, stop_reason, signal, tok, ms)
    if reaction is None or (reaction.soft and signal != "think"):  # no marker to read back from
        return onset_index(post, signal, tok, lead)
    text = tok.decode(post[:reaction.token_index])
    starts = [m.start() for k in _KEYWORDS[signal] for m in re.finditer(rf"\b{re.escape(k)}\b", text, re.IGNORECASE)]
    return _sentence_start(post, text, max(starts), tok, lead) if starts else None


def first_text_index(post: Sequence[int], tok) -> int | None:
    """Definition (a), the audit's measure: index of the first token that is not whitespace only."""
    return next((i for i, token in enumerate(post) if tok.decode([token]).strip()), None)


def evaluation_onsets(folder: Path, tok) -> dict:
    """The three onset definitions over the hits of a saved paired evaluation (`records.jsonl` and
    `generations.jsonl` of `prio.eval.run`, standard or harness mode): the main kinds' `event` runs
    with the channel or the notice on whose reaction is the event signal's. The reading behind AP
    reading 13 (phase-1 full evaluation: (a) 550 / 550, (b) 549 / 550, (c) 544 / 550 at index 0 or 1)."""
    folder = Path(folder)
    records = {(r["episode_id"], r["condition"], r["paired"]): r for r in _read(folder / "records.jsonl")}
    saved = {(g["episode_id"], g["condition"], g["paired"]): g for g in _read(folder / "generations.jsonl")}
    rows = []
    for key, g in saved.items():
        r = records.get(key)
        if (r is None or r["condition"] != "on" or r["paired"] != "event" or r["kind"] not in ("event", "gating_event")
                or not r["event_occurred"] or r["reaction_signal"] != r["signal"]):
            continue
        ids = g["token_ids"]
        rows.append({"episode_id": r["episode_id"], "kind": r["kind"], "signal": r["signal"],
                     "b": onset_index(ids, r["signal"], tok), "c": onset_index(ids, r["signal"], tok, lead=False),
                     "a": first_text_index(ids, tok)})
    out: dict = {"folder": folder.as_posix(), "hits": len(rows)}
    for name in ("b", "c", "a"):
        out[name] = _histogram([row[name] for row in rows])
        out[f"{name}_at_most_1"] = sum(1 for row in rows if row[name] is not None and row[name] <= 1)
    out["b_by_kind"] = {kind: _histogram([row["b"] for row in rows if row["kind"] == kind])
                        for kind in sorted({row["kind"] for row in rows})}
    out["b_by_signal"] = {signal: _histogram([row["b"] for row in rows if row["signal"] == signal])
                          for signal in sorted({row["signal"] for row in rows})}
    out["b_late"] = [row for row in rows if row["b"] is None or row["b"] > 1]
    return out


# --- the judge -------------------------------------------------------------------------------------

def occurs(pre: Sequence[int], k: int, signal: str, stops, ms) -> tuple[bool, str | None]:
    """The crossing on token k occurs: the run wrote k tokens, none of them ends the turn and, for
    `think`, none of them is `</think>` (a thinking budget cannot run out after the thinking is over)."""
    first = list(pre[:k])
    if len(first) < k or any(t in stops for t in first):
        return False, "ended"
    if signal == "think" and ms.think_close in first:
        return False, "thinking_over"
    return True, None


def _pseudo_episode(plan: Plan, pre: list[int], phase: str, specs: Sequence[SignalSpec]) -> Episode:
    """The episode `classify` needs: the first k tokens as the forced prefix, the crossing on its last token."""
    return Episode(
        episode_id=plan.task_id, task_id=plan.task_id, family=plan.family, split="test", kind="event", signal=plan.signal,
        phase=phase, prompt_ids=[], target_ids=pre, signals=np.zeros((len(pre), len(specs)), dtype=np.float32),
        t_cross=len(pre) - 1, t_decision=None, templated=False)


def _generation(tokens: list[int], stop_reason: str, specs: Sequence[SignalSpec]) -> GenResult:
    return GenResult(tokens, np.zeros((len(tokens), len(specs)), dtype=np.float32), stop_reason)


def _close_reaction(tokens: list[int], stop_reason: str, tok, ms) -> Reaction | None:
    """Anchor (c): a counted thinking-budget reaction at the first `</think>` of `tokens`, wherever
    it falls: the classifier anchored 47 tokens before the close (the widest window it has), so the
    budget must be named within the 47 tokens before `</think>`."""
    if ms.think_close not in tokens:
        return None
    start = max(0, tokens.index(ms.think_close) - (REACTION_WINDOW - 1))
    found = find_reaction(tokens, stop_reason, "think", tok, ms, start=start)
    return found if found is not None and counts(found) else None


def find_alarms(plan: Plan, pre: list[int], post: list[int], stop_reason: str, tok, ms,
                specs: Sequence[SignalSpec]) -> dict[str, tuple[str, bool] | None]:
    """Twin and off runs: (signal, named) of the counted reaction at each anchor of the classifier, or None.
    `start` (a): the first generated token, over the whole run. `at_k` (b): the crossing position,
    i.e. the part after token k judged by `classify` on the event run's pseudo-episode as a run with
    nothing to react to; only when the run has its k tokens. `at_close` (c): the close of the thinking."""
    whole = pre + post
    first = first_reaction(whole, stop_reason, specs, tok, ms)
    at_k = None
    if len(pre) >= plan.k:
        phase = "answer" if ms.think_close in pre[:plan.k] else "thinking"
        record = classify(_pseudo_episode(plan, pre, phase, specs), "nominal", "on", _generation(post, stop_reason, specs),
                          tok, ms, specs)
        if record.reaction_signal is not None:
            at_k = (record.reaction_signal, record.named)
    close = _close_reaction(whole, stop_reason, tok, ms)
    return {"start": (first.signal, first.named) if first is not None else None, "at_k": at_k,
            "at_close": (close.signal, close.named) if close is not None else None}


def alarm_token(k: int, pre: Sequence[int], post: Sequence[int], stop_reason: str, tok, ms,
                specs: Sequence[SignalSpec]) -> int | None:
    """Index, in the whole run, of the marker of the false alarm that `find_alarms` reports first
    (for reading the run by hand): anchor (a), else (b), else (c)."""
    pre, post = list(pre), list(post)
    first = first_reaction(pre + post, stop_reason, specs, tok, ms)
    if first is not None:
        return first.token_index
    if len(pre) >= k:
        at_k = first_reaction(post, stop_reason, specs, tok, ms)  # what `classify` looks for in a run with nothing to react to
        if at_k is not None:
            return len(pre) + at_k.token_index
    close = _close_reaction(pre + post, stop_reason, tok, ms)
    return close.token_index if close is not None else None


def judge(plan: Plan, run_type: str, model: str, pre: Sequence[int], post: Sequence[int], stop_reason: str, tok, ms,
          specs: Sequence[SignalSpec], *, crossing_at: int | None = None, budget: int | None = None,
          start: int | None = None, same_prefix: bool | None = None) -> StreamRecord:
    """One run as a record. `pre`: the first k generated tokens (fewer when the turn ended); `post`:
    what followed (after the notice, for the text-notice model); `crossing_at`: read from the rows."""
    pre, post = [int(t) for t in pre], [int(t) for t in post]
    whole = pre + post
    phase = "answer" if ms.think_close in pre[:plan.k] else "thinking"
    fields = dict(
        task_id=plan.task_id, family=plan.family, arm=plan.arm, model=model, signal=plan.signal, run_type=run_type, k=plan.k,
        phase=phase, crossing_at=crossing_at, same_prefix=same_prefix,
        n_pre=len(pre), n_post=len(post), stop_reason=stop_reason, budget=budget, start=start,
        unnamed_call=unnamed_call(whole, stop_reason, specs, tok, ms), delay=None, onset=None, onset_word=None,
        onset_first_text=None,
        deadline_near_end=bool(stop_reason == "eos" and names_signal(tok.decode(whole[-TIME_END_WINDOW:]), "time")),
    )
    if run_type != "event":  # nothing to react to: a counted reaction at any of the three anchors is a false alarm
        found = find_alarms(plan, pre, post, stop_reason, tok, ms, specs)
        signal, named = next((v for v in found.values() if v is not None), (None, False))
        return StreamRecord(**fields, occurred=False, why_not=None, hit=False, premature=False,
                            reaction_signal=signal, named=named, false_alarm=signal is not None,
                            alarm_start=found["start"] is not None, alarm_at_k=found["at_k"] is not None,
                            alarm_at_close=found["at_close"] is not None)

    stops = stop_ids(tok, ms)
    occurred, why_not = occurs(pre, plan.k, plan.signal, stops, ms)
    if occurred and plan.arm in LIVE_ARMS:  # read from the rows: the first past-limit value must be on token k
        if crossing_at is None:
            occurred, why_not = False, "no_crossing"
        elif crossing_at != plan.k - 1:
            occurred, why_not = False, "crossing_off_k"
    stop_inside = "eos" if any(t in stops for t in pre) else "max_new_tokens"
    premature = (first_reaction(pre, stop_inside, specs, tok, ms) is not None  # anchor (a) on the first k tokens
                 or _close_reaction(pre, stop_inside, tok, ms) is not None)  # anchor (c): the thinking closes inside them
    hit, reaction_signal, named = False, None, False
    if occurred:
        record = classify(_pseudo_episode(plan, pre, phase, specs), "event", "on", _generation(post, stop_reason, specs),
                          tok, ms, specs)
        reaction_signal, named = record.reaction_signal, record.named
        hit = bool(not premature and record.event_occurred and record.reaction_signal == plan.signal)
        if hit:
            fields.update(delay=record.delay, onset=onset_index(post, plan.signal, tok),
                          onset_word=onset_index(post, plan.signal, tok, lead=False),
                          onset_first_text=first_text_index(post, tok),
                          onset_last=onset_last_index(post, plan.signal, stop_reason, tok, ms))
        if plan.signal == "time":  # the convention of scripts/time_misses.py
            late = (not premature and stop_reason == "eos" and names_signal(tok.decode(post[:_NAMING_SPAN]), "time")
                    and not parse_tool_calls(post, tok, ms))
            fields["practical"] = bool(hit or late)
    return StreamRecord(**fields, occurred=occurred, why_not=why_not, hit=hit, premature=premature,
                        reaction_signal=reaction_signal, named=named, false_alarm=None)


# --- rows ------------------------------------------------------------------------------------------

def free_trajectories(specs: Sequence[SignalSpec], n_prompt: int, plan: Plan, set_code: int) -> dict[str, np.ndarray]:
    """Arm `free`: {"event", "twin"} rows [n_prompt + k + horizon, N]. Rows 0..n_prompt-1 go with the
    prompt; row n_prompt + j goes with generated token j, so the k-th generated token is the first
    that carries the past-limit value. Twin: the event signal held, from the crossing position on,
    at its last value before the crossing."""
    names = [s.name for s in specs]
    column, cross = names.index(plan.signal), n_prompt + plan.k - 1
    rng = np.random.default_rng([SEED, set_code, STREAM_CODE["free"], plan.index, 1])
    event = sample_trajectory(specs, n_prompt + plan.k + plan.horizon, rng, event=column, t_cross=cross).values
    twin = event.copy()
    twin[cross:, column] = event[cross - 1, column]
    return {"event": event, "twin": twin}


def _milli(values: np.ndarray, specs: Sequence[SignalSpec], names: Sequence[str]) -> dict[str, list[int]]:
    """Columns of `values` as whole thousandths (every value is a 1% step center or 0 / 1)."""
    index = {s.name: i for i, s in enumerate(specs)}
    return {name: [int(round(float(v) * 1000)) for v in values[:, index[name]]] for name in names}


def recomputed_rows_match(row: dict, n_prompt: int, plan: Plan, set_code: int, specs: Sequence[SignalSpec], ms) -> bool:
    """The rows saved with a generations.jsonl row equal the rows recomputed from its tokens alone.
    Live arms: the three counters (channel model: the rows that were fed; text-notice model: the
    counters the harness computed over the first k tokens). Arm `free`, channel model: the fed
    column of the event signal against the planned trajectory, `think` following the text. A row
    that saved no rows has nothing to compare."""
    saved = row.get("rows_milli")
    if saved is None:
        return True
    which = "twin" if row["run_type"] == "twin" else "event"
    length = len(next(iter(saved.values())))
    tokens = (list(row["pre"]) + list(row["post"]))[:length]
    if len(tokens) != length:
        return False
    if plan.arm in LIVE_ARMS:
        budgets = live.draw_budgets(plan.arm, plan.signal, n_prompt, plan.k, plan.horizon, set_code, plan.index)
        stored = live.base_rows(plan.arm, specs, n_prompt, plan.k, plan.horizon, set_code, plan.index)
        return _milli(live.counter_rows(n_prompt, tokens, budgets[which], stored, ms), specs, live.LIVE) == saved
    think = _think_index(specs)
    values = free_trajectories(specs, n_prompt, plan, set_code)[which]
    signal_fn = follow_thinking(values, n_prompt, think, specs[think].nominal if think is not None else 1.0, ms.think_close)
    again = np.stack([np.asarray(signal_fn(j, tokens[:j + 1]), dtype=np.float32) for j in range(length)]) if length else \
        np.zeros((0, len(specs)), dtype=np.float32)
    return _milli(again, specs, (plan.signal,)) == saved


# --- files -----------------------------------------------------------------------------------------

def _append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


class _Store:
    """Records and generations of one results folder, appended run by run."""

    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        # Before a resume reads or appends: a last line torn by a kill is moved to `<file>.torn`, and a
        # last row without its pair in the other file is dropped back so that its run is made again.
        self.repairs = torn.repair_stream_folder(self.out)
        self.records = {(r["task_id"], r["run_type"]): StreamRecord(**r) for r in _read(self.out / "records.partial.jsonl")}

    def has(self, plan: Plan, run_type: str) -> bool:
        return (plan.task_id, run_type) in self.records

    def add(self, record: StreamRecord, row: dict) -> None:
        # The generation first: a record in the file always has its tokens (a cut between the two
        # writes leaves a generation without a record; the next resume drops that row back and makes the run again).
        _append(self.out / "generations.jsonl", {"task_id": record.task_id, "run_type": record.run_type, **row})
        _append(self.out / "records.partial.jsonl", asdict(record))
        self.records[(record.task_id, record.run_type)] = record

    def finish(self, plans: Sequence[Plan], model: str) -> list[StreamRecord]:
        """The records of `plans` in plan order (event, twin, off), the twins' `same_prefix` filled
        in (channel model), written with their summary."""
        pre = {(g["task_id"], g["run_type"]): g["pre"] for g in _read(self.out / "generations.jsonl")}
        ordered = []
        for plan in plans:
            for run_type in RUN_TYPES:
                record = self.records.get((plan.task_id, run_type))
                if record is None:
                    continue
                if run_type == "twin" and model == "channel":
                    event, twin = pre.get((plan.task_id, "event")), pre.get((plan.task_id, "twin"))
                    record.same_prefix = None if event is None or twin is None else twin == event
                ordered.append(record)
        write_stream_summary(ordered, self.out)
        return ordered


# --- the channel model: one pass per run -------------------------------------------------------------

def run_channel(model, tok, ms, specs: Sequence[SignalSpec], plans: Sequence[Plan], prompts: dict[str, list[int]],
                set_code: int, out_dir: Path, batch_size: int = 16, run_types: Sequence[str] = RUN_TYPES) -> list[StreamRecord]:
    """Every (plan, run type) of the channel model; `prompts`: task id -> prompt ids. Runs already
    in the folder are not repeated."""
    store = _Store(out_dir)
    runs = _Runs(model, stop_ids(tok, ms), batch_size)
    think = _think_index(specs)
    nominal_think = specs[think].nominal if think is not None else 1.0

    def done(plan: Plan, run_type: str, info: dict, gen: GenResult) -> None:
        tokens = [int(t) for t in gen.token_ids]
        crossing = live.first_past(gen.signals, plan.signal, specs)
        record = judge(plan, run_type, "channel", tokens[:plan.k], tokens[plan.k:], gen.stop_reason, tok, ms, specs,
                       crossing_at=crossing, budget=info.get("budget"), start=info.get("start"))
        fed = _milli(gen.signals, specs, live.LIVE if plan.arm in LIVE_ARMS else (plan.signal,))
        store.add(record, {"n_prompt": info["n_prompt"], "pre": tokens[:plan.k], "post": tokens[plan.k:],
                           "stop_reason": gen.stop_reason, "notice": None, "notice_text": None, "rows_milli": fed,
                           "budgets": info.get("budgets")})

    for plan in plans:
        todo = [t for t in run_types if not store.has(plan, t)]
        if not todo:
            continue
        prompt = [int(t) for t in prompts[plan.task_id]]
        n_prompt = len(prompt)
        if plan.arm in LIVE_ARMS:
            budgets = live.draw_budgets(plan.arm, plan.signal, n_prompt, plan.k, plan.horizon, set_code, plan.index)
            stored = live.base_rows(plan.arm, specs, n_prompt, plan.k, plan.horizon, set_code, plan.index)
        else:
            rows = free_trajectories(specs, n_prompt, plan, set_code)
        for run_type in todo:
            which = "twin" if run_type == "twin" else "event"  # off: the event rows, channel off
            if plan.arm in LIVE_ARMS:
                b = budgets[which]
                values = live.prompt_rows(n_prompt, b, stored)
                signal_fn = live.live_signal_fn(n_prompt, b, stored, ms)
                info = {"n_prompt": n_prompt, "budget": getattr(b, plan.signal), "budgets": asdict(b),
                        "start": 0 if plan.signal == "ctx" else getattr(b, f"{plan.signal}_start")}
            else:
                values = rows[which]
                signal_fn = follow_thinking(values, n_prompt, think, nominal_think, ms.think_close)
                info = {"n_prompt": n_prompt}
            runs.add("zeroed" if run_type == "off" else "on", prompt, values, plan.k + plan.horizon, signal_fn, None,
                     functools.partial(done, plan, run_type, info))
    runs.finish()
    return store.finish(plans, "channel")


# --- the text-notice model: two passes ---------------------------------------------------------------

def run_notice(model, tok, ms, specs: Sequence[SignalSpec], plans: Sequence[Plan], prompts: dict[str, list[int]],
               set_code: int, out_dir: Path, batch_size: int = 16, run_types: Sequence[str] = RUN_TYPES) -> list[StreamRecord]:
    """Every (plan, run type) of the text-notice model. Pass 1, once per task: k tokens from the
    prompt, no notice (`pass1.jsonl`; every run type continues these same tokens). Pass 2, per run
    type while the turn is still open: prompt + the k tokens + the notice (event: the value at the
    crossing, only when the crossing occurs; twin: a below-limit value; off: nothing), then
    `horizon` tokens. The notice ids are marked as harness tokens. For a model that reads no state
    before the notice, the two passes are exactly the run a splicing harness would produce."""
    store = _Store(out_dir)
    stops = stop_ids(tok, ms)
    names = [s.name for s in specs]
    nominal = np.array([s.nominal for s in specs], dtype=np.float32)

    def quiet(n: int) -> np.ndarray:
        return np.tile(nominal, (n, 1))

    def signal_fn(step: int, generated: list[int]) -> np.ndarray:
        return nominal

    first = {row["task_id"]: (row["pre"], row["stop_reason"]) for row in _read(store.out / "pass1.jsonl")}

    def first_done(plan: Plan, gen: GenResult) -> None:
        pre = [int(t) for t in gen.token_ids]
        first[plan.task_id] = (pre, gen.stop_reason)
        _append(store.out / "pass1.jsonl", {"task_id": plan.task_id, "pre": pre, "stop_reason": gen.stop_reason})

    runs = _Runs(model, stops, batch_size)
    for plan in plans:
        if plan.task_id not in first and not all(store.has(plan, t) for t in run_types):
            prompt = [int(t) for t in prompts[plan.task_id]]
            runs.add("on", prompt, quiet(len(prompt)), plan.k, signal_fn, None, functools.partial(first_done, plan))
    runs.finish()

    def save(plan: Plan, run_type: str, pre: list[int], notice: list[int] | None, text: str | None, info: dict,
             post: list[int], stop_reason: str) -> None:
        record = judge(plan, run_type, "notice", pre, post, stop_reason, tok, ms, specs, crossing_at=info.get("crossing_at"),
                       budget=info.get("budget"), start=info.get("start"))
        store.add(record, {"n_prompt": info["n_prompt"], "pre": pre, "post": post, "stop_reason": stop_reason,
                           "notice": notice, "notice_text": text, "budgets": info.get("budgets"),
                           # live arms: the counters the harness computed over the k tokens (they decide the event's notice)
                           "rows_milli": info.get("rows_milli") if run_type == "event" else None})

    def second_done(plan: Plan, run_type: str, pre: list[int], notice: list[int], text: str | None, info: dict,
                    gen: GenResult) -> None:
        save(plan, run_type, pre, notice or None, text, info, [int(t) for t in gen.token_ids], gen.stop_reason)

    runs = _Runs(model, stops, batch_size)
    for plan in plans:
        todo = [t for t in run_types if not store.has(plan, t)]
        if not todo:
            continue
        prompt = [int(t) for t in prompts[plan.task_id]]
        n_prompt = len(prompt)
        pre, stop_reason = first[plan.task_id]
        column, spec = names.index(plan.signal), specs[names.index(plan.signal)]
        open_turn = not any(t in stops for t in pre)
        info: dict = {"n_prompt": n_prompt}
        if plan.arm in LIVE_ARMS:  # the harness computes the counters from the k tokens the model wrote
            budgets = live.draw_budgets(plan.arm, plan.signal, n_prompt, plan.k, plan.horizon, set_code, plan.index)["event"]
            stored = live.base_rows(plan.arm, specs, n_prompt, plan.k, plan.horizon, set_code, plan.index)
            counters = live.counter_rows(n_prompt, pre, budgets, stored, ms)
            # The first token on which the computed counter is past its limit: the k-th as planned, none
            # (`no_crossing`), or an earlier one (`crossing_off_k`: no notice, the run is not judged).
            first_past = live.first_past(counters, plan.signal, specs)
            crossing_value = float(counters[plan.k - 1, column]) if first_past == plan.k - 1 else None
            info = {"n_prompt": n_prompt, "budget": getattr(budgets, plan.signal), "budgets": asdict(budgets),
                    "start": 0 if plan.signal == "ctx" else getattr(budgets, f"{plan.signal}_start"),
                    "rows_milli": _milli(counters, specs, live.LIVE)}
        else:
            first_past = plan.k - 1  # arm free: the planned trajectory crosses on token k
            crossing_value = float(free_trajectories(specs, n_prompt, plan, set_code)["event"][n_prompt + plan.k - 1, column])
        for run_type in todo:
            mine = dict(info)
            if run_type == "event":
                ok, _ = occurs(pre, plan.k, plan.signal, stops, ms)
                mine["crossing_at"] = first_past if ok else None  # with the notice: right after the k-th token
                if not ok or crossing_value is None:  # no crossing on token k: no notice, nothing to judge
                    save(plan, run_type, pre, None, None, mine, [], stop_reason)
                    continue
                text = variants.notice_text(spec, crossing_value)
            elif not open_turn:  # the turn ended within its first k tokens: that is the whole run
                save(plan, run_type, pre, None, None, mine, [], stop_reason)
                continue
            elif run_type == "twin":
                rng = np.random.default_rng([SEED, set_code, STREAM_CODE[plan.arm], plan.index, 2 if plan.arm == "free" else 4])
                text = variants.notice_text(spec, variants.below_limit_value(spec, rng))
            else:
                text = None
            notice = variants.encode(tok, text) if text else []
            harness = [False] * (n_prompt + len(pre)) + [True] * len(notice) if notice else None
            full = prompt + pre + notice
            runs.add("on", full, quiet(len(full)), plan.horizon, signal_fn, harness,
                     functools.partial(second_done, plan, run_type, pre, notice, text, mine))
    runs.finish()
    return store.finish(plans, "notice")


# --- summary of one folder ---------------------------------------------------------------------------

def _core(rows: list[StreamRecord]) -> dict:
    occurred = [r for r in rows if r.occurred]
    return {"planned": len(rows), "occurred": len(occurred), "hits": sum(r.hit for r in occurred)}


def _groups(rows: list[StreamRecord], key) -> dict:
    out: dict[str, list[StreamRecord]] = {}
    for r in rows:
        out.setdefault(str(key(r)), []).append(r)
    return {name: _core(group) for name, group in out.items()}


def _histogram(values: list) -> dict[str, int]:
    counts = Counter("none" if v is None else str(v) for v in values)
    return dict(sorted(counts.items(), key=lambda kv: (kv[0] == "none", int(kv[0]) if kv[0] != "none" else 0)))


def _alarms(rows: list[StreamRecord]) -> dict:
    """False alarms of twin or off runs: the union of the three anchors, and what each anchor found
    (`at_start` = what a reading from the first generated token alone would have counted)."""
    flags = [bool(r.false_alarm) for r in rows]
    return {"n": len(rows), "false_alarms": sum(flags), "false_alarm_rate": sum(flags) / len(flags) if flags else None,
            "false_alarm_ci": list(wilson_ci(flags)) if flags else None,
            "at_start": sum(bool(r.alarm_start) for r in rows), "at_k": sum(bool(r.alarm_at_k) for r in rows),
            "at_close": sum(bool(r.alarm_at_close) for r in rows)}


def stream_summary(records: Sequence[StreamRecord]) -> dict:
    """Counts of one results folder (one arm, one model). Not a line: the pre-registered lines are
    computed by `prio.r12.lines`."""
    events = [r for r in records if r.run_type == "event"]
    occurred = [r for r in events if r.occurred]
    hits = [r for r in occurred if r.hit]
    delays = [r.delay for r in hits if r.delay is not None]
    twins = [r for r in records if r.run_type == "twin"]
    paired = [r for r in twins if r.same_prefix is not None]
    flags = [r.hit for r in occurred]
    return {
        "arm": "/".join(sorted({r.arm for r in records})), "model": "/".join(sorted({r.model for r in records})),
        "event": {
            **_core(events),
            "not_occurred": {why: sum(1 for r in events if not r.occurred and r.why_not == why) for why in WHY_NOT},
            "reaction_rate": sum(flags) / len(flags) if flags else None,
            "reaction_ci": list(wilson_ci(flags)) if flags else None,
            "premature": sum(r.premature for r in events),
            # Crossings that were not on token k (expected 0): live arms, the runs excluded under that reason;
            # arm free, a run that occurred although its saved crossing is elsewhere (a defect of the rows).
            "crossing_off_k": sum(1 for r in events if r.why_not == "crossing_off_k" or (r.occurred and r.crossing_at != r.k - 1)),
            "by_signal": _groups(events, lambda r: r.signal), "by_family": _groups(events, lambda r: r.family),
            "by_phase": _groups(events, lambda r: r.phase),
            "delay_median": float(np.median(delays)) if delays else None,
            "delay_p90": float(np.percentile(delays, 90)) if delays else None,
            "onset": {"b": _histogram([r.onset for r in hits]), "c": _histogram([r.onset_word for r in hits]),
                      "a": _histogram([r.onset_first_text for r in hits]),
                      "b_at_most_1": sum(1 for r in hits if r.onset is not None and r.onset <= 1),
                      # the second reading beside L2b: the index at the last keyword before the reaction's marker
                      "last": _histogram([r.onset_last for r in hits]),
                      "last_at_most_1": sum(1 for r in hits if r.onset_last is not None and r.onset_last <= 1)},
            "time_practical": {"occurred": sum(1 for r in occurred if r.signal == "time"),
                               "strict": sum(1 for r in occurred if r.signal == "time" and r.hit),
                               "practical": sum(1 for r in occurred if r.signal == "time" and r.practical)},
        },
        "twin": {**_alarms(twins), "same_prefix": sum(bool(r.same_prefix) for r in paired), "same_prefix_of": len(paired)},
        "off": _alarms([r for r in records if r.run_type == "off"]),
        # Keyword match only, not a counted reaction: turns that ended and name the deadline in their last 128 tokens.
        "deadline_near_end": {"event": sum(r.deadline_near_end for r in events),
                              "event_not_hit": sum(r.deadline_near_end and not r.hit for r in events),
                              "twin": sum(r.deadline_near_end for r in twins),
                              "off": sum(r.deadline_near_end for r in records if r.run_type == "off")},
    }


def _pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


def write_stream_summary(records: Sequence[StreamRecord], out_dir: Path) -> dict:
    """records.jsonl, stream_summary.json and stream_summary.md of one folder. (No `summary.json`:
    `prio.registry` reads that name as a standard evaluation.)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    s = stream_summary(records)
    (out_dir / "stream_summary.json").write_bytes((json.dumps(s, indent=1) + "\n").encode("utf-8"))
    e, t, o = s["event"], s["twin"], s["off"]
    ci = e["reaction_ci"]
    lines = [
        f"# R12 arm `{s['arm']}`, model `{s['model']}`: counts of this folder (not a line)", "",
        "- The pre-registered lines are computed over these records by `prio.r12.lines`; this file only counts.",
        "- Crossing that occurs: the run wrote k tokens, none ends the turn, and for `think` the thinking is still open "
        "(live arms: the first past-limit value of the counter is on token k). Hit: the counted reaction to the crossing "
        "signal in the tokens after the k-th, nothing premature.",
        "- Twin and off runs: a false alarm is a counted reaction with the classifier anchored at the first generated "
        "token, at the crossing position (the part after token k, judged as the event run's) or at the close of the "
        "thinking; the count from the first token alone is beside it. A premature reaction is read at the first token "
        "or at the close of the thinking within the first k tokens.", "",
        "| Run type | Runs | Crossings that occurred | Hits | Reaction | 95% CI | Premature | False alarms (any anchor) | Rate | "
        "From the first token alone | At the crossing position | At the close of the thinking |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| event | {e['planned']} | {e['occurred']} | {e['hits']} | {_pct(e['reaction_rate'])} | "
        f"{'-' if ci is None else f'[{ci[0]:.1%}, {ci[1]:.1%}]'} | {e['premature']} | - | - | - | - | - |",
        f"| twin | {t['n']} | - | - | - | - | - | {t['false_alarms']} | {_pct(t['false_alarm_rate'])} | {t['at_start']} | "
        f"{t['at_k']} | {t['at_close']} |",
        f"| off | {o['n']} | - | - | - | - | - | {o['false_alarms']} | {_pct(o['false_alarm_rate'])} | {o['at_start']} | "
        f"{o['at_k']} | {o['at_close']} |", "",
        f"- Crossings that did not occur: {', '.join(f'{why} {n}' for why, n in e['not_occurred'].items())}.",
        f"- Crossings not on token k (expected 0; live arms: excluded under `crossing_off_k`): {e['crossing_off_k']}.",
        f"- Twins that share their first k tokens with the event run: {t['same_prefix']} / {t['same_prefix_of']}.",
        f"- Onset index of the hits, definition (b): {e['onset']['b']}; at most 1: {e['onset']['b_at_most_1']} / {e['hits']}. "
        f"(c): {e['onset']['c']}; (a): {e['onset']['a']}. At the last keyword before the reaction's marker (a second "
        f"reading, not a line): {e['onset']['last']}; at most 1: {e['onset']['last_at_most_1']} / {e['hits']}.",
        "- Turns that ended and name the deadline in their last 128 tokens (keyword match only, not a counted reaction; a "
        "deadline reaction that starts at none of the anchors is not seen by the frozen classifier): "
        f"event runs {s['deadline_near_end']['event']} (of them not hits: {s['deadline_near_end']['event_not_hit']}), twin runs "
        f"{s['deadline_near_end']['twin']}, off runs {s['deadline_near_end']['off']}.",
        f"- Delay (tokens until the signal is named): median {e['delay_median']}, 90th percentile {e['delay_p90']}.",
        f"- `time`, crossings that occurred: {e['time_practical']['occurred']}; strict rule {e['time_practical']['strict']}; "
        f"practical reading (deadline named, turn ended inside the horizon) {e['time_practical']['practical']}.", "",
        "| Signal | Runs | Occurred | Hits |", "|---|---|---|---|",
        *[f"| {name} | {v['planned']} | {v['occurred']} | {v['hits']} |" for name, v in e["by_signal"].items()], "",
        "| Family | Runs | Occurred | Hits |", "|---|---|---|---|",
        *[f"| {name} | {v['planned']} | {v['occurred']} | {v['hits']} |" for name, v in e["by_family"].items()], "",
    ]
    (out_dir / "stream_summary.md").write_bytes("\n".join(lines).encode("utf-8"))
    return s
