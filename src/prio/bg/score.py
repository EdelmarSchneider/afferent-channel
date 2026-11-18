"""Scoring of one evaluation run of the background-task model (spec section 6.6).

A reaction counts as in phase 1: a sentence that names the task, then the action (the commands),
starting within the reaction window. Perception (reacting, naming the right task) and wording
(copying the right fact into the report) are scored apart, so a copying slip of the small model
does not count as a missed perception.
"""

import re
from dataclasses import dataclass

from prio.bg.protocol import FIRST_COMMANDS, REACTION_TOOLS
from prio.datagen.episodes import Episode
from prio.eval.run import extract_number
from prio.reactions import ToolCall

WINDOW = 48  # tokens after the event within which the naming must come
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_ALARM = re.compile(r"\b(finished|failed|waiting for an answer|stuck)\b", re.IGNORECASE)


@dataclass
class BgRun:
    """What happened after the forced prefix: model and harness tokens in order. `rounds` holds the
    tool calls of each model segment, with token indices relative to that segment."""
    token_ids: list[int]
    segments: list[tuple[str, int, int]]  # ("model" | "harness", start, end) in token_ids
    rounds: list[list[ToolCall]]
    stop: str  # "eos" | "budget"


@dataclass
class BgRecord:
    episode_id: str
    kind: str  # "event" | "twin" | "opening"
    condition: str  # "on" | "zeroed" | "swap" | "base"
    event_kind: str | None = None
    phase: str | None = None
    held_out: bool | None = None
    n_watched: int | None = None
    near_miss: bool | None = None
    family: str | None = None
    reacted: bool | None = None  # naming sentence + the right command names (spec 6.6)
    reacted_loose: bool | None = None  # naming sentence + any reaction command
    named_task: str | None = None
    right_task: bool | None = None
    commands_ok: bool | None = None
    delay: int | None = None
    unnamed_call: bool | None = None
    false_alarm: bool | None = None
    report_ok: bool | None = None
    closing_ok: bool | None = None
    resumed: bool | None = None
    math_correct: bool | None = None
    opening_started: int | None = None
    opening_watched: int | None = None


def _tid(call: ToolCall) -> str | None:
    """A call's task id as a string (the model may write a list or a number)."""
    value = call.arguments.get("task_id")
    return value if isinstance(value, str) or value is None else str(value)


def named_tasks(text: str, tasks: list[dict]) -> list[str]:
    """Task ids in order of first mention; a task is mentioned by its head noun or its kind noun."""
    first = {}
    for t in tasks:
        words = [w for w in (t.get("head"), t.get("noun")) if w]
        hits = [m.start() for w in words for m in re.finditer(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)]
        if hits:
            first[t["task_id"]] = min(hits)
    return sorted(first, key=first.get)


def _model_segments(run: BgRun) -> list[tuple[list[int], list[ToolCall]]]:
    models = [(a, b) for kind, a, b in run.segments if kind == "model"]
    return [(run.token_ids[a:b], calls) for (a, b), calls in zip(models, run.rounds)]


def _after_first_harness(run: BgRun, tok) -> str | None:
    seen = False
    texts = []
    for kind, a, b in run.segments:
        if kind == "harness":
            seen = True
        elif seen:
            texts.append(tok.decode(run.token_ids[a:b]))
    return " ".join(texts) if seen else None


def _visible(text: str) -> str:
    return _THINK.sub(" ", text)


def _base(ep: Episode, condition: str, kind: str) -> BgRecord:
    meta = ep.meta
    return BgRecord(episode_id=ep.episode_id, kind=kind, condition=condition, event_kind=ep.signal, phase=ep.phase,
                    held_out=meta.get("held_out"), n_watched=len(meta["tasks"]), near_miss=meta.get("near_miss"),
                    family=meta.get("family"))


def _fact_found(text: str, kind: str, task: dict, policy: str) -> bool | None:
    if kind == "done":
        value = task.get("detail_value")
        if not value:
            return None
        return value in text or value.replace(",", "") in text
    if kind == "error":
        return task["reason"] in text
    if kind == "waiting":
        return re.search(rf"\b{policy}\b", text, re.IGNORECASE) is not None
    return None


def _closing_ok(run: BgRun, ep: Episode, tok) -> bool | None:
    if run.stop != "eos":
        return None
    unwatched = {_tid(c) for calls in run.rounds for c in calls if c.name == "unwatch"}
    running = {t["task_id"] for t in ep.meta["tasks"] if t["task_id"] not in unwatched}
    last_text = _visible(tok.decode(_model_segments(run)[-1][0]))
    sentences = [s for s in re.split(r"[.!?;]", last_text) if "still running" in s]
    named = {i for s in sentences for i in named_tasks(s, ep.meta["tasks"])}
    return named == running if running else not sentences


def score_event(run: BgRun, ep: Episode, condition: str, tok, ms, expected_task: str) -> BgRecord:
    r = _base(ep, condition, "event")
    kind, tasks = ep.signal, ep.meta["tasks"]
    segments = _model_segments(run)
    first_ids, first_calls = segments[0] if segments else ([], [])
    reaction_calls = [c for c in first_calls if c.name in REACTION_TOOLS]
    end = reaction_calls[0].token_index if reaction_calls else len(first_ids)
    mention = next((k for k in range(min(end, WINDOW + 1))
                    if named_tasks(tok.decode(first_ids[:k + 1]), tasks)), None)
    want = list(FIRST_COMMANDS[kind])
    named = mention is not None
    r.reacted_loose = bool(reaction_calls) and named
    r.reacted = named and [c.name for c in first_calls] == want  # spec 6.6: naming sentence + right commands
    # a reaction command with no task named before it, in the reaction round (a later round, e.g. the answer
    # to a waiting task, follows the harness's answer and needs no new naming)
    r.unnamed_call = bool(reaction_calls) and not named_tasks(tok.decode(first_ids[:end]), tasks)
    if r.reacted_loose:
        r.delay = mention
        r.named_task = named_tasks(tok.decode(first_ids[:end]), tasks)[0]
        r.right_task = r.named_task == expected_task
    r.commands_ok = bool(first_calls) and [c.name for c in first_calls] == want \
        and all(_tid(c) == expected_task for c in first_calls)
    if kind == "waiting" and r.commands_ok:
        second = segments[1][1] if len(segments) > 1 else []
        r.commands_ok = [(c.name, _tid(c), c.arguments.get("answer")) for c in second] == [
            ("answer_task", expected_task, ep.meta["policy"])]
    after = _after_first_harness(run, tok)
    task = next(t for t in tasks if t["task_id"] == expected_task)
    if after is not None:
        r.report_ok = _fact_found(_visible(after), kind, task, ep.meta["policy"])
    r.closing_ok = _closing_ok(run, ep, tok)
    r.resumed, r.math_correct = score_resumption(run, ep, tok)
    return r


def score_twin(run: BgRun, ep: Episode, condition: str, tok, ms) -> BgRecord:
    r = _base(ep, condition, "twin")
    called = any(c.name in REACTION_TOOLS for calls in run.rounds for c in calls)
    text = " ".join(tok.decode(ids) for ids, _ in _model_segments(run))
    said = any(_ALARM.search(s) and named_tasks(s, ep.meta["tasks"]) for s in re.split(r"[.!?;]", text))
    r.false_alarm = called or said
    r.resumed, r.math_correct = score_resumption(run, ep, tok)
    return r


def _ngrams(words: list[str], n: int = 8) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def score_resumption(run: BgRun, ep: Episode, tok) -> tuple[bool | None, bool | None]:
    """(resumed, math_correct). Math: the run's final number against the gold answer. Explanations:
    after the reaction the text does not repeat what came before the event (8-gram overlap < 0.2)
    and the run ends normally."""
    if ep.meta.get("family") == "math":
        text = " ".join(tok.decode(ids) for ids, _ in _model_segments(run))
        if "</think>" not in text and ep.phase == "thinking":
            return None, False  # the thinking never closed: no answer
        number = extract_number(text)
        return None, number is not None and number == float(ep.meta["answer"])
    after = _after_first_harness(run, tok)
    if after is None:
        return None, None
    before = tok.decode(ep.target_ids[ep.meta["watch_end"]:ep.t_cross + 1])
    grams = _ngrams(after.split())
    overlap = len(grams & _ngrams(before.split())) / len(grams) if grams else 0.0
    return overlap < 0.2 and run.stop == "eos", None


def score_opening(run: BgRun, ep: Episode, tok, ms) -> BgRecord:
    r = _base(ep, "on", "opening")
    calls = [c for calls in run.rounds for c in calls]
    descriptions = [str(c.arguments.get("description", "")).lower() for c in calls if c.name == "start_task"]
    r.opening_started = sum(any(t["head"].lower() in d for d in descriptions) for t in ep.meta["tasks"])
    returned = {f"t{i}" for i in range(1, len(descriptions) + 1)}
    r.opening_watched = len({_tid(c) for c in calls if c.name == "watch"} & returned)
    return r
