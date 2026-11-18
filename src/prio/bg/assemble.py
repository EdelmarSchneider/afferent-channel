"""Assemble background-task episodes from phase-1 base runs (spec section 6.4, approach A).

An episode is one token stream: the user's request, then assistant turns alternating with the
harness's tool answers (masked from the loss). The other work is a base run of the unmodified
model, kept token for token, so the resumption after an event has an exact reference.
"""

from dataclasses import dataclass, replace

import numpy as np

from prio.bg.curves import progress_at, sample_curve
from prio.bg.kinds import BgTask
from prio.bg.protocol import DONE_PROGRESS, FIRST_COMMANDS, N_INPUTS, STATE_VALUE, slot_columns
from prio.bg.sentences import (bridge, closing_line, reaction_sentence, report_line, request_text,
                               waiting_answer_sentence)
from prio.datagen.build import BaseRun
from prio.datagen.episodes import Episode
from prio.modelspec import ModelSpec

_OPENING = {
    1: ["I will start the task and watch it, then work on the question.",
        "First I start the task and ask to watch it; then I turn to the question."],
    2: ["I will start the tasks and watch them, then work on the question.",
        "First I start the tasks and ask to watch them; then I turn to the question."],
}


@dataclass
class BgEvent:
    kind: str  # "done" | "error" | "waiting" | "stalled"
    task_index: int  # 0-based into BgPlan.tasks
    cut: int  # index into the base run's target_ids: last token of the sentence after which the event lands
    phase: str  # "thinking" | "answer"


@dataclass
class BgPlan:
    episode_id: str
    run: BaseRun
    split: str
    tasks: list[BgTask]
    policy: str  # "yes" | "no": the answer the request gives for "replace a file?"
    event: BgEvent | None
    twin: bool = False  # no event, but `event` kept for its cut
    near_miss: bool = False
    seed: int = 0
    opening: bool = False  # opening probe: prompt only, the model must start and watch the tasks itself
    slot_of: str | None = None  # slot twin: id of the event plan whose event this one moves to another task


def _rng(plan: BgPlan, stream: int) -> np.random.Generator:
    """Independent random streams per part of the episode, so that e.g. the request text is the
    same whatever event the episode carries (spec anti-goal: the text must not announce the event)."""
    return np.random.default_rng([plan.seed, stream])


class _Stream:
    """target_ids under construction, with the spans to mask and to weight."""

    def __init__(self):
        self.ids: list[int] = []
        self.harness: list[list[int]] = []
        self.focus: list[list[int]] = []

    def add(self, ids: list[int], harness: bool = False, focus: bool = False) -> int:
        start = len(self.ids)
        self.ids += ids
        if harness:
            self.harness.append([start, len(self.ids)])
        if focus:
            self.focus.append([start, len(self.ids)])
        return start


def _opening(plan: BgPlan, fmt, out: _Stream) -> None:
    """Start every task, then watch every task id (slots answered in order)."""
    rng = _rng(plan, 1)
    n = len(plan.tasks)
    out.add(fmt.encode(fmt.think(_OPENING[min(n, 2)][int(rng.integers(2))])
                       + fmt.calls_text([("start_task", {"description": t.describe()}) for t in plan.tasks])))
    out.add(fmt.turn_end)
    out.add(fmt.segment([{"task_id": t.task_id} for t in plan.tasks]), harness=True)
    out.add(fmt.encode(fmt.EMPTY_THINK + fmt.calls_text([("watch", {"task_id": t.task_id}) for t in plan.tasks])))
    out.add(fmt.turn_end)
    out.add(fmt.segment([{"slot": k} for k in range(1, n + 1)]), harness=True)


def _closing(plan: BgPlan, fmt, out: _Stream, running: list[BgTask]) -> None:
    if running:
        ids = fmt.encode("\n\n" + closing_line(running, _rng(plan, 5)))  # its own paragraph after the answer
        start = out.add(ids)
        out.focus.append([start, start + 1])  # whether a task still runs is read from the channel
    out.add(fmt.turn_end)


PAYLOADS = {  # the harness's answers to the first round of each reaction (spec 6.1)
    "done": lambda t: [{"ok": True}, {"result": t.result}],
    "error": lambda t: [{"state": "error", "reason": t.reason}, {"ok": True}],
    "waiting": lambda t: [{"state": "waiting", "question": t.question}],
    "stalled": lambda t: [{"state": "running", "note": "no progress for a while"}],
}
STALL_HOLD = 64  # tokens the progress of a stalled task stays flat after its status was checked


def first_difference(event: Episode, twin: Episode) -> int:
    """First token after the event where the two texts part. The reaction and the base continuation
    often share their first token (" The"); the decision is the next one."""
    d = event.t_cross + 1
    while d < min(len(event.target_ids), len(twin.target_ids)) and event.target_ids[d] == twin.target_ids[d]:
        d += 1
    return d


def align_divergence(event: Episode, twin: Episode) -> int:
    """Set both episodes' t_diverge to the first token where their texts really part."""
    event.t_diverge = twin.t_diverge = first_difference(event, twin)
    return event.t_diverge


def twin_plan(plan: BgPlan, near_miss: bool = False) -> BgPlan:
    """The no-event twin of an event plan: same text up to the event, original continuation."""
    return BgPlan(episode_id=plan.episode_id + "-twin", run=plan.run, split=plan.split, tasks=plan.tasks,
                  policy=plan.policy, event=plan.event, twin=True, near_miss=near_miss, seed=plan.seed)


def slot_plan(plan: BgPlan) -> BgPlan:
    """The slot twin of an event plan with 2-3 tasks: same text, curves and channel up to the event,
    the event on another task (drawn from the plan's own seed). Only the channel tells which task
    to name, as only the channel tells the event from its no-event twin."""
    others = [i for i in range(len(plan.tasks)) if i != plan.event.task_index]
    other = others[int(_rng(plan, 7).integers(len(others)))]
    return BgPlan(episode_id=plan.episode_id + "-slot", run=plan.run, split=plan.split, tasks=plan.tasks,
                  policy=plan.policy, event=replace(plan.event, task_index=other), seed=plan.seed,
                  slot_of=plan.episode_id)


def _event_body(plan: BgPlan, fmt, tok, ms: ModelSpec, out: _Stream, base: list[int], watch_end: int) -> dict:
    """Base text up to the cut, the reaction rounds, the report, the bridge and the rest of the base
    text. Returns positions (indices into target_ids) for the signals and the metadata."""
    ev, task = plan.event, plan.tasks[plan.event.task_index]
    rng = _rng(plan, 3)
    out.add(base[:ev.cut + 1])
    info = {"t_cross": watch_end + ev.cut}
    rest = base[ev.cut + 1:]
    at_line_end = tok.decode(base[ev.cut:ev.cut + 1]).endswith("\n")
    lead = "" if at_line_end else " "  # a reaction after a line break starts the new line
    # after the bridge: back on a new line if the cut was one, else one space unless the rest has its own
    after_bridge = "\n" if at_line_end else ("" if tok.decode(rest[:1]).startswith((" ", "\n")) else " ")
    sentence = reaction_sentence(ev.kind, task, rng)
    calls = fmt.calls_text([(name, {"task_id": task.task_id}) for name in FIRST_COMMANDS[ev.kind]])
    joint = "\n" if ev.phase == "answer" else fmt.THINK_CLOSE
    out.add(fmt.encode(lead + sentence + joint + calls) + fmt.turn_end, focus=True)
    out.add(fmt.segment(PAYLOADS[ev.kind](task)), harness=True)
    info["first_answer_end"] = len(out.ids)
    if ev.kind == "waiting":
        call = fmt.calls_text([("answer_task", {"task_id": task.task_id, "answer": plan.policy})])
        out.add(fmt.encode(fmt.EMPTY_THINK + waiting_answer_sentence(task, plan.policy, rng) + "\n" + call)
                + fmt.turn_end, focus=True)
        out.add(fmt.segment([{"ok": True}]), harness=True)
        info["second_answer_end"] = len(out.ids)
    report = report_line(ev.kind, task, _rng(plan, 4), answer=plan.policy)
    if ev.phase == "answer":
        out.add(fmt.encode(fmt.EMPTY_THINK))
        start = out.add(fmt.encode(report), focus=True)
        info["report_span"] = [start, len(out.ids)]
        out.add(fmt.encode(" " + bridge("answer", rng) + after_bridge))
        info["resume_start"] = len(out.ids)
        out.add(rest)
    else:  # the thinking reopens with the bridge; the user is told at the start of the answer
        out.add(fmt.encode(fmt.THINK_OPEN + bridge("thinking", rng) + after_bridge))
        info["resume_start"] = len(out.ids)
        close = rest.index(ms.think_close) + 1
        if close < len(rest) and tok.decode(rest[close:close + 1]).strip() == "":
            close += 1  # keep the base run's own "\n\n" after </think> before the report
        out.add(rest[:close])
        start = out.add(fmt.encode(report + "\n\n"), focus=True)
        info["report_span"] = [start, len(out.ids)]
        out.add(rest[close:])
    return info


def _signals(plan: BgPlan, n_prompt: int, length: int, watch_end: int, base_len: int,
             info: dict | None) -> tuple[np.ndarray, list[dict]]:
    """Channel values for every token. Curves reach their cap at the end of the base text, which an
    event episode and its twin share, so the two carry the same values up to the event row."""
    rng = _rng(plan, 2)
    start, total = n_prompt + watch_end, n_prompt + length
    curves = [sample_curve(rng, start=start, end_hint=start + base_len) for _ in plan.tasks]
    if plan.twin and plan.near_miss:  # the event task gets close to the end at the event, never finishes
        i = plan.event.task_index
        curves[i] = sample_curve(_rng(plan, 6), start=start, end_hint=n_prompt + watch_end + plan.event.cut,
                                 near_miss=True)
    values = np.zeros((total, N_INPUTS), dtype=np.float32)
    for k in range(1, 4):
        values[:, slot_columns(k)[1]] = STATE_VALUE["off"]
    for k, curve in enumerate(curves, start=1):
        p, s = slot_columns(k)
        values[start:, s] = STATE_VALUE["running"]
        values[start:, p] = [progress_at(curve, pos) for pos in range(start, total)]
    if info is None or plan.twin:
        return values, curves
    kind, curve = plan.event.kind, curves[plan.event.task_index]
    p, s = slot_columns(plan.event.task_index + 1)
    e = n_prompt + info["t_cross"]
    frozen = progress_at(curve, e - 1)
    values[e:, s] = STATE_VALUE[kind]
    values[e:, p] = DONE_PROGRESS if kind == "done" else frozen

    def resume(at: int) -> None:  # running again, progress continues where it stopped
        values[at:, s] = STATE_VALUE["running"]
        values[at:, p] = [max(frozen, progress_at(curve, pos - (at - e))) for pos in range(at, total)]

    first = n_prompt + info["first_answer_end"]
    if kind in ("done", "error"):  # unwatched: the slot is free from the next token
        values[first:, s] = STATE_VALUE["off"]
        values[first:, p] = 0.0
    elif kind == "waiting":
        resume(n_prompt + info["second_answer_end"])
    else:  # stalled: status checked, the state is back to running, progress flat a while longer
        values[first:, s] = STATE_VALUE["running"]
        resume(first + STALL_HOLD)
    return values, curves


def _opening_probe(plan: BgPlan, prompt: list[int]) -> Episode:
    values = np.zeros((len(prompt), N_INPUTS), dtype=np.float32)
    for k in range(1, 4):
        values[:, slot_columns(k)[1]] = STATE_VALUE["off"]
    run = plan.run
    meta = {"tasks": [t.to_dict() for t in plan.tasks], "policy": plan.policy, "event": None,
            "family": run.task.family, "answer": run.task.answer, "base_task_id": run.task.task_id,
            "near_miss": False, "twin_of": None, "held_out": any(t.held_out for t in plan.tasks)}
    return Episode(
        episode_id=plan.episode_id, task_id=run.task.task_id, family=run.task.family, split=plan.split,
        kind="bg_opening", signal=None, phase=None, prompt_ids=prompt, target_ids=[], signals=values,
        t_cross=None, t_decision=None, templated=False, meta=meta)


def assemble(plan: BgPlan, fmt, tok, ms: ModelSpec) -> Episode:
    run, ev = plan.run, plan.event
    prompt = fmt.prompt(request_text(plan.tasks, run.task.prompt, plan.policy, _rng(plan, 0)))
    if plan.opening:
        return _opening_probe(plan, prompt)
    out = _Stream()
    _opening(plan, fmt, out)
    watch_end = len(out.ids)
    base = run.target_ids[:-1]
    info, running = None, list(plan.tasks)
    if ev is not None and not plan.twin:
        info = _event_body(plan, fmt, tok, ms, out, base, watch_end)
        if ev.kind in ("done", "error"):
            running = [t for i, t in enumerate(plan.tasks) if i != ev.task_index]
    else:
        out.add(base)
    _closing(plan, fmt, out, running)
    signals, curves = _signals(plan, len(prompt), len(out.ids), watch_end, len(base), info)
    t_cross = None if ev is None else watch_end + ev.cut
    meta = {
        "tasks": [t.to_dict() | {"slot": k, "curve": c} for k, (t, c) in enumerate(zip(plan.tasks, curves), start=1)],
        "policy": plan.policy, "event": None, "family": run.task.family, "answer": run.task.answer,
        "base_task_id": run.task.task_id, "near_miss": plan.near_miss, "twin_of": None, "slot_of": plan.slot_of,
        "watch_end": watch_end, "resume_start": watch_end, "held_out": any(t.held_out for t in plan.tasks),
    }
    if ev is not None:
        meta["cut_position"] = len(prompt) + t_cross
        meta["resume_start"] = info["resume_start"] if info else t_cross + 1
    if info:
        meta["event"] = {"kind": ev.kind, "task_index": ev.task_index, "slot": ev.task_index + 1,
                         "position": len(prompt) + t_cross}
        meta["report_span"] = info["report_span"]
        meta["answer_ends"] = [info["first_answer_end"]] + ([info["second_answer_end"]] if "second_answer_end" in info else [])
    if plan.twin:
        meta["twin_of"] = plan.episode_id.removesuffix("-twin")
    kind = "bg_nominal" if ev is None else ("bg_twin" if plan.twin else "bg_event")
    return Episode(
        episode_id=plan.episode_id, task_id=run.task.task_id, family=run.task.family, split=plan.split,
        kind=kind, signal=ev.kind if ev else None, phase=ev.phase if ev else None, prompt_ids=prompt,
        target_ids=out.ids, signals=signals, t_cross=t_cross, t_decision=None, templated=kind == "bg_event",
        t_diverge=None if t_cross is None else t_cross + 1, harness_spans=out.harness, focus_spans=out.focus,
        meta=meta)
