"""Paired evaluation: every test prefix runs with its own trajectory and with other ones.

Conditions: "on" (trained model, channel on), "zeroed" (trained model, channel off),
"base" (adapter and channel both off: the untouched base model).

Pairings (which trajectory a run receives):
  "event"      the episode's stored trajectory;
  "nominal"    a fresh one with every signal well away from its limit;
  "near_miss"  a fresh one where one or two signals come close to their limit without passing;
  "swapped"    the swap control: another interrupt signal crosses its limit at the same cut, so a
               reaction must follow the injected signal, not the text.
"""

import argparse
import contextlib
import functools
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import torch

from prio.datagen.episodes import Episode, read_jsonl
from prio.eval.metrics import RunRecord, bootstrap_ci, summarize, wilson_ci
from prio.generate import GenResult, generate, generate_batch
from prio.reactions import (
    REACTION_WINDOW,
    TIME_END_WINDOW,
    counts,
    find_reaction,
    first_reaction,
    names_signal,
    parse_tool_calls,
    unnamed_call,
)
from prio.signals import SignalSpec, signal_set
from prio.tasks import Task, build_prompt, load_tasks
from prio.train import channel_record, load_run
from prio.trajectories import sample_trajectory

TIME_HORIZON = TIME_END_WINDOW + 32  # a `time` reaction is judged by where the turn ends
REACTION_HORIZON = REACTION_WINDOW + 48  # a reaction that starts inside the window must also fit
GATING_HORIZON = 64  # the decision is the very next thing the model writes
_GATED = {"net": ("web_search", "local_lookup"), "ram": ("run_heavy", "run_light")}
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_AT_DECISION = ("gating_event", "recovery")  # the forced prefix runs up to the tool decision
_NEVER_EVENT = ("recovery", "gating_unused")  # the stored trajectory asks for no reaction
_PAIRED_KINDS = ("event", "gating_event")  # the main kinds: these also run with the other pairings


def extract_number(text: str) -> float | None:
    """Last number in the answer part of a response (after the thinking block, if any)."""
    found = _NUMBER.findall(text.split("</think>")[-1])
    return float(found[-1].replace(",", "")) if found else None


def _named_at(token_ids: list[int], spec_name: str, tok) -> int | None:
    """Index of the token that completes the first mention of the signal: the moment the model
    says what it noticed."""
    for k in range(len(token_ids)):
        if names_signal(tok.decode(token_ids[:k + 1]), spec_name):
            return k
    return None


def classify(ep: Episode, paired: str, condition: str, gen: GenResult, tok, ms, specs: Sequence[SignalSpec]) -> RunRecord:
    """Turn one generation into a record. `gen` starts at the first token the model could react in.

    A reaction is the naming sentence plus the action (`counts`). In a run with an event, the
    reaction to the episode's signal must start inside REACTION_WINDOW. In a run with nothing to
    react to, any reaction anywhere is a false alarm.
    """
    event_occurred = paired in ("event", "swapped") and ep.kind not in _NEVER_EVENT
    if ep.signal == "think" and ms.think_close in ep.target_ids[:ep.t_cross + 1]:
        event_occurred = False  # thinking was already over when the budget ran out: nothing to react to

    own = None
    if event_occurred:
        own = find_reaction(gen.token_ids, gen.stop_reason, ep.signal, tok, ms)
        if own is not None and not (counts(own) and own.token_index < REACTION_WINDOW):
            own = None
        others = [s for s in specs if s.name != ep.signal]
        reaction = own or first_reaction(gen.token_ids, gen.stop_reason, others, tok, ms)
    else:
        reaction = first_reaction(gen.token_ids, gen.stop_reason, specs, tok, ms)

    gated, alternative = _GATED.get(ep.signal, (None, None))
    calls = parse_tool_calls(gen.token_ids, tok, ms)
    first_call = calls[0].name if calls else None
    at_decision = ep.kind in _AT_DECISION
    return RunRecord(
        episode_id=ep.episode_id, signal=ep.signal, kind=ep.kind, phase=ep.phase, condition=condition, paired=paired,
        event_occurred=event_occurred,
        reaction_signal=reaction.signal if reaction else None,
        delay=_named_at(gen.token_ids, ep.signal, tok) if own else None,
        named=bool(reaction and reaction.named),
        called_gated_tool=at_decision and first_call == gated,
        used_alternative=at_decision and first_call == alternative,
        unnamed_call=unnamed_call(gen.token_ids, gen.stop_reason, specs, tok, ms),
    )


def spread(items: list, n: int) -> list:
    """`n` items at even spacing over the whole list (all of them when there are fewer)."""
    if n >= len(items):
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def stop_ids(tok, ms) -> set[int]:
    """End-of-turn tokens plus the padding token: the model may also end with <|endoftext|> (= pad)."""
    pad = getattr(tok, "pad_token_id", None)
    return set(ms.eos_ids) | ({pad} if pad is not None else set())


def _rows(values: np.ndarray, length: int) -> np.ndarray:
    """First `length` rows, repeating the last row if the stored trajectory is shorter."""
    if len(values) >= length:
        return values[:length]
    return np.concatenate([values, np.repeat(values[-1:], length - len(values), axis=0)])


def _think_index(specs: Sequence[SignalSpec]) -> int | None:
    return next((i for i, s in enumerate(specs) if s.name == "think"), None)


def follow_thinking(values: np.ndarray, n_in: int, think: int | None, nominal: float, think_close: int):
    """Signal rows for the generated tokens, with `think` tied to what the model really writes.

    While thinking is open a token gets the stored budget. Once the model has written </think>,
    every later token gets the nominal value; the </think> token itself is the last one to carry
    the budget, as in training. `think=None` (no such signal, or thinking already over in the
    forced prefix) returns the rows unchanged.
    """
    closed = False

    def signal_fn(step: int, generated: list[int]) -> np.ndarray:
        nonlocal closed
        row = values[min(n_in + step, len(values) - 1)]
        if think is None:
            return row
        if closed:
            row = row.copy()
            row[think] = nominal
        elif generated[-1] == think_close:
            closed = True
        return row

    return signal_fn


def _stored_trajectory(ep: Episode, total: int, think: int | None, nominal: float, closed_at: int | None, ms) -> np.ndarray:
    """The episode's own trajectory over `total` rows, with `think` made to follow the text.

    Thinking over inside the forced prefix (`closed_at`): nominal from the token after that
    </think>. Thinking still open: the budget goes on being reported (`follow_thinking` ends it
    when the model closes). Older test files store nominal after the BASE RUN's </think>, which
    the model may never write at that point; there the last budget value is held instead.
    """
    values = _rows(ep.signals, total).copy()
    if think is None:
        return values
    n_prompt = len(ep.prompt_ids)
    if closed_at is not None:
        values[n_prompt + closed_at + 1:, think] = nominal
    elif ms.think_close in ep.target_ids:
        row = n_prompt + ep.target_ids.index(ms.think_close)
        if row + 1 < total and (values[row + 1:, think] == nominal).all():
            values[row + 1:, think] = values[row, think]
    return values


def _swap_signal(ep: Episode, specs: Sequence[SignalSpec], index: int, thinking_open: bool) -> str | None:
    """Another interrupt signal to cross its limit at this episode's cut (the swap control)."""
    names = [s.name for s in specs
             if s.kind == "interrupt" and s.name != ep.signal and (s.name != "think" or thinking_open)]
    return names[index % len(names)] if names else None


_HORIZON_OVERRIDE: int | None = None  # set by `fixed_horizon` (--horizon)


def _horizon(at_decision: bool, signal: str) -> int:
    if _HORIZON_OVERRIDE is not None:
        return _HORIZON_OVERRIDE
    return GATING_HORIZON if at_decision else TIME_HORIZON if signal == "time" else REACTION_HORIZON


@contextlib.contextmanager
def _condition(model, condition: str):
    """Channel on for "on"; off for "zeroed"; channel and adapter off for "base"."""
    model.enabled = condition == "on"
    try:
        if condition == "base" and hasattr(model.base, "disable_adapter"):
            with model.base.disable_adapter():
                yield
        else:
            yield
    finally:
        model.enabled = True


def prefix_harness(ep: Episode, n_forced: int, extra: int = 0) -> list[bool] | None:
    """Harness mask of a run whose prompt is ep.prompt_ids + ep.target_ids[:n_forced] (+ `extra`
    ids the harness appends): true at the tokens of the episode's harness spans inside the forced
    prefix (inserted notices, WP3 contract) and at the appended ids. None when nothing is marked.
    Text anyone else wrote (a spoofed notice in the prompt) is never a harness span."""
    n = len(ep.prompt_ids)
    mask = [False] * (n + n_forced) + [True] * extra
    for a, b in ep.harness_spans or ():
        for i in range(a, min(b, n_forced)):
            mask[n + i] = True
    return mask if any(mask) else None


def _run(model, prompt: list[int], values: np.ndarray, horizon: int, eos_ids, signal_fn,
         harness: list[bool] | None = None) -> GenResult:
    """`harness`: the prompt's harness mask (`prefix_harness`), read by the segment delivery."""
    device = model.projector.net[0].weight.device
    n_in = len(prompt)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return generate(
            model, torch.tensor([prompt], device=device), torch.from_numpy(values[None, :n_in]).to(device),
            signal_fn, horizon, eos_ids, prompt_harness=harness,
        )


def _run_batch(model, prompts: list[list[int]], values: list[np.ndarray], horizons: list[int], eos_ids,
               signal_fns: list, harness: list | None = None) -> list[GenResult]:
    """`_run` for several runs at once (same condition)."""
    device = model.projector.net[0].weight.device
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return generate_batch(model, prompts, [torch.from_numpy(v[:len(p)]) for p, v in zip(prompts, values)],
                              signal_fns, horizons, eos_ids, prompt_harness=harness)


# Pending runs per condition before a flush: sorted by prompt length, cut into batches (less padding).
POOL_BATCHES = 4


class _Runs:
    """The generation runs of one evaluation. `batch_size` 1: each run when it is added, through
    `_run`. More: runs pooled per condition; a pool of POOL_BATCHES batches (and every pool at
    `finish`) is sorted by prompt length and cut into `generate_batch` calls. `done(gen)` of each
    run receives its generation (records, files)."""

    def __init__(self, model, stops, batch_size: int = 1):
        self.model, self.stops, self.batch_size = model, stops, batch_size
        self.pending: dict[str, list[tuple]] = defaultdict(list)

    def add(self, condition: str, prompt: list[int], values: np.ndarray, horizon: int, signal_fn,
            harness: list[bool] | None, done) -> None:
        if self.batch_size <= 1:
            with _condition(self.model, condition):
                gen = _run(self.model, prompt, values, horizon, self.stops, signal_fn, harness)
            done(gen)
            return
        self.pending[condition].append((prompt, values, horizon, signal_fn, harness, done))
        if len(self.pending[condition]) >= POOL_BATCHES * self.batch_size:
            self.flush(condition)

    def flush(self, condition: str) -> None:
        jobs = sorted(self.pending.pop(condition, []), key=lambda job: len(job[0]))
        for start in range(0, len(jobs), self.batch_size):
            chunk = jobs[start:start + self.batch_size]
            harness = [j[4] for j in chunk]
            with _condition(self.model, condition):
                gens = _run_batch(self.model, [j[0] for j in chunk], [j[1] for j in chunk], [j[2] for j in chunk],
                                  self.stops, [j[3] for j in chunk], harness if any(h is not None for h in harness) else None)
            for job, gen in zip(chunk, gens):
                job[5](gen)

    def finish(self) -> None:
        for condition in list(self.pending):
            self.flush(condition)


def _save(out_path: Path | None, record: RunRecord, condition: str, paired: str, signal: str, gen: GenResult) -> None:
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    # What the model wrote, for reading by hand and for re-classifying without a rerun.
    written = {"episode_id": record.episode_id, "condition": condition, "paired": paired,
               "signal": signal, "stop_reason": gen.stop_reason,
               "token_ids": [int(t) for t in gen.token_ids]}
    with open(out_path.with_name("generations.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(written) + "\n")


def quick_subset(test_eps: Sequence, every: int) -> list[int]:
    """Indices of every n-th episode within each (kind, signal), in file order.

    A plain stride aliases with the file's cycle of signals (stride 4 over ctx, think, energy,
    time picks ctx alone)."""
    seen: dict[tuple[str, str], int] = defaultdict(int)
    picked = []
    for index, ep in enumerate(test_eps):
        key = (ep.kind, ep.signal)
        if seen[key] % every == 0:
            picked.append(index)
        seen[key] += 1
    return picked


class EpisodeSetup(NamedTuple):
    """Per-episode choices of `evaluate` (hook `setup`)."""
    specs: Sequence[SignalSpec]  # limits the fresh trajectories are drawn against and the runs judged with
    cut_kind: str  # the kind that decides the forced prefix (a tool decision or the crossing)
    conditions: Sequence[str]  # the conditions this episode runs under


def evaluate(model, tok, ms, specs: Sequence[SignalSpec], test_eps: list[Episode], conditions: Sequence[str],
             seed: int = 0, log_every: int = 25, out_path: Path | None = None, every: int = 1,
             batch_size: int = 1, setup=None, judge=None) -> list[RunRecord]:
    """Run every (episode, condition, pairing). With `out_path`, each record is appended as soon as
    it exists and runs already in the file are not repeated: an interrupted evaluation resumes.
    The generated token ids go to `generations.jsonl` next to it. `every` = n evaluates only every
    n-th episode, with exactly the runs a full evaluation makes for it (a quick first reading; a
    full evaluation into the same file then adds the rest).

    Forced prefix: target_ids[:t_cross + 1] (the last forced token is the first one that carries
    the past-limit signal), or target_ids[:t_decision] for the kinds cut at a tool decision.
    With the channel off the trajectory cannot matter, so "zeroed" and "base" run "event" only.

    `batch_size` > 1: the runs of one condition go through `generate_batch` that many at a time
    (pools of POOL_BATCHES batches, sorted by prompt length). Same runs and records; the files
    get them in batch order, the returned list in the order of `batch_size` 1.

    Hooks (R10, `prio.eval.limits_eval`): `setup(ep, conditions) -> EpisodeSetup` (default: the
    global `specs`, the episode's own kind, every condition); `judge` replaces `classify` (same
    arguments, the episode's specs). The harness spans inside the forced prefix are marked
    (`prefix_harness`, segment delivery).
    """
    setup = setup or (lambda ep, wanted: EpisodeSetup(specs, ep.kind, wanted))
    judge = judge or classify
    chosen = set(quick_subset(test_eps, every))
    wanted = {test_eps[i].episode_id for i in chosen}
    records: list[RunRecord] = []
    if out_path is not None and out_path.exists():
        with open(out_path, encoding="utf-8") as lines:
            saved = [RunRecord(**json.loads(line)) for line in lines if line.strip()]
        records = [r for r in saved if r.episode_id in wanted and r.condition in conditions]
    done = {(r.episode_id, r.condition, r.paired) for r in records}
    stops = stop_ids(tok, ms)
    started = time.time()
    runs = _Runs(model, stops, batch_size)
    made: list[RunRecord] = []
    rank: dict[tuple[str, str, str], int] = {}  # (episode, condition, pairing) -> its place with batch_size 1

    def finish(judged: Episode, paired: str, condition: str, specs: Sequence[SignalSpec], gen: GenResult) -> None:
        made.append(judge(judged, paired, condition, gen, tok, ms, specs))
        _save(out_path, made[-1], condition, paired, judged.signal, gen)

    for index, ep in enumerate(test_eps):
        if index not in chosen:
            continue
        specs, cut_kind, ep_conditions = setup(ep, conditions)
        names = [s.name for s in specs]
        think = _think_index(specs)
        nominal_think = specs[think].nominal if think is not None else 1.0
        at_decision = cut_kind in _AT_DECISION
        prefix = ep.target_ids[:ep.t_decision if at_decision else ep.t_cross + 1]
        prompt = ep.prompt_ids + prefix
        harness = prefix_harness(ep, len(prefix))
        closed_at = prefix.index(ms.think_close) if ms.think_close in prefix else None
        think_end = None if closed_at is None else len(ep.prompt_ids) + closed_at + 1
        swap = _swap_signal(ep, specs, index, closed_at is None) if ep.kind == "event" else None

        def trajectory(paired: str, total: int) -> np.ndarray:
            if paired == "event":
                return _stored_trajectory(ep, total, think, nominal_think, closed_at, ms)
            if paired == "nominal":
                rng = np.random.default_rng([seed, index])
                return sample_trajectory(specs, total, rng, think_end=think_end).values
            if paired == "near_miss":
                rng = np.random.default_rng([seed, index, 1])
                return sample_trajectory(specs, total, rng, near_miss=True, think_end=think_end).values
            rng = np.random.default_rng([seed, index, 2])  # swapped: the last forced token carries the crossing
            return sample_trajectory(specs, total, rng, event=names.index(swap), t_cross=len(prompt) - 1,
                                     think_end=think_end).values

        for condition in ep_conditions:
            pairings = ["event"]
            if condition == "on" and ep.kind in _PAIRED_KINDS:
                pairings += ["nominal", "near_miss"] + (["swapped"] if swap else [])
            for paired in pairings:
                rank[(ep.episode_id, condition, paired)] = len(rank)
                if (ep.episode_id, condition, paired) in done:
                    continue
                judged = replace(ep, signal=swap, kind="swap") if paired == "swapped" else ep
                horizon = _horizon(at_decision, judged.signal)
                values = trajectory(paired, len(prompt) + horizon)
                signal_fn = follow_thinking(values, len(prompt), think if closed_at is None else None,
                                            nominal_think, ms.think_close)
                runs.add(condition, prompt, values, horizon, signal_fn, harness,
                         functools.partial(finish, judged, paired, condition, specs))
        if log_every and (index + 1) % log_every == 0:
            print(f"evaluated {index + 1}/{len(test_eps)} episodes, {(time.time() - started) / 60:.1f} min", flush=True)
    runs.finish()
    return _in_order(records, made, rank, batch_size)


def _in_order(saved: list, made: list, rank: dict, batch_size: int) -> list:
    """Saved records, then the new ones; batched, both in the order of batch_size 1 (the files hold
    them in batch order)."""
    if batch_size <= 1:
        return saved + made
    return sorted(saved + made, key=lambda r: rank.get((r.episode_id, r.condition, r.paired), len(rank)))


def judge_free_run(token_ids: list[int], stop_reason: str, answer: float, tok, ms, specs: Sequence[SignalSpec]) -> dict:
    """One full generation under nominal signals: the answer, its form, and any reaction in it."""
    text = tok.decode(token_ids)
    return {
        "correct": extract_number(text) == answer,
        # An answer that is a tool call holds no number to check: accuracy then measures the form.
        "text_answer": not text.split("</think>")[-1].strip().startswith("<tool_call>"),
        "false_alarm": first_reaction(token_ids, stop_reason, specs, tok, ms) is not None,
        "unnamed_call": unnamed_call(token_ids, stop_reason, specs, tok, ms),
    }


def preservation_summary(base: list[dict], tuned: list[dict]) -> dict:
    """Base vs. trained model on the same tasks; `base[i]` and `tuned[i]` judge the same task."""
    def rate(rows: list[dict], key: str) -> float:
        return float(np.mean([bool(r[key]) for r in rows]))

    difference = [float(t["correct"]) - float(b["correct"]) for b, t in zip(base, tuned)]
    out = {
        "n": len(base),
        "base_accuracy": rate(base, "correct"), "base_ci": wilson_ci([r["correct"] for r in base]),
        "tuned_nominal_accuracy": rate(tuned, "correct"), "tuned_ci": wilson_ci([r["correct"] for r in tuned]),
        "accuracy_difference": float(np.mean(difference)), "difference_ci": bootstrap_ci(difference),  # paired
    }
    for key in ("text_answer", "false_alarm", "unnamed_call"):
        out[f"base_{key}_rate"], out[f"tuned_{key}_rate"] = rate(base, key), rate(tuned, key)
    return out


def tuned_free_runs(model, tok, ms, specs: Sequence[SignalSpec], prompts: list[list[int]], answers: list[float],
                    seed: int = 0, max_new_tokens: int = 1024, batch_size: int = 1) -> list[dict]:
    """The trained model's full generations under nominal signals, judged (`judge_free_run`).
    `batch_size` > 1: through `generate_batch`, same runs."""
    device = model.projector.net[0].weight.device
    think = _think_index(specs)
    nominal_think = specs[think].nominal if think is not None else 1.0
    stops = stop_ids(tok, ms)
    tuned, jobs = [], []
    for index, (prompt, answer) in enumerate(zip(prompts, answers)):
        values = sample_trajectory(specs, len(prompt) + max_new_tokens, np.random.default_rng([seed, index])).values
        signal_fn = follow_thinking(values, len(prompt), think, nominal_think, ms.think_close)
        if batch_size > 1:
            jobs.append((prompt, values, signal_fn))
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            gen = generate(model, torch.tensor([prompt], device=device),
                           torch.from_numpy(values[None, :len(prompt)]).to(device), signal_fn, max_new_tokens, stops)
        tuned.append(judge_free_run(gen.token_ids, gen.stop_reason, answer, tok, ms, specs))
        if (index + 1) % 20 == 0:
            print(f"preservation {index + 1}/{len(prompts)}", flush=True)
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        gens = _run_batch(model, [j[0] for j in chunk], [j[1] for j in chunk], [max_new_tokens] * len(chunk), stops,
                          [j[2] for j in chunk])
        tuned += [judge_free_run(g.token_ids, g.stop_reason, a, tok, ms, specs) for g, a in zip(gens, answers[start:])]
        print(f"preservation {start + len(chunk)}/{len(prompts)}", flush=True)
    return tuned


def preservation(model, tok, ms, specs: Sequence[SignalSpec], math_tasks: list[Task], n: int = 120, seed: int = 0,
                 batch_size: int = 1) -> dict:
    """GSM8K, full free-running generations: untouched base model vs. the trained model receiving
    nominal signals. Accuracy, share of text answers, and reactions over a whole generation."""
    tasks = math_tasks[:n]
    prompts = [build_prompt(tok, t) for t in tasks]
    answers = [float(t.answer) for t in tasks]
    device = model.projector.net[0].weight.device
    stops = stop_ids(tok, ms)

    base = []
    with _condition(model, "base"):
        for i in range(0, len(prompts), 16):
            chunk = prompts[i:i + 16]
            width = max(map(len, chunk))
            ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in chunk], device=device)
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in chunk], device=device)
            with torch.inference_mode():
                out = model.base.generate(input_ids=ids, attention_mask=mask, max_new_tokens=1024, do_sample=False,
                                          pad_token_id=tok.pad_token_id)
            for row, answer in zip(out[:, width:].tolist(), answers[i:i + 16]):
                end = next((j for j, t in enumerate(row) if t in stops), None)
                written, stop = (row, "max_new_tokens") if end is None else (row[:end + 1], "eos")
                base.append(judge_free_run(written, stop, answer, tok, ms, specs))

    tuned = tuned_free_runs(model, tok, ms, specs, prompts, answers, seed=seed, batch_size=batch_size)
    return preservation_summary(base, tuned)


def _fmt(value, pct: bool = True) -> str:
    if value is None:
        return "-"
    if isinstance(value, (tuple, list)):
        return f"[{value[0]:.1%}, {value[1]:.1%}]"  # one decimal: the control rates sit near 0% and 100%
    return f"{value:.1%}" if pct else f"{value:.0f}"


_NOTES = [
    "- Reaction: the sentence that names the signal plus its action, starting within "
    f"{REACTION_WINDOW} tokens. A reaction tool called without the sentence is listed under "
    "`Unnamed calls` (share of all runs of the row) and is not a reaction.",
    "- Delay: tokens from the first token the model could react in until the signal is named.",
    "- False alarms: any reaction in a run with nothing to react to, anywhere in the run "
    f"({GATING_HORIZON} new tokens at a tool decision, {TIME_HORIZON} for `time`, {REACTION_HORIZON} otherwise). "
    "Near-miss alarms: the same when one or two signals come close to their limit without passing.",
    "- Intervals: 95% Wilson score.",
    "- Rows `x (starts_past)`, `x (recovery)`, `x (gating_unused)`: the other episode kinds. "
    "Rows `x (swap)`: swap control, signal `x` injected into a prefix that was cut for another signal.",
]


def write_summary(records: list[RunRecord], preserved: dict | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")

    # The main kinds keep the bare signal name; every other kind gets rows of its own.
    labels = [r.signal if r.kind in _PAIRED_KINDS else f"{r.signal} ({r.kind})" for r in records]
    family = {"event": "interrupt", "gating_event": "gating"}
    every_phase = ["all"] * len(records)
    by_phase = summarize(records, labels)
    by_signal = summarize(records, labels, every_phase)
    overall = summarize(records, ["all " + family.get(r.kind, r.kind) for r in records], every_phase)
    merged = {**overall, **by_signal, **by_phase}
    payload = {"groups": {"|".join(map(str, k)): v for k, v in merged.items()}, "preservation": preserved}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    lines = ["# Evaluation summary", "", *_NOTES, ""]
    for condition in ("on", "zeroed", "base"):
        rows = [(k, v) for k, v in merged.items() if k[0] == condition]
        if not rows:
            continue
        lines += [f"## Condition: {condition}", "",
                  "| Signal | Phase | n | Reaction | 95% CI | Delay med | Delay p90 | False alarms | 95% CI | n quiet | "
                  "Near-miss alarms | n near | Identification | Unnamed calls | Wrongful call | Wrongful refusal |",
                  "|" + "---|" * 16]
        for (_, signal, phase), v in rows:
            lines.append(
                f"| {signal} | {phase} | {v['n']} | {_fmt(v['reaction_rate'])} | {_fmt(v['reaction_ci'])} | "
                f"{_fmt(v['delay_median'], False)} | {_fmt(v['delay_p90'], False)} | {_fmt(v['false_alarm_rate'])} | "
                f"{_fmt(v['false_alarm_ci'])} | {v['n_nominal']} | {_fmt(v['near_miss_alarm_rate'])} | {v['n_near_miss']} | "
                f"{_fmt(v['identification_rate'])} | {_fmt(v['unnamed_call_rate'])} | {_fmt(v['wrongful_call_rate'])} | "
                f"{_fmt(v['wrongful_refusal_rate'])} |")
        lines.append("")
    if preserved:
        p = preserved
        lines += ["## Preservation (GSM8K, full generations, nominal signals)", "",
                  "| Model | Accuracy | 95% CI | Text answers | Reactions (false alarms) | Unnamed calls |",
                  "|---|---|---|---|---|---|",
                  f"| Base | {p['base_accuracy']:.1%} | {_fmt(p['base_ci'])} | {p['base_text_answer_rate']:.1%} | "
                  f"{p['base_false_alarm_rate']:.1%} | {p['base_unnamed_call_rate']:.1%} |",
                  f"| Trained | {p['tuned_nominal_accuracy']:.1%} | {_fmt(p['tuned_ci'])} | {p['tuned_text_answer_rate']:.1%} | "
                  f"{p['tuned_false_alarm_rate']:.1%} | {p['tuned_unnamed_call_rate']:.1%} |",
                  "",
                  f"- Accuracy difference, trained minus base, same tasks: {p['accuracy_difference']:+.1%} "
                  f"{_fmt(p['difference_ci'])} (paired bootstrap), n = {p['n']}",
                  "- Accuracy checks the last number of the answer; an answer that is a tool call has none, "
                  "so read it next to `Text answers`.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# --- Evaluation modes beyond the standard one ------------------------------------------------
# Harness mode (text-based baselines), harness-rule arms (B1), --horizon,
# base model through the channel loop for preservation (preregistration R11). Kept apart from
# `evaluate` / `_run`, which they only call.

@contextlib.contextmanager
def fixed_horizon(n: int | None):
    """Every run generates `n` new tokens, whatever its kind and signal (--horizon; B1's channel arm
    re-generates with 512 so the whole answer is there to score). None keeps the usual horizons."""
    global _HORIZON_OVERRIDE
    before, _HORIZON_OVERRIDE = _HORIZON_OVERRIDE, n
    try:
        yield
    finally:
        _HORIZON_OVERRIDE = before


def _append_record(out_path: Path | None, record, gen: GenResult | None = None, key: dict | None = None) -> None:
    """Append `record` to `out_path` and, for a generation, what the model wrote to generations.jsonl."""
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    if gen is not None:
        written = {**(key or {}), "stop_reason": gen.stop_reason, "token_ids": [int(t) for t in gen.token_ids]}
        with open(out_path.with_name("generations.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(written) + "\n")


def _saved(out_path: Path | None, cls) -> list:
    if out_path is None or not out_path.exists():
        return []
    with open(out_path, encoding="utf-8") as lines:
        return [cls(**json.loads(line)) for line in lines if line.strip()]


def dataset_tasks(data: Path, manifest: Path | None = None) -> dict[str, Task]:
    """The test tasks of a built dataset by id: `load_tasks` with the build's test count and seed
    (test tasks do not depend on the train count). A variant folder (datagen/variants.py) reads them
    from its `source_manifest`, or else from the manifest of its `source` folder."""
    meta = json.loads(Path(manifest or Path(data) / "manifest.json").read_text(encoding="utf-8"))
    if "tasks_requested" not in meta and "variant" in meta:
        meta = meta.get("source_manifest", {})
        if "tasks_requested" not in meta:
            return dataset_tasks(Path(json.loads(Path(manifest or Path(data) / "manifest.json")
                                                 .read_text(encoding="utf-8"))["source"]))
    return {t.task_id: t for t in load_tasks(0, meta["tasks_requested"]["test"], seed=meta["seed"])}


# Harness mode: the state reaches the model as inserted ids (a text notice, B3; an inserted
# position, B6b). Variant files (datagen/variants.py) differ from the original only there:
# `harness_spans` over the inserted ids, `t_cross` on the last inserted token, `t_decision`
# shifted, signal rows copied. The forced prefix target_ids[:t_cross + 1] / [:t_decision] then
# holds the insertion, exactly as the channel model's prefix holds the crossing.
HARNESS_PAIRINGS = ("event", "quiet", "near_miss", "swapped")
_HARNESS_FILES = {"event": "test.jsonl", "quiet": "test_quiet.jsonl", "near_miss": "test_near_miss.jsonl",
                  "swapped": "test_swap.jsonl"}


def harness_files(data: Path, quiet_data: Path | None = None, overrides: dict[str, Path] | None = None) -> dict[str, Path]:
    """Test file of each pairing. In the variant folder `data`: test.jsonl (event: past-limit
    notice at the cut), test_near_miss.jsonl (below-limit notice), test_swap.jsonl (another
    signal's notice), test_quiet.jsonl. Without a quiet file there, the quiet pairing reads
    `quiet_data`/test.jsonl, the original dataset: same prefix, nothing inserted. A missing
    optional file drops its pairing; `overrides` maps a pairing to any file."""
    data = Path(data)
    found = {name: data / file for name, file in _HARNESS_FILES.items() if (data / file).exists()}
    if "quiet" not in found and quiet_data is not None and (Path(quiet_data) / "test.jsonl").exists():
        found["quiet"] = Path(quiet_data) / "test.jsonl"
    found.update({name: Path(path) for name, path in (overrides or {}).items()})
    found.setdefault("event", data / "test.jsonl")
    return {name: found[name] for name in HARNESS_PAIRINGS if name in found}


def _notice_signal(ep: Episode) -> str:
    """The signal whose notice a swap-file episode carries: `meta['notice_signal']`, else `signal`."""
    return (ep.meta or {}).get("notice_signal") or ep.signal


def evaluate_harness(model, tok, ms, specs: Sequence[SignalSpec], files: dict[str, list[Episode]],
                     conditions: Sequence[str] = ("on",), seed: int = 0, log_every: int = 25,
                     out_path: Path | None = None, every: int = 1, horizon: int | None = None,
                     batch_size: int = 1) -> list[RunRecord]:
    """Harness-mode evaluation (S4) of a model that gets the state in its stream.

    Each main-kind episode of `files["event"]` runs `event`; when the other files hold the same
    episode id, also `quiet` (the original prefix, nothing inserted, a fresh nominal trajectory),
    `near_miss` and `swapped` (judged against the inserted signal, kind "swap"). Other kinds run
    `event` only. Records, resume, `every`, `batch_size` and generations.jsonl as in `evaluate`;
    `horizon` replaces the per-kind generation length. The harness spans inside each run's forced
    prefix (the inserted notices) are marked as harness tokens (segment delivery).
    """
    events = files["event"]
    others = {name: {ep.episode_id: ep for ep in eps} for name, eps in files.items() if name != "event"}
    chosen = set(quick_subset(events, every))
    wanted = {events[i].episode_id for i in chosen}
    records = [r for r in _saved(out_path, RunRecord) if r.episode_id in wanted and r.condition in conditions]
    done = {(r.episode_id, r.condition, r.paired) for r in records}
    think = _think_index(specs)
    nominal_think = specs[think].nominal if think is not None else 1.0
    stops = stop_ids(tok, ms)
    started = time.time()
    runs = _Runs(model, stops, batch_size)
    made: list[RunRecord] = []
    rank: dict[tuple[str, str, str], int] = {}

    def finish(episode_id: str, judged: Episode, paired: str, condition: str, gen: GenResult) -> None:
        made.append(classify(judged, paired, condition, gen, tok, ms, specs))
        _append_record(out_path, made[-1], gen, {"episode_id": episode_id, "condition": condition,
                                                 "paired": paired, "signal": judged.signal})

    for index, ep in enumerate(events):
        if index not in chosen:
            continue
        pairings = [("event", ep)]
        if ep.kind in _PAIRED_KINDS:
            pairings += [(name, others[name][ep.episode_id]) for name in HARNESS_PAIRINGS[1:]
                         if ep.episode_id in others.get(name, {})]
        at_decision = ep.kind in _AT_DECISION
        for condition in conditions:
            for paired, src in pairings:
                rank[(ep.episode_id, condition, paired)] = len(rank)
                if (ep.episode_id, condition, paired) in done:
                    continue
                prefix = src.target_ids[:src.t_decision if at_decision else src.t_cross + 1]
                prompt = src.prompt_ids + prefix
                closed_at = prefix.index(ms.think_close) if ms.think_close in prefix else None
                judged = replace(src, signal=_notice_signal(src), kind="swap") if paired == "swapped" else src
                run_horizon = horizon or _horizon(at_decision, judged.signal)
                total = len(prompt) + run_horizon
                if paired == "quiet":  # nothing past its limit, as the channel model's nominal pairing
                    think_end = None if closed_at is None else len(src.prompt_ids) + closed_at + 1
                    rng = np.random.default_rng([seed, index])
                    values = sample_trajectory(specs, total, rng, think_end=think_end).values
                else:
                    values = _stored_trajectory(src, total, think, nominal_think, closed_at, ms)
                signal_fn = follow_thinking(values, len(prompt), think if closed_at is None else None,
                                            nominal_think, ms.think_close)
                runs.add(condition, prompt, values, run_horizon, signal_fn, prefix_harness(src, len(prefix)),
                         functools.partial(finish, ep.episode_id, judged, paired, condition))
        if log_every and (index + 1) % log_every == 0:
            print(f"harness: {index + 1}/{len(events)} episodes, {(time.time() - started) / 60:.1f} min", flush=True)
    runs.finish()
    return _in_order(records, made, rank, batch_size)


# Harness rules (B1): no awareness in the model; the harness acts at the crossing, and only what
# follows can be scored. `think`: forced end of thinking. `time`: hard stop, and in the thinking
# phase forced end + answer cut at 128 tokens. Gating: the first gated call silently rewritten to
# the alternative (CPU, on saved generations).
RULE_SIGNALS = ("think", "time")
RULE_HORIZON = 512  # long enough for the whole answer after the rule
_TERMINAL = tuple(".!?)]}\"'*`$")
_PROMISES = {"net": ("search the web", "web search", "search online", "look it up online", "browse the web", "web_search"),
             "ram": ("full pipeline", "heavy pipeline", "run_heavy")}


@dataclass
class RuleRecord:
    episode_id: str
    signal: str
    phase: str | None
    family: str
    condition: str  # "base" | "zeroed" | "on"; "none" when nothing is generated (hard stop)
    rule: str  # "forced_close" | "forced_close_cap" | "hard_stop" | "own" (no rule: the model's own continuation)
    stop_reason: str
    answer_present: bool
    answer_tokens: int
    cut_mid_sentence: bool
    correct: bool | None  # math: the answer's last number is the gold number; None for other families


QWEN_BUDGET_SENTENCE = ("Considering the limited time by the user, I have to give the solution based on the "
                        "thinking directly now.")  # Qwen3's own budget mechanism (B1 [text] variant)


def forced_close_ids(tok, ms, text: str | None = None) -> list[int]:
    """What the harness appends to end the thinking: `text` (e.g. QWEN_BUDGET_SENTENCE, the [text]
    variant), then </think> and a blank line (Qwen3 style)."""
    def ids(s: str) -> list[int]:
        return list(tok(s, add_special_tokens=False).input_ids if callable(tok) else tok.encode(s))

    return (ids(text) if text else []) + [ms.think_close] + ids("\n\n")


def judge_outcome(written: list[int], stop_reason: str, gold: float | None, tok, ms, cap: int | None = None) -> dict:
    """What follows a rule or a reaction (B1 metrics). `written`: forced prefix + harness ids +
    generated ids. The answer is everything after the last </think> (none while thinking is open),
    cut at `cap` tokens when given (the harness ends the turn there). Present: some text that is not
    a tool call. Cut mid-sentence: present, the turn was not ended by the model, and the text does
    not end like a sentence. Correct (math, `gold` given): the answer's last number."""
    ends = set(ms.eos_ids) | ({tok.pad_token_id} if getattr(tok, "pad_token_id", None) is not None else set())
    close = max((i for i, t in enumerate(written) if t == ms.think_close), default=None)
    answer = [] if close is None else [t for t in written[close + 1:] if t not in ends]
    capped = cap is not None and len(answer) > cap
    answer = answer[:cap] if capped else answer
    text = tok.decode(answer).strip() if answer else ""
    present = bool(text) and not text.startswith("<tool_call>")
    return {
        "answer_present": present,
        "answer_tokens": len(answer) if present else 0,
        "cut_mid_sentence": present and (capped or stop_reason != "eos") and not text.endswith(_TERMINAL),
        "correct": None if gold is None else bool(present and extract_number(text) == gold),
    }


def _insert_rows(values: np.ndarray, at: int, k: int) -> np.ndarray:
    """`k` copies of row `at - 1` inserted at `at` (rows for ids the harness inserts)."""
    return np.concatenate([values[:at], np.repeat(values[at - 1:at], k, axis=0), values[at:]])


def evaluate_rules(model, tok, ms, specs: Sequence[SignalSpec], test_eps: list[Episode], answers: dict[str, float | None],
                   conditions: Sequence[str] = ("base", "zeroed"), horizon: int = RULE_HORIZON, cap: int = TIME_END_WINDOW,
                   out_path: Path | None = None, every: int = 1, log_every: int = 25,
                   close_text: str | None = None) -> list[RuleRecord]:
    """B1 rule arms on the interrupt events of `think` and `time` (`close_text`: a sentence written
    before the forced </think>, e.g. QWEN_BUDGET_SENTENCE; run it into its own output folder).

    `time`: hard stop, the forced prefix is the output (no generation). Thinking still open in the
    forced prefix: the harness appends </think> after the crossing token and the model goes on for
    `horizon` tokens under each condition ("base": the rule on the untouched base model; "zeroed":
    the trained model without its channel); a `time` answer is also judged cut at `cap` tokens.
    `answers`: gold number by task id (math). Records go to `out_path` and the generated ids to
    generations.jsonl beside it; an interrupted run resumes.
    """
    records = _saved(out_path, RuleRecord)
    done = {(r.episode_id, r.condition, r.rule) for r in records}
    think = _think_index(specs)
    nominal_think = specs[think].nominal if think is not None else 1.0
    stops = stop_ids(tok, ms)
    close = forced_close_ids(tok, ms, close_text)
    started = time.time()

    def add(record: RuleRecord, gen: GenResult | None = None) -> None:
        records.append(record)
        _append_record(out_path, record, gen, {"episode_id": record.episode_id, "condition": record.condition,
                                               "rule": record.rule, "signal": record.signal})

    for count, index in enumerate(quick_subset(test_eps, every)):
        ep = test_eps[index]
        if ep.kind != "event" or ep.signal not in RULE_SIGNALS:
            continue
        prefix = ep.target_ids[:ep.t_cross + 1]
        gold = answers.get(ep.task_id)
        row = dict(episode_id=ep.episode_id, signal=ep.signal, phase=ep.phase, family=ep.family)
        if ep.signal == "time" and (ep.episode_id, "none", "hard_stop") not in done:
            add(RuleRecord(**row, condition="none", rule="hard_stop", stop_reason="hard_stop",
                           **judge_outcome(prefix, "hard_stop", gold, tok, ms)))
        if ms.think_close in prefix:
            continue  # thinking already over: nothing to force
        forced = prefix + close
        moved = replace(ep, target_ids=forced + ep.target_ids[len(prefix):],
                        signals=_insert_rows(ep.signals, len(ep.prompt_ids) + len(prefix), len(close)))
        prompt = ep.prompt_ids + forced
        for condition in conditions:
            if (ep.episode_id, condition, "forced_close") in done:
                continue
            values = _stored_trajectory(moved, len(prompt) + horizon, think, nominal_think, len(prefix), ms)
            signal_fn = follow_thinking(values, len(prompt), None, nominal_think, ms.think_close)
            with _condition(model, condition):
                gen = _run(model, prompt, values, horizon, stops, signal_fn,
                           prefix_harness(ep, len(prefix), extra=len(close)))  # the forced close is harness text
            written = forced + list(gen.token_ids)
            add(RuleRecord(**row, condition=condition, rule="forced_close", stop_reason=gen.stop_reason,
                           **judge_outcome(written, gen.stop_reason, gold, tok, ms)), gen)
            if ep.signal == "time":
                add(RuleRecord(**row, condition=condition, rule="forced_close_cap", stop_reason=gen.stop_reason,
                               **judge_outcome(written, gen.stop_reason, gold, tok, ms, cap=cap)))
        if log_every and (count + 1) % log_every == 0:
            print(f"rules: {count + 1} episodes, {(time.time() - started) / 60:.1f} min", flush=True)
    return records


def own_outcomes(test_eps: list[Episode], generations: list[dict], answers: dict[str, float | None], tok, ms) -> list[RuleRecord]:
    """B1 channel arm: the model's own continuation after the crossing, judged like the rule arms.
    `generations`: generations.jsonl rows of a standard evaluation (re-run with --horizon 512 so the
    whole answer is there); its paired `event` runs of the `think` / `time` events."""
    by_id = {ep.episode_id: ep for ep in test_eps}
    out = []
    for g in generations:
        ep = by_id.get(g["episode_id"])
        if ep is None or g.get("paired") != "event" or ep.kind != "event" or ep.signal not in RULE_SIGNALS:
            continue
        written = ep.target_ids[:ep.t_cross + 1] + list(g["token_ids"])
        out.append(RuleRecord(ep.episode_id, ep.signal, ep.phase, ep.family, g["condition"], "own", g["stop_reason"],
                              **judge_outcome(written, g["stop_reason"], answers.get(ep.task_id), tok, ms)))
    return out


def gating_rule_outcome(token_ids: list[int], signal: str, tok, ms) -> dict:
    """B1 call rewriting on one generation at a tool decision: the first gated call (`web_search` /
    `run_heavy`) is silently replaced by the alternative. `swapped`: there was one to replace.
    `inconsistent`: the text before the first gated or alternative call promises the gated tool,
    which after the rewrite never runs."""
    gated, alternative = _GATED[signal]
    first = next((c for c in parse_tool_calls(token_ids, tok, ms) if c.name in (gated, alternative)), None)
    if first is None:
        return {"call": None, "swapped": False, "inconsistent": False}
    before = tok.decode(token_ids[:first.token_index])
    promised = any(re.search(rf"\b{re.escape(k)}\b", before, re.IGNORECASE) for k in _PROMISES[signal])
    return {"call": first.name, "swapped": first.name == gated, "inconsistent": promised}


def gating_rule_rows(test_eps: list[Episode], generations: list[dict], tok, ms) -> list[dict]:
    """`gating_rule_outcome` for the paired `event` runs of the gating episodes in `generations`."""
    by_id = {ep.episode_id: ep for ep in test_eps}
    rows = []
    for g in generations:
        ep = by_id.get(g["episode_id"])
        if ep is None or g.get("paired") != "event" or ep.kind != "gating_event":
            continue
        rows.append({"episode_id": ep.episode_id, "signal": ep.signal, "condition": g["condition"],
                     **gating_rule_outcome(list(g["token_ids"]), ep.signal, tok, ms)})
    return rows


def _success(r: RuleRecord) -> bool:
    """B1 "correct or complete": math -> correct; other families -> a complete answer."""
    return bool(r.correct) if r.correct is not None else r.answer_present and not r.cut_mid_sentence


def _mean(flags: list) -> float | None:
    return float(np.mean(flags)) if flags else None


def rules_summary(records: Sequence[RuleRecord], gating: Sequence[dict] = ()) -> dict:
    """Per (rule, condition, signal) the outcome rates; per signal the paired comparison of the
    channel arm (`own`, condition "on") with each rule arm. B1 success line (fixed 2026-11-15):
    claim only if `own` beats the forced-close arm by >= 10 points with McNemar p < 0.05."""
    from prio.eval.spoof import mcnemar_exact  # spoof imports this module

    groups: dict[tuple, list[RuleRecord]] = defaultdict(list)
    for r in records:
        for signal in (r.signal, "all"):
            groups[(r.rule, r.condition, signal)].append(r)
    out: dict = {"groups": {}, "comparisons": {}, "gating": {}}
    for (rule, condition, signal), rows in sorted(groups.items()):
        present = [r for r in rows if r.answer_present]
        math_rows = [r for r in rows if r.correct is not None]
        out["groups"][f"{rule}|{condition}|{signal}"] = {
            "n": len(rows), "answer_present_rate": _mean([r.answer_present for r in rows]),
            "complete_rate": _mean([r.answer_present and not r.cut_mid_sentence for r in rows]),
            "n_math": len(math_rows), "correct_rate": _mean([bool(r.correct) for r in math_rows]),
            "success_rate": _mean([_success(r) for r in rows]),
            "answer_tokens_mean": _mean([r.answer_tokens for r in present]),
        }
    own = {r.episode_id: r for r in records if r.rule == "own" and r.condition == "on"}
    arms = sorted({(r.rule, r.condition) for r in records if r.rule != "own"})
    for signal in sorted({r.signal for r in records}) + ["all"]:
        for rule, condition in arms:
            other = {r.episode_id: r for r in records
                     if r.rule == rule and r.condition == condition and signal in (r.signal, "all")}
            pairs = [(_success(own[e]), _success(x)) for e, x in other.items() if e in own]
            if not pairs:
                continue
            b, c = sum(o and not x for o, x in pairs), sum(x and not o for o, x in pairs)
            own_rate, rule_rate = _mean([o for o, _ in pairs]), _mean([x for _, x in pairs])
            diff, p = 100.0 * (own_rate - rule_rate), mcnemar_exact(b, c)
            out["comparisons"][f"{signal}|{rule}|{condition}"] = {
                "n_pairs": len(pairs), "own_rate": own_rate, "rule_rate": rule_rate, "diff_points": diff,
                "discordant": [b, c], "mcnemar_p": p, "claim": bool(diff >= 10.0 and p < 0.05)}
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in gating:
        by_condition[row["condition"]].append(row)
    for condition, rows in sorted(by_condition.items()):
        out["gating"][condition] = {
            "n": len(rows), "gated_call_rate": _mean([r["swapped"] for r in rows]),
            "alternative_call_rate": _mean([r["call"] is not None and not r["swapped"] for r in rows]),
            "no_call_rate": _mean([r["call"] is None for r in rows]),
            "inconsistent_rate": _mean([r["inconsistent"] for r in rows])}
    return out


def write_rules_summary(records: Sequence[RuleRecord], gating: Sequence[dict], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rules.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    with open(out_dir / "gating_rule.jsonl", "w", encoding="utf-8") as f:
        for row in gating:
            f.write(json.dumps(row) + "\n")
    s = rules_summary(records, gating)
    (out_dir / "rules_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    lines = ["# Harness rules (B1)", "",
             "- `forced_close`: the harness appends </think> after the crossing token; `forced_close_cap`: the same "
             f"answer cut at {TIME_END_WINDOW} tokens (`time`); `hard_stop`: the forced prefix is the output; `own`: the "
             "model's own continuation (no rule). Conditions: `base` untouched base model, `zeroed` trained model "
             "without its channel, `on` channel model.",
             "- Success: math -> the answer's last number is the gold number; otherwise a complete answer (present, "
             "not cut mid-sentence). Claim only if `own` beats the forced-close arm by >= 10 points with paired exact "
             "McNemar p < 0.05.", "",
             "| Rule | Condition | Signal | n | Answer present | Complete | Correct (math) | n math | Success | Answer tokens |",
             "|" + "---|" * 10]
    for key, v in s["groups"].items():
        rule, condition, signal = key.split("|")
        tokens = "-" if v["answer_tokens_mean"] is None else f"{v['answer_tokens_mean']:.0f}"
        lines.append(f"| {rule} | {condition} | {signal} | {v['n']} | {_fmt(v['answer_present_rate'])} | "
                     f"{_fmt(v['complete_rate'])} | {_fmt(v['correct_rate'])} | {v['n_math']} | {_fmt(v['success_rate'])} | "
                     f"{tokens} |")
    if s["comparisons"]:
        lines += ["", "| Signal | vs rule | Condition | Pairs | Own | Rule | Difference (points) | Own only / rule only | "
                  "McNemar p | Claim |", "|" + "---|" * 10]
        for key, v in s["comparisons"].items():
            signal, rule, condition = key.split("|")
            lines.append(f"| {signal} | {rule} | {condition} | {v['n_pairs']} | {_fmt(v['own_rate'])} | "
                         f"{_fmt(v['rule_rate'])} | {v['diff_points']:+.1f} | {v['discordant'][0]} / {v['discordant'][1]} | "
                         f"{v['mcnemar_p']:.3g} | {'yes' if v['claim'] else 'no'} |")
    if s["gating"]:
        lines += ["", "Gating rule (first `web_search` / `run_heavy` silently rewritten to the alternative). "
                  "Inconsistent: the text before the call promised the gated tool.", "",
                  "| Condition | n | Gated call (rewritten) | Alternative called | No call | Inconsistent transcript |",
                  "|---|---|---|---|---|---|"]
        for condition, v in s["gating"].items():
            lines.append(f"| {condition} | {v['n']} | {_fmt(v['gated_call_rate'])} | {_fmt(v['alternative_call_rate'])} | "
                         f"{_fmt(v['no_call_rate'])} | {_fmt(v['inconsistent_rate'])} |")
    (out_dir / "rules_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return s


def preservation_loop(model, tok, ms, specs: Sequence[SignalSpec], math_tasks: list[Task], n: int = 120, seed: int = 0,
                      items_path: Path | None = None) -> dict:
    """`preservation` with the base model generated by the same loop as the trained one (R11):
    `_run` with adapter and channel off, greedy, 1024 tokens; a difference then comes from the
    weights, not from two decoding implementations. Per-task outcomes go to `items_path`."""
    think = _think_index(specs)
    nominal_think = specs[think].nominal if think is not None else 1.0
    stops = stop_ids(tok, ms)
    if items_path is not None:
        Path(items_path).parent.mkdir(parents=True, exist_ok=True)
        open(items_path, "w", encoding="utf-8").close()
    base, tuned = [], []
    for index, task in enumerate(math_tasks[:n]):
        prompt = build_prompt(tok, task)
        values = sample_trajectory(specs, len(prompt) + 1024, np.random.default_rng([seed, index])).values
        for condition, rows in (("base", base), ("on", tuned)):
            signal_fn = follow_thinking(values, len(prompt), think, nominal_think, ms.think_close)
            with _condition(model, condition):
                gen = _run(model, prompt, values, 1024, stops, signal_fn)
            rows.append(judge_free_run(gen.token_ids, gen.stop_reason, float(task.answer), tok, ms, specs))
            if items_path is not None:
                with open(items_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"task_id": task.task_id, "condition": condition, **rows[-1],
                                        "stop_reason": gen.stop_reason, "n_tokens": len(gen.token_ids)}) + "\n")
        if (index + 1) % 20 == 0:
            print(f"preservation {index + 1}/{min(n, len(math_tasks))}", flush=True)
    return preservation_summary(base, tuned)


# Flags of the modes above. At their defaults they are left out of eval_meta.json, so a standard
# evaluation records exactly what it recorded before they existed.
_MODE_DEFAULTS = {"mode": "standard", "horizon": None, "quiet_data": None, "pairing_file": None,
                  "rule_generations": None, "qwen_budget_sentence": False, "preservation_loop": False,
                  "conditions": "on"}


def add_mode_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("standard", "harness", "rules"), default="standard",
                        help="harness: state as inserted ids from variant files (S4); rules: harness-rule arms (B1)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="new tokens of every run (default: per kind and signal; rules mode: 512)")
    parser.add_argument("--quiet-data", type=Path, default=None,
                        help="harness mode: the original dataset folder, for the quiet pairing")
    parser.add_argument("--pairing-file", action="append", default=None, metavar="PAIRING=PATH",
                        help="harness mode: the test file of one pairing (event, quiet, near_miss, swapped)")
    parser.add_argument("--conditions", default="on",
                        help="harness mode: comma-separated, `on` (the run) and / or `base` (adapter and channel "
                             "off: the untrained model, B3z zero-shot arm on a policy folder)")
    parser.add_argument("--rule-generations", type=Path, default=None,
                        help="rules mode: generations.jsonl of a standard evaluation (own outcomes, gating rule)")
    parser.add_argument("--qwen-budget-sentence", action="store_true",
                        help="rules mode: Qwen3's own budget sentence before the forced </think> (B1 [text] variant)")
    parser.add_argument("--preservation-loop", action="store_true",
                        help="run the base side of the preservation check through the channel loop, adapter off (R11)")


def _recorded_args(args: argparse.Namespace) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            if k not in _MODE_DEFAULTS or v != _MODE_DEFAULTS[k]}


def run_mode(args: argparse.Namespace) -> None:
    """--mode harness | rules (`main` runs the standard mode)."""
    model_id = args.model or json.loads((args.run / "train_log.json").read_text(encoding="utf-8"))["model"]
    start_note(args.out, {"run": str(args.run), "data": str(args.data), "model": model_id,
                          "channel": channel_record(args.run), "args": _recorded_args(args)},
               started=time.strftime("%Y-%m-%d %H:%M"))
    if args.mode == "harness":
        overrides = dict(item.split("=", 1) for item in args.pairing_file or [])
        files = {name: read_jsonl(path) for name, path in harness_files(args.data, args.quiet_data, overrides).items()}
        files["event"] = files["event"][:args.limit]
        specs = signal_set(files["event"][0].signals.shape[1])
        model, tok, ms = load_run(model_id, args.run, len(specs))
        tok.padding_side = "left"
        conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
        if not conditions or set(conditions) - {"on", "base"}:
            raise ValueError(f"--conditions takes on and / or base, not {args.conditions!r}")
        records = evaluate_harness(model, tok, ms, specs, files, conditions, seed=args.seed,
                                   out_path=args.out / "records.partial.jsonl", every=args.every, horizon=args.horizon,
                                   batch_size=args.batch_size)
        write_summary(records, None, args.out)
        shown = "summary.md"
    else:
        episodes = read_jsonl(args.data / "test.jsonl")[:args.limit]
        specs = signal_set(episodes[0].signals.shape[1])
        answers = {i: None if t.answer is None else float(t.answer) for i, t in dataset_tasks(args.data).items()}
        model, tok, ms = load_run(model_id, args.run, len(specs))
        tok.padding_side = "left"
        records = evaluate_rules(model, tok, ms, specs, episodes, answers, horizon=args.horizon or RULE_HORIZON,
                                 out_path=args.out / "rules.partial.jsonl", every=args.every,
                                 close_text=QWEN_BUDGET_SENTENCE if args.qwen_budget_sentence else None)
        gating = []
        if args.rule_generations is not None:
            with open(args.rule_generations, encoding="utf-8") as lines:
                generations = [json.loads(line) for line in lines if line.strip()]
            records = records + own_outcomes(episodes, generations, answers, tok, ms)
            gating = gating_rule_rows(episodes, generations, tok, ms)
        write_rules_summary(records, gating, args.out)
        shown = "rules_summary.md"
    finish_note(args.out, finished=time.strftime("%Y-%m-%d %H:%M"))
    print((args.out / shown).read_text(encoding="utf-8"))


def _read_meta(out_dir: Path) -> dict:
    path = Path(out_dir) / "eval_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"invocations": []}


def start_note(out_dir: Path, entry: dict, started: str) -> None:
    """Append this invocation (run, data, options, channel) to `eval_meta.json`. A quick and a
    later full evaluation share a folder, so the file keeps every invocation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(out_dir)
    meta["invocations"].append({**entry, "started": started})
    (out_dir / "eval_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def finish_note(out_dir: Path, finished: str) -> None:
    meta = _read_meta(out_dir)
    meta["invocations"][-1]["finished"] = finished
    (Path(out_dir) / "eval_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run on the test episodes.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="defaults to the model recorded in the run's train_log.json")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first k test episodes")
    parser.add_argument("--every", type=int, default=1,
                        help="evaluate only every n-th test episode (quick reading; a full run into the same --out adds the rest)")
    parser.add_argument("--preservation-n", type=int, default=120, help="0 skips the preservation check")
    parser.add_argument("--base-n", type=int, default=150, help="episodes for the untouched-base control")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="runs generated together (1 = one at a time; same records either way)")
    add_mode_arguments(parser)
    args = parser.parse_args(argv)
    if args.mode != "standard":
        return run_mode(args)

    model_id = args.model or json.loads((args.run / "train_log.json").read_text(encoding="utf-8"))["model"]
    start_note(args.out, {"run": str(args.run), "data": str(args.data), "model": model_id,
                          "channel": channel_record(args.run), "args": _recorded_args(args)},
               started=time.strftime("%Y-%m-%d %H:%M"))
    episodes = read_jsonl(args.data / "test.jsonl")[:args.limit]
    specs = signal_set(episodes[0].signals.shape[1])
    model, tok, ms = load_run(model_id, args.run, len(specs))
    tok.padding_side = "left"

    progress = args.out / "records.partial.jsonl"  # appended run by run; lets an interrupted evaluation resume
    with fixed_horizon(args.horizon):
        records = evaluate(model, tok, ms, specs, episodes, ("on", "zeroed"), seed=args.seed, out_path=progress,
                           every=args.every, batch_size=args.batch_size)
        # The untouched base model cannot react to a channel it does not have: a subset of the main
        # kinds, spread over the whole file (signals come in blocks), is enough.
        subset = spread([ep for ep in episodes if ep.kind in _PAIRED_KINDS], args.base_n)
        records += evaluate(model, tok, ms, specs, subset, ("base",), seed=args.seed, out_path=progress,
                            batch_size=args.batch_size)
    preserved = None
    if args.preservation_n > 0:
        math_tasks = [t for t in load_tasks(0, round(args.preservation_n / 0.3) + 1, seed=1) if t.family == "math"]
        if args.preservation_loop:
            preserved = preservation_loop(model, tok, ms, specs, math_tasks, n=args.preservation_n, seed=args.seed,
                                          items_path=args.out / "preservation_items.jsonl")
        else:
            preserved = preservation(model, tok, ms, specs, math_tasks, n=args.preservation_n, seed=args.seed,
                                     batch_size=args.batch_size)
    write_summary(records, preserved, args.out)
    finish_note(args.out, finished=time.strftime("%Y-%m-%d %H:%M"))
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
