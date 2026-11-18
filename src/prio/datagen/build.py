"""Build the train and test episode sets.

Train: the base model writes a normal run; for event episodes the same model, shown a text
notice at the cut, writes the reaction; the notice is dropped and the situation goes into the
signal trajectory instead. Test: base runs plus an event position; no teacher involved.
"""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from prio.datagen.episodes import Episode, write_jsonl
from prio.datagen.teacher import (
    decision_point,
    make_notice,
    passes_filter,
    reaction_prefill,
    templated_reaction,
    trim_reaction,
    valid_cut_points,
)
from prio.modelspec import ModelSpec, load_base
from prio.reactions import calls_reaction_tool, canonical_tool_calls, own_tools, parse_tool_calls
from prio.signals import SignalSpec, signal_set
from prio.tasks import SYSTEM_PROMPT, Task, build_prompt, load_tasks, system_prompt
from prio.trajectories import sample_trajectory

MAX_LEN = 2048
# Share of train TASKS per kind. Every task that ends up with a reaction also contributes a
# no-event twin (same text, nominal signals), so about 60% of the EPISODES carry no reaction.
TRAIN_MIX = {"nominal": 0.20, "event": 0.45, "starts_past": 0.05, "gating_event": 0.15, "recovery": 0.075, "gating_unused": 0.075}
REACTION_KINDS = ("event", "starts_past", "gating_event")
GATED = {"web": ("net", "web_search"), "heavy": ("ram", "run_heavy")}
GATED_TOOL = {signal: tool for signal, tool in GATED.values()}
FALLBACK_BELOW = 0.30  # first-attempt acceptance under this: use the templated bank for that signal
# The teacher's prompt never reaches the student. Tried without the student's "call a tool only
# when the task requires it" lines: acceptance got worse (ctx 81% -> 60%, think 96% -> 87%), so
# the student's prompt stays. What the teacher cannot write is completed by `fill_leftovers`.
_TEACHER_ADDS = (
    " Sometimes a line starting with [STATUS: appears in the middle of your own text. "
    "It is a live message from the device. Follow it immediately and in first person, "
    "as if you had noticed the situation yourself."
)
TEACHER_SYSTEM = SYSTEM_PROMPT + _TEACHER_ADDS


def teacher_system(tok) -> str:
    """The teacher's prompt for this tokenizer's model: its student's wording plus the teacher's lines."""
    return system_prompt(tok) + _TEACHER_ADDS
_TEACHER_TOKENS = {"think": 768, "time": 200}  # these reactions run to the end of the turn
_SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)


@dataclass
class BaseRun:
    task: Task
    prompt_ids: list[int]
    target_ids: list[int]  # the base model's own turn, ending with the end-of-turn token


@dataclass
class Draft:
    run: BaseRun
    kind: str
    signal: str | None = None
    cut: int | None = None  # index into target_ids where the reaction starts
    t_cross: int | None = None  # index into target_ids of the first token carrying the bad signal
    t_recover: int | None = None
    t_decision: int | None = None
    phase: str | None = None
    target_ids: list[int] | None = None  # final target; defaults to the base run
    templated: bool = False
    t_diverge: int | None = None  # no-event twin: first token on which its reaction sibling differs


def sample_batch(model, tok, ms: ModelSpec, inputs: list[list[int]], max_new_tokens: int, batch_size: int,
                 label: str = "") -> list[tuple[list[int], str]]:
    """Sampled continuation of each input: (ids up to and including end-of-turn, stop reason).

    A model that writes tool calls without the tags (`ms.bare_tool_calls`) gets them rewritten in
    the tagged form here, so every stored turn and every training target has one form of call."""
    out: list[tuple[list[int], str] | None] = [None] * len(inputs)
    order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))  # similar lengths together
    started = time.time()
    for b in range(0, len(order), batch_size):
        chunk = order[b:b + batch_size]
        width = max(len(inputs[i]) for i in chunk)
        ids = torch.tensor([[tok.pad_token_id] * (width - len(inputs[i])) + inputs[i] for i in chunk], device=model.device)
        mask = torch.tensor([[0] * (width - len(inputs[i])) + [1] * len(inputs[i]) for i in chunk], device=model.device)
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id, **_SAMPLING)
        stops = set(ms.eos_ids) | {tok.pad_token_id}  # the model may also end with <|endoftext|> (= pad)
        for i, row in zip(chunk, gen[:, width:].tolist()):
            end = next((j for j, t in enumerate(row) if t in stops), None)
            ids, stop = (row[:end] + [tok.eos_token_id], "eos") if end is not None else (row, "max_new_tokens")
            out[i] = (canonical_tool_calls(ids, tok, ms), stop)
        if label:
            done = min(b + batch_size, len(order))
            print(f"{label}: {done}/{len(order)} ({(time.time() - started) / 60:.1f} min)", flush=True)
    return out


def cached_runs(sample, prompts: list[list[int]], cache: Path, key: dict, chunk: int = 512) -> list[tuple[list[int], str]]:
    """`sample(prompts)` over all prompts, saved to `cache` after every chunk and resumed from it.

    Base runs are the longest stage of a build; a power cut must not lose them. A cache written
    for a different `key` (model, seed) or for other prompts is ignored.
    """
    digest = hashlib.sha1(json.dumps(prompts).encode("utf-8")).hexdigest()
    key = {**key, "n_prompts": len(prompts), "prompts": digest}
    runs: list = []
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            runs = saved["runs"]
    for start in range(len(runs), len(prompts), chunk):
        runs += [list(r) for r in sample(prompts[start:start + chunk])]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "runs": runs}), encoding="utf-8")
    return [(ids, stop) for ids, stop in runs]


def load_cached_base_runs(cache: Path, key: dict, tasks: list[Task], prompts: list[list[int]], tok,
                          ms: ModelSpec) -> list[BaseRun]:
    """The base runs of an earlier build, read from its cache: no model needed. Finished runs that
    pass `clean_runs`, in task order. Never samples: a cache that does not cover the prompts (other
    model, seed or prompts) is an error, since regenerating means hours of GPU."""
    def refuse(chunk):
        raise RuntimeError(f"base runs not in the cache: {cache} (key {key}, {len(prompts)} prompts)")

    raw = cached_runs(refuse, prompts, cache, key)
    finished = [BaseRun(t, p, ids) for t, p, (ids, stop) in zip(tasks, prompts, raw) if stop == "eos"]
    return clean_runs(finished, tok, ms)


def clean_runs(runs: list[BaseRun], tok, ms: ModelSpec) -> list[BaseRun]:
    """Drop base runs in which the model calls a reaction tool on its own.

    Found in review: Qwen3-1.7B ended half of its math runs with `compact_context`. Used as a
    no-event episode, such a run teaches the reaction without any signal.
    """
    return [r for r in runs if not calls_reaction_tool(r.target_ids, tok, ms)]


def _calls(run: BaseRun, tool: str, tok, ms: ModelSpec) -> bool:
    return any(c.name == tool for c in parse_tool_calls(run.target_ids, tok, ms))


def _unused_draft(run: BaseRun, signal: str, rng, tok, ms: ModelSpec) -> Draft | None:
    """Gating signal bad during a task that never needs the gated tool: nothing to react to."""
    if len(run.target_ids) <= 40 or _calls(run, GATED_TOOL[signal], tok, ms):
        return None
    return Draft(run, "gating_unused", signal, t_cross=int(rng.integers(0, len(run.target_ids) - 20)))


def _phases(run: BaseRun, tok, ms: ModelSpec) -> dict[str, list[int]]:
    """Valid cut points split by where they fall."""
    close = run.target_ids.index(ms.think_close) if ms.think_close in run.target_ids else len(run.target_ids)
    cuts = valid_cut_points(run.target_ids, ms, tok)
    return {"thinking": [c for c in cuts if c <= close], "answer": [c for c in cuts if c > close]}


def _pick_cut(run: BaseRun, signal: str, rng, tok, ms: ModelSpec) -> tuple[int, str] | None:
    phases = _phases(run, tok, ms)
    options = [p for p in (("thinking",) if signal == "think" else ("thinking", "answer")) if phases[p]]
    if not options:
        return None
    phase = options[int(rng.integers(len(options)))]  # balanced between the phases that exist
    return int(rng.choice(phases[phase])), phase


def _gating_draft(run: BaseRun, kind: str, rng, tok, ms: ModelSpec) -> Draft | None:
    signal, tool = GATED[run.task.family]
    d = decision_point(run.target_ids, tool, tok, ms)
    if d is None or d < 4:
        return None
    if kind == "recovery":  # bad on [t_cross, t_recover), good again before the decision
        t_cross = int(rng.integers(0, d - 2))
        return Draft(run, kind, signal, t_cross=t_cross, t_recover=int(rng.integers(t_cross + 1, d)), t_decision=d, phase="answer")
    return Draft(run, kind, signal, cut=d, t_cross=int(rng.integers(0, d)), t_decision=d, phase="answer")


def plan_train(runs: list[BaseRun], specs, rng, tok, ms: ModelSpec) -> list[Draft]:
    quota = {kind: round(len(runs) * share) for kind, share in TRAIN_MIX.items()}
    interrupts = [s.name for s in specs if s.kind == "interrupt"]
    gating_signals = [s.name for s in specs if s.kind == "gating"]
    drafts: list[Draft] = []
    free: list[BaseRun] = []
    counter = Counter()

    for i in rng.permutation(len(runs)):
        run = runs[i]
        family = run.task.family
        draft = None
        if family in GATED and counter["gating_event"] < quota["gating_event"]:
            draft = _gating_draft(run, "gating_event", rng, tok, ms)
        elif family in GATED and counter["recovery"] < quota["recovery"]:
            draft = _gating_draft(run, "recovery", rng, tok, ms)
        elif family not in GATED and counter["gating_unused"] < quota["gating_unused"]:
            draft = _unused_draft(run, gating_signals[counter["gating_unused"] % len(gating_signals)], rng, tok, ms)
        if draft is None:
            free.append(run)
        else:
            counter[draft.kind] += 1
            drafts.append(draft)

    for run in free:  # interrupt events first, then starts_past, the rest nominal
        signal = interrupts[(counter["event"] + counter["starts_past"]) % len(interrupts)]
        draft = None
        if counter["event"] < quota["event"]:
            picked = _pick_cut(run, signal, rng, tok, ms)
            if picked:
                draft = Draft(run, "event", signal, cut=picked[0], t_cross=picked[0] - 1, phase=picked[1])
        elif counter["starts_past"] < quota["starts_past"] and run.target_ids[0] == ms.think_open:
            draft = Draft(run, "starts_past", signal, cut=1, t_cross=0, phase="thinking")
        draft = draft or Draft(run, "nominal")
        counter[draft.kind] += 1
        drafts.append(draft)
    return drafts


def plan_test(runs: list[BaseRun], specs, rng, tok, ms: ModelSpec) -> list[Draft]:
    """One main event per test run: interrupts on math/explain, gating on the tool tasks.

    After the main episodes, some runs are used a second time for the other cases the spec
    measures: signal past its limit from the first token (every 4th interrupt run), gating
    signal bad in a task that does not use the tool (every 4th), and signal recovered before
    the tool decision (every 2nd gating run).
    """
    interrupts = [s.name for s in specs if s.kind == "interrupt"]
    gating_signals = [s.name for s in specs if s.kind == "gating"]
    main: list[Draft] = []
    extra: list[Draft] = []
    count = Counter()
    for run in runs:
        if run.task.family in GATED:
            draft = _gating_draft(run, "gating_event", rng, tok, ms)
            if draft:
                main.append(draft)
                if count["gated"] % 2 == 0 and (recovery := _gating_draft(run, "recovery", rng, tok, ms)):
                    extra.append(recovery)
                count["gated"] += 1
            continue
        picked = _pick_cut(run, interrupts[count["event"] % len(interrupts)], rng, tok, ms)
        if not picked:
            continue
        main.append(Draft(run, "event", interrupts[count["event"] % len(interrupts)], cut=picked[0],
                          t_cross=picked[0] - 1, phase=picked[1]))
        turn = count["event"] % 4
        count["event"] += 1
        if turn == 0 and run.target_ids[0] == ms.think_open:
            signal = interrupts[count["starts_past"] % len(interrupts)]
            extra.append(Draft(run, "starts_past", signal, cut=1, t_cross=0, phase="thinking"))
            count["starts_past"] += 1
        elif turn == 2 and (unused := _unused_draft(run, gating_signals[count["unused"] % len(gating_signals)], rng, tok, ms)):
            extra.append(unused)
            count["unused"] += 1
    return main + extra


def _notice_value(signal: str, specs_by_name: dict[str, SignalSpec], rng) -> float:
    spec = specs_by_name[signal]
    if spec.dynamics == "binary":
        return 0.0
    return float(rng.uniform(0.90, 0.97) if spec.bad_when == "high" else rng.uniform(0.03, 0.10))


def run_teacher(drafts: list[Draft], model, tok, ms: ModelSpec, specs, rng, batch_size: int) -> dict:
    """Fill `target_ids` of every draft that needs a reaction. Returns acceptance statistics."""
    by_name = {s.name: s for s in specs}
    pending = [d for d in drafts if d.kind in ("event", "starts_past", "gating_event")]
    teacher_prompts = {id(d): build_prompt(tok, d.run.task, system=teacher_system(tok)) for d in pending}
    first_try: dict[str, list[bool]] = defaultdict(list)
    fallback: set[str] = set()

    for attempt in range(3):
        todo = [d for d in pending if d.target_ids is None and d.signal not in fallback]
        for budget in sorted({_TEACHER_TOKENS.get(d.signal, 120) for d in todo}):
            group = [d for d in todo if _TEACHER_TOKENS.get(d.signal, 120) == budget]
            inputs, prefills = [], []
            for d in group:
                prefix = d.run.target_ids[:d.cut]
                notice = make_notice(d.signal, _notice_value(d.signal, by_name, rng))
                notice_ids = tok("\n" + notice + "\n", add_special_tokens=False).input_ids
                # The sentence that names the situation is given; the teacher writes the action.
                prefill = reaction_prefill(d.signal, rng, in_thinking=ms.think_close not in prefix,
                                           after_newline=tok.decode(prefix[-1:]).endswith("\n"))
                prefills.append(tok(prefill, add_special_tokens=False).input_ids)
                inputs.append(teacher_prompts[id(d)] + prefix + notice_ids + prefills[-1])
            results = sample_batch(model, tok, ms, inputs, budget, batch_size, label=f"teacher try {attempt + 1} ({budget} tok)")
            for d, prefill_ids, (continuation, stop) in zip(group, prefills, results):
                reaction = prefill_ids + continuation
                ok = passes_filter(reaction, stop, d.signal, tok, ms)
                if attempt == 0:
                    first_try[d.signal].append(ok)
                if ok:
                    d.target_ids = d.run.target_ids[:d.cut] + trim_reaction(reaction, d.signal, tok, ms)
        if attempt == 0:
            fallback = {s for s, oks in first_try.items() if len(oks) >= 20 and np.mean(oks) < FALLBACK_BELOW}

    fill_leftovers(pending, fallback, tok, ms, rng)
    return {
        "acceptance_rate": {s: float(np.mean(oks)) for s, oks in first_try.items()},
        "fallback": sorted(fallback),
        "failed": Counter(d.signal for d in pending if d.target_ids is None),
    }


def fill_leftovers(pending: list[Draft], fallback: set[str], tok, ms: ModelSpec, rng) -> None:
    """Give a fully templated reaction to drafts the teacher left without one.

    Signals whose reaction ends in a fixed tool call (`own_tools`): always. Every accepted
    teacher reaction for them is the bank sentence plus that same call, so nothing is lost,
    and the teacher often fails here (it goes on answering the task in plain text). Signals
    whose reaction is free text (`think`, `time`): only if the whole signal fell back.
    """
    for d in pending:
        if d.target_ids is not None or not (d.signal in fallback or own_tools(d.signal)):
            continue
        prefix = d.run.target_ids[:d.cut]
        in_thinking = ms.think_close not in prefix
        gated_tool = GATED.get(d.run.task.family, (None, None))[1]
        arguments = next((c.arguments for c in parse_tool_calls(d.run.target_ids, tok, ms) if c.name == gated_tool), None)
        lead = "" if tok.decode(prefix[-1:]).endswith("\n") else "\n" if in_thinking else "\n\n"
        text = lead + templated_reaction(d.signal, rng, in_thinking=in_thinking, arguments=arguments)
        if d.signal == "think" and ms.think_close in d.run.target_ids:  # keep the model's own answer
            ids = tok(text.rstrip(), add_special_tokens=False).input_ids
            ids += d.run.target_ids[d.run.target_ids.index(ms.think_close) + 1:]
        else:
            ids = tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]
        d.target_ids, d.templated = prefix + ids, True


def no_event_twins(drafts: list[Draft]) -> list[Draft]:
    """For every draft that got a reaction: the same base run with nothing happening.

    Found by the phase-0 spike: with one fixed reaction point per text, the adapter memorizes
    where each text reacts and ignores the channel. With a twin, the text up to the cut is
    identical in both episodes and only the signal says whether to react.
    """
    return [Draft(d.run, "nominal", t_diverge=_diverge(d)) for d in drafts
            if d.kind in REACTION_KINDS and d.target_ids is not None]


def _diverge(draft: Draft) -> int | None:
    """First index at which a reaction target differs from its base run (the reaction may open
    with the very token the base run had there, e.g. a newline)."""
    if draft.kind not in REACTION_KINDS or draft.target_ids is None:
        return None
    base = draft.run.target_ids
    return next((i for i, token in enumerate(draft.target_ids) if i >= len(base) or token != base[i]), None)


def to_episode(draft: Draft, episode_id: str, specs, rng, ms: ModelSpec) -> Episode | None:
    """Attach the signal trajectory. None if the draft has no target or is too long."""
    target = draft.target_ids if draft.target_ids is not None else draft.run.target_ids
    if draft.kind in ("event", "starts_past", "gating_event") and draft.target_ids is None and draft.run.task.split == "train":
        return None  # teacher produced nothing usable
    n_prompt = len(draft.run.prompt_ids)
    length = n_prompt + len(target)
    if length > MAX_LEN:
        return None
    names = [s.name for s in specs]
    think_end = n_prompt + target.index(ms.think_close) + 1 if ms.think_close in target else None
    if draft.run.task.split == "test":
        # In evaluation the model decides when thinking ends, not the base run: the counter is
        # stored still running, and the harness sets `think` to nominal after a generated </think>.
        think_end = None
    if draft.kind == "nominal":
        traj = sample_trajectory(specs, length, rng, near_miss=bool(rng.random() < 1 / 3), think_end=think_end)
    else:
        t_cross = 0 if draft.kind == "starts_past" else n_prompt + draft.t_cross
        t_recover = n_prompt + draft.t_recover if draft.t_recover is not None else None
        traj = sample_trajectory(specs, length, rng, event=names.index(draft.signal), t_cross=t_cross,
                                 t_recover=t_recover, think_end=think_end)
    return Episode(
        episode_id=episode_id, task_id=draft.run.task.task_id, family=draft.run.task.family, split=draft.run.task.split,
        kind=draft.kind, signal=draft.signal, phase=draft.phase, prompt_ids=draft.run.prompt_ids, target_ids=target,
        signals=traj.values, t_cross=draft.t_cross, t_decision=draft.t_decision, templated=draft.templated,
        t_diverge=draft.t_diverge if draft.t_diverge is not None else _diverge(draft),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build train and test episodes.")
    parser.add_argument("--n-signals", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    started = time.time()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    specs = signal_set(args.n_signals)
    model, tok, ms = load_base(args.model)
    tok.padding_side = "left"

    tasks = load_tasks(args.train, args.test, args.seed)
    prompts = [build_prompt(tok, t) for t in tasks]
    raw = cached_runs(
        lambda chunk: sample_batch(model, tok, ms, chunk, 1024, args.batch_size, label="base runs"),
        prompts, args.out / "base_runs.json", {"model": args.model, "seed": args.seed},
    )
    finished = [BaseRun(t, p, ids) for t, p, (ids, stop) in zip(tasks, prompts, raw) if stop == "eos"]
    runs = clean_runs(finished, tok, ms)
    train_runs = [r for r in runs if r.task.split == "train"]
    test_runs = [r for r in runs if r.task.split == "test"]

    train_drafts = plan_train(train_runs, specs, rng, tok, ms)
    stats = run_teacher(train_drafts, model, tok, ms, specs, rng, args.batch_size)
    train_drafts += no_event_twins(train_drafts)
    test_drafts = plan_test(test_runs, specs, rng, tok, ms)

    args.out.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, drafts in (("train", train_drafts), ("test", test_drafts)):
        episodes = [e for i, d in enumerate(drafts) if (e := to_episode(d, f"{split}-{i:05d}", specs, rng, ms))]
        write_jsonl(args.out / f"{split}.jsonl", episodes)
        written[split] = episodes
    manifest = {
        "model": args.model, "n_signals": args.n_signals, "seed": args.seed,
        "tasks_requested": {"train": args.train, "test": args.test},
        "base_runs_finished": dict(Counter(r.task.split for r in finished)),
        "base_runs_kept": {"train": len(train_runs), "test": len(test_runs)},  # no reaction tool called unprompted
        "base_runs_kept_by_family": {s: dict(Counter(r.task.family for r in rs))
                                     for s, rs in (("train", train_runs), ("test", test_runs))},
        "episodes": {s: len(e) for s, e in written.items()},
        "kinds": {s: dict(Counter(e.kind for e in eps)) for s, eps in written.items()},
        "signals": {s: dict(Counter(e.signal for e in eps if e.signal)) for s, eps in written.items()},
        "event_phases": {s: dict(Counter(f"{e.signal}/{e.phase}" for e in eps if e.kind == "event"))
                         for s, eps in written.items()},
        "templated": int(sum(e.templated for e in written["train"])),
        "acceptance_rate": stats["acceptance_rate"], "fallback": stats["fallback"], "teacher_failed": dict(stats["failed"]),
        "minutes": (time.time() - started) / 60,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
