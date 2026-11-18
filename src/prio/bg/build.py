"""Build the background-task dataset from the phase-1 base runs (spec section 6.4). CPU only.

uv run python -m prio.bg.build --base data/n6 --model Qwen/Qwen3-1.7B --out data/bg3 --seed 0 \
    --train 4500 --test-events 400 --test-openings 100
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from prio.bg.assemble import BgEvent, BgPlan, align_divergence, assemble, first_difference, slot_plan, twin_plan
from prio.bg.kinds import HELD_OUT_KINDS, TRAIN_KINDS, sample_tasks
from prio.datagen.build import BaseRun, load_cached_base_runs
from prio.datagen.episodes import write_jsonl
from prio.modelspec import ModelSpec

EVENT_KINDS = ("done", "error", "waiting", "stalled")
TRAIN_EVENT_SHARE = {"done": 0.40, "error": 0.20, "waiting": 0.20, "stalled": 0.20}
PAIR_SHARE = 0.85  # train: share of episodes that are an event or its twin; the rest are nominal
NEAR_MISS_SHARE = 0.30  # share of twins whose event task gets to 95-99% at the event without finishing
MAX_USES = 2  # a base run carries at most this many plans (not counting twins)


def sentence_ends(target_ids: list[int], tok) -> list[int]:
    """Tokens after which a sentence ends: they end with . ! or ? followed by whitespace (not "3.5"),
    or contain a newline."""
    ends = []
    for i in range(len(target_ids) - 1):
        one = tok.decode(target_ids[i:i + 1])
        following = tok.decode(target_ids[i:i + 2])[len(one):]
        if "\n" in one or (one.rstrip(" ").endswith((".", "!", "?")) and (not following or following[0].isspace())):
            ends.append(i)
    return ends


def cut_candidates(run: BaseRun, tok, ms: ModelSpec, phase: str, min_before: int = 48,
                   min_after: int = 64) -> list[int]:
    """Sentence ends inside `phase` with room on both sides: more than `min_before` tokens of that
    phase before, at least `min_after` tokens of the base run after (end-of-turn excluded). Math
    answers: only before the answer's last number, so the report never follows the final number."""
    ids = run.target_ids
    specials = {ms.think_open, ms.think_close, *ms.eos_ids}
    if ms.think_close not in ids:
        return []
    close = ids.index(ms.think_close)
    opener = ids.index(ms.think_open) if ms.think_open in ids else -1
    lo, hi = (opener, close) if phase == "thinking" else (close, len(ids) - 1)
    limit = hi
    if phase == "answer" and run.task.family == "math":
        numbers = [i for i in range(close + 1, len(ids) - 1) if any(ch.isdigit() for ch in tok.decode(ids[i:i + 1]))]
        limit = numbers[-1] if numbers else hi
    return [i for i in sentence_ends(ids, tok)
            if lo < i < limit and i - lo > min_before and len(ids) - 1 - i >= min_after
            and ids[i] not in specials and ids[i + 1] not in specials]


class _Cuts:
    """Cut candidates per (base run, phase), computed once (decoding is the slow part)."""

    def __init__(self, tok, ms: ModelSpec, min_before: int, min_after: int):
        self.tok, self.ms, self.min_before, self.min_after = tok, ms, min_before, min_after
        self.cache: dict[tuple[int, str], list[int]] = {}

    def __call__(self, run: BaseRun, phase: str) -> list[int]:
        key = (id(run), phase)
        if key not in self.cache:
            self.cache[key] = cut_candidates(run, self.tok, self.ms, phase, self.min_before, self.min_after)
        return self.cache[key]


def _draw(pool: list[BaseRun], order: list[int], cuts: _Cuts, phase: str | None) -> tuple[BaseRun, str | None] | None:
    """Take the next run of the shuffled pool that has room for an event in `phase` (the other phase
    if no run has); runs passed over stay in the pool. `phase` None: any run."""
    if phase is None:
        return (pool[order.pop()], None) if order else None
    for wanted in (phase, "answer" if phase == "thinking" else "thinking"):
        for j in range(len(order) - 1, -1, -1):
            if cuts(pool[order[j]], wanted):
                return pool[order.pop(j)], wanted
    return None


def plan_split(runs: list[BaseRun], split: str, rng, tok, ms: ModelSpec, n_episodes: int | None = None,
               n_events: int | None = None, n_openings: int = 0, min_before: int = 48,
               min_after: int = 64, slot_twins: bool = False) -> list[BgPlan]:
    """Train (`n_episodes`): event+twin pairs and nominal episodes, train kinds only; with `slot_twins`
    every event with 2-3 tasks also gets its slot twin (added on top of `n_episodes`; drawn from the
    plan's own seed, so `rng` and every other plan stay as without them). Test (`n_events`): events
    balanced over the four kinds, half with held-out kinds, each with its twin, plus `n_openings`
    opening probes."""
    pool = [r for r in runs for _ in range(MAX_USES)]
    order = list(rng.permutation(len(pool)))
    if split == "train":
        n_pairs = round(n_episodes * PAIR_SHARE / 2)
        n_nominal = n_episodes - 2 * n_pairs
        kinds = list(rng.choice(EVENT_KINDS, size=n_pairs, p=[TRAIN_EVENT_SHARE[k] for k in EVENT_KINDS]))
        wants = [("event", k, False) for k in kinds] + [("nominal", None, False)] * n_nominal
    else:
        kinds = [EVENT_KINDS[i % 4] for i in range(n_events)]
        held = [(i // 4) % 2 == 1 for i in range(n_events)]  # every event kind half seen, half held out
        wants = [("event", k, h) for k, h in zip(kinds, held)] + [("opening", None, i % 2 == 1) for i in range(n_openings)]
    wants = [wants[i] for i in rng.permutation(len(wants))]
    cuts = _Cuts(tok, ms, min_before, min_after)
    phases = iter(["thinking", "answer"] * len(wants))  # events alternate, so the two phases are balanced
    plans: list[BgPlan] = []
    n_slots = 0
    for want, kind, held_out in wants:
        drawn = _draw(pool, order, cuts, next(phases) if want == "event" else None)
        if drawn is None:
            break  # pool exhausted
        run, phase = drawn
        n = int(rng.integers(1, 4))
        tasks = sample_tasks(n, HELD_OUT_KINDS if held_out else TRAIN_KINDS, rng, held_out=held_out)
        policy = "yes" if rng.random() < 0.5 else "no"
        # numbered without the slot twins (they take their original's id + "-slot")
        plan = BgPlan(episode_id=f"{split}-{len(plans) - n_slots:05d}", run=run, split=split, tasks=tasks,
                      policy=policy, event=None, seed=int(rng.integers(2**31)))
        if want == "event":
            options = cuts(run, phase)
            plan.event = BgEvent(str(kind), int(rng.integers(n)), int(options[int(rng.integers(len(options)))]), phase)
            plans += [plan, twin_plan(plan, near_miss=bool(rng.random() < NEAR_MISS_SHARE))]
            if slot_twins and split == "train" and n > 1:
                plans.append(slot_plan(plan))
                n_slots += 1
        else:
            plan.opening = want == "opening"
            plans.append(plan)
    return plans


def usable_runs(runs: list[BaseRun], ms: ModelSpec) -> list[BaseRun]:
    """Math and explain base runs without any tool call: a phase-1 tool (e.g. web_search) is not
    offered in the background-task prompt, so a run that calls one would teach calling it."""
    return [r for r in runs if r.task.family in ("math", "explain") and ms.tool_open not in r.target_ids]


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec

    tok = AutoTokenizer.from_pretrained(model)
    return tok, resolve_spec(model, tok)


def _make_format(tok):
    from prio.bg.protocol import QwenFormat

    return QwenFormat(tok)


def _load_runs(args, tok, ms: ModelSpec) -> list[BaseRun]:
    from prio.tasks import build_prompt, load_tasks

    tasks = load_tasks(args.base_train, args.base_test, args.base_seed)
    prompts = [build_prompt(tok, t) for t in tasks]
    return load_cached_base_runs(args.base / "base_runs.json", {"model": args.model, "seed": args.base_seed},
                                 tasks, prompts, tok, ms)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the background-task episodes from cached base runs.")
    parser.add_argument("--base", type=Path, required=True, help="phase-1 data folder holding base_runs.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=4500)
    parser.add_argument("--test-events", type=int, default=400)
    parser.add_argument("--test-openings", type=int, default=100)
    parser.add_argument("--base-train", type=int, default=3000, help="tasks of the phase-1 build (its cache key)")
    parser.add_argument("--base-test", type=int, default=600)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--min-before", type=int, default=48)
    parser.add_argument("--min-after", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--slot-twins", action="store_true",
                        help="train: add the slot twin of every event with 2-3 tasks (event moved to another task)")
    args = parser.parse_args(argv)

    started = time.time()
    rng = np.random.default_rng(args.seed)
    tok, ms = _load_tokenizer(args.model)
    fmt = _make_format(tok)
    runs = usable_runs(_load_runs(args, tok, ms), ms)
    by_split = {s: [r for r in runs if r.task.split == s] for s in ("train", "test")}
    margins = {"min_before": args.min_before, "min_after": args.min_after}
    plans = {"train": plan_split(by_split["train"], "train", rng, tok, ms, n_episodes=args.train,
                                 slot_twins=args.slot_twins, **margins),
             "test": plan_split(by_split["test"], "test", rng, tok, ms, n_events=args.test_events,
                                n_openings=args.test_openings, **margins)}
    args.out.mkdir(parents=True, exist_ok=True)
    written, dropped = {}, {}
    for split, split_plans in plans.items():
        episodes = [assemble(p, fmt, tok, ms) for p in split_plans]
        by_id = {e.episode_id: e for e in episodes}
        for e in episodes:
            if e.kind == "bg_twin":
                align_divergence(by_id[e.meta["twin_of"]], e)
        for e in episodes:  # a slot twin's decision: where it parts from the base text (its original's twin)
            if e.meta.get("slot_of"):
                e.t_diverge = first_difference(e, by_id[e.meta["slot_of"] + "-twin"])
        kept = [e for e in episodes if len(e.prompt_ids) + len(e.target_ids) <= args.max_len]
        dropped[split] = len(episodes) - len(kept)
        write_jsonl(args.out / f"{split}.jsonl", kept)
        written[split] = kept
    lengths = [len(e.prompt_ids) + len(e.target_ids) for eps in written.values() for e in eps]
    events = lambda eps: [e for e in eps if e.kind == "bg_event"]
    manifest = {
        "model": args.model, "seed": args.seed, "base": str(args.base),
        "base_runs_used": {s: len({p.run.task.task_id for p in ps}) for s, ps in plans.items()},
        "episodes": {s: len(e) for s, e in written.items()},
        "kinds": {s: dict(Counter(e.kind for e in eps)) for s, eps in written.items()},
        "event_kinds": {s: dict(Counter(e.signal for e in events(eps))) for s, eps in written.items()},
        "phases": {s: dict(Counter(e.phase for e in events(eps))) for s, eps in written.items()},
        "watched": {s: dict(Counter(len(e.meta["tasks"]) for e in eps)) for s, eps in written.items()},
        "held_out_events": sum(e.meta["held_out"] for e in events(written["test"])),
        "near_miss_twins": {s: sum(e.kind == "bg_twin" and e.meta["near_miss"] for e in eps) for s, eps in written.items()},
        "slot_twins": {s: sum(bool(e.meta.get("slot_of")) for e in eps) for s, eps in written.items()},
        "dropped_long": dropped, "max_len": args.max_len,
        "length_p50": float(np.median(lengths)) if lengths else None, "length_max": max(lengths, default=None),
        "minutes": (time.time() - started) / 60,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
