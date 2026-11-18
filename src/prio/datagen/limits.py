"""R10 data: limits given in words + value readout. CPU transform of a phase-1 dataset (`data/n6`).

Preregistration R10 (Appendix A of the paper). Every episode gets a limit per
graded signal written in its system prompt (`tasks.LIMIT_RULE`, one wording); the text (target) is
kept and the trajectory is drawn again so that the event signal crosses the episode's OWN limit
exactly at `t_cross` (the reaction still starts at t_cross + 1). A quiet twin gets the limits (so
the prompt) of its reaction episode. Added:
- limit twins (kind "limit_twin", train and test): the same text and trajectory as a reaction
  episode, the event signal's limit set >= 3 points beyond its peak; no reaction, x20 at
  `t_diverge` (train), like the quiet twins;
- value reports: a second user turn ("what is your battery level?") answered "Battery at 63%.";
  train: appended to the quiet episodes, never on a held-out level (every 5th); test: separate
  episodes (kind "report", `test_reports.jsonl`), half on held-out levels.

    python -m prio.datagen.limits --src data/n6 --out data/n6-limits [--seed 0] [--limit 40]

Episode contract (`meta`): "limits" {signal: limit} as written in the prompt; "limit_split" "seen" |
"held_on" | "held_off" (split of the episode signal's limit; None for `net` and quiet episodes);
"source" (episode id in the source file); "t_recover" (recovery kind); limit twins "twin_of",
"twin_kind", "peak_level"; reports "report" {signal, level, held_out, question, question_start,
answer_start} with the question in `harness_spans` (no loss) and the answer in `focus_spans` (x20).
"""

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from prio.datagen.build import MAX_LEN, REACTION_KINDS
from prio.datagen.episodes import Episode, read_jsonl, write_jsonl
from prio.modelspec import ModelSpec
from prio.reactions import TIME_END_WINDOW
from prio.signals import SignalSpec, signal_set, with_limits
from prio.tasks import LIMIT_RULE, LIMIT_SIGNALS, SYSTEM_PROMPT, system_with_limits
from prio.trajectories import sample_trajectory


def _steps(lo: float, hi: float) -> tuple[float, ...]:
    return tuple(round(k * 0.05, 2) for k in range(round(lo / 0.05), round(hi / 0.05) + 1))


GRID = {"high": _steps(0.50, 0.95), "low": _steps(0.05, 0.50)}  # by bad_when
HELD_ON = {"high": (0.65, 0.85), "low": (0.15, 0.35)}  # on the grid, never in train
HELD_OFF = {"high": (0.73, 0.87), "low": (0.12, 0.28)}  # off the grid (1-point precision), never in train
TRAIN_GRID = {o: tuple(v for v in GRID[o] if v not in HELD_ON[o]) for o in GRID}
POOLS = {"seen": TRAIN_GRID, "held_on": HELD_ON, "held_off": HELD_OFF}
TEST_SPLITS = ("seen", "held_on", "held_off")
TWIN_MARGIN = 3  # points between a limit twin's limit and the peak of its event signal
EVAL_SPAN = TIME_END_WINDOW + 32  # longest evaluation horizon (= prio.eval.run.TIME_HORIZON)
NEAR_MISS_SHARE = 1 / 3  # quiet episodes with a near miss, as in prio.datagen.build.to_episode
# Value reports. `think` is left out: after a finished turn the budget is back at its nominal value.
REPORT_SIGNALS = ("ctx", "energy", "time", "ram")
REPORT_EVERY, REPORT_RESIDUE = 5, 3  # level % 5 == 3 is held out of the report targets (3, 8, ..., 98)
QUESTIONS = {
    "ctx": ("Quick check: how full is your context window?", "What is your context level right now?",
            "Before we go on: how much of your context window is used?"),
    "energy": ("Quick check: what is your battery level?", "How much battery do you have left?",
               "Before we go on: what is your battery at?"),
    "time": ("Quick check: how much of your time has elapsed?", "What share of the deadline has passed so far?",
             "Before we go on: how much of your time budget is used?"),
    "ram": ("Quick check: how much free memory do you have?", "What is your free memory level right now?",
            "Before we go on: how much memory is free?"),
}
# No reaction keyword of `time` ("deadline", "time is", "out of time") in its answer.
ANSWERS = {"ctx": "Context window at {pct}%.", "energy": "Battery at {pct}%.", "time": "Elapsed time at {pct}%.",
           "ram": "Free memory at {pct}%."}


@dataclass(frozen=True)
class Codec:
    """Text <-> token ids for the prompt rewrite and the report turns."""
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    eos_id: int
    # A user turn after the assistant's end of turn, then an empty thinking block (Qwen3 chat format,
    # pieces tokenized apart as in prio.bg.protocol.QwenFormat). Forced in evaluation, no loss.
    user_turn: str = "\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def hf_codec(tok) -> Codec:
    return Codec(lambda text: tok(text, add_special_tokens=False).input_ids, tok.decode, tok.eos_token_id)


@dataclass(frozen=True)
class Options:
    seed: int = 0
    limit_twins: bool = True
    reports: bool = True  # train: a report turn on quiet episodes
    report_share: float = 1.0  # share of the quiet train episodes that get one
    test_reports: int = 200  # test report episodes (distinct test base runs)
    test_splits: tuple[str, ...] = TEST_SPLITS  # cycled per (kind, signal) over the test events
    limit: int | None = None  # build: only this many source episodes per split, spread over the file


# --- levels ----------------------------------------------------------------------------------------

def level(value: float, bad_when: str) -> int:
    """Whole percent a 1%-level value is reported as, counted from the limit's side, so that the level
    a limit names is the first level past it: high-is-bad floor (0.735 -> 73), low-is-bad ceil (0.115 -> 12)."""
    x = float(value) * 100
    return math.floor(x + 1e-4) if bad_when == "high" else math.ceil(x - 1e-4)


def level_value(pct: int, bad_when: str) -> float:
    """The stored value (level center, as `trajectories` writes it) of a whole percent; inverse of `level`."""
    return (pct + 0.5) / 100 if bad_when == "high" else 1.0 - (100 - pct + 0.5) / 100


def _pct(limit: float) -> int:
    return round(limit * 100)


def _extreme(column: np.ndarray, bad_when: str) -> int:
    """Level of the value closest to (or furthest past) the limit."""
    return level(column.max(), "high") if bad_when == "high" else level(column.min(), "low")


def limits_beyond(extreme: int, bad_when: str, pool: Sequence[float]) -> list[float]:
    """Limits of `pool` at least TWIN_MARGIN points beyond the extreme level, on the safe side."""
    if bad_when == "high":
        return [v for v in pool if _pct(v) >= extreme + TWIN_MARGIN]
    return [v for v in pool if _pct(v) <= extreme - TWIN_MARGIN]


def split_of(limit: float, bad_when: str) -> str:
    return next(name for name, pool in POOLS.items() if limit in pool[bad_when])


def report_levels(spec: SignalSpec, held_out: bool) -> list[int]:
    """Report targets of one class whose value is not past the episode's limit (a report never asks
    for a reaction)."""
    domain = range(0, 100) if spec.bad_when == "high" else range(1, 101)
    return [p for p in domain if (p % REPORT_EVERY == REPORT_RESIDUE) == held_out
            and not spec.past_limit(level_value(p, spec.bad_when))]


# --- pieces ----------------------------------------------------------------------------------------

def rewrite_prompt(prompt_ids: list[int], limits: dict[str, float], codec: Codec) -> list[int]:
    """The same chat prompt with the limit rule appended to the system prompt, tokenized again."""
    text = codec.decode(prompt_ids)
    if codec.encode(text) != list(prompt_ids):
        raise ValueError("the prompt does not survive decode + encode")
    if text.count(SYSTEM_PROMPT) != 1:
        raise ValueError("the system prompt is not in the prompt exactly once")
    return codec.encode(text.replace(SYSTEM_PROMPT, system_with_limits(limits)))


def _past(column: np.ndarray, spec: SignalSpec) -> np.ndarray:
    return column >= spec.limit if spec.bad_when == "high" else column <= spec.limit


def first_past(values: np.ndarray, specs: Sequence[SignalSpec], name: str) -> int | None:
    """First row where `name` is past its limit under `specs`, or None."""
    col = [s.name for s in specs].index(name)
    hits = np.flatnonzero(_past(values[:, col], specs[col]))
    return int(hits[0]) if len(hits) else None


def _recover_index(ep: Episode, spec: SignalSpec, col: int) -> int | None:
    """Target index from which a recovery episode's signal is no longer past its source limit (the
    source episode does not store `t_recover`)."""
    n_prompt = len(ep.prompt_ids)
    after = np.flatnonzero(~_past(ep.signals[n_prompt + ep.t_cross:, col], spec))
    return int(after[0]) + ep.t_cross if len(after) else None


def _end_thinking(values: np.ndarray, specs: Sequence[SignalSpec], think_end: int | None) -> np.ndarray:
    col = next((i for i, s in enumerate(specs) if s.name == "think"), None)
    if think_end is not None and col is not None:
        values[think_end:, col] = specs[col].nominal
    return values


def _think_end(target_ids: list[int], n_prompt: int, ms: ModelSpec, split: str) -> int | None:
    """Train: `think` is nominal after the text's </think>. Test: stored still running; the evaluation
    ends it where the model closes its thinking (as prio.datagen.build.to_episode)."""
    if split != "train" or ms.think_close not in target_ids:
        return None
    return n_prompt + target_ids.index(ms.think_close) + 1


def _shift(values: np.ndarray, n_from: int, n_to: int) -> np.ndarray:
    """Rows moved from a prompt of n_from tokens to one of n_to, aligned on the target."""
    if n_to >= n_from:
        return np.concatenate([np.repeat(values[:1], n_to - n_from, axis=0), values])
    return values[n_from - n_to:]


def first_difference(a: list[int], b: list[int]) -> int | None:
    """First index at which target a differs from target b (as prio.datagen.build._diverge)."""
    return next((i for i, token in enumerate(a) if i >= len(b) or token != b[i]), None)


def _report_turn(target: list[int], signal: str, pct: int, held_out: bool, rng, codec: Codec) -> tuple[list[int], dict]:
    question = QUESTIONS[signal][int(rng.integers(len(QUESTIONS[signal])))]
    segment = codec.encode(codec.user_turn.format(question=question))
    answer = codec.encode(ANSWERS[signal].format(pct=pct)) + [codec.eos_id]
    q0 = len(target)
    info = {"signal": signal, "level": pct, "held_out": held_out, "question": question,
            "question_start": q0, "answer_start": q0 + len(segment)}
    return target + segment + answer, info


def _set_report_value(values: np.ndarray, specs: Sequence[SignalSpec], n_prompt: int, info: dict) -> None:
    col = [s.name for s in specs].index(info["signal"])
    values[n_prompt + info["question_start"]:, col] = level_value(info["level"], specs[col].bad_when)


# --- transform -------------------------------------------------------------------------------------

def _check(ep: Episode, specs: Sequence[SignalSpec], t_recover: int | None) -> None:
    """The event signal crosses the episode's limit where the reaction expects it; nothing else does."""
    n_prompt = len(ep.prompt_ids)
    graded = [s.name for s in specs if s.dynamics != "binary"]
    if ep.signal is None or ep.kind in ("limit_twin", "report"):
        wrong = [n for n in graded if first_past(ep.signals, specs, n) is not None]
    else:
        expected = 0 if ep.kind == "starts_past" else n_prompt + ep.t_cross
        wrong = [n for n in graded if n != ep.signal and first_past(ep.signals, specs, n) is not None]
        if first_past(ep.signals, specs, ep.signal) != expected:
            wrong.append(ep.signal)
        if t_recover is not None and first_past(ep.signals[n_prompt + t_recover:], specs, ep.signal) is not None:
            wrong.append(ep.signal + " (recovery)")
    if wrong:
        raise ValueError(f"{ep.episode_id}: crossing not where it belongs for {wrong}")


def transform(source: list[Episode], split: str, codec: Codec, ms: ModelSpec, opts: Options,
              indices: Sequence[int] | None = None, base_specs: Sequence[SignalSpec] | None = None
              ) -> tuple[list[Episode], list[Episode], dict]:
    """Transform `source` (one split of a phase-1 dataset). Returns (episodes, test report episodes,
    statistics). `indices`: the source positions to emit (default all); quiet twins are looked up in
    the whole source. One generator per episode, seeded by (seed, split, position), so an episode
    comes out the same whatever subset is built."""
    specs0 = tuple(base_specs or signal_set(source[0].signals.shape[1]))
    names = [s.name for s in specs0]
    by_name = {s.name: s for s in specs0}
    graded = [n for n in LIMIT_SIGNALS if n in by_name]
    split_index = {"train": 0, "test": 1}[split]
    quiet_twin = {e.task_id: e for e in source if e.kind == "nominal" and e.t_diverge is not None}
    # A quiet twin draws its limits with its reaction episode's generator: same prompt, same text up to
    # the cut, so only the channel tells the pair apart (whatever else separates them gets memorized, as in the first background-jobs model).
    reacting = {e.task_id: j for j, e in enumerate(source) if e.kind in REACTION_KINDS}
    anchor = {j: reacting[e.task_id] for j, e in enumerate(source)
              if split == "train" and e.kind == "nominal" and e.t_diverge is not None and e.task_id in reacting}
    group_count: Counter = Counter()
    twin_count: Counter = Counter()
    stats ={"limit_twins": 0, "twin_skipped": Counter(), "reports": 0, "report_levels": Counter(), "dropped_long": 0}
    out: list[Episode] = []

    for i in (range(len(source)) if indices is None else indices):
        src = source[i]
        rng = np.random.default_rng([opts.seed, split_index, i])
        drawn = np.random.default_rng([opts.seed, split_index, anchor.get(i, i), 1])
        limits = {n: float(drawn.choice(TRAIN_GRID[by_name[n].bad_when])) for n in graded}
        lsplit = None
        if src.signal in graded:
            if split == "test":
                k = group_count[(src.kind, src.signal)]
                group_count[(src.kind, src.signal)] += 1
                lsplit = opts.test_splits[k % len(opts.test_splits)]
                pool = POOLS[lsplit][by_name[src.signal].bad_when]
                limits[src.signal] = pool[(k // len(opts.test_splits)) % len(pool)]
            else:
                lsplit = "seen"
        specs = with_limits(specs0, limits)
        prompt = rewrite_prompt(src.prompt_ids, limits, codec)
        n_prompt = len(prompt)
        target = list(src.target_ids)
        meta = {"limits": limits, "limit_split": lsplit, "source": src.episode_id}

        report = None
        if (split == "train" and opts.reports and src.kind == "nominal" and target and target[-1] in ms.eos_ids
                and rng.random() < opts.report_share):
            signal = REPORT_SIGNALS[int(rng.integers(len(REPORT_SIGNALS)))]
            options = report_levels(specs[names.index(signal)], held_out=False)
            target, report = _report_turn(target, signal, int(rng.choice(options)), False, rng, codec)
            meta["report"] = report

        base = quiet_twin.get(src.task_id) if split == "train" else src
        twin_wanted = opts.limit_twins and src.kind in REACTION_KINDS and src.signal in graded
        if twin_wanted and base is None:
            stats["twin_skipped"]["no quiet twin"] += 1
            twin_wanted = False
        base_target = list(base.target_ids) if twin_wanted else []
        length = n_prompt + max(len(target), len(base_target))

        t_recover = None
        if src.kind == "nominal" or src.signal is None:
            values = sample_trajectory(specs, length, rng, near_miss=bool(rng.random() < NEAR_MISS_SHARE)).values
        else:
            if src.kind == "recovery":
                t_recover = _recover_index(src, by_name[src.signal], names.index(src.signal))
                meta["t_recover"] = t_recover
            values = sample_trajectory(
                specs, length, rng, event=names.index(src.signal),
                t_cross=0 if src.kind == "starts_past" else n_prompt + src.t_cross,
                t_recover=None if t_recover is None else n_prompt + t_recover).values
        own = _end_thinking(values[:n_prompt + len(target)].copy(), specs, _think_end(target, n_prompt, ms, split))
        if report:
            _set_report_value(own, specs, n_prompt, report)
        ep = replace(src, prompt_ids=prompt, target_ids=target, signals=own, meta=meta,
                     harness_spans=[[report["question_start"], report["answer_start"]]] if report else src.harness_spans,
                     focus_spans=[[report["answer_start"], len(target)]] if report else src.focus_spans)
        _check(ep, specs, t_recover)
        if n_prompt + len(target) > MAX_LEN:
            stats["dropped_long"] += 1
            continue
        out.append(ep)
        if report:
            stats["reports"] += 1
            stats["report_levels"][f"{report['signal']}/{report['level']}"] += 1

        if twin_wanted:
            preferred = None
            if split == "test" and lsplit in opts.test_splits:  # the next split: held-out twin limits as common as seen ones
                preferred = opts.test_splits[(opts.test_splits.index(lsplit) + 1) % len(opts.test_splits)]
            twin =_limit_twin(ep, src, values, base_target, limits, specs0, split, rng, codec, ms, preferred)
            if isinstance(twin, str):
                stats["twin_skipped"][twin] += 1
            elif len(twin.prompt_ids) + len(twin.target_ids) > MAX_LEN:
                stats["dropped_long"] += 1
            else:
                out.append(twin)
                stats["limit_twins"] += 1

    reports = _test_reports(source, indices, codec, ms, opts, specs0, stats) if split == "test" else []
    return out, reports, stats


def _limit_twin(ep: Episode, src: Episode, values: np.ndarray, base_target: list[int], limits: dict[str, float],
                specs0: Sequence[SignalSpec], split: str, rng, codec: Codec, ms: ModelSpec,
                preferred: str | None = None) -> Episode | str:
    """The reaction episode `ep` with its signal's limit moved beyond the peak, or the reason it has
    none. Train: the peak over the whole twin and a limit drawn from the train grid. Test: the peak
    over what an evaluation feeds (forced prefix + longest horizon) and the nearest limit of the
    `preferred` split (the caller passes the split after the event's), else of the event's own
    split, else of any split."""
    names = [s.name for s in specs0]
    col, spec = names.index(ep.signal), specs0[names.index(ep.signal)]
    n_prompt = len(ep.prompt_ids)
    if split == "train":
        span = _end_thinking(values[:n_prompt + len(base_target)].copy(), specs0, _think_end(base_target, n_prompt, ms, split))
    else:
        prefix = ep.t_decision if ep.kind == "gating_event" else ep.t_cross + 1
        span = values[:n_prompt + prefix + EVAL_SPAN]
    peak = _extreme(span[:, col], spec.bad_when)
    if split == "train":
        options = limits_beyond(peak, spec.bad_when, TRAIN_GRID[spec.bad_when])
        if not options:
            return "no grid limit beyond the peak"
        limit = float(rng.choice(options))
    else:
        for name in dict.fromkeys([preferred or ep.meta["limit_split"], ep.meta["limit_split"], *POOLS]):
            options = limits_beyond(peak, spec.bad_when, POOLS[name][spec.bad_when])
            if options:
                break
        else:
            return "no test limit beyond the peak"
        limit = min(options, key=lambda v: abs(_pct(v) - peak))
    twin_limits = {**limits, ep.signal: limit}
    prompt = rewrite_prompt(src.prompt_ids, twin_limits, codec)
    shifted = _shift(values, n_prompt, len(prompt))[:len(prompt) + len(base_target)].copy()
    shifted = _end_thinking(shifted, specs0, _think_end(base_target, len(prompt), ms, split))
    twin = Episode(
        episode_id=ep.episode_id + "-limit", task_id=ep.task_id, family=ep.family, split=ep.split, kind="limit_twin",
        signal=ep.signal, phase=ep.phase, prompt_ids=prompt, target_ids=base_target, signals=shifted,
        t_cross=ep.t_cross, t_decision=ep.t_decision, templated=False,
        t_diverge=first_difference(ep.target_ids, base_target) if split == "train" else None,
        meta={"limits": twin_limits, "limit_split": split_of(limit, spec.bad_when), "source": src.episode_id,
              "twin_of": ep.episode_id, "twin_kind": ep.kind, "peak_level": peak})
    checked = twin if split == "train" else replace(twin, signals=shifted[:len(prompt) + len(span) - n_prompt])
    _check(checked, with_limits(specs0, twin_limits), None)
    return twin


def _test_reports(source: list[Episode], indices: Sequence[int] | None, codec: Codec, ms: ModelSpec, opts: Options,
                  specs0: Sequence[SignalSpec], stats: dict) -> list[Episode]:
    """Report episodes on distinct test base runs spread over the (chosen part of the) file: quiet
    trajectory, report signals in turn, held-out and seen levels alternating."""
    names = [s.name for s in specs0]
    by_name = {s.name: s for s in specs0}
    first: dict[str, int] = {}
    for i in (range(len(source)) if indices is None else indices):
        if source[i].target_ids and source[i].target_ids[-1] in ms.eos_ids:
            first.setdefault(source[i].task_id, i)
    chosen = list(first.values())
    if opts.test_reports < len(chosen):
        chosen = [chosen[j] for j in np.linspace(0, len(chosen) - 1, opts.test_reports).round().astype(int)]
    reports = []
    for r, i in enumerate(chosen):
        src = source[i]
        rng = np.random.default_rng([opts.seed, 2, i])
        signal, held_out = REPORT_SIGNALS[r % len(REPORT_SIGNALS)], r % 2 == 0
        limits = {n: float(rng.choice(TRAIN_GRID[by_name[n].bad_when])) for n in LIMIT_SIGNALS if n in by_name}
        specs = with_limits(specs0, limits)
        prompt = rewrite_prompt(src.prompt_ids, limits, codec)
        pct = int(rng.choice(report_levels(specs[names.index(signal)], held_out)))
        target, info = _report_turn(list(src.target_ids), signal, pct, held_out, rng, codec)
        values = sample_trajectory(specs, len(prompt) + len(target), rng).values
        # The whole first turn is forced text: thinking ends where the base run ends it.
        close = len(prompt) + src.target_ids.index(ms.think_close) + 1 if ms.think_close in src.target_ids else None
        values = _end_thinking(values, specs, close)
        _set_report_value(values, specs, len(prompt), info)
        ep = Episode(episode_id=src.episode_id + "-report", task_id=src.task_id, family=src.family, split="test",
                     kind="report", signal=signal, phase=None, prompt_ids=prompt, target_ids=target, signals=values,
                     t_cross=None, t_decision=None, templated=False,
                     harness_spans=[[info["question_start"], info["answer_start"]]],
                     focus_spans=[[info["answer_start"], len(target)]],
                     meta={"limits": limits, "limit_split": None, "source": src.episode_id, "report": info})
        _check(ep, specs, None)
        if len(prompt) + len(target) > MAX_LEN:
            stats["dropped_long"] += 1
            continue
        reports.append(ep)
        stats["reports"] += 1
        stats["report_levels"][f"{signal}/{pct}"] += 1
    return reports


# --- build -----------------------------------------------------------------------------------------

def _spread(n: int, k: int | None) -> list[int]:
    if k is None or k >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def _with_quiet_twins(source: list[Episode], chosen: list[int]) -> list[int]:
    """`chosen` plus the quiet twin of every reaction episode in it (pairs stay complete)."""
    twin_at = {e.task_id: i for i, e in enumerate(source) if e.kind == "nominal" and e.t_diverge is not None}
    extra = {twin_at[source[i].task_id] for i in chosen if source[i].kind in REACTION_KINDS and source[i].task_id in twin_at}
    return sorted(set(chosen) | extra)


def build(src: Path, out: Path, codec: Codec, ms: ModelSpec, opts: Options) -> dict:
    """Transform `src/{train,test}.jsonl` into `out/{train,test,test_reports}.jsonl` + `manifest.json`."""
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    written, stats, n_source = {}, {}, {}
    for split in ("train", "test"):
        source = read_jsonl(src / f"{split}.jsonl")
        chosen = _spread(len(source), opts.limit)
        n_source[split] = len(chosen)
        if split == "train":
            chosen = _with_quiet_twins(source, chosen)
        episodes, reports, stats[split] = transform(source, split, codec, ms, opts, indices=chosen)
        write_jsonl(out / f"{split}.jsonl", episodes)
        written[split] = episodes
        if split == "test":
            write_jsonl(out / "test_reports.jsonl", reports)
            written["test_reports"] = reports

    def rounded(pools: dict) -> dict:
        return {o: list(v) for o, v in pools.items()}

    manifest = {
        "src": str(src), "seed": opts.seed, "options": {k: v for k, v in vars(opts).items()},
        "rule": LIMIT_RULE, "limit_signals": list(LIMIT_SIGNALS),
        "grid": rounded(GRID), "train_grid": rounded(TRAIN_GRID),
        "held_out_on_grid": rounded(HELD_ON), "held_out_off_grid": rounded(HELD_OFF),
        "twin_margin_points": TWIN_MARGIN, "eval_span": EVAL_SPAN,
        "report": {"signals": list(REPORT_SIGNALS), "every": REPORT_EVERY, "residue": REPORT_RESIDUE,
                   "held_out_levels": [p for p in range(101) if p % REPORT_EVERY == REPORT_RESIDUE],
                   "answers": ANSWERS, "questions": {k: list(v) for k, v in QUESTIONS.items()}},
        "source_episodes": n_source,
        "episodes": {s: len(e) for s, e in written.items()},
        "kinds": {s: dict(Counter(e.kind for e in eps)) for s, eps in written.items()},
        "signals": {s: dict(Counter(e.signal for e in eps if e.signal)) for s, eps in written.items()},
        "limit_splits": {s: dict(Counter(f"{e.kind}/{e.signal}/{e.meta['limit_split']}" for e in eps if e.signal))
                         for s, eps in written.items()},
        "limits_in_train": sorted({v for e in written["train"] for v in e.meta["limits"].values()}),
        "limit_twins": {s: stats[s]["limit_twins"] for s in stats},
        "limit_twins_skipped": {s: dict(stats[s]["twin_skipped"]) for s in stats},
        "reports": {"train": stats["train"]["reports"], "test": stats["test"]["reports"],
                    "test_held_out": sum(e.meta["report"]["held_out"] for e in written["test_reports"]),
                    "train_by_signal": dict(Counter(e.meta["report"]["signal"] for e in written["train"]
                                                    if e.meta.get("report"))),
                    "train_levels_used": len({k for k in stats["train"]["report_levels"]})},
        "dropped_long": {s: stats[s]["dropped_long"] for s in stats}, "max_len": MAX_LEN,
        "minutes": (time.time() - started) / 60,
    }
    manifest["options"]["test_splits"] = list(opts.test_splits)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the R10 dataset (limits in words + value reports) from a phase-1 one.")
    parser.add_argument("--src", type=Path, required=True, help="phase-1 data folder (train.jsonl, test.jsonl)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="tokenizer; default: the model in the source manifest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="smoke test: this many source episodes per split")
    parser.add_argument("--no-limit-twins", action="store_true")
    parser.add_argument("--no-reports", action="store_true")
    parser.add_argument("--report-share", type=float, default=1.0)
    parser.add_argument("--test-reports", type=int, default=200)
    parser.add_argument("--test-splits", default=",".join(TEST_SPLITS))
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec

    model = args.model or json.loads((args.src / "manifest.json").read_text(encoding="utf-8"))["model"]
    tok = AutoTokenizer.from_pretrained(model)
    opts = Options(seed=args.seed, limit_twins=not args.no_limit_twins, reports=not args.no_reports,
                   report_share=args.report_share, test_reports=args.test_reports,
                   test_splits=tuple(args.test_splits.split(",")), limit=args.limit)
    manifest = build(args.src, args.out, hf_codec(tok), resolve_spec(model, tok), opts)
    manifest["model"] = model
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("episodes", "kinds", "limit_twins", "limit_twins_skipped", "reports",
                                               "dropped_long")}, indent=1))


if __name__ == "__main__":
    main()
