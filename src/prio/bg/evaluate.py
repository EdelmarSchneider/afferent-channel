"""Evaluation of the background-task model with a simulated harness (spec section 6.6).

uv run python -m prio.bg.evaluate --run runs/bg3-wide-levels-s0 --data data/bg3 --out results/bg-wide-levels [--every 4]
    [--batch-size 16]

The prefix of each test episode is forced up to and including the event token; then the model
generates freely, the harness executes its tool calls, writes the answers into the stream and feeds
the channel values at every token, until a turn ends without a tool call or the budget is spent.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from prio.bg.harness import SimHarness
from prio.bg.score import WINDOW, BgRecord, BgRun, score_event, score_opening, score_twin
from prio.datagen.episodes import Episode, read_jsonl
from prio.eval.metrics import bootstrap_ci, wilson_ci
from prio.eval.run import _condition, finish_note, spread, start_note
from prio.generate import generate, generate_batch
from prio.reactions import parse_tool_calls

# New model tokens per run (harness tokens not counted). A run that must reach the end of the work (event
# runs, math twins: resumption is scored at the end) gets the rest of its reference continuation x 1.25 + 64,
# at least its floor below and at most `cap`.
BUDGETS = {"event": 256, "twin": 96, "zeroed": 64, "swap": 96, "base": 64, "opening": 160, "cap": 1024}
EVENT_KINDS = ("done", "error", "waiting", "stalled")


def _device(model):
    return model.projector.net[0].weight.device if model is not None else torch.device("cpu")


def _stops(tok, ms) -> set[int]:
    pad = getattr(tok, "pad_token_id", None)
    return set(ms.eos_ids) | ({pad} if pad is not None else set())


class EpisodeLoop:
    """One run of the harness loop as a state machine, one model turn at a time: `request()` gives
    the context to generate from, `advance(gen)` takes the model's turn and runs the harness on
    it. Several loops can share one `generate_batch` call per turn."""

    def __init__(self, fmt, tok, ms, ep: Episode, harness: SimHarness, prefix_len: int, budget: int):
        self.fmt, self.tok, self.ms, self.harness, self.budget = fmt, tok, ms, harness, budget
        self.ids = list(ep.prompt_ids) + list(ep.target_ids[:prefix_len])
        self.rows = [harness.row(p) for p in range(len(self.ids))]
        self.out, self.segments, self.rounds = [], [], []
        self.spent, self.stop = 0, "budget"
        self.done = self.spent >= budget

    def request(self):
        """(context ids, their rows [T, N], signal_fn, max_new_tokens) for the next model turn."""
        start, harness = len(self.ids), self.harness
        return (self.ids, np.asarray(self.rows, dtype=np.float32), lambda step, tokens: harness.row(start + step),
                self.budget - self.spent)

    def advance(self, gen) -> None:
        tokens = list(gen.token_ids)
        start = len(self.ids)
        self.spent += len(tokens)
        self.segments.append(("model", len(self.out), len(self.out) + len(tokens)))
        self.out += tokens
        self.ids += tokens
        self.rows += [self.harness.row(p) for p in range(start, len(self.ids))]
        calls = parse_tool_calls(tokens, self.tok, self.ms)
        self.rounds.append(calls)
        if gen.stop_reason != "eos" or not calls:
            self.stop, self.done = "budget" if gen.stop_reason != "eos" else "eos", True
            return
        payloads = self.harness.respond(calls)
        seg = self.fmt.segment(payloads)
        # Rows before the change: as in training.
        self.rows += [self.harness.row(p) for p in range(len(self.ids), len(self.ids) + len(seg))]
        self.harness.commit(len(self.ids) + len(seg) - 1)
        self.segments.append(("harness", len(self.out), len(self.out) + len(seg)))
        self.out += seg
        self.ids += seg
        self.done = self.spent >= self.budget

    def result(self) -> BgRun:
        return BgRun(self.out, self.segments, self.rounds, self.stop)


def run_episode(model, fmt, tok, ms, ep: Episode, harness: SimHarness, prefix_len: int, budget: int,
                condition: str) -> BgRun:
    """Force the prefix, then generate / execute tool calls / generate ... `condition` only labels
    the run: the caller switches the channel and the adapter (`_condition`)."""
    loop = EpisodeLoop(fmt, tok, ms, ep, harness, prefix_len, budget)
    device = _device(model)
    stops = _stops(tok, ms)
    while not loop.done:
        ids, rows, signal_fn, max_new = loop.request()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            gen = generate(model, torch.tensor([ids], device=device), torch.from_numpy(rows[None]).to(device),
                           signal_fn, max_new, stops)
        loop.advance(gen)
    return loop.result()


def run_episodes(model, tok, ms, loops: list[EpisodeLoop], batch_size: int, on_done) -> None:
    """Drive several loops (same condition) with `generate_batch`: up to `batch_size` loops share
    each call, one model turn each; a finished loop goes to `on_done(index, loop)` and its place
    goes to the next waiting one."""
    device = _device(model)
    stops = _stops(tok, ms)
    waiting = list(range(len(loops)))[::-1]
    active: list[int] = []
    while waiting or active:
        while waiting and len(active) < batch_size:
            index = waiting.pop()
            if loops[index].done:
                on_done(index, loops[index])
            else:
                active.append(index)
        if not active:
            continue
        requests = [loops[i].request() for i in active]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            gens = generate_batch(model, [r[0] for r in requests], [torch.from_numpy(r[1]) for r in requests],
                                  [r[2] for r in requests], [r[3] for r in requests], stops)
        for i, gen in zip(active, gens):
            loops[i].advance(gen)
        for i in [i for i in active if loops[i].done]:
            on_done(i, loops[i])
        active = [i for i in active if not loops[i].done]


def _jobs(episodes: list[Episode], every: int, base_n: int) -> list[tuple[str, Episode, int | None]]:
    """(condition, episode, swap slot) in a stable order: each selected event with its controls,
    then its twin; then the opening probes."""
    events = [e for e in episodes if e.kind == "bg_event"]
    groups = defaultdict(list)
    for e in events:
        groups[(e.signal, e.meta["held_out"])].append(e)
    chosen = {e.episode_id for g in groups.values() for e in g[::every]}
    events = [e for e in events if e.episode_id in chosen]
    twins = {e.meta["twin_of"]: e for e in episodes if e.kind == "bg_twin"}
    base = {e.episode_id for e in spread(events, base_n)} if base_n > 0 else set()
    jobs = []
    for e in events:
        jobs += [("on", e, None), ("zeroed", e, None)]
        n = len(e.meta["tasks"])
        if n >= 2:
            jobs.append(("swap", e, e.meta["event"]["slot"] % n + 1))
        if e.episode_id in base:
            jobs.append(("base", e, None))
        if e.episode_id in twins:
            jobs.append(("on", twins[e.episode_id], None))
    jobs += [("on", e, None) for e in [o for o in episodes if o.kind == "bg_opening"][::every]]
    return jobs


def _remaining(ep: Episode) -> int:
    """Model tokens of the reference continuation after the event token (harness tokens left out)."""
    harness = sum(b - a for a, b in ep.harness_spans or () if a > ep.t_cross)
    return len(ep.target_ids) - (ep.t_cross + 1) - harness


def _budget(budgets: dict, condition: str, ep: Episode) -> int:
    math = ep.meta.get("family") == "math"
    to_the_end = lambda floor: max(floor, min(budgets["cap"], int(1.25 * _remaining(ep)) + 64))
    if ep.kind == "bg_opening":
        return budgets["opening"]
    if ep.kind == "bg_twin":
        return to_the_end(budgets["twin"]) if math else budgets["twin"]
    if condition == "on":
        return to_the_end(budgets["event"])
    return budgets[condition]


def _score_and_save(run: BgRun, ep: Episode, condition: str, swap: int | None, tok, ms, out_dir: Path) -> BgRecord:
    if ep.kind == "bg_event":
        slot = swap or ep.meta["event"]["slot"]
        expected = next(t["task_id"] for t in ep.meta["tasks"] if t["slot"] == slot)
        record = score_event(run, ep, condition, tok, ms, expected)
    elif ep.kind == "bg_twin":
        record = score_twin(run, ep, condition, tok, ms)
    else:
        record = score_opening(run, ep, tok, ms)
    with open(out_dir / "records.partial.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    with open(out_dir / "generations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"episode_id": ep.episode_id, "condition": condition, "token_ids": run.token_ids,
                            "segments": run.segments, "stop": run.stop}) + "\n")
    return record


def evaluate_bg(model, fmt, tok, ms, episodes: list[Episode], out_dir: Path, every: int = 1, base_n: int = 100,
                budgets: dict | None = None, batch_size: int = 1) -> list[BgRecord]:
    """Run and score every job; resumable from `records.partial.jsonl` (one record per job).

    `batch_size` > 1: the jobs of one channel condition run `batch_size` at a time through
    `run_episodes` (one `generate_batch` call per model turn). Same records; the files get them
    as they finish, the returned list in the order of `batch_size` 1."""
    budgets = budgets or BUDGETS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "records.partial.jsonl"
    records = [BgRecord(**json.loads(line)) for line in partial.read_text(encoding="utf-8").splitlines()
               if line.strip()] if partial.exists() else []
    done = {(r.episode_id, r.condition) for r in records}
    jobs = _jobs(episodes, every, base_n)
    if batch_size > 1:
        return _evaluate_batched(model, fmt, tok, ms, jobs, records, done, budgets, out_dir, batch_size)
    for condition, ep, swap in jobs:
        if (ep.episode_id, condition) in done:
            continue
        harness = SimHarness.from_episode(ep, swap_slot=swap)
        prefix = 0 if ep.kind == "bg_opening" else ep.t_cross + 1
        with _condition(model, "on" if condition == "swap" else condition):  # swap: channel on, event moved
            run = run_episode(model, fmt, tok, ms, ep, harness, prefix, _budget(budgets, condition, ep), condition)
        records.append(_score_and_save(run, ep, condition, swap, tok, ms, out_dir))
        done.add((ep.episode_id, condition))
    return records


def _evaluate_batched(model, fmt, tok, ms, jobs, records, done, budgets, out_dir: Path, batch_size: int):
    rank = {(ep.episode_id, condition): i for i, (condition, ep, _) in enumerate(jobs)}
    todo = []
    for condition, ep, swap in jobs:
        if (ep.episode_id, condition) not in done:
            todo.append((condition, ep, swap))
            done.add((ep.episode_id, condition))
    for channel in ("on", "zeroed", "base"):  # swap: channel on, event moved
        group = [job for job in todo if ("on" if job[0] == "swap" else job[0]) == channel]
        loops = [EpisodeLoop(fmt, tok, ms, ep, SimHarness.from_episode(ep, swap_slot=swap),
                             0 if ep.kind == "bg_opening" else ep.t_cross + 1, _budget(budgets, condition, ep))
                 for condition, ep, swap in group]

        def finish(index: int, loop: EpisodeLoop) -> None:
            condition, ep, swap = group[index]
            records.append(_score_and_save(loop.result(), ep, condition, swap, tok, ms, out_dir))

        with _condition(model, channel):
            run_episodes(model, tok, ms, loops, batch_size, finish)
    return sorted(records, key=lambda r: rank.get((r.episode_id, r.condition), len(rank)))


def _rate(values) -> float | None:
    values = [bool(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _event_group(rs: list[BgRecord]) -> dict:
    reacted = [r for r in rs if r.reacted]
    delays = [r.delay for r in reacted if r.delay is not None]
    return {"n": len(rs), "reaction_rate": _rate([r.reacted for r in rs]),
            "reaction_ci": list(wilson_ci([bool(r.reacted) for r in rs])),
            "reaction_loose_rate": _rate([r.reacted_loose for r in rs]),
            "identification_rate": _rate([r.right_task for r in reacted]),
            "commands_rate": _rate([r.commands_ok for r in rs]),
            "report_rate": _rate([r.report_ok for r in reacted]), "closing_rate": _rate([r.closing_ok for r in rs]),
            "delay_median": float(np.median(delays)) if delays else None,
            "unnamed_call_rate": _rate([r.unnamed_call for r in rs])}


def _twin_group(rs: list[BgRecord]) -> dict:
    return {"n": len(rs), "false_alarm_rate": _rate([r.false_alarm for r in rs]),
            "false_alarm_ci": list(wilson_ci([bool(r.false_alarm) for r in rs]))}


def summarize_bg(records: list[BgRecord]) -> dict:
    groups = {}
    events = [r for r in records if r.kind == "event"]
    for condition in ("on", "zeroed", "swap", "base"):
        rs = [r for r in events if r.condition == condition]
        for kind in ("all",) + EVENT_KINDS:
            by_kind = [r for r in rs if kind == "all" or r.event_kind == kind]
            for split, keep in (("all", lambda r: True), ("seen", lambda r: not r.held_out),
                                ("held_out", lambda r: r.held_out), ("thinking", lambda r: r.phase == "thinking"),
                                ("answer", lambda r: r.phase == "answer"),
                                ("single", lambda r: r.n_watched == 1), ("multi", lambda r: (r.n_watched or 0) >= 2)):
                chosen = [r for r in by_kind if keep(r)]
                if chosen:
                    groups[f"{condition}|{kind}|{split}"] = _event_group(chosen)
    twins = [r for r in records if r.kind == "twin" and r.condition == "on"]
    for split, keep in (("all", lambda r: True), ("seen", lambda r: not r.held_out), ("held_out", lambda r: r.held_out)):
        chosen = [r for r in twins if keep(r)]
        if chosen:
            groups[f"on|twin|{split}"] = _twin_group(chosen)
    near = [r for r in twins if r.near_miss]
    if near:
        groups["on|near_miss|all"] = _twin_group(near)
    on = {r.episode_id: r for r in events if r.condition == "on"}
    pairs = [(on[r.episode_id.removesuffix("-twin")], r) for r in twins
             if r.family == "math" and r.episode_id.removesuffix("-twin") in on]
    diffs = [float(bool(e.math_correct)) - float(bool(t.math_correct)) for e, t in pairs]
    correct_twin = [e for e, t in pairs if t.math_correct]
    math = {"n_pairs": len(pairs), "difference": float(np.mean(diffs)) if diffs else None,
            "difference_ci": list(bootstrap_ci(diffs)) if diffs else None,
            "event_accuracy": _rate([e.math_correct for e, _ in pairs]),
            "twin_accuracy": _rate([t.math_correct for _, t in pairs]),
            "n_correct_twin": len(correct_twin), "accuracy_when_twin_correct": _rate([e.math_correct for e in correct_twin])}
    explain = [r.resumed for r in events if r.condition == "on" and r.family != "math"]
    openings = [r for r in records if r.kind == "opening"]
    opening = {"n": len(openings),
               "started_rate": _rate([r.opening_started == r.n_watched for r in openings]),
               "watched_rate": _rate([r.opening_watched == r.n_watched for r in openings])}
    return {"kind": "bg", "groups": groups,
            "resumption": {"math": math, "explain_rate": _rate(explain), "explain_n": sum(v is not None for v in explain)},
            "opening": opening}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def write_bg_summary(records: list[BgRecord], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    summary = summarize_bg(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    lines = ["# Background-task evaluation summary", "",
             "- Reaction: a sentence naming a watched task, then a reaction command, starting within "
             f"{WINDOW} tokens of the event. Identification: the named task is the one whose slot changed. "
             "Commands: the first round is exactly the expected commands on that task.",
             "- Intervals: 95% Wilson score; math resumption difference: paired bootstrap.", "",
             "| Group | n | Reaction | 95% CI | Identification | Commands | Report | Closing | Delay med | Unnamed calls |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, g in summary["groups"].items():
        if "reaction_rate" in g:
            lo, hi = g["reaction_ci"]
            lines.append(f"| {key} | {g['n']} | {_pct(g['reaction_rate'])} | [{100 * lo:.1f}, {100 * hi:.1f}] | "
                         f"{_pct(g['identification_rate'])} | {_pct(g['commands_rate'])} | {_pct(g['report_rate'])} | "
                         f"{_pct(g['closing_rate'])} | {g['delay_median'] if g['delay_median'] is not None else '-'} | "
                         f"{_pct(g['unnamed_call_rate'])} |")
    lines += ["", "| Twins (nothing happens) | n | False alarms | 95% CI |", "|---|---|---|---|"]
    for key, g in summary["groups"].items():
        if "false_alarm_rate" in g:
            lo, hi = g["false_alarm_ci"]
            lines.append(f"| {key} | {g['n']} | {_pct(g['false_alarm_rate'])} | [{100 * lo:.1f}, {100 * hi:.1f}] |")
    m = summary["resumption"]["math"]
    difference = "-" if m["difference"] is None else f"{100 * m['difference']:+.1f} points"
    if m["difference_ci"]:
        lo, hi = m["difference_ci"]
        difference += f" [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    lines += ["", "## Resumption", "",
              f"- Math, paired with the twin (same problem, no event): {m['n_pairs']} pairs; accuracy with the event "
              f"{_pct(m['event_accuracy'])}, without {_pct(m['twin_accuracy'])}; difference {difference}; "
              f"accuracy when the twin is right: {_pct(m['accuracy_when_twin_correct'])} (n = {m['n_correct_twin']}).",
              f"- Explanations resumed without restarting: {_pct(summary['resumption']['explain_rate'])} "
              f"(n = {summary['resumption']['explain_n']}).", "",
              "## Opening (the model starts and watches the tasks itself)", "",
              f"- n = {summary['opening']['n']}; all tasks started: {_pct(summary['opening']['started_rate'])}; "
              f"all watched: {_pct(summary['opening']['watched_rate'])}.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    from prio.bg.protocol import N_INPUTS, QwenFormat
    from prio.train import load_run

    parser = argparse.ArgumentParser(description="Evaluate a background-task run on its test episodes.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="defaults to the model recorded in the run's train_log.json")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--base-n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="episodes advanced together, one model turn each (1 = one at a time; same records)")
    args = parser.parse_args(argv)

    from prio.train import channel_record

    model_id = args.model or json.loads((args.run / "train_log.json").read_text(encoding="utf-8"))["model"]
    start_note(args.out, {"run": str(args.run), "data": str(args.data), "model": model_id, "kind": "bg",
                          "channel": channel_record(args.run),
                          "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
               started=time.strftime("%Y-%m-%d %H:%M"))
    model, tok, ms = load_run(model_id, args.run, N_INPUTS)
    records = evaluate_bg(model, QwenFormat(tok), tok, ms, read_jsonl(args.data / "test.jsonl"), args.out,
                          every=args.every, base_n=args.base_n, batch_size=args.batch_size)
    write_bg_summary(records, args.out)
    finish_note(args.out, finished=time.strftime("%Y-%m-%d %H:%M"))
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
