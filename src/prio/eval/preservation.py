"""R11 capability preservation (preregistration R11; reading fixed in amendment A4, 2026-11-16).

Untrained base model (adapter and channel off) and trained model through the same batched loop:
greedy, 1,024 new tokens, nominal signals, no notice.
- GSM8K: the first 300 `math` test tasks in `load_tasks`' order (seed 1); correct = the last number
  of the answer equals the reference.
- Explain-format check: the first 100 `explain` test tasks; format kept = thinking closed, non-empty
  text answer, no tool call, no reaction.
- Line for both: paired difference (trained minus base) >= -2 points and its 95% CI lower bound
  >= -5 (paired bootstrap); otherwise reported as a cost.
- Beside, no line: reactions (false alarms), unnamed calls, tool calls, turns ended within the
  limit, median answer length.
Resumable: `items.jsonl` holds one row per task and condition, `generations.jsonl` the text of each turn
(to read flagged items by hand). One batch size per folder (batched generation can flip rare near-tie tokens).

python -m prio.eval.preservation --run runs/n6-wide-levels-s0 --out results/r11-wide-levels-s0-b16 --batch-size 16
"""

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from prio.eval.metrics import bootstrap_ci, wilson_ci
from prio.eval.run import (
    _condition,
    _run_batch,
    _think_index,
    extract_number,
    finish_note,
    follow_thinking,
    start_note,
    stop_ids,
)
from prio.reactions import first_reaction, unnamed_call
from prio.signals import SignalSpec, signal_set
from prio.tasks import Task, build_prompt, family_tasks
from prio.trajectories import sample_trajectory

MAX_NEW, MIN_DIFFERENCE, MIN_LOWER = 1024, -0.02, -0.05
CONDITIONS = ("base", "on")  # base = adapter and channel off, through the same loop
TASK_SEED = 1  # the phase-1 preservation check drew its GSM8K problems with this seed


def judge_format(token_ids: list[int], stop_reason: str, tok, ms, specs: Sequence[SignalSpec]) -> dict:
    """Form of one full turn: is it a closed thought followed by a text answer, with no tool and no reaction."""
    ids = list(token_ids)
    closed = ms.think_close in ids or ms.think_open not in ids
    start = len(ids) - ids[::-1].index(ms.think_close) if ms.think_close in ids else 0
    stops = stop_ids(tok, ms)
    answer = [t for t in ids[start:] if t not in stops] if closed else []
    tool = ms.tool_open in ids
    reaction = first_reaction(ids, stop_reason, specs, tok, ms) is not None
    return {"format": bool(closed and tok.decode(answer).strip() and not tool and not reaction),
            "ended": stop_reason == "eos", "tool_call": tool, "reaction": reaction, "answer_tokens": len(answer)}


def preserved(difference: float, ci_low: float) -> bool:
    return difference >= MIN_DIFFERENCE - 1e-9 and ci_low >= MIN_LOWER - 1e-9


def paired(base: list[bool], tuned: list[bool]) -> dict:
    """Base and trained model on the same items: both rates, the paired difference and the line."""
    difference = [float(t) - float(b) for b, t in zip(base, tuned)]
    mean, ci = float(np.mean(difference)), bootstrap_ci(difference)
    return {"n": len(base), "base": float(np.mean(base)), "base_ci": list(wilson_ci(base)),
            "trained": float(np.mean(tuned)), "trained_ci": list(wilson_ci(tuned)),
            "difference": mean, "ci": list(ci), "preserved": preserved(mean, ci[0])}


def _judge(family: str, task: Task, gen, tok, ms, specs) -> dict:
    row = judge_format(gen.token_ids, gen.stop_reason, tok, ms, specs)
    row["unnamed_call"] = bool(unnamed_call(gen.token_ids, gen.stop_reason, specs, tok, ms))
    if family == "math":
        row["correct"] = extract_number(tok.decode(gen.token_ids)) == float(task.answer)
    return {**row, "stop_reason": gen.stop_reason, "n_tokens": len(gen.token_ids)}


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: dict, math_tasks: list[Task], explain_tasks: list[Task]) -> dict:
    def column(tasks, condition, key):
        return [bool(rows[(t.task_id, condition)][key]) for t in tasks]

    beside = {}
    for family, tasks in (("math", math_tasks), ("explain", explain_tasks)):
        beside[family] = {
            condition: {**{key: float(np.mean(column(tasks, condition, key)))
                           for key in ("format", "ended", "tool_call", "reaction", "unnamed_call")},
                        "median_answer_tokens": float(np.median([rows[(t.task_id, condition)]["answer_tokens"] for t in tasks]))}
            for condition in CONDITIONS}
    return {"gsm8k": paired(column(math_tasks, "base", "correct"), column(math_tasks, "on", "correct")),
            "explain": paired(column(explain_tasks, "base", "format"), column(explain_tasks, "on", "format")),
            "beside": beside}


def run(model, tok, ms, specs: Sequence[SignalSpec], math_tasks: list[Task], explain_tasks: list[Task], out_dir: Path, *,
        seed: int = 0, batch_size: int = 16, max_new_tokens: int = MAX_NEW) -> dict:
    """Generate what `items.jsonl` does not hold yet (per family: base, then trained) and summarize."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = out_dir / "items.jsonl"
    rows = {(r["task_id"], r["condition"]): r for r in _read(items)}
    think = _think_index(specs)
    nominal_think = specs[think].nominal if think is not None else 1.0
    stops, started = stop_ids(tok, ms), time.time()
    for family, tasks in (("math", math_tasks), ("explain", explain_tasks)):
        jobs = []
        for index, task in enumerate(tasks):
            prompt = build_prompt(tok, task)
            values = sample_trajectory(specs, len(prompt) + max_new_tokens, np.random.default_rng([seed, index])).values
            jobs.append((task, prompt, values))
        for condition in CONDITIONS:
            todo = [job for job in jobs if (job[0].task_id, condition) not in rows]
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                signal_fns = [follow_thinking(v, len(p), think, nominal_think, ms.think_close) for _, p, v in chunk]
                with _condition(model, condition):
                    gens = _run_batch(model, [p for _, p, _ in chunk], [v for _, _, v in chunk],
                                      [max_new_tokens] * len(chunk), stops, signal_fns)
                # the text first: an item row never exists without the turn it judges (read by hand)
                with open(out_dir / "generations.jsonl", "a", encoding="utf-8") as texts:
                    for (task, _, _), gen in zip(chunk, gens):
                        texts.write(json.dumps({"task_id": task.task_id, "condition": condition,
                                                "text": tok.decode(gen.token_ids)}) + "\n")
                with open(items, "a", encoding="utf-8") as f:
                    for (task, _, _), gen in zip(chunk, gens):
                        row = {"task_id": task.task_id, "family": family, "condition": condition,
                               **_judge(family, task, gen, tok, ms, specs)}
                        rows[(task.task_id, condition)] = row
                        f.write(json.dumps(row) + "\n")
                print(f"preservation {family} {condition}: {min(start + batch_size, len(todo))}/{len(todo)}, "
                      f"{(time.time() - started) / 60:.1f} min", flush=True)
    return summarize(rows, math_tasks, explain_tasks)


def _line(name: str, s: dict) -> str:
    return (f"| {name} | {s['n']} | {s['base']:.1%} [{s['base_ci'][0]:.1%}, {s['base_ci'][1]:.1%}] | "
            f"{s['trained']:.1%} [{s['trained_ci'][0]:.1%}, {s['trained_ci'][1]:.1%}] | {100 * s['difference']:+.1f} "
            f"[{100 * s['ci'][0]:+.1f}, {100 * s['ci'][1]:+.1f}] | {'preserved' if s['preserved'] else 'COST'} |")


def write_summary(summary: dict, out_dir: Path, run_name: str) -> str:
    lines = [f"# Capability preservation (R11, amendment A4): {run_name}", "",
             "- Base = the same weights with adapter and channel off, through the same batched loop (greedy, "
             f"{MAX_NEW} new tokens, nominal signals, no notice).",
             "- GSM8K: the last number of the answer equals the reference. Explain: format kept = thinking closed, "
             "non-empty text answer, no tool call, no reaction.",
             f"- Line (fixed before any run): preserved if the paired difference is >= {100 * MIN_DIFFERENCE:.0f} points and "
             f"its 95% CI lower bound >= {100 * MIN_LOWER:.0f} (paired bootstrap); otherwise a cost.", "",
             "| Check | n | Base [95% Wilson] | Trained [95% Wilson] | Difference, points [95% CI] | Line |", "|---|---|---|---|---|---|",
             _line("GSM8K accuracy", summary["gsm8k"]), _line("Explain format kept", summary["explain"]), "",
             "## Beside (no line)", "",
             "| Tasks | Model | Format kept | Ended within the limit | Tool call | Reaction (false alarm) | Unnamed call | "
             "Median answer tokens |", "|---|---|---|---|---|---|---|---|"]
    for family, by_condition in summary["beside"].items():
        for condition, v in by_condition.items():
            lines.append(f"| {family} | {'base' if condition == 'base' else 'trained'} | {v['format']:.1%} | {v['ended']:.1%} | "
                         f"{v['tool_call']:.1%} | {v['reaction']:.1%} | {v['unnamed_call']:.1%} | {v['median_answer_tokens']:.0f} |")
    text = "\n".join(lines) + "\n"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preservation_summary.md").write_bytes(text.encode("utf-8"))
    (out_dir / "preservation_summary.json").write_bytes((json.dumps(summary, indent=1) + "\n").encode("utf-8"))
    return text


def main(argv: list[str] | None = None) -> None:
    from prio.train import channel_record, load_run  # here: pulls in the training stack

    parser = argparse.ArgumentParser(description="R11: base vs trained model on GSM8K and the explain-format check.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="defaults to the model recorded in the run's train_log.json")
    parser.add_argument("--n", type=int, default=300, help="GSM8K problems")
    parser.add_argument("--explain-n", type=int, default=100)
    parser.add_argument("--n-signals", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW, help="smoke checks only; the reading uses the default")
    args = parser.parse_args(argv)
    model_id = args.model or json.loads((args.run / "train_log.json").read_text(encoding="utf-8"))["model"]
    start_note(args.out, {"run": str(args.run), "model": model_id, "channel": channel_record(args.run),
                          "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
               started=time.strftime("%Y-%m-%d %H:%M"))
    specs = signal_set(args.n_signals)
    model, tok, ms = load_run(model_id, args.run, len(specs))
    tok.padding_side = "left"
    summary = run(model, tok, ms, specs, family_tasks("math", "test", args.n, TASK_SEED),
                  family_tasks("explain", "test", args.explain_n, TASK_SEED), args.out, seed=args.seed,
                  batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    print(write_summary(summary, args.out, str(args.run)))
    finish_note(args.out, finished=time.strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
