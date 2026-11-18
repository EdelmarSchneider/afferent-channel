"""R10 evaluation: reactions judged against each episode's own limits, limit twins, value reports.

Data from `prio.datagen.limits`. Runs through `prio.eval.run.evaluate` (its records, files, resume,
`--batch-size`) with its `setup` and `judge` hooks, for three changes:
- every fresh trajectory (nominal, near-miss and swapped pairings) is drawn against the limits
  written in the episode's prompt (`episode_specs`), and runs are classified with them;
- a limit twin runs once (channel on, its stored trajectory: the event's values under a limit
  beyond their peak) and any reaction in it is a false alarm; its prefix and tool checks are
  those of its sibling's kind (`meta["twin_kind"]`);
- report episodes (`test_reports.jsonl`) are run up to the answer and scored by the absolute error
  of the reported level, in points.
Rows of `limits_summary.md` are split by the split of the episode's limit (seen / held_on /
held_off; "-" for `net`); limit-twin rows hold the limit-twin false alarms.

    python -m prio.eval.limits_eval --run runs/<run> --data data/n6-limits --out results/<folder> [--every 4] [--batch-size 16]
"""

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from prio.datagen.episodes import Episode, read_jsonl
from prio.eval import run as base_eval
from prio.eval.metrics import RunRecord, summarize
from prio.signals import SignalSpec, signal_set, with_limits

REPORT_HORIZON = 16  # "Battery at 63%." and the end of turn take ~8 tokens
_PERCENT = re.compile(r"(\d{1,3})\s*%")


def episode_specs(ep: Episode, base_specs: Sequence[SignalSpec]) -> tuple[SignalSpec, ...]:
    """The signal specs with the limits written in this episode's prompt (unchanged without any)."""
    limits = (ep.meta or {}).get("limits")
    return with_limits(base_specs, limits) if limits else tuple(base_specs)


def limit_split(ep: Episode) -> str | None:
    return (ep.meta or {}).get("limit_split")


def _kind(ep: Episode) -> str:
    """The kind that decides prefix and tool checks: a limit twin is cut like its sibling."""
    return ep.meta["twin_kind"] if ep.kind == "limit_twin" else ep.kind


def classify_limits(ep: Episode, paired: str, condition: str, gen, tok, ms, specs: Sequence[SignalSpec]) -> RunRecord:
    """`prio.eval.run.classify`, with a limit twin judged as a quiet run (any reaction anywhere is a
    false alarm) and recorded as kind "limit_twin" with the pairing it really ran."""
    if ep.kind != "limit_twin":
        return base_eval.classify(ep, paired, condition, gen, tok, ms, specs)
    record = base_eval.classify(replace(ep, kind=_kind(ep)), "nominal", condition, gen, tok, ms, specs)
    return replace(record, kind="limit_twin", paired=paired)


def _resume(out_path: Path | None, wanted: set[str], conditions: Sequence[str], make):
    if out_path is None or not out_path.exists():
        return []
    with open(out_path, encoding="utf-8") as lines:
        saved = [make(json.loads(line)) for line in lines if line.strip()]
    return [r for r in saved if _get(r, "episode_id") in wanted and _get(r, "condition") in conditions]


def _get(record, key):
    return record[key] if isinstance(record, dict) else getattr(record, key)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def episode_setup(base_specs: Sequence[SignalSpec]):
    """The `setup` hook of `run.evaluate` for R10: the episode's own limits; a limit twin is cut like
    its sibling's kind and runs with the channel on only (its pairing: its stored trajectory)."""
    def setup(ep: Episode, conditions: Sequence[str]) -> base_eval.EpisodeSetup:
        wanted = [c for c in conditions if c == "on"] if ep.kind == "limit_twin" else list(conditions)
        return base_eval.EpisodeSetup(episode_specs(ep, base_specs), _kind(ep), wanted)

    return setup


def evaluate_limits(model, tok, ms, base_specs: Sequence[SignalSpec], test_eps: list[Episode], conditions: Sequence[str],
                    seed: int = 0, log_every: int = 25, out_path: Path | None = None, every: int = 1,
                    batch_size: int = 1) -> list[RunRecord]:
    """`prio.eval.run.evaluate` with per-episode limits and limit twins (module docstring). Resumable
    through `out_path`, `every` and `batch_size` as there; generations go to `generations.jsonl`."""
    return base_eval.evaluate(model, tok, ms, base_specs, test_eps, conditions, seed=seed, log_every=log_every,
                              out_path=out_path, every=every, batch_size=batch_size, setup=episode_setup(base_specs),
                              judge=classify_limits)


# --- value reports ---------------------------------------------------------------------------------

def parse_report(text: str) -> int | None:
    """The first "NN%" of the answer (after the thinking block, if any)."""
    found = _PERCENT.search(text.split("</think>")[-1])
    return int(found.group(1)) if found else None


def evaluate_reports(model, tok, ms, report_eps: list[Episode], conditions: Sequence[str],
                     out_path: Path | None = None, every: int = 1) -> list[dict]:
    """Run every report episode up to its answer (first turn, question and empty thinking forced) and
    read the level it gives. One row per (episode, condition); resumable through `out_path`."""
    chosen = set(base_eval.quick_subset(report_eps, every))
    rows = _resume(out_path, {report_eps[i].episode_id for i in chosen}, conditions, lambda row: row)
    done = {(r["episode_id"], r["condition"]) for r in rows}
    stops = base_eval.stop_ids(tok, ms)
    for index, ep in enumerate(report_eps):
        if index not in chosen:
            continue
        info = ep.meta["report"]
        prompt = ep.prompt_ids + ep.target_ids[:info["answer_start"]]
        harness = base_eval.prefix_harness(ep, info["answer_start"])  # the question is the harness's
        values = base_eval._rows(ep.signals, len(prompt) + REPORT_HORIZON)
        signal_fn = base_eval.follow_thinking(values, len(prompt), None, 1.0, ms.think_close)
        for condition in conditions:
            if (ep.episode_id, condition) in done:
                continue
            with base_eval._condition(model, condition):
                gen = base_eval._run(model, prompt, values, REPORT_HORIZON, stops, signal_fn, harness)
            text = tok.decode(gen.token_ids)
            rows.append({"episode_id": ep.episode_id, "condition": condition, "signal": info["signal"],
                         "level": info["level"], "held_out": info["held_out"], "predicted": parse_report(text),
                         "text": text})
            if out_path is not None:
                _append(out_path, rows[-1])
    return rows


def score_reports(rows: list[dict]) -> dict:
    """Per "condition|signal|held_out" (signal "all" for every signal; "seen" for the levels that
    were report targets in training): n, share parsed, and over the parsed answers the mean and
    median absolute error in points, share exact and share within 3 points."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        part = "held_out" if r["held_out"] else "seen"
        groups[f"{r['condition']}|all|{part}"].append(r)
        groups[f"{r['condition']}|{r['signal']}|{part}"].append(r)
    out = {}
    for key, group in groups.items():
        errors = [abs(r["predicted"] - r["level"]) for r in group if r["predicted"] is not None]
        out[key] = {
            "n": len(group), "parsed": len(errors) / len(group),
            "mae": float(np.mean(errors)) if errors else None,
            "median_error": float(np.median(errors)) if errors else None,
            "exact": float(np.mean([e == 0 for e in errors])) if errors else None,
            "within_3": float(np.mean([e <= 3 for e in errors])) if errors else None,
        }
    return out


# --- summary ---------------------------------------------------------------------------------------

def split_labels(records: list[RunRecord], split_of: dict[str, str | None]) -> list[str]:
    labels = []
    for r in records:
        row = r.signal if r.kind in base_eval._PAIRED_KINDS else f"{r.signal} ({r.kind})"
        labels.append(f"{row} [{split_of.get(r.episode_id) or '-'}]")
    return labels


def write_limits_summary(records: list[RunRecord], split_of: dict[str, str | None], report_rows: list[dict],
                         out_dir: Path) -> None:
    """The standard summary (`prio.eval.run.write_summary`: records.jsonl, summary.md|json, rows by
    kind) plus `limits_summary.md|json`: the same rates per limit split, and the report scores."""
    out_dir = Path(out_dir)
    base_eval.write_summary(records, None, out_dir)
    every = ["all"] * len(records)
    family = {"event": "interrupt", "gating_event": "gating"}
    overall = [f"all {family.get(r.kind, r.kind)} [{split_of.get(r.episode_id) or '-'}]" for r in records]
    merged = {**summarize(records, overall, every), **summarize(records, split_labels(records, split_of), every)}
    reports = score_reports(report_rows)
    payload = {"groups": {"|".join(map(str, k)): v for k, v in merged.items()}, "reports": reports}
    (out_dir / "limits_summary.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    fmt = base_eval._fmt
    lines = ["# R10 evaluation: rates per limit split", "",
             "- Split of the limit written for the episode's signal: `seen` (training grid), `held_on` (held-out "
             "grid values), `held_off` (off-grid values), `-` (`net`, binary, no limit).",
             "- Rows `x (limit_twin)`: the event's trajectory under a limit beyond its peak; `False alarms` there is "
             "the limit-twin alarm rate (split of the twin's own limit).",
             "- Reports: absolute error between the level reported and the level fed, in points, over the answers "
             "that hold a percent; `held_out` = levels never used as report targets in training.", ""]
    for condition in ("on", "zeroed", "base"):
        rows = sorted((k, v) for k, v in merged.items() if k[0] == condition)
        if not rows:
            continue
        lines += [f"## Condition: {condition}", "",
                  "| Row | n | Reaction | 95% CI | False alarms | 95% CI | n quiet | Near-miss alarms | n near |",
                  "|" + "---|" * 9]
        for (_, label, _), v in rows:
            lines.append(f"| {label} | {v['n']} | {fmt(v['reaction_rate'])} | {fmt(v['reaction_ci'])} | "
                         f"{fmt(v['false_alarm_rate'])} | {fmt(v['false_alarm_ci'])} | {v['n_nominal']} | "
                         f"{fmt(v['near_miss_alarm_rate'])} | {v['n_near_miss']} |")
        lines.append("")
    if reports:
        lines += ["## Value reports", "", "| Condition / signal / levels | n | Parsed | MAE | Median | Exact | Within 3 |",
                  "|---|---|---|---|---|---|---|"]
        for key, v in sorted(reports.items()):
            mae = "-" if v["mae"] is None else f"{v['mae']:.2f}"
            median = "-" if v["median_error"] is None else f"{v['median_error']:.1f}"
            lines.append(f"| {key.replace('|', ' / ')} | {v['n']} | {v['parsed']:.1%} | {mae} | {median} | "
                         f"{fmt(v['exact'])} | {fmt(v['within_3'])} |")
        lines.append("")
    (out_dir / "limits_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run on the R10 test episodes and reports.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="defaults to the model recorded in the run's train_log.json")
    parser.add_argument("--every", type=int, default=1, help="evaluate only every n-th episode of each kind and signal")
    parser.add_argument("--base-n", type=int, default=150, help="episodes for the untouched-base control (0: none)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="runs generated together (1 = one at a time; same records either way)")
    args = parser.parse_args(argv)

    model_id = args.model or json.loads((args.run / "train_log.json").read_text(encoding="utf-8"))["model"]
    base_eval.start_note(args.out, {"run": str(args.run), "data": str(args.data), "model": model_id, "kind": "limits",
                                    "channel": base_eval.channel_record(args.run),
                                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                         started=time.strftime("%Y-%m-%d %H:%M"))
    episodes = read_jsonl(args.data / "test.jsonl")
    reports_path = args.data / "test_reports.jsonl"
    report_eps = read_jsonl(reports_path) if reports_path.exists() else []
    specs = signal_set(episodes[0].signals.shape[1])
    model, tok, ms = base_eval.load_run(model_id, args.run, len(specs))
    tok.padding_side = "left"

    progress = args.out / "records.partial.jsonl"
    records = evaluate_limits(model, tok, ms, specs, episodes, ("on", "zeroed"), seed=args.seed, out_path=progress,
                              every=args.every, batch_size=args.batch_size)
    if args.base_n > 0:
        subset = base_eval.spread([ep for ep in episodes if ep.kind in base_eval._PAIRED_KINDS], args.base_n)
        records += evaluate_limits(model, tok, ms, specs, subset, ("base",), seed=args.seed, out_path=progress,
                                   batch_size=args.batch_size)
    report_rows = evaluate_reports(model, tok, ms, report_eps, ("on", "zeroed"), out_path=args.out / "reports.partial.jsonl",
                                   every=args.every)
    write_limits_summary(records, {ep.episode_id: limit_split(ep) for ep in episodes}, report_rows, args.out)
    base_eval.finish_note(args.out, finished=time.strftime("%Y-%m-%d %H:%M"))
    print((args.out / "limits_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
