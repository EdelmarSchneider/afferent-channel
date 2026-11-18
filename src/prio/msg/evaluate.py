"""Evaluate a messenger run (R23, preregistration amendment A8).

uv run python -m prio.msg.evaluate --run runs/n7msg-wide-levels-s0 --data data/n7msg --out results/<folder>
    [--every K] [--batch-size 16] [--conditions on,off,base] [--forgery-every 2]

Lines L1-L5 through `prio.table.evaluate` (every sibling of the test families under `on`, `off` and
`base`; forged-status cells on every `--forgery-every`-th family), plus L6, one agent: the same
`net` state leads to different actions by task. On n6's web tasks with `net` down the first command
is still `local_lookup` (n6's quick gating subset >= 90%, false alarms <= 5%), and on messenger
tasks `local_lookup` is called in <= 2% of runs.

L6's n6 part reruns what `prio.eval.run` does for the `net` gating events of n6's quick subset
(every 4th; same prefixes, pairings, seeds and classifier), with `cell` at its nominal; its records
and generations go to `<out>/l6/`. One batch size per results folder.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from prio.datagen.episodes import Episode, read_jsonl
from prio.eval.metrics import RunRecord
from prio.eval.run import EpisodeSetup, classify, evaluate, finish_note, start_note
from prio.msg.build import widen
from prio.msg.domain import CELL, MESSENGER, SPECS
from prio.signals import CORE_SIGNALS
from prio.table.evaluate import CONDITIONS, evaluate_table, line, row, summarize, write_summary
from prio.table.score import TableRecord

N6_EVERY = 4  # n6's quick subset: every 4th episode of each kind and signal
L6_LOOKUP_MIN, L6_ALARM_MAX, L6_MSG_LOOKUP_MAX = 0.90, 0.05, 0.02


def n6_test_episodes(n6: Path) -> list[Episode]:
    """n6's test episodes with the `cell` column appended at its nominal."""
    return [widen(ep, CELL.nominal) for ep in read_jsonl(Path(n6) / "test.jsonl")]


def evaluate_n6(model, tok, ms, episodes: list[Episode], out_dir: Path | None = None, batch_size: int = 1,
                seed: int = 0) -> list[RunRecord]:
    """`prio.eval.run.evaluate` on the `net` gating events of the quick subset of `episodes` (n6's
    whole test file, seven columns), condition `on`: the stored trajectory (`net` down), a quiet one
    and a near-miss one. Judged by the n6 classifier against the six n6 signals."""
    def setup(ep: Episode, conditions: Sequence[str]) -> EpisodeSetup:
        return EpisodeSetup(SPECS, ep.kind, conditions if (ep.kind, ep.signal) == ("gating_event", "net") else ())

    def judge(ep, paired, condition, gen, tok, ms, specs) -> RunRecord:
        return classify(ep, paired, condition, gen, tok, ms, CORE_SIGNALS)

    out_path = None if out_dir is None else Path(out_dir) / "l6" / "records.partial.jsonl"
    return evaluate(model, tok, ms, SPECS, episodes, ("on",), seed=seed, log_every=0, out_path=out_path,
                    every=N6_EVERY, batch_size=batch_size, setup=setup, judge=judge)


def line_l6(n6_records: Sequence[RunRecord], records: Sequence[TableRecord]) -> dict:
    """L6 from the n6 runs (`evaluate_n6`) and the messenger's own runs (main cell, channel on)."""
    paired = lambda name: [r for r in n6_records if r.paired == name]  # noqa: E731
    alarm = lambda rs: [r.reaction_signal is not None for r in rs]  # noqa: E731
    messenger = [r for r in records if r.cell == "main" and r.condition == "on"]
    return line(
        "one agent: on n6's web tasks with `net` down the first command is still `local_lookup` (n6's quick gating "
        "subset >= 90%, false alarms <= 5%), and on messenger tasks `local_lookup` is called in <= 2% of runs",
        [row("n6 web tasks, `net` down: the first command is `local_lookup`",
             [r.used_alternative for r in paired("event")], L6_LOOKUP_MIN),
         row("the same prefixes, nothing past its limit: false alarms (n6 classifier)", alarm(paired("nominal")),
             L6_ALARM_MAX, True),
         row("messenger runs (channel on) that call `local_lookup`", ["local_lookup" in r.commands for r in messenger],
             L6_MSG_LOOKUP_MAX, True),
         row("the same prefixes, a signal close to its limit: false alarms", alarm(paired("near_miss"))),
         row("n6 web tasks, `net` down: the counted n6 reaction (naming sentence + `local_lookup`)",
             [r.reaction_signal == "net" for r in paired("event")])])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a messenger run (R23): lines L1-L6 of amendment A8.")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="messenger dataset folder (test.jsonl, manifest.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=None, help="defaults to the model recorded in the run's train_log.json")
    p.add_argument("--n6", type=Path, default=None, help="n6 data folder for L6 (default: the one in the data manifest)")
    p.add_argument("--every", type=int, default=1, help="every n-th test family (a full run into the same --out adds the rest)")
    p.add_argument("--batch-size", type=int, default=16, help="runs generated together (one size per results folder)")
    p.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated: on, off (channel off), base (untrained model)")
    p.add_argument("--forgery-every", type=int, default=2, help="forged-status cells on every n-th evaluated family (0: none)")
    return p


def main(argv: list[str] | None = None) -> None:
    from prio.train import channel_record, load_run

    cli = parser()
    args = cli.parse_args(argv)
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    if not conditions or set(conditions) - set(CONDITIONS):
        cli.error(f"--conditions takes {', '.join(CONDITIONS)}, not {args.conditions!r}")
    model_id = args.model or json.loads((args.run / "train_log.json").read_text(encoding="utf-8"))["model"]
    n6 = args.n6 or Path(json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))["n6"])
    start_note(args.out, {"run": str(args.run), "data": str(args.data), "model": model_id, "kind": "table",
                          "domain": MESSENGER.name, "channel": channel_record(args.run),
                          "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
               started=time.strftime("%Y-%m-%d %H:%M"))
    episodes = read_jsonl(args.data / "test.jsonl")
    model, tok, ms = load_run(model_id, args.run, len(SPECS))
    tok.padding_side = "left"
    records = evaluate_table(model, tok, ms, MESSENGER, episodes, args.out, conditions, every=args.every,
                             batch_size=args.batch_size, forgery_every=args.forgery_every)
    n6_records = evaluate_n6(model, tok, ms, n6_test_episodes(n6), args.out, args.batch_size) if "on" in conditions else []
    summary = summarize(records, MESSENGER)
    summary["lines"]["L6"] = line_l6(n6_records, records)
    write_summary(summary, records, args.out)
    finish_note(args.out, finished=time.strftime("%Y-%m-%d %H:%M"))
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
