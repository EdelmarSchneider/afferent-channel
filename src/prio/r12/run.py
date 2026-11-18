"""One evaluation of R12 per (set, arm, model), started through the run-once marker file.

    uv run python -m prio.r12.run --set pilot|confirm --arm paired|free|live|Ls --model channel|notice
                                  [--batch-size 16] [--limit N] [--run-types event,twin,off] [--rerun-reason TEXT]

Arm `paired`: `prio.eval.run` on the set's test episodes (channel model: standard mode, no
preservation check, the untrained floor at its default 150 episodes; text-notice model: harness
mode on the notice files). Arms `free`, `live`, `Ls`: `prio.r12.stream` on the set's plans.

Results: `results/r12-<arm>-<tag>-b16/` (confirmatory), `results/r12p-<arm>-<tag>-b<batch>/` (pilot).

The confirmatory set runs whole, once, at batch 16: `--batch-size` other than 16, `--limit` and
`--run-types` are for the pilot. A start that the run-once rule forbids (`prio.r12.ledger`) ends
with exit code 3 before any model is loaded and before anything is written. A set whose build
manifest is missing (the builder writes it last) is not run: exit code 3 for the confirmatory
set, a plain error for the pilot. Before a run goes on, a last line of its partial files torn by
a kill or a power cut is moved aside (`prio.r12.torn`) and the repair is written into its entry.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from prio.eval.run import finish_note, spread, start_note
from prio.eval.run import main as eval_main
from prio.r12 import ARMS, BATCH_SIZE, PREFIX, RUNS, SET_CODE, TAGS, ledger, torn, utf8_console
from prio.r12.stream import RUN_TYPES, load_plans, run_channel, run_notice
from prio.r12.tasks import read_tasks
from prio.signals import signal_set
from prio.tasks import build_prompt
from prio.train import channel_record, load_run

_RESULTS = {"confirm": "r12", "pilot": "r12p"}
_NOTICE_FILES = ("test.jsonl", "test_quiet.jsonl", "test_near_miss.jsonl", "test_swap.jsonl")


def out_folder(set_name: str, arm: str, model: str, batch_size: int = BATCH_SIZE, rerun: int = 0) -> Path:
    """Results folder of a run, relative to the project root (a rerun: `-rerun<n>`)."""
    name = f"{_RESULTS[set_name]}-{arm}-{TAGS[model]}-b{batch_size}"
    return Path("results") / (name + (f"-rerun{rerun}" if rerun else ""))


def data_files(set_name: str, arm: str, model: str, root: Path = Path(".")) -> list[Path]:
    """Every data file the run reads (hashed into the marker file)."""
    data = Path(root) / "data" / PREFIX[set_name]
    if arm != "paired":
        return [data / "tasks.json", data / "stream.json"]
    if model == "channel":
        return [data / "paired" / "test.jsonl"]
    return [data / "paired-notice" / name for name in _NOTICE_FILES]


def _show(path: Path) -> None:
    if path.exists():
        print(path.read_text(encoding="utf-8"))


def _paired(args, root: Path, out: Path) -> None:
    data = root / "data" / PREFIX[args.set_name] / ("paired" if args.model == "channel" else "paired-notice")
    argv = ["--run", str(root / RUNS[args.model]), "--data", str(data), "--out", str(out), "--batch-size", str(args.batch_size)]
    argv += ["--preservation-n", "0"] if args.model == "channel" else ["--mode", "harness"]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    eval_main(argv)  # prints its summary


def _stream(args, root: Path, out: Path, run_types: tuple[str, ...]) -> None:
    run = root / RUNS[args.model]
    data = root / "data" / PREFIX[args.set_name]
    model_id = json.loads((run / "train_log.json").read_text(encoding="utf-8"))["model"]
    plans = load_plans(data / "stream.json", args.arm)
    if args.limit is not None:  # pilot only: plans at even spacing over the arm, so that a slice covers its families and signals
        plans = spread(plans, args.limit)
    tasks = {t.task_id: t for t in read_tasks(data)[args.arm]}
    start_note(out, {"run": RUNS[args.model], "data": (Path("data") / PREFIX[args.set_name]).as_posix(), "model": model_id,
                     "channel": channel_record(run), "mode": "r12-stream", "set": args.set_name, "arm": args.arm,
                     "args": {"batch_size": args.batch_size, "limit": args.limit, "run_types": list(run_types)}},
               started=time.strftime("%Y-%m-%d %H:%M"))
    specs = signal_set(6)
    model, tok, ms = load_run(model_id, run, len(specs))
    tok.padding_side = "left"
    prompts = {plan.task_id: build_prompt(tok, tasks[plan.task_id]) for plan in plans}
    runner = run_channel if args.model == "channel" else run_notice
    runner(model, tok, ms, specs, plans, prompts, SET_CODE[args.set_name], out, batch_size=args.batch_size, run_types=run_types)
    finish_note(out, finished=time.strftime("%Y-%m-%d %H:%M"))
    _show(out / "stream_summary.md")


def main(argv: list[str] | None = None) -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description="Run one R12 evaluation (set, arm, model).")
    parser.add_argument("--set", choices=tuple(SET_CODE), required=True, dest="set_name")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", choices=tuple(RUNS), required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="pilot only: the first N episodes (arm paired), or N plans at even spacing over the arm")
    parser.add_argument("--run-types", default=None, help="pilot only, arms free / live / Ls: comma-separated event,twin,off")
    parser.add_argument("--rerun-reason", default=None, help="confirmatory set: run again into a new folder, for this reason")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (holds data/, runs/ and results/)")
    args = parser.parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    run_types = RUN_TYPES
    if args.run_types is not None:
        run_types = tuple(t.strip() for t in args.run_types.split(",") if t.strip())
        if args.arm == "paired" or not run_types or set(run_types) - set(RUN_TYPES):
            parser.error(f"--run-types takes {', '.join(RUN_TYPES)} and applies to the arms free, live and Ls")
    root = args.root
    manifest = root / "data" / PREFIX[args.set_name] / "manifest.json"
    try:
        if args.set_name == "confirm" and (args.batch_size != BATCH_SIZE or args.limit is not None or args.run_types is not None):
            raise ledger.RunOnceError(f"--batch-size other than {BATCH_SIZE}, --limit and --run-types are for the pilot: "
                                      "the confirmatory set is run whole, once")
        if not manifest.exists():  # the builder writes it last: without it the set may be half built
            unfinished = f"no {manifest.as_posix()}: the build of the {args.set_name} set is not finished (prio.r12.build)"
            if args.set_name == "confirm":
                raise ledger.RunOnceError(unfinished)
            raise SystemExit(f"error: {unfinished}")
        entry = ledger.begin(args.set_name, args.arm, args.model, root / out_folder(args.set_name, args.arm, args.model, args.batch_size),
                             data_files(args.set_name, args.arm, args.model, root), argv, batch_size=args.batch_size,
                             rerun_reason=args.rerun_reason, root=root)
    except ledger.RunOnceError as error:
        print(f"refused: {error}", file=sys.stderr)
        sys.exit(3)
    out = root / entry["out"]
    # A last line torn by a kill or a power cut would break the resume (the evaluators read their partial
    # files line by line): it is moved to `<file>.torn` first, and the repair goes into the marker entry.
    repairs = torn.repair_paired_folder(out) if args.arm == "paired" else torn.repair_stream_folder(out)
    ledger.add_repairs(entry, [{**repair, "file": f"{entry['out']}/{repair['file']}"} for repair in repairs], root)
    if args.arm == "paired":
        _paired(args, root, out)
    else:
        _stream(args, root, out, run_types)
    ledger.finish(entry, out / "records.jsonl", root)


if __name__ == "__main__":
    main()
