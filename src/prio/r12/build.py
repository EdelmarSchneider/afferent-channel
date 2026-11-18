"""Build the data of one R12 task set (AP 3.5, 4, 5, 6): the paired arm's base runs and test
episodes, the notice files of the text-notice model, and the plans of the arms without a forced
prefix (`free`, `live`, `Ls`).

    uv run python -m prio.r12.build --set pilot --model Qwen/Qwen3-1.7B --batch-size 32     -> data/r12p/
    uv run python -m prio.r12.build --set confirm --model Qwen/Qwen3-1.7B --batch-size 32   -> data/r12c/

Base runs: the untrained model, sampled with the builder's settings, at most 1,024 new tokens,
`torch.manual_seed(1200 + set code)`, cached in `paired/base_runs.json` (resumable): exactly how
`data/n6` was built. The model is loaded only when a base run is missing.

A task of the paired arm gives no episode when, and only when: its base run does not end
(`not_ended`), calls a reaction tool on its own (`reaction_tool`), has no position for the event
(`no_position`), or prompt + turn exceed 2,048 tokens (`too_long`). Each case is counted per
family; no task is replaced.

The confirmatory set (`--set confirm`) is built only from a clean source tree on which the
pilot's smoke checks passed, and never rebuilt (exit code 3). A freshness match stops the build
(exit code 4).
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from prio.datagen import variants
from prio.datagen.build import BaseRun, cached_runs, clean_runs, plan_test, sample_batch, to_episode
from prio.datagen.episodes import write_jsonl
from prio.eval.run import _horizon
from prio.modelspec import load_base, resolve_spec
from prio.r12 import (FAMILIES, FREE_K, INTERRUPTS, LIVE, LIVE_K, LIVE_ZERO, MODEL_ID, PREFIX, SEED, SET_CODE,
                      STREAM_ARMS, STREAM_CODE, freshness, ledger, utf8_console)
from prio.r12.tasks import task_sets, write_tasks
from prio.signals import signal_set
from prio.tasks import Task, build_prompt

MAX_NEW_TOKENS = 1024
MAIN_KINDS = ("event", "gating_event")  # one per kept base run
REASONS = ("not_ended", "reaction_tool", "no_position", "too_long")
KEPT_MIN = 0.7  # AP 9: under it a family's rows carry the note "under the 70% rule"
_NOTICE_FILES = ("test.jsonl", "test_quiet.jsonl", "test_near_miss.jsonl", "test_swap.jsonl")
_TARGETS = {"free": INTERRUPTS, "live": LIVE, "Ls": LIVE_ZERO}
_K = {"free": FREE_K, "live": LIVE_K, "Ls": LIVE_K}


class FreshnessError(RuntimeError):
    """An R12 task was found in an earlier draw or in an existing data file."""


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=1) + "\n").encode("utf-8"))


def build_paired(tok, ms, sample, tasks: list[Task], out: Path, set_name: str, model_id: str) -> dict:
    """Base runs (through `sample`, cached) and the test episodes of the paired arm: <out>/base_runs.json,
    test.jsonl, manifest.json. Episodes by the existing planner, generator [SEED, set code, 1]."""
    out = Path(out)
    code = SET_CODE[set_name]
    specs = signal_set(6)
    prompts = [build_prompt(tok, t) for t in tasks]
    torch.manual_seed(1200 + code)
    raw = cached_runs(sample, prompts, out / "base_runs.json", {"model": model_id, "seed": 1200 + code})
    dropped = {family: Counter() for family in FAMILIES}
    for task, (_, stop) in zip(tasks, raw):
        if stop != "eos":
            dropped[task.family]["not_ended"] += 1
    finished = [BaseRun(t, p, ids) for t, p, (ids, stop) in zip(tasks, prompts, raw) if stop == "eos"]
    kept = clean_runs(finished, tok, ms)
    clean = {id(run) for run in kept}
    for run in finished:
        if id(run) not in clean:
            dropped[run.task.family]["reaction_tool"] += 1

    rng = np.random.default_rng([SEED, code, 1])
    drafts = plan_test(kept, specs, rng, tok, ms)
    placed = {id(d.run) for d in drafts if d.kind in MAIN_KINDS}
    for run in kept:
        if id(run) not in placed:
            dropped[run.task.family]["no_position"] += 1
    episodes = []
    for i, draft in enumerate(drafts):
        episode = to_episode(draft, f"{PREFIX[set_name]}-P-{i:05d}", specs, rng, ms)
        if episode is not None:
            episodes.append(episode)
        elif draft.kind in MAIN_KINDS:
            dropped[draft.run.task.family]["too_long"] += 1
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "test.jsonl", episodes)

    requested = Counter(t.family for t in tasks)
    main = Counter(e.family for e in episodes if e.kind in MAIN_KINDS)
    share = {f: main[f] / requested[f] for f in FAMILIES if requested[f]}
    manifest = {
        "set": set_name, "model": model_id, "n_signals": len(specs), "seed": SEED, "set_code": code,
        "base_run_seed": 1200 + code, "episode_generator": [SEED, code, 1], "max_new_tokens": MAX_NEW_TOKENS,
        "tasks_requested": {f: requested[f] for f in FAMILIES if requested[f]},
        "base_runs_finished": dict(Counter(r.task.family for r in finished)),
        "base_runs_kept": dict(Counter(r.task.family for r in kept)),  # no reaction tool called unprompted
        "main_episodes": {f: main[f] for f in share},
        "dropped": {f: {reason: dropped[f][reason] for reason in REASONS} for f in share},
        "kept_share": share, "kept_min": KEPT_MIN, "under_70_percent": [f for f in share if share[f] < KEPT_MIN],
        "episodes": len(episodes), "kinds": dict(Counter(e.kind for e in episodes)),
        "kinds_by_family": {f: dict(Counter(e.kind for e in episodes if e.family == f)) for f in share},
        "signals": dict(Counter(e.signal for e in episodes if e.signal)),
        "event_phases": dict(Counter(f"{e.signal}/{e.phase}" for e in episodes if e.kind == "event")),
        "sha256": {name: ledger.sha256(out / name) for name in ("test.jsonl", "base_runs.json")},
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def build_notice(src: Path, out: Path, tok, ms) -> dict:
    """The text-notice model's files of the paired arm (harness mode): the event file with the
    past-limit notice after the crossing token, and the swap, near-miss and quiet files beside it."""
    src, out = Path(src), Path(out)
    variants.build_variant(src, out, "notice", tok, ms, seed=0)
    for name in ("swap", "near_miss", "quiet"):
        variants.build_variant(src, None, name, tok, ms, eval_seed=0, into=out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["sha256"] = {name: ledger.sha256(out / name) for name in _NOTICE_FILES}
    _write_json(out / "manifest.json", manifest)
    return manifest


def stream_plans(free: list[Task], live: list[Task], zero: list[Task], set_name: str) -> list[dict]:
    """One plan per task of the arms `free`, `live` and `Ls` (i = position in the arm's list):
    the signal that crosses (in rotation), the generated token k on which it crosses (generator
    [SEED, set code, arm code, i]) and the tokens generated after it."""
    code = SET_CODE[set_name]
    plans = []
    for arm, tasks in zip(STREAM_ARMS, (free, live, zero), strict=True):
        targets, (lo, hi) = _TARGETS[arm], _K[arm]
        for i, task in enumerate(tasks):
            signal = targets[i % len(targets)]
            k = int(np.random.default_rng([SEED, code, STREAM_CODE[arm], i]).integers(lo, hi + 1))
            plans.append({"arm": arm, "index": i, "task_id": task.task_id, "family": task.family, "signal": signal,
                          "k": k, "horizon": _horizon(False, signal)})
    return plans


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return tok, resolve_spec(model_id, tok)


def _sampler(model_id: str, tok, ms, batch_size: int, seed: int):
    """The base-run sampler; the model is loaded at the first call (never, when the cache holds every run)."""
    loaded = {}

    def sample(prompts: list[list[int]]):
        if "model" not in loaded:
            loaded["model"] = load_base(model_id)[0]
            torch.manual_seed(seed)  # the draws do not depend on what loading the model consumed
        return sample_batch(loaded["model"], tok, ms, prompts, MAX_NEW_TOKENS, batch_size, label="base runs")

    return sample


def build(set_name: str, model_id: str = MODEL_ID, batch_size: int = 32, root: Path = Path(".")) -> dict:
    """Everything of one set under <root>/data/<r12p | r12c>/: tasks.json, freshness.json, paired/,
    paired-notice/, stream.json, manifest.json (written last)."""
    root = Path(root)
    data = root / "data"
    out = data / PREFIX[set_name]
    clean = ledger.tree_clean()
    if set_name == "confirm":
        if (out / "manifest.json").exists():
            raise ledger.RunOnceError(f"{out / 'manifest.json'} exists: the confirmatory set is built once and never rebuilt")
        if not clean:
            raise ledger.RunOnceError("src/prio has uncommitted changes: the confirmatory set is built from a committed tree")
        ledger.pilot_passed(root)
    tok, ms = _load_tokenizer(model_id)
    arms = task_sets()[set_name]
    written = write_tasks(set_name, out)
    audit = freshness.audit(data, tok, out)
    if not audit["passed"]:
        found = audit["by_seed"]["used"] + audit["by_seed"]["in_both_sets"] + audit["by_file"]["matches"]
        raise FreshnessError(f"the freshness audit found {len(found)} match(es), first: {found[:3]}; see {out / 'freshness.json'}")
    code = SET_CODE[set_name]
    paired = build_paired(tok, ms, _sampler(model_id, tok, ms, batch_size, 1200 + code), arms["paired"], out / "paired",
                          set_name, model_id)
    notice = build_notice(out / "paired", out / "paired-notice", tok, ms)
    plans = stream_plans(arms["free"], arms["live"], arms["Ls"], set_name)
    _write_json(out / "stream.json", plans)
    manifest = {
        "set": set_name, "model": model_id, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": ledger.commit(), "tree": ledger.code_tree(), "clean": clean,
        "tasks": written["by_arm"], "freshness_passed": True,
        "freshness": {"earlier_prompts": audit["by_seed"]["earlier_prompts"], "folders": audit["by_file"]["folders"],
                      "files": audit["by_file"]["files"], "episodes": audit["by_file"]["episodes"]},
        "plans": dict(Counter(p["arm"] for p in plans)),
        "paired": paired,
        "notice": {"episodes": notice["episodes"], "kinds": notice["kinds"],
                   "test_only": {name: entry["episodes"] for name, entry in notice["test_only"].items()},
                   "sha256": notice["sha256"]},
        "sha256": {name: ledger.sha256(out / name) for name in ("tasks.json", "stream.json", "freshness.json")},
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description="Build the data of one R12 task set.")
    parser.add_argument("--set", choices=tuple(SET_CODE), required=True, dest="set_name")
    parser.add_argument("--model", default=MODEL_ID, help="the untrained model of the base runs and its tokenizer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (holds data/ and results/)")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.set_name, args.model, args.batch_size, args.root)
    except ledger.RunOnceError as error:
        print(f"refused: {error}", file=sys.stderr)
        sys.exit(3)
    except FreshnessError as error:
        print(f"stopped: {error}", file=sys.stderr)
        sys.exit(4)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
