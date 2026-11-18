"""Build the messenger dataset (R23, preregistration amendment A8).

uv run python -m prio.msg.build --n6 data/n6 --model Qwen/Qwen3-1.7B --out data/n7msg --seed 0 \
    --train 1500 --test 300 --batch-size 32

Base runs of the untrained model on the messenger tasks (GPU; cached in `<out>/base_runs.json`
chunk by chunk, so a rerun continues and a complete cache needs no model), cleaning
(`domain.CLEAN_RULE`), then one family of siblings per kept run, cut at its first command
(`domain.plan`: all four state classes, timing mix of A8).
  train.jsonl    n6's training episodes with a nominal `cell` column appended, then the messenger
                 training episodes;
  test.jsonl     the messenger test episodes;
  manifest.json  counts: tasks requested, base runs finished / kept (A8 stop rule: kept < 70% of
                 the tasks requested), episodes by class and timing kind, n6 episodes carried.
"""

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from prio.datagen.build import BaseRun, cached_runs, sample_batch
from prio.datagen.episodes import Episode, read_jsonl, write_jsonl
from prio.modelspec import ModelSpec, load_base
from prio.msg.domain import CELL, CLEAN_RULE, MESSENGER, SPECS, dropped_for, load_tasks, plan
from prio.reactions import parse_tool_calls
from prio.table.core import build_prompt, family

MAX_NEW_TOKENS = 512  # a base run that has not ended by then is not used
KEPT_MIN = 0.7  # A8 stop rule: base runs kept, as a share of the tasks requested
SPLITS = ("train", "test")


def widen(ep: Episode, value: float) -> Episode:
    """An n6 episode with one more signal column at `value` (`cell` at its nominal); the rest as stored."""
    if ep.signals.shape[1] != len(SPECS) - 1:
        raise ValueError(f"{ep.episode_id}: {ep.signals.shape[1]} signal columns, expected {len(SPECS) - 1}")
    column = np.full((len(ep.signals), 1), value, dtype=np.float32)
    return replace(ep, signals=np.concatenate([ep.signals, column], axis=1))


def split_episodes(runs: list[BaseRun], rng: np.random.Generator, tok, ms: ModelSpec) -> tuple[list[Episode], int]:
    """(the sibling families of the kept runs of one split, families dropped for their length)."""
    episodes, dropped = [], 0
    for run in runs:
        cut = parse_tool_calls(run.target_ids, tok, ms)[0].token_index  # the decision: the first command
        siblings = family(MESSENGER, run, cut, plan(cut, rng), rng, tok, ms, family_id=run.task.task_id)
        episodes += siblings
        dropped += not siblings
    return episodes, dropped


def build(tok, ms: ModelSpec, sample, n6: Path, out: Path, n_train: int, n_test: int, seed: int, model_id: str) -> dict:
    """Write the dataset into `out` and return its manifest. `sample(prompts)` gives the base runs
    as (ids up to and including the end of the turn, stop reason), as `datagen.build.sample_batch`."""
    n6, out = Path(n6), Path(out)
    if out.resolve() == n6.resolve():
        raise ValueError("--out must not be the n6 folder: its train.jsonl would be overwritten")
    started = time.time()
    tasks = load_tasks(n_train, n_test, seed)
    prompts = [build_prompt(tok, MESSENGER, t.user) for t in tasks]
    raw = cached_runs(sample, prompts, out / "base_runs.json",
                      {"model": model_id, "seed": seed, "domain": MESSENGER.name, "max_new_tokens": MAX_NEW_TOKENS})
    reasons = [dropped_for(ids, stop, tok, ms) for ids, stop in raw]
    runs = {s: [BaseRun(t, p, ids) for t, p, (ids, _), why in zip(tasks, prompts, raw, reasons)
                if why is None and t.split == s] for s in SPLITS}
    built = {s: split_episodes(runs[s], np.random.default_rng([seed, i]), tok, ms) for i, s in enumerate(SPLITS)}
    own = {s: Counter(ep.meta["table"]["own_command"]["name"] for ep in built[s][0] if ep.meta["table"]["variant"] == "base")
           for s in SPLITS}
    carried = [widen(ep, CELL.nominal) for ep in read_jsonl(n6 / "train.jsonl")]
    written = {"train": carried + built["train"][0], "test": built["test"][0]}
    out.mkdir(parents=True, exist_ok=True)
    for split, episodes in written.items():
        write_jsonl(out / f"{split}.jsonl", episodes)
    requested = {"train": n_train, "test": n_test}
    of = lambda split, key: [ep.meta["table"][key] for ep in built[split][0]]  # noqa: E731
    lengths = [len(ep.prompt_ids) + len(ep.target_ids) for s in SPLITS for ep in built[s][0]]
    share = {s: len(runs[s]) / requested[s] if requested[s] else None for s in SPLITS}
    manifest = {
        "domain": MESSENGER.name, "model": model_id, "seed": seed, "n_signals": len(SPECS),
        "signals": [s.name for s in SPECS], "n6": str(n6), "n6_episodes_carried": len(carried),
        "tasks_requested": requested, "max_new_tokens": MAX_NEW_TOKENS,
        "base_runs_finished": {s: sum(stop == "eos" for t, (_, stop) in zip(tasks, raw) if t.split == s) for s in SPLITS},
        "base_runs_kept": {s: len(runs[s]) for s in SPLITS}, "base_runs_kept_share": share,
        "stop_rule": {"base_runs_kept_min": KEPT_MIN, "passed": all(v is None or v >= KEPT_MIN for v in share.values())},
        "clean_rule": CLEAN_RULE,
        "dropped": {s: dict(Counter(why for t, why in zip(tasks, reasons) if why and t.split == s)) for s in SPLITS},
        "own_command": {s: dict(own[s]) for s in SPLITS},
        "families": {s: len(set(of(s, "family"))) for s in SPLITS},
        "families_dropped_long": {s: built[s][1] for s in SPLITS},
        "episodes": {s: len(eps) for s, eps in written.items()},
        "msg_episodes": {s: len(built[s][0]) for s in SPLITS},
        "kinds": {s: dict(Counter(ep.kind for ep in eps)) for s, eps in written.items()},
        "by_class_timing": {s: dict(Counter(f"{c}/{t or 'none'}" for c, t in zip(of(s, "cls"), of(s, "timing")))) for s in SPLITS},
        "variants": {s: dict(Counter(of(s, "variant"))) for s in SPLITS},
        "length_p50": float(np.median(lengths)) if lengths else None, "length_max": max(lengths, default=None),
        "minutes": (time.time() - started) / 60,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec

    tok = AutoTokenizer.from_pretrained(model)
    return tok, resolve_spec(model, tok)


def _sampler(model_id: str, tok, ms: ModelSpec, batch_size: int, seed: int):
    """`sample` for `build`: the untrained model, loaded when the first missing base run is asked for."""
    loaded = []

    def sample(prompts: list[list[int]]):
        if not loaded:
            torch.manual_seed(seed)
            loaded.append(load_base(model_id)[0])
        return sample_batch(loaded[0], tok, ms, prompts, MAX_NEW_TOKENS, batch_size, label="base runs")

    return sample


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the messenger dataset (R23): base runs, sibling families, n6 merge.")
    p.add_argument("--n6", type=Path, required=True, help="n6 data folder: its train.jsonl is carried, `cell` nominal")
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", type=int, default=1500, help="messenger train tasks")
    p.add_argument("--test", type=int, default=300, help="messenger test tasks")
    p.add_argument("--batch-size", type=int, default=32, help="base runs sampled together")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    tok, ms = _load_tokenizer(args.model)
    manifest = build(tok, ms, _sampler(args.model, tok, ms, args.batch_size, args.seed), args.n6, args.out,
                     args.train, args.test, args.seed, args.model)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
