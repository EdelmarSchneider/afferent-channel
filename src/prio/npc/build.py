"""Build the game-character dataset (R22, preregistration amendment A8).

uv run python -m prio.npc.build --model Qwen/Qwen3-1.7B --out data/npc6 --seed 0 --train 1250 --test 300 \
    --batch-size 32

Base runs of the untrained model on the character tasks (GPU; cached in `<out>/base_runs.json`
chunk by chunk, so a rerun continues and a complete cache needs no model), cleaning
(`domain.CLEAN_RULE`), then one family of siblings per kept run (`domain.plan`: five siblings at
one cut; the six kinds of family, `domain.SLOTS`, in a drawn order per block of six runs).
  train.jsonl    the training episodes (one state per episode);
  test.jsonl     the test episodes: the same siblings plus, per family, one or two with two states
                 high at once (variants `pair-...`: an extra reading, never part of a line);
  manifest.json  tasks requested, base runs finished / kept (A8 stop rule: kept < 70% of the tasks
                 requested), drop reasons, how often the untrained model's own text already matches
                 a keyword list (`false_naming_floor`), episodes per task and per family, episodes
                 by class and timing kind, by role, by phase.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from prio.datagen.build import BaseRun, cached_runs, sample_batch
from prio.datagen.episodes import Episode, write_jsonl
from prio.modelspec import ModelSpec, load_base
from prio.npc import sentences as S
from prio.npc.domain import CLEAN_RULE, MAX_NEW_TOKENS, NPC, SLOTS, SPECS, dropped_for, load_tasks, plan
from prio.table.core import build_prompt, family
from prio.table.evaluate import families
from prio.table.score import names, spoken

KEPT_MIN = 0.7  # A8 stop rule: base runs kept, as a share of the tasks requested
SPLITS = ("train", "test")


def split_episodes(runs: list[BaseRun], rng: np.random.Generator, tok, ms: ModelSpec, pairs: bool) -> tuple[list[Episode], dict]:
    """(the sibling families of the kept runs of one split, families not built by reason: `no_cut`
    = the run offers no cut for its kind of family, `long` = a sibling would pass the length limit)."""
    episodes, lost = [], Counter()
    kinds = [SLOTS[j] for _ in range(0, len(runs), len(SLOTS)) for j in rng.permutation(len(SLOTS))]
    for run, (felt, timing) in zip(runs, kinds):
        planned = plan(run, felt, timing, rng, tok, ms, pairs)
        siblings = family(NPC, run, *planned, rng, tok, ms, family_id=run.task.task_id) if planned else []
        episodes += siblings
        if not siblings:
            lost["long" if planned else "no_cut"] += 1
    return episodes, dict(lost)


def naming_floor(runs: list[BaseRun], tok, ms: ModelSpec) -> dict:
    """How many kept base runs already hold a word of each keyword list in what they think and say:
    a `base` sibling of such a run counts as naming a state whatever the model does."""
    lists = {**S.THREAT_WORDS, "thirst": S.THIRST_WORDS, "hunger": S.HUNGER_WORDS}
    hits = [{name for name, listed in lists.items() if names(tok.decode(spoken(run.target_ids, ms)), [listed])} for run in runs]
    return {"runs": len(runs), "any": sum(bool(h) for h in hits), "by_list": {name: sum(name in h for h in hits) for name in lists}}


def build(tok, ms: ModelSpec, sample, out: Path, n_train: int, n_test: int, seed: int, model_id: str) -> dict:
    """Write the dataset into `out` and return its manifest. `sample(prompts)` gives the base runs
    as (ids up to and including the end of the turn, stop reason), as `datagen.build.sample_batch`."""
    out, started = Path(out), time.time()
    tasks = load_tasks(n_train, n_test, seed)
    prompts = [build_prompt(tok, NPC, t.user, t.system) for t in tasks]
    raw = cached_runs(sample, prompts, out / "base_runs.json",
                      {"model": model_id, "seed": seed, "domain": NPC.name, "max_new_tokens": MAX_NEW_TOKENS})
    reasons = [dropped_for(ids, stop, tok, ms) for ids, stop in raw]
    runs = {s: [BaseRun(t, p, ids) for t, p, (ids, _), why in zip(tasks, prompts, raw, reasons)
                if why is None and t.split == s] for s in SPLITS}
    built = {s: split_episodes(runs[s], np.random.default_rng([seed, i]), tok, ms, pairs=s == "test")
             for i, s in enumerate(SPLITS)}
    written = {s: built[s][0] for s in SPLITS}
    out.mkdir(parents=True, exist_ok=True)
    for split, episodes in written.items():
        write_jsonl(out / f"{split}.jsonl", episodes)
    requested = {"train": n_train, "test": n_test}
    groups = {s: families(written[s]) for s in SPLITS}
    count = lambda split, key: dict(Counter(key(ep) for ep in written[split]))  # noqa: E731
    per = lambda split, n: len(written[split]) / n if n else None  # noqa: E731
    share = {s: len(runs[s]) / requested[s] if requested[s] else None for s in SPLITS}
    lengths = [len(ep.prompt_ids) + len(ep.target_ids) for s in SPLITS for ep in written[s]]
    own = lambda ep: (ep.meta["table"]["own_command"] or {"name": "none"})["name"]  # noqa: E731
    manifest = {
        "domain": NPC.name, "model": model_id, "seed": seed, "n_signals": len(SPECS), "signals": [s.name for s in SPECS],
        "tasks_requested": requested, "max_new_tokens": MAX_NEW_TOKENS,
        "base_runs_finished": {s: sum(stop == "eos" for t, (_, stop) in zip(tasks, raw) if t.split == s) for s in SPLITS},
        "base_runs_kept": {s: len(runs[s]) for s in SPLITS}, "base_runs_kept_share": share,
        "stop_rule": {"base_runs_kept_min": KEPT_MIN, "passed": all(v is None or v >= KEPT_MIN for v in share.values())},
        "clean_rule": CLEAN_RULE,
        "dropped": {s: dict(Counter(why for t, why in zip(tasks, reasons) if why and t.split == s)) for s in SPLITS},
        "false_naming_floor": {s: naming_floor(runs[s], tok, ms) for s in SPLITS},
        "own_command": {s: dict(Counter(own(siblings[0]) for siblings in groups[s].values())) for s in SPLITS},
        "families": {s: len(groups[s]) for s in SPLITS}, "families_dropped": {s: built[s][1] for s in SPLITS},
        "families_without_mention": {s: sum(all(ep.kind != "table_mention" for ep in siblings) for siblings in groups[s].values())
                                     for s in SPLITS},
        "episodes": {s: len(written[s]) for s in SPLITS},
        "episodes_per_task": {s: per(s, requested[s]) for s in SPLITS},
        "episodes_per_family": {s: per(s, len(groups[s])) for s in SPLITS},
        "kinds": {s: count(s, lambda ep: ep.kind) for s in SPLITS},
        "by_class_timing": {s: count(s, lambda ep: f"{ep.meta['table']['cls']}/{ep.meta['table']['timing'] or 'none'}") for s in SPLITS},
        "by_role": {s: count(s, lambda ep: ep.meta["table"]["facts"]["role"]) for s in SPLITS},
        "by_phase": {s: count(s, lambda ep: ep.phase) for s in SPLITS},
        "variants": {s: count(s, lambda ep: ep.meta["table"]["variant"]) for s in SPLITS},
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
    p = argparse.ArgumentParser(description="Build the game-character dataset (R22): base runs, sibling families.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", type=int, default=1250, help="train tasks (about 4 episodes each: see episodes_per_task)")
    p.add_argument("--test", type=int, default=300, help="test tasks")
    p.add_argument("--batch-size", type=int, default=32, help="base runs sampled together")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    tok, ms = _load_tokenizer(args.model)
    manifest = build(tok, ms, _sampler(args.model, tok, ms, args.batch_size, args.seed), args.out, args.train, args.test,
                     args.seed, args.model)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
