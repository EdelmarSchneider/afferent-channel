"""Several-flips test set (preregistration amendment A2; Appendix A of the paper).

From each gating event of the quick subset of <src>/test.jsonl: the forced prefix up to the tool
decision, with the gating signal past its limit on m separate intervals before the decision
(m in MS), ending bad (kind `gating_event`: name it, use the alternative tool) or good (kind
`recovery`: call the gated tool). m = 1 are the trained kinds. Test only; CPU, seconds.

- Change points: distinct target positions drawn uniformly before the decision, per (episode, m,
  final state, seed): the same for every carrier. Only the gating signal's column is redrawn
  (`trajectories.sample_trajectory(intervals=...)`); every other column is the source's.
- Channel form (no `--carrier`): signals only. `--carrier notice | insert`: one notice / one
  inserted position right after the first token of every new state (contract of
  `datagen/variants.py`: harness spans, shifted `t_decision`, `meta["variant"]`).
- Episode ids `<source id>-m<m>-<bad|good>`; `meta["flips"]`: m, final, intervals [a, b) in
  source target indices (b null = up to the decision), source id, decision index.
- Evaluate with `prio.eval.run --mode harness` (one stored trajectory per run, any model);
  summary: `python -m prio.eval.flips`.

python -m prio.datagen.flips --src data/n6 --out data/n6-flips
python -m prio.datagen.flips --src data/n6 --out data/n6-notice-flips --carrier notice
python -m prio.datagen.flips --src data/n6 --out data/n6-insert-flips --carrier insert --insert-id 151669

Training set of the control P-e1b (amendment A3): `--train-with <the carrier's training folder>`
writes <out>/train.jsonl = that folder's train.jsonl unchanged, then one pair per gating event of
<src>/train.jsonl that has a no-event twin, nothing cut: the event with m past-limit intervals
before its decision, the last one open to the end (final bad, same reaction), and its twin with
m closed intervals (final good, same gated tool call; kind `nominal`, `t_diverge` = the decision).
Pair ids `<source id>-m<m>-<bad|good>`; `meta["flips"]` also names the signal.

python -m prio.datagen.flips --src data/n6 --out data/n6-notice-m2 --carrier notice --train-with data/n6-notice
python -m prio.datagen.flips --src data/n6 --out data/n6-flips-m4 --ms 4
"""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from prio.datagen.episodes import Episode, read_jsonl, write_jsonl
from prio.datagen.variants import Carrier, _noticed, _rng, _row, insert, manifest_model
from prio.signals import signal_set
from prio.trajectories import sample_trajectory

MS = (2, 3)
EVERY = 4  # the quick subset of eval/run.py
CARRIERS = (None, "notice", "insert")


def n_changes(m: int, final_bad: bool) -> int:
    return 2 * m - 1 if final_bad else 2 * m


def change_points(d: int, m: int, final_bad: bool, rng) -> list[tuple[int, int | None]]:
    """m past-limit intervals [a, b) over the target indices before the decision `d`; the last one
    stays open (b None) when the final state is bad."""
    k = n_changes(m, final_bad)
    if d < k:
        raise ValueError(f"{k} state changes do not fit in the {d} tokens before the decision")
    p = sorted(int(x) for x in rng.choice(d, size=k, replace=False))
    return [(p[2 * i], p[2 * i + 1] if 2 * i + 1 < k else None) for i in range(m)]


def flips_episode(ep: Episode, specs, m: int, final_bad: bool, seed: int = 0) -> Episode:
    """Channel form: `ep` (a gating event) cut at its tool decision, its gating signal redrawn."""
    if ep.kind != "gating_event" or ep.t_decision is None:
        raise ValueError(f"{ep.episode_id}: a flips episode is built from a gating event, not from kind {ep.kind}")
    index = [s.name for s in specs].index(ep.signal)
    n_prompt, d, final = len(ep.prompt_ids), ep.t_decision, "bad" if final_bad else "good"
    rng = _rng(seed, ep.episode_id, m, final)
    intervals = change_points(d, m, final_bad, rng)
    signals = _redrawn(ep, specs, index, intervals, rng, n_prompt + d)
    info = {"m": m, "final": final, "intervals": [list(x) for x in intervals], "source": ep.episode_id, "decision": d}
    return replace(ep, episode_id=f"{ep.episode_id}-m{m}-{final}", kind="gating_event" if final_bad else "recovery",
                   target_ids=list(ep.target_ids[:d]), signals=signals, t_cross=intervals[0][0], t_diverge=None,
                   meta={**(ep.meta or {}), "flips": info})


def _redrawn(ep: Episode, specs, index: int, intervals, rng, length: int):
    """`ep.signals[:length]` with column `index` past its limit on `intervals` (target indices; open = to the end)."""
    n_prompt = len(ep.prompt_ids)
    rows = [(n_prompt + a, None if b is None else n_prompt + b) for a, b in intervals]
    signals = ep.signals[:length].copy()
    signals[:, index] = sample_trajectory(specs, length, rng, event=index, intervals=rows).values[:, index]
    return signals


def train_pair(event: Episode, twin: Episode, specs, m: int = 2, seed: int = 0) -> tuple[Episode, Episode]:
    """Training pair of P-e1b (amendment A3), channel form, nothing cut: `event` (a gating event)
    ending bad after m intervals and its no-event `twin` ending good after m closed ones."""
    if event.kind != "gating_event" or event.t_decision is None:
        raise ValueError(f"{event.episode_id}: a training pair is built from a gating event, not from kind {event.kind}")
    d = event.t_decision
    if (twin.kind != "nominal" or twin.t_diverge != d or twin.task_id != event.task_id
            or twin.prompt_ids != event.prompt_ids or twin.target_ids[:d] != event.target_ids[:d]):
        raise ValueError(f"{twin.episode_id} is not the no-event twin of {event.episode_id}")
    index, pair = [s.name for s in specs].index(event.signal), []
    for ep, final_bad in ((event, True), (twin, False)):
        final = "bad" if final_bad else "good"
        rng = _rng(seed, ep.episode_id, m, final)
        intervals = change_points(d, m, final_bad, rng)
        signals = _redrawn(ep, specs, index, intervals, rng, len(ep.prompt_ids) + len(ep.target_ids))
        info = {"m": m, "final": final, "intervals": [list(x) for x in intervals], "source": ep.episode_id, "decision": d,
                "signal": event.signal}
        pair.append(replace(ep, episode_id=f"{ep.episode_id}-m{m}-{final}", signals=signals,
                            t_cross=intervals[0][0] if final_bad else None, meta={**(ep.meta or {}), "flips": info}))
    return pair[0], pair[1]


def noticed(ep: Episode, specs, carrier: Carrier) -> Episode:
    """`ep` (channel form) with one notice / inserted position after the first token of every new state."""
    name = ep.meta["flips"].get("signal", ep.signal)  # a training twin (kind nominal) carries no signal of its own
    index = [s.name for s in specs].index(name)
    changes = [p for a, b in ep.meta["flips"]["intervals"] for p in (a, b) if p is not None]
    insertions = []
    for k, p in enumerate(changes):
        row, role = _row(ep, p + 1), "recovery" if k % 2 else "crossing"
        x = _noticed(carrier, specs, p + 1, index, float(row[index]), row, role, anchor=k == 0 and ep.t_cross is not None)
        if specs[index].past_limit(float(x.shown)) != (role == "crossing"):
            raise ValueError(f"{ep.episode_id}: {name} = {x.shown} at change {k} is on the wrong side of its limit")
        insertions.append(x)
    return insert(ep, insertions, "flips")


def build(src: Path, out: Path, tok, *, carrier: str | None = None, insert_id: int | None = None, seed: int = 0,
          ms=MS, every: int = EVERY) -> dict:
    """Write <out>/test.jsonl and <out>/manifest.json. Deterministic: a re-run writes the same bytes."""
    from prio.eval.run import quick_subset  # here: pulls in the model stack

    src, out = Path(src), Path(out)
    if src.resolve() == out.resolve():
        raise ValueError("--out must differ from --src")
    if carrier not in CARRIERS:
        raise ValueError(f"unknown carrier {carrier!r}; one of {CARRIERS}")
    if carrier == "insert" and insert_id is None:
        raise ValueError("the insert carrier needs --insert-id (the delivery's INSERT_ID)")
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8")) if (src / "manifest.json").exists() else {}
    if "variant" in src_manifest:
        raise ValueError(f"{src} is a variant ({src_manifest['variant']}): build from the source dataset")
    episodes = read_jsonl(src / "test.jsonl")
    specs = signal_set(episodes[0].signals.shape[1])
    render = None if carrier is None else Carrier(tok, insert_id if carrier == "insert" else None)
    result, skipped = [], Counter()
    for i in quick_subset(episodes, every):
        ep = episodes[i]
        if ep.kind != "gating_event":
            continue
        for m in ms:
            for final_bad in (True, False):
                if ep.t_decision < n_changes(m, final_bad):
                    skipped["too_short"] += 1
                    continue
                made = flips_episode(ep, specs, m, final_bad, seed)
                result.append(noticed(made, specs, render) if render else made)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "test.jsonl", result)
    records = [r for e in result for r in e.meta.get("variant", {}).get("insertions", [])]
    kept = ("model", "n_signals", "seed", "tasks_requested")
    manifest = {"variant": "flips", "source": str(src),
                "source_manifest": {k: src_manifest[k] for k in kept if k in src_manifest},
                "options": {"seed": seed, "ms": list(ms), "every": every, "carrier": carrier, "insert_id": insert_id},
                "episodes": {"test": len(result)}, "kinds": {"test": dict(Counter(e.kind for e in result))},
                "sources": {"test": len({e.meta["flips"]["source"] for e in result})},
                "insertions": {"test": dict(Counter(r["role"] for r in records))},
                "harness_tokens": {"test": int(sum(r["end"] - r["at"] for r in records))},
                "skipped": dict(skipped)}
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    return manifest


def build_train(src: Path, base: Path, out: Path, tok, *, carrier: str | None, insert_id: int | None = None, seed: int = 0,
                m: int = 2) -> dict:
    """Write <out>/train.jsonl (P-e1b, amendment A3): <base>/train.jsonl, the carrier's own training
    file, unchanged, then one noticed pair per gating event of <src>/train.jsonl that has a twin."""
    src, base, out = Path(src), Path(base), Path(out)
    if out.resolve() in (src.resolve(), base.resolve()):
        raise ValueError("--out must differ from --src and from --train-with")
    if carrier not in ("notice", "insert"):
        raise ValueError(f"a training set needs a carrier (notice | insert), not {carrier!r}")
    if carrier == "insert" and insert_id is None:
        raise ValueError("the insert carrier needs --insert-id (the delivery's INSERT_ID)")
    options = json.loads((base / "manifest.json").read_text(encoding="utf-8")).get("options", {})
    if options.get("carrier") != carrier or (carrier == "insert" and options.get("insert_id") != insert_id):
        raise ValueError(f"{base} was not built for the {carrier} carrier (its options: {options})")
    episodes = read_jsonl(src / "train.jsonl")
    specs = signal_set(episodes[0].signals.shape[1])
    render = Carrier(tok, insert_id if carrier == "insert" else None)
    twins = {e.task_id: e for e in episodes if e.kind == "nominal" and e.t_diverge is not None}
    pairs, skipped = [], Counter()
    for ep in episodes:
        if ep.kind != "gating_event" or ep.task_id not in twins:
            continue
        if ep.t_decision < n_changes(m, False):
            skipped["too_short"] += 1
            continue
        pairs += [noticed(made, specs, render) for made in train_pair(ep, twins[ep.task_id], specs, m, seed)]
    own = read_jsonl(base / "train.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", own + pairs)
    records = [r for e in pairs for r in e.meta["variant"]["insertions"]]
    manifest = {"variant": "flips-train", "source": str(src), "base": str(base),
                "options": {"seed": seed, "m": m, "carrier": carrier, "insert_id": insert_id},
                "episodes": {"train": len(own) + len(pairs)}, "base_episodes": len(own), "pairs": len(pairs) // 2,
                "pair_signals": dict(Counter(e.signal for e in pairs if e.signal)),
                "insertions": {"train": dict(Counter(r["role"] for r in records))},
                "harness_tokens": {"train": int(sum(r["end"] - r["at"] for r in records))},
                "skipped": dict(skipped)}
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the several-flips test set (P-e1) or the P-e1b training set.")
    parser.add_argument("--src", type=Path, required=True, help="source dataset (data/n6), not a variant")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--carrier", choices=("notice", "insert"), default=None, help="default: channel form, signals only")
    parser.add_argument("--insert-id", type=int, default=None, help="--carrier insert: the delivery's INSERT_ID")
    parser.add_argument("--model", default=None, help="tokenizer (notice carrier); defaults to the model in <src>/manifest.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ms", type=int, nargs="+", default=None,
                        help="numbers of past-limit intervals; default 2 3 (test set), 2 (training set, one value)")
    parser.add_argument("--train-with", type=Path, default=None,
                        help="build the P-e1b training set instead: this folder's train.jsonl + one pair per twinned gating event")
    args = parser.parse_args(argv)
    if args.train_with is not None and args.ms is not None and len(args.ms) != 1:
        parser.error("--train-with takes one value in --ms")
    tok = None
    if args.carrier == "notice":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            args.model or manifest_model(json.loads((args.src / "manifest.json").read_text(encoding="utf-8"))))
    if args.train_with is not None:
        manifest = build_train(args.src, args.train_with, args.out, tok, carrier=args.carrier, insert_id=args.insert_id,
                               seed=args.seed, m=args.ms[0] if args.ms else 2)
    else:
        manifest = build(args.src, args.out, tok, carrier=args.carrier, insert_id=args.insert_id, seed=args.seed,
                         ms=tuple(args.ms) if args.ms else MS)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
