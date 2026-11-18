"""Context eviction on the several-flips test files (R21; preregistration amendment A7, 2026-11-16).

A feed that is never cleared drops its oldest tokens. Here the prompt stays whole (system prompt,
tools, user message, assistant opener) and of the model's own turn before the tool decision only
the last W tokens stay, counted in tokens of the original turn (inserted notices do not count), so
the cut is at the same place for every carrier:
- with d tokens before the decision, the first c = max(0, d - W) go, with their signal rows;
- a notice or inserted position that sits before token c goes whole, one at or after it stays whole;
- `meta["flips"]` is untouched (`prio.eval.flips` reads m, final state and distances from it);
  `meta["evict"]`: window, cut, decision, changes, changes_inside (state changes whose notice
  position p + 1 is at or after the cut; always the last ones), last_inside.

python -m prio.datagen.evict --src data/n6-notice-flips --out data/n6-notice-flips-w16 --window 16
"""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from prio.datagen.episodes import Episode, read_jsonl, write_jsonl


def _cut_index(ep: Episode, cut: int) -> int:
    """Index into target_ids of the first token that stays: base token `cut`, moved back over the
    notices inserted right before it."""
    spans = ep.harness_spans or []
    base = [i for i in range(len(ep.target_ids)) if not any(a <= i < b for a, b in spans)]
    s = base[cut]
    while any(b == s for _, b in spans):
        s = next(a for a, b in spans if b == s)
    return s


def evict(ep: Episode, window: int) -> Episode:
    """`ep` (a flips episode, any carrier) with only the last `window` tokens of the turn before its decision."""
    if window < 1:
        raise ValueError("the window keeps at least one token")
    if not ep.meta or "flips" not in ep.meta or ep.t_decision is None:
        raise ValueError(f"{ep.episode_id}: eviction is defined on the several-flips test episodes")
    info = ep.meta["flips"]
    d = info["decision"]
    cut = max(0, d - window)
    s = _cut_index(ep, cut)
    inside = [p + 1 >= cut for a, b in info["intervals"] for p in (a, b) if p is not None]
    n_prompt = len(ep.prompt_ids)
    meta = {**ep.meta, "evict": {"window": window, "cut": cut, "decision": d, "changes": len(inside),
                                 "changes_inside": int(sum(inside)), "last_inside": bool(inside[-1])}}
    if "variant" in ep.meta:
        kept = [{**r, "at": r["at"] - s, "end": r["end"] - s} for r in ep.meta["variant"]["insertions"] if r["at"] >= s]
        meta["variant"] = {**ep.meta["variant"], "insertions": kept}
    spans = [[a - s, b - s] for a, b in ep.harness_spans or () if a >= s]
    return replace(
        ep, target_ids=list(ep.target_ids[s:]),
        signals=np.concatenate([ep.signals[:n_prompt], ep.signals[n_prompt + s:]]).astype(np.float32),
        t_decision=ep.t_decision - s, t_cross=None if ep.t_cross is None else max(ep.t_cross - s, 0),
        harness_spans=spans or None, meta=meta,
    )


def build(src: Path, out: Path, window: int) -> dict:
    """Write <out>/test.jsonl and <out>/manifest.json from a flips folder. Deterministic."""
    src, out = Path(src), Path(out)
    if out.resolve() == src.resolve():
        raise ValueError("the output folder is the source folder")
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8")) if (src / "manifest.json").exists() else {}
    if src_manifest.get("variant") != "flips":
        raise ValueError(f"{src} is not a several-flips folder (prio.datagen.flips)")
    episodes = [evict(ep, window) for ep in read_jsonl(src / "test.jsonl")]
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "test.jsonl", episodes)
    inside = Counter("2+" if e.meta["evict"]["changes_inside"] >= 2 else str(e.meta["evict"]["changes_inside"]) for e in episodes)
    manifest = {"variant": "evict", "source": src_manifest.get("source"), "evicted_from": str(src),
                "source_manifest": src_manifest.get("source_manifest", {}),
                "options": {**src_manifest.get("options", {}), "window": window},
                "episodes": {"test": len(episodes)}, "kinds": {"test": dict(Counter(e.kind for e in episodes))},
                "changes_inside": {"test": dict(sorted(inside.items()))},
                "last_inside": {"test": int(sum(e.meta["evict"]["last_inside"] for e in episodes))}}
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="R21: keep only the last W tokens of the turn before the decision.")
    parser.add_argument("--src", type=Path, required=True, help="a several-flips folder (any carrier)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.src, args.out, args.window), indent=1))


if __name__ == "__main__":
    main()
