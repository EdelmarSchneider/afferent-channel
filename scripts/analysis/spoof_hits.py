"""Decode the counted reactions (hits) of forged-status cells for reading by hand (CPU, seconds).

usage: uv run python scripts/analysis/spoof_hits.py <results folder> <cell,cell,...> [--max N] [--model ID] [--chars N]

Prints, per hit of the trigger arm (condition `on`): episode, forged signal, signal reacted to, first tool call, and the
end of the decoded run (forged text + generation). A hit = a record with a non-empty `reaction_signal`
(`prio.eval.spoof`). Backs the hand readings of R5 (Qwen3-0.6B). Qwen3 sizes share one tokenizer.
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def _load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("cells")
    p.add_argument("--max", type=int, default=99, help="hits shown per cell")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="tokenizer")
    p.add_argument("--chars", type=int, default=700, help="characters shown from the end of each run")
    a = p.parse_args(argv)
    recs = _load(a.folder / "records.jsonl")
    gens = {(g["episode_id"], g["cell"], g["condition"], g["arm"]): g for g in _load(a.folder / "generations.jsonl")}
    tok = AutoTokenizer.from_pretrained(a.model)
    for cell in a.cells.split(","):
        hits = [r for r in recs if r["cell"] == cell and r["condition"] == "on" and r["arm"] == "trigger"
                and r.get("reaction_signal")]
        print(f"===== {cell}: {len(hits)} hits")
        for r in hits[: a.max]:
            g = gens.get((r["episode_id"], r["cell"], r["condition"], r["arm"]))
            text = tok.decode(g["token_ids"]) if g else "(no generation)"
            print(f"--- {r['episode_id']} forged={r.get('spoof_signal')} reacted={r['reaction_signal']} "
                  f"first_call={r.get('first_call')}")
            print(text[-a.chars:].replace("\n", " | "))


if __name__ == "__main__":
    main()
