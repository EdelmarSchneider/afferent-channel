"""Batch-size equivalence: the same runs evaluated at batch 1 (reference) and batched (candidate).

Greedy bf16 decoding can flip near-tie tokens across batch sizes,
so a batched evaluation is reported only beside its agreement with batch 1 on the same episodes.
Works on any evaluation folder with records(.partial).jsonl + generations.jsonl: standard and harness
mode (`prio.eval.run`), R10 (`prio.eval.limits_eval`), background tasks (`prio.bg.evaluate`), spoof.
Runs are matched by the key fields the records carry (episode, condition, pairing / kind / arm / cell).

    python scripts/analysis/batch_equivalence.py results/<batch-1 folder> results/<batched folder> [--out eq.json]

Prints and returns: matched runs, runs in one folder only, share of identical records, agreement per
field, share of identical generations, median index of the first differing token, the first 20
differing records. CPU, seconds.
"""

import argparse
import json
import statistics
from pathlib import Path

KEYS = ("episode_id", "condition", "paired", "kind", "arm", "cell", "rule")


def _rows(folder: Path, names: tuple[str, ...]) -> list[dict]:
    for name in names:
        path = Path(folder) / name
        if path.exists():
            with open(path, encoding="utf-8") as lines:
                return [json.loads(line) for line in lines if line.strip()]
    return []


def _by_key(rows: list[dict]) -> tuple[list[str], dict[tuple, dict]]:
    key = [k for k in KEYS if rows and k in rows[0]]
    return key, {tuple(r.get(k) for k in key): r for r in rows}  # a resumed run's duplicate: the last one


def _first_difference(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare(reference: Path, candidate: Path, shown: int = 20) -> dict:
    key, ref = _by_key(_rows(reference, ("records.partial.jsonl", "records.jsonl")))
    _, new = _by_key(_rows(candidate, ("records.partial.jsonl", "records.jsonl")))
    matched = [k for k in ref if k in new]
    fields = sorted({f for k in matched for f in ref[k]} - set(key))
    differing = []
    for k in matched:
        changed = {f: [ref[k].get(f), new[k].get(f)] for f in fields if ref[k].get(f) != new[k].get(f)}
        if changed:
            differing.append({"key": list(k), "fields": changed})
    gen_key, gen_ref = _by_key(_rows(reference, ("generations.jsonl",)))
    _, gen_new = _by_key(_rows(candidate, ("generations.jsonl",)))
    gen_matched = [k for k in gen_ref if k in gen_new]
    firsts = [_first_difference(gen_ref[k]["token_ids"], gen_new[k]["token_ids"]) for k in gen_matched
              if gen_ref[k]["token_ids"] != gen_new[k]["token_ids"]]
    share = lambda n, total: n / total if total else None  # noqa: E731
    return {
        "reference": str(reference), "candidate": str(candidate), "key": key,
        "matched": len(matched), "only_reference": len(ref) - len(matched), "only_candidate": len(new) - len(matched),
        "records_identical": share(len(matched) - len(differing), len(matched)),
        "fields": {f: share(sum(ref[k].get(f) == new[k].get(f) for k in matched), len(matched)) for f in fields},
        "generations_matched": len(gen_matched),
        "generations_identical": share(len(gen_matched) - len(firsts), len(gen_matched)),
        "first_difference_median": statistics.median(firsts) if firsts else None,
        "differing": differing[:shown],
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Agreement of a batched evaluation with its batch-1 reference.")
    parser.add_argument("reference", type=Path, help="evaluation folder at batch 1")
    parser.add_argument("candidate", type=Path, help="the same evaluation, batched (its own folder)")
    parser.add_argument("--out", type=Path, default=None, help="write the comparison as JSON")
    args = parser.parse_args(argv)
    out = compare(args.reference, args.candidate)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    pct = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    print(f"matched runs {out['matched']} (only in reference {out['only_reference']}, only in candidate "
          f"{out['only_candidate']}); identical records {pct(out['records_identical'])}; identical generations "
          f"{pct(out['generations_identical'])} of {out['generations_matched']}; first differing token, median "
          f"{out['first_difference_median']}")
    for field, agree in out["fields"].items():
        print(f"  {field}: {pct(agree)}")
    for row in out["differing"]:
        print(f"  differs {row['key']}: {row['fields']}")
    return out


if __name__ == "__main__":
    main()
