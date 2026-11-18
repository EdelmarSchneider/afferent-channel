"""R11 (ordinary answers), a reading beside the line of amendment A4: do two trained models differ from EACH
OTHER on the preservation check? CPU, seconds.

The line of R11 compares each trained model with the untrained base model. When one model meets it
and the other does not, that is not yet a difference between the two models. Both folders hold the
same items (`prio.eval.preservation`: the same 300 GSM8K problems and 100 explain tasks, the same
batched loop), so the trained rows can be paired item by item: counts of items that only one model
gets right, and an exact McNemar test. The base rows of the two folders are the same model on the
same input and should agree; their disagreements are printed (batched bfloat16 generation can flip rare near-tie tokens).

Usage (project root):
  uv run python scripts/analysis/preservation_compare.py results/r11-wide-levels-s0-b16 \
      results/r11-notice-none-s0-b16 [--out preservation_compare.md]
"""
import argparse
import json
from pathlib import Path

from prio.eval.spoof import mcnemar_exact

CHECKS = (("math", "correct", "GSM8K accuracy"), ("explain", "format", "Explain format kept"))


def _rows(items, family: str, condition_is_base: bool, field: str) -> dict[str, bool]:
    return {r["task_id"]: bool(r[field]) for r in items
            if r["family"] == family and (r["condition"] == "base") == condition_is_base}


def paired(items_a, items_b, family: str, field: str) -> dict:
    """The trained rows of two folders on the items both hold: counts and an exact McNemar test."""
    a, b = _rows(items_a, family, False, field), _rows(items_b, family, False, field)
    shared = [t for t in a if t in b]
    both = sum(a[t] and b[t] for t in shared)
    only_a = sum(a[t] and not b[t] for t in shared)
    only_b = sum(b[t] and not a[t] for t in shared)
    n = len(shared)
    return {"n": n, "both": both, "only_a": only_a, "only_b": only_b, "neither": n - both - only_a - only_b,
            "rate_a": (both + only_a) / n, "rate_b": (both + only_b) / n, "p": mcnemar_exact(only_a, only_b)}


def base_disagreements(items_a, items_b, family: str, field: str) -> int:
    """Items on which the untrained model's outcome differs between the two folders."""
    a, b = _rows(items_a, family, True, field), _rows(items_b, family, True, field)
    return sum(a[t] != b[t] for t in a if t in b)


def _load(folder: Path):
    with open(folder / "items.jsonl", encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def report(folder_a: Path, folder_b: Path) -> str:
    a, b = _load(folder_a), _load(folder_b)
    text = [f"# Preservation check: {folder_a.as_posix()} (A) against {folder_b.as_posix()} (B)", "",
            "The trained models of the two folders on the same items. A reading beside the line of amendment A4, "
            "which compares each model with the untrained base model.", "",
            "| Check | n | A | B | Both | Only A | Only B | Neither | Exact McNemar p | Base rows that differ |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for family, field, label in CHECKS:
        p = paired(a, b, family, field)
        text.append(f"| {label} | {p['n']} | {p['rate_a']:.1%} | {p['rate_b']:.1%} | {p['both']} | {p['only_a']} | "
                    f"{p['only_b']} | {p['neither']} | {p['p']:.3g} | {base_disagreements(a, b, family, field)} |")
    return "\n".join(text) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--out", help="file name inside folder B")
    args = parser.parse_args(argv)
    text = report(args.a, args.b)
    print(text)
    if args.out:
        (args.b / args.out).write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
