"""Freshness audit of the R12 task sets (AP 3.2): no R12 task was used in any training set or
earlier test set. Established twice, both before any confirmatory base run:

- by seed: every earlier draw is recomputed from its seed; no R12 prompt is in any of them; the
  pilot and the confirmatory set share no prompt; no new item is in `topics.json`;
- by file: the token ids of every R12 prompt against the prompt of every episode in every
  `data/*/` folder outside `data/r12*`.

A match in either stops the build.

    python -m prio.r12.freshness --set confirm      (tokenizer only, CPU)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

from prio import tasks as T
from prio.r12 import MODEL_ID, PREFIX, utf8_console
from prio.r12.tasks import earlier_draws, new_items, task_sets, used_prompts
from prio.tasks import build_prompt

# The episode lines carry kilobytes of packed signals: the prompt ids are cut out of the raw line.
_PROMPT_IDS = re.compile(rb'"prompt_ids": \[([^\]]*)\]')


def _rows(sets: dict) -> list[tuple[str, str, object]]:
    return [(name, arm, task) for name, arms in sets.items() for arm, tasks in arms.items() for task in tasks]


def _old_items() -> list[str]:
    old = json.loads(T.TOPICS_PATH.read_text(encoding="utf-8"))
    return [item.lower() for lists in old.values() for split in lists.values() for item in split]


def by_seed(sets: dict, used: set[str] | None = None) -> dict:
    """Every R12 prompt against the prompts of every earlier draw (`used`, default: recomputed
    from the seeds), the pilot set against the confirmatory set, the new items against the old lists."""
    labels = None
    if used is None:
        used = used_prompts()
    rows = _rows(sets)
    hits = []
    for name, arm, task in rows:
        if task.prompt in used:
            if labels is None:
                labels = {label: {t.prompt for t in tasks} for label, tasks in earlier_draws().items()}
            hits.append({"task_id": task.task_id, "set": name, "arm": arm,
                         "draws": [label for label, prompts in labels.items() if task.prompt in prompts]})
    pilot = {task.prompt: task.task_id for name, _, task in rows if name == "pilot"}
    both = [{"pilot": pilot[task.prompt], "confirm": task.task_id}
            for name, _, task in rows if name == "confirm" and task.prompt in pilot]
    old = _old_items()
    new = new_items()
    stale = [item for item in new["topics"] + new["entities"] if item.lower() in old]
    stale += [dataset for dataset in new["datasets"] if any(dataset.lower() in job for job in old)]
    return {"passed": not hits and not both and not stale,
            "tasks": {name: sum(len(tasks) for tasks in arms.values()) for name, arms in sets.items()},
            "earlier_prompts": len(used), "used": hits, "in_both_sets": both, "new_items_in_old_lists": stale}


def prompts_of(path: Path) -> Iterator[tuple[int, tuple[int, ...]]]:
    """(line number, prompt ids) of every line of an episode file that has prompt ids; the line is
    not parsed as JSON."""
    with open(path, "rb") as lines:
        for number, line in enumerate(lines, 1):
            found = _PROMPT_IDS.search(line)
            if found is not None:
                yield number, tuple(int(x) for x in found.group(1).split(b",") if x.strip())


def by_file(sets: dict, data_root: Path, tok) -> dict:
    """The token ids of every R12 prompt against the prompt ids of every episode line of every
    `<data_root>/*/**/*.jsonl` outside the R12 folders."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"no data folder at {data_root}")
    wanted = {tuple(build_prompt(tok, task)): (name, arm, task.task_id) for name, arm, task in _rows(sets)}
    folders = sorted(p for p in data_root.iterdir() if p.is_dir())
    skipped = [p.name for p in folders if p.name.startswith("r12")]  # the R12 sets' own folders
    scanned = [p for p in folders if not p.name.startswith("r12")]
    distinct: set[tuple[int, ...]] = set()
    matches, files, episodes, blank = [], 0, 0, 0
    for folder in scanned:
        for path in sorted(folder.rglob("*.jsonl")):
            files += 1
            with open(path, "rb") as lines:
                total = sum(1 for line in lines if line.strip())
            seen = 0
            for number, ids in prompts_of(path):
                seen += 1
                distinct.add(ids)
                if ids in wanted:
                    name, arm, task_id = wanted[ids]
                    matches.append({"task_id": task_id, "set": name, "arm": arm,
                                    "file": path.relative_to(data_root).as_posix(), "line": number})
            episodes += seen
            blank += total - seen
    return {"passed": not matches, "folders": len(scanned), "files": files, "episodes": episodes,
            "lines_without_prompt": blank, "distinct_prompts": len(distinct), "r12_prompts": len(wanted),
            "matches": matches, "skipped_folders": skipped}


def audit(data_root: Path, tok, out: Path, sets: dict | None = None, used: set[str] | None = None) -> dict:
    """Both audits over both task sets; writes <out>/freshness.json. `passed` only when neither
    finds a match."""
    sets = task_sets() if sets is None else sets
    seed, file = by_seed(sets, used), by_file(sets, data_root, tok)
    result = {"by_seed": seed, "by_file": file, "passed": bool(seed["passed"] and file["passed"])}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "freshness.json").write_bytes((json.dumps(result, indent=1) + "\n").encode("utf-8"))
    return result


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def main(argv: list[str] | None = None) -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description="Freshness audit of the R12 task sets (both sets, by seed and by file).")
    parser.add_argument("--set", choices=tuple(PREFIX), required=True, help="the set whose folder gets freshness.json")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", default=MODEL_ID, help="tokenizer of the prompts")
    args = parser.parse_args(argv)
    result = audit(args.data, _load_tokenizer(args.model), args.data / PREFIX[args.set], task_sets(), used_prompts())
    print(json.dumps(result, indent=1))
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
