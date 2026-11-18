"""A last line torn by a kill or a power cut in the middle of a write.

The results files of a run are appended line by line (`records.partial.jsonl`, `generations.jsonl`,
`pass1.jsonl`) and read line by line at a resume: one half-written line breaks every later resume.
Before a resume reads or appends, the tail is repaired:

- a file that does not end with a line break, or whose last line does not parse, has that last
  piece moved (its bytes, with a time stamp line) to `<file>.torn` beside it and continues from its
  last whole line. Nothing is deleted;
- records and generations stay paired: a last row of one file whose run has no row in the other
  (the process died between the two writes, or the other row was the torn one) is dropped back the
  same way, so that the run is made again.

Every repair is returned as `{file, bytes, time, what}`; `prio.r12.run` writes them into the marker
entry of the run. One kill leaves at most one row unpaired: files that differ by more are left as
they are and the repair stops with `TornError`, for a reading by hand.
"""

import json
import time
from pathlib import Path

STREAM_KEY = ("task_id", "run_type")  # arms free, live, Ls (`prio.r12.stream`)
PAIRED_KEY = ("episode_id", "condition", "paired")  # arm paired (`prio.eval.run`)
MAX_UNPAIRED = 1  # rows one kill can leave without their pair, per file


class TornError(RuntimeError):
    """Results files in a state that a single interrupted write cannot explain."""


def _parses(line: bytes) -> bool:
    try:
        return isinstance(json.loads(line.decode("utf-8")), dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False


def _move(path: Path, keep: int, piece: bytes, what: str) -> dict:
    """Append `piece` to `<path>.torn` under a time stamp line and cut `path` back to its first `keep` bytes."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path.with_name(path.name + ".torn"), "ab") as aside:
        aside.write(f"# {stamp} {what} of {path.name}, {len(piece)} bytes\n".encode("utf-8"))
        aside.write(piece if piece.endswith(b"\n") else piece + b"\n")
    with open(path, "r+b") as f:
        f.truncate(keep)
    return {"file": path.name, "bytes": len(piece), "time": stamp, "what": what}


def _last_line_start(data: bytes) -> int:
    """Offset of the last line of `data` (which ends with a line break)."""
    return data.rfind(b"\n", 0, len(data) - 1) + 1


def repair_tail(path: Path) -> list[dict]:
    """Move a torn last line of `path` aside; the file then ends with its last whole line."""
    path = Path(path)
    if not path.exists():
        return []
    data, repairs = path.read_bytes(), []
    while data:
        if data.endswith(b"\n"):
            start = _last_line_start(data)
            line = data[start:-1]
            if not line.strip() or _parses(line):
                break
        else:
            start = data.rfind(b"\n") + 1
        repairs.append(_move(path, start, data[start:], "torn last line"))
        data = data[:start]
    return repairs


def _keys(path: Path, key: tuple[str, ...]) -> list[tuple]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as lines:
        return [tuple(row.get(k) for k in key) for row in map(json.loads, (line for line in lines if line.strip()))]


def repair_pair(records: Path, generations: Path, key: tuple[str, ...]) -> list[dict]:
    """Repair the tails of a records file and its generations file, then drop back a last row of
    either file whose run has no row in the other."""
    records, generations = Path(records), Path(generations)
    repairs = repair_tail(records) + repair_tail(generations)
    record_keys, generation_keys = _keys(records, key), _keys(generations, key)
    drop = {"records": 0, "generations": 0}
    while True:  # counted on the keys first: nothing is touched when the files differ by more than one kill explains
        left, right = record_keys[:len(record_keys) - drop["records"]], generation_keys[:len(generation_keys) - drop["generations"]]
        if right and right[-1] not in set(left):
            drop["generations"] += 1
        elif left and left[-1] not in set(right):
            drop["records"] += 1
        else:
            break
        if max(drop.values()) > MAX_UNPAIRED:
            raise TornError(f"{records} and {generations} do not pair: more than {MAX_UNPAIRED} row(s) at the end of one file "
                            "have no row in the other; nothing was changed")
    for path, n, what in ((generations, drop["generations"], "generation row without its record"),
                          (records, drop["records"], "record without its generation row")):
        for _ in range(n):
            data = path.read_bytes()
            start = _last_line_start(data)
            repairs.append(_move(path, start, data[start:], what))
    return repairs


def repair_stream_folder(folder: Path) -> list[dict]:
    """A results folder of the arms free, live, Ls: `pass1.jsonl`, `records.partial.jsonl`, `generations.jsonl`."""
    folder = Path(folder)
    return repair_tail(folder / "pass1.jsonl") + repair_pair(folder / "records.partial.jsonl", folder / "generations.jsonl", STREAM_KEY)


def repair_paired_folder(folder: Path) -> list[dict]:
    """A results folder of arm paired (`prio.eval.run` reads its partial file line by line too)."""
    folder = Path(folder)
    return repair_pair(folder / "records.partial.jsonl", folder / "generations.jsonl", PAIRED_KEY)
