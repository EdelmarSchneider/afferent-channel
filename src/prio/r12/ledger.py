"""The run-once marker file of R12 (AP 9, "Run once, checkable").

`results/r12/RUN_ONCE.json` (tracked by git): one entry per (arm, model) of the confirmatory set,
written when the run starts (time, commit, git tree id of `src/prio`, sha256 of every data file
read, command, batch size) and completed at the end with the sha256 of the records file.

Rules for the confirmatory set (`begin` refuses with `RunOnceError`; nothing is written then):
- the source tree must be clean and the pilot's smoke checks must have passed on that same tree;
- a second start for the same (arm, model) is refused;
- exception, recorded as a resume: the latest entry is unfinished (crash, power cut) and the tree
  id and the data hashes are identical;
- a second run needs a reason: it gets an entry of kind "rerun" and a new folder (`...-rerun<n>`),
  never overwrites, and is reported beside the first run. The confirmatory result is the first run.

Pilot runs are recorded in `results/r12p/RUN_ONCE.json` and never refused.

A resume that finds a results file with a last line torn by a kill or a power cut repairs it
(`prio.r12.torn`: the piece is moved to `<file>.torn`); the repair is written into the entry
(`repairs: [{file, bytes, time, what}]`).

Git is only read (`GIT_COMMANDS`).
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

GIT_COMMANDS = (("rev-parse", "HEAD:src/prio"), ("rev-parse", "HEAD"),
                ("--no-optional-locks", "status", "--porcelain", "--", "src/prio"))
_FOLDER = {"confirm": "r12", "pilot": "r12p"}
_RECORD_FILES = ("records.jsonl", "records.partial.jsonl")


class RunOnceError(RuntimeError):
    """A start that the run-once rule does not allow."""


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def code_tree() -> str:
    """Git tree id of the committed `src/prio`."""
    return _git(*GIT_COMMANDS[0])


def commit() -> str:
    return _git(*GIT_COMMANDS[1])


def tree_clean() -> bool:
    """No uncommitted change and no untracked file under `src/prio`."""
    return _git(*GIT_COMMANDS[2]) == ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def marker_file(set_name: str, root: Path = Path(".")) -> Path:
    return Path(root) / "results" / _FOLDER[set_name] / "RUN_ONCE.json"


def smoke_file(root: Path = Path(".")) -> Path:
    """Where the pilot's smoke checks record their verdict and the tree they ran on."""
    return Path(root) / "results" / "r12p" / "smoke.json"


def entries(set_name: str, root: Path = Path(".")) -> list[dict]:
    path = marker_file(set_name, root)
    return json.loads(path.read_text(encoding="utf-8"))["entries"] if path.exists() else []


def _write(set_name: str, root: Path, rows: list[dict]) -> None:
    path = marker_file(set_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_bytes((json.dumps({"entries": rows}, indent=1) + "\n").encode("utf-8"))
    os.replace(partial, path)  # a power cut leaves the old file or the new one, never half of one


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _relative(path: Path, root: Path) -> str:
    path, root = Path(path), Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def first_run(rows: list[dict], arm: str, model: str) -> dict | None:
    """The confirmatory result of (arm, model): its first entry."""
    return next((e for e in rows if (e["arm"], e["model"]) == (arm, model) and e["kind"] == "run"), None)


def reruns(rows: list[dict], arm: str, model: str) -> list[dict]:
    return [e for e in rows if (e["arm"], e["model"]) == (arm, model) and e["kind"] == "rerun"]


def pilot_passed(root: Path = Path(".")) -> dict:
    """The pilot's smoke checks passed on the `src/prio` tree that is checked out now; else RunOnceError."""
    path = smoke_file(root)
    if not path.exists():
        raise RunOnceError(f"no smoke file at {path}: the pilot's smoke checks come first")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    if smoke.get("passed") is not True:
        raise RunOnceError(f"{path}: the pilot's smoke checks have not passed")
    if smoke.get("tree") != code_tree():
        raise RunOnceError(f"{path}: the smoke checks passed on another src/prio tree ({smoke.get('tree')}) "
                           f"than the one checked out ({code_tree()})")
    return smoke


def begin(set_name: str, arm: str, model: str, out: Path, data_files: list[Path], argv: list[str], *,
          batch_size: int = 16, rerun_reason: str | None = None, root: Path = Path(".")) -> dict:
    """Record the start of a run and return its entry (`resumed`: an unfinished entry goes on;
    `out`: the folder the run writes into, relative to `root`). Raises RunOnceError when the
    run-once rule forbids the start; the marker file is then left as it was."""
    root = Path(root)
    confirm = set_name == "confirm"
    if set_name not in _FOLDER:
        raise ValueError(f"unknown set {set_name!r}")
    if rerun_reason is not None and not confirm:
        raise RunOnceError("--rerun-reason is for the confirmatory set; pilot runs are recorded and never refused")
    if rerun_reason is not None and not rerun_reason.strip():
        raise RunOnceError("a rerun needs a reason in words")
    clean = tree_clean()
    if confirm:
        if not clean:
            raise RunOnceError("src/prio has uncommitted changes: the confirmatory set runs from a committed tree")
        pilot_passed(root)
    data = {_relative(path, root): sha256(path) for path in data_files}
    tree = code_tree()
    rows = entries(set_name, root)
    mine = [e for e in rows if (e["arm"], e["model"]) == (arm, model)] if confirm else []
    folder = _relative(out, root)

    if confirm and rerun_reason is None and mine:
        last = mine[-1]
        if last["finished"] is not None:
            raise RunOnceError(f"{arm} / {model} was already run (started {last['started']}, finished {last['finished']}, "
                               f"{last['out']}): the confirmatory result is the first run; a second run needs --rerun-reason")
        if last["tree"] != tree:
            raise RunOnceError(f"{arm} / {model} has an unfinished entry of src/prio tree {last['tree']}; the tree is now "
                               f"{tree}: not a resume; a second run needs --rerun-reason")
        if last["data"] != data:
            changed = sorted(k for k in set(last["data"]) | set(data) if last["data"].get(k) != data.get(k))
            raise RunOnceError(f"{arm} / {model} has an unfinished entry and its data files changed since ({changed}): "
                               f"not a resume; a second run needs --rerun-reason")
        last["resumes"].append(_now())
        _write(set_name, root, rows)
        return {**last, "resumed": True}

    number = 0
    if rerun_reason is not None:
        if not mine:
            raise RunOnceError(f"{arm} / {model} has no first run: nothing to run again")
        number = len(reruns(rows, arm, model)) + 1
        folder = f"{folder}-rerun{number}"
        if (root / folder).exists():
            raise RunOnceError(f"{folder} exists: a rerun never writes into an existing folder")
    elif confirm and any((root / folder / name).exists() for name in _RECORD_FILES):
        raise RunOnceError(f"{folder} holds records but {marker_file(set_name, root)} has no entry for {arm} / {model}")
    entry = {"set": set_name, "arm": arm, "model": model, "kind": "rerun" if number else "run", "rerun": number,
             "rerun_reason": rerun_reason, "out": folder, "started": _now(), "commit": commit(), "tree": tree,
             "clean": clean, "data": data, "argv": list(argv), "batch_size": batch_size, "resumes": [],
             "repairs": [], "finished": None, "records_sha256": None}
    _write(set_name, root, rows + [entry])
    return {**entry, "resumed": False}


def _stored(entry: dict, root: Path) -> tuple[list[dict], dict]:
    """The rows of the marker file and, among them, the row of `entry`."""
    rows = entries(entry["set"], root)
    key = (entry["arm"], entry["model"], entry["out"], entry["started"])
    for row in reversed(rows):
        if (row["arm"], row["model"], row["out"], row["started"]) == key:
            return rows, row
    raise RunOnceError(f"no entry for {key} in {marker_file(entry['set'], root)}")


def add_repairs(entry: dict, repairs: list[dict], root: Path = Path(".")) -> None:
    """Write the repairs of torn results files (`prio.r12.torn`: {file, bytes, time, what}) into `entry`."""
    if not repairs:
        return
    rows, row = _stored(entry, root)
    row.setdefault("repairs", []).extend(repairs)
    _write(entry["set"], root, rows)


def finish(entry: dict, records_file: Path, root: Path = Path(".")) -> dict:
    """Complete `entry` with the end time and the sha256 of its records file."""
    rows, row = _stored(entry, root)
    row["finished"], row["records_sha256"] = _now(), sha256(records_file)
    _write(entry["set"], root, rows)
    return row
