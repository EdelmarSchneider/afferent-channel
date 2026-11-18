"""The episode record and its JSONL storage."""

import base64
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

Kind = Literal["nominal", "event", "starts_past", "gating_event", "recovery", "gating_unused"]


@dataclass(eq=False)
class Episode:
    episode_id: str
    task_id: str
    family: str
    split: str
    kind: Kind
    signal: str | None  # the signal that goes past its limit, if any
    phase: Literal["thinking", "answer"] | None  # where the event lands
    prompt_ids: list[int]
    target_ids: list[int]
    signals: np.ndarray  # float32 [len(prompt_ids) + len(target_ids), N]
    # Index into target_ids of the first token that carries the past-limit signal as it enters.
    # The model can only react in its next prediction: a reaction starts at t_cross + 1.
    t_cross: int | None
    t_decision: int | None  # gating: index into target_ids of the <tool_call> token
    templated: bool  # reaction came from the fallback bank, not from the teacher
    # Reaction episodes and their no-event twins: index into target_ids of the first token on
    # which the two differ. Same text before it, so only the channel can decide that token.
    t_diverge: int | None = None
    # Background-task episodes (prio.bg): half-open [start, end) ranges of target_ids written by
    # the harness (tool answers: context, no loss) and ranges weighted like a reaction.
    harness_spans: list[list[int]] | None = None
    focus_spans: list[list[int]] | None = None
    meta: dict | None = None  # JSON-serializable description of the episode (tasks, event, positions)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Episode):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f.name), getattr(other, f.name)) if f.name == "signals"
            else getattr(self, f.name) == getattr(other, f.name)
            for f in fields(self)
        )


def reaction_start(ep: Episode) -> int | None:
    """Index into target_ids of the first reaction token, for episodes built with a reaction."""
    if ep.kind == "gating_event":
        return ep.t_decision
    if ep.kind in ("event", "starts_past"):
        return ep.t_cross + 1
    return None


def write_jsonl(path: Path, episodes: Iterable[Episode]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for ep in episodes:
            row = asdict(ep)
            values = np.ascontiguousarray(ep.signals, dtype=np.float32)
            row["signals"] = {"shape": list(values.shape), "b64": base64.b64encode(values.tobytes()).decode("ascii")}
            out.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[Episode]:
    episodes = []
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            packed = row["signals"]
            raw = np.frombuffer(base64.b64decode(packed["b64"]), dtype=np.float32)
            row["signals"] = raw.reshape(packed["shape"]).copy()
            episodes.append(Episode(**row))
    return episodes
