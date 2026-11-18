"""Simulated harness for the evaluation of background tasks (spec section 6.6).

It answers the model's tool calls and gives the channel values for every token position, with the
same rules `prio.bg.assemble` used to write the training episodes: replaying an episode's own calls
through the harness reproduces its signals exactly (tested).
"""

import numpy as np

from prio.bg.assemble import PAYLOADS, STALL_HOLD
from prio.bg.curves import progress_at
from prio.bg.kinds import BgTask
from prio.bg.protocol import DONE_PROGRESS, N_INPUTS, N_SLOTS, STATE_VALUE, slot_columns
from prio.datagen.episodes import Episode
from prio.reactions import ToolCall


class SimHarness:
    def __init__(self, tasks: list[BgTask], curves: dict[str, dict], policy: str, event: dict | None):
        self.tasks = {t.task_id: t for t in tasks}
        self.curves = dict(curves)
        self.policy = policy
        self.event = event  # {"task_id", "kind", "position"} or None
        self.next_id = 1
        self.slot: dict[str, int] = {}  # task id -> slot
        self.slot_from: dict[str, int] = {}  # task id -> first position with the slot in use
        self.unwatch_from: dict[str, int] = {}
        self.resume_from: int | None = None  # waiting: running again from here
        self.ack_from: int | None = None  # stalled: status checked, running again from here
        self.pending: list[tuple[str, str]] = []  # (change, task id) waiting for `commit`
        self.now = 0  # last position asked for by `row`

    @staticmethod
    def from_episode(ep: Episode, swap_slot: int | None = None) -> "SimHarness":
        """A harness for a stream whose opening already happened (tasks started and watched as in
        the episode). `swap_slot`: the event moves to the task watched in that slot."""
        meta = ep.meta
        tasks = [BgTask.from_dict(d) for d in meta["tasks"]]
        event = None
        if meta.get("event"):
            index = meta["event"]["task_index"] if swap_slot is None else swap_slot - 1
            event = {"task_id": tasks[index].task_id, "kind": meta["event"]["kind"],
                     "position": meta["event"]["position"]}
        h = SimHarness(tasks, {d["task_id"]: d["curve"] for d in meta["tasks"] if "curve" in d}, meta["policy"], event)
        started = any("slot" in d for d in meta["tasks"])  # opening probes: nothing started yet, ids from t1
        h.next_id = len(tasks) + 1 if started else 1
        for d in meta["tasks"]:
            if "slot" in d:
                h.slot[d["task_id"]] = d["slot"]
                h.slot_from[d["task_id"]] = d["curve"]["start"]
        return h

    # --- tool calls --------------------------------------------------------------------------

    def _fired(self, task_id: str, position: int) -> bool:
        return bool(self.event) and self.event["task_id"] == task_id and position >= self.event["position"]

    def _answer(self, call: ToolCall, position: int) -> dict:
        name, args = call.name, call.arguments
        if name == "start_task":
            task_id = f"t{self.next_id}"
            self.next_id += 1
            if task_id not in self.tasks:  # a task the episode did not plan: never progresses, never ends
                self.tasks[task_id] = BgTask(task_id, "unplanned", False, "", "", str(args.get("description", "")),
                                             "", "", None, None, "done", "", "")
            return {"task_id": task_id}
        if name not in ("watch", "unwatch", "task_result", "task_status", "answer_task"):
            return {"error": "unknown tool"}
        task_id = args.get("task_id")
        if not isinstance(task_id, str) or task_id not in self.tasks:  # a list or a number: off-script output
            return {"error": "unknown task"}
        task = self.tasks[task_id]
        fired = self._fired(task_id, position)
        kind = self.event["kind"] if fired else None
        if name == "watch":
            if task_id in self.slot and task_id not in self.unwatch_from:
                return {"slot": self.slot[task_id]}
            used = {s for t, s in self.slot.items() if t not in self.unwatch_from}
            free = [s for s in range(1, N_SLOTS + 1) if s not in used]
            if not free:
                return {"error": "no free slot"}
            self.slot[task_id] = free[0]
            self.unwatch_from.pop(task_id, None)
            self.pending.append(("watch", task_id))
            return {"slot": free[0]}
        if name == "unwatch":
            self.pending.append(("unwatch", task_id))
            return {"ok": True}
        if name == "task_result":
            return PAYLOADS["done"](task)[1] if kind == "done" else {"state": kind or "running"}
        if name == "task_status":
            if kind in ("error", "waiting", "stalled"):
                if kind == "stalled":
                    self.pending.append(("ack", task_id))
                return PAYLOADS[kind](task)[0]
            return {"state": kind or "running"}
        # answer_task
        if kind != "waiting":
            return {"error": "the task is not waiting for an answer"}
        self.pending.append(("resume", task_id))
        return {"ok": True}

    def respond(self, calls: list[ToolCall], position: int | None = None) -> list[dict]:
        """One payload per call. The state changes they cause take effect from `position + 1`, the
        token after the harness segment; without `position`, call `commit` once the segment's
        length is known."""
        payloads = [self._answer(c, self.now if position is None else position) for c in calls]
        if position is not None:
            self.commit(position)
        return payloads

    def commit(self, position: int) -> None:
        start = position + 1
        for change, task_id in self.pending:
            if change == "watch":
                self.slot_from[task_id] = start
            elif change == "unwatch":
                self.unwatch_from[task_id] = start
            elif change == "resume":
                self.resume_from = start
            elif change == "ack":
                self.ack_from = start
        self.pending = []

    # --- channel values ------------------------------------------------------------------------

    def _progress(self, task_id: str, position: int) -> float:
        curve = self.curves.get(task_id)
        return progress_at(curve, position) if curve else 0.0

    def row(self, position: int) -> np.ndarray:
        self.now = max(self.now, position)
        values = np.zeros(N_INPUTS, dtype=np.float32)
        for k in range(1, N_SLOTS + 1):
            values[slot_columns(k)[1]] = STATE_VALUE["off"]
        for task_id, slot in self.slot.items():
            if position < self.slot_from.get(task_id, 1 << 30):
                continue
            if task_id in self.unwatch_from and position >= self.unwatch_from[task_id]:
                continue
            p, s = slot_columns(slot)
            state, progress = "running", self._progress(task_id, position)
            if self._fired(task_id, position):
                e, kind = self.event["position"], self.event["kind"]
                frozen = self._progress(task_id, e - 1)
                state, progress = kind, (DONE_PROGRESS if kind == "done" else frozen)
                if kind == "waiting" and self.resume_from is not None and position >= self.resume_from:
                    shift = self.resume_from - e
                    state, progress = "running", max(frozen, self._progress(task_id, position - shift))
                if kind == "stalled" and self.ack_from is not None and position >= self.ack_from:
                    state = "running"
                    again = self.ack_from + STALL_HOLD
                    if position >= again:
                        progress = max(frozen, self._progress(task_id, position - (again - e)))
            values[p], values[s] = progress, STATE_VALUE[state]
        return values
