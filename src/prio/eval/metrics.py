"""Metrics over evaluation runs. Pure functions: no model, no tokenizer."""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


@dataclass
class RunRecord:
    episode_id: str
    signal: str  # the signal this run is judged against
    kind: str  # "event", "starts_past", "gating_event", "recovery", "gating_unused", or "swap" (swap control)
    phase: str | None  # "thinking" / "answer": where the event lands
    condition: Literal["on", "zeroed", "base"]
    paired: Literal["event", "nominal", "near_miss", "swapped"]  # which trajectory this run received
    event_occurred: bool  # the signal really was past its limit while it could matter
    reaction_signal: str | None  # signal the model reacted to (sentence + action), if any
    delay: int | None  # tokens from the first possible moment until the signal is named (own signal only)
    named: bool  # the reaction names its signal
    called_gated_tool: bool  # gating: first tool call was the gated tool
    used_alternative: bool  # gating: first tool call was the alternative
    unnamed_call: bool = False  # a reaction tool was called without the naming sentence: not a reaction


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def wilson_ci(flags: Sequence[bool], z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a proportion.

    Used for the rates instead of the percentile bootstrap, which collapses to a point at 0% and
    100%: exactly where the control rates are expected to sit (0 of 80 gives [0%, 4.6%] here).
    """
    n = len(flags)
    p = sum(bool(f) for f in flags) / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if p == 0.0 else max(0.0, center - half)), (1.0 if p == 1.0 else min(1.0, center + half))


def _rate(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def _ci(flags: list[bool]) -> tuple[float, float] | None:
    return wilson_ci(flags) if flags else None


def summarize(records: list[RunRecord], labels: Sequence[str] | None = None,
              phases: Sequence[str | None] | None = None) -> dict[tuple[str, str, str | None], dict]:
    """Per (condition, label, phase): reaction, delay, false-alarm, identification and gating rates.

    `labels[i]` / `phases[i]` name the row of `records[i]` (default: its signal / phase). A run is
    always judged against its own signal, whatever row it lands in.
    Rates over an empty set are None. `n` counts runs where the event occurred, `n_nominal` the
    quiet runs (nothing to react to) and `n_near_miss` the quiet runs where a signal came close to
    its limit; near-miss alarms are kept apart from plain false alarms.
    """
    labels = [r.signal for r in records] if labels is None else labels
    phases = [r.phase for r in records] if phases is None else phases
    groups: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r, label, phase in zip(records, labels, phases, strict=True):
        groups[(r.condition, label, phase)].append(r)

    out = {}
    for key, group in groups.items():
        events = [r for r in group if r.event_occurred]
        nominal = [r for r in group if not r.event_occurred and r.paired != "near_miss"]
        near = [r for r in group if not r.event_occurred and r.paired == "near_miss"]
        hits = [r.reaction_signal == r.signal for r in events]
        delays = [r.delay for r in events if r.reaction_signal == r.signal and r.delay is not None]
        reacted = [r for r in events if r.reaction_signal is not None]
        alarms = [r.reaction_signal is not None for r in nominal]
        near_alarms = [r.reaction_signal is not None for r in near]
        gating = any(r.kind in ("gating_event", "recovery") for r in group)  # runs cut at a tool decision
        out[key] = {
            "n": len(events),
            "n_nominal": len(nominal),
            "n_near_miss": len(near),
            "reaction_rate": _rate(hits),
            "reaction_ci": _ci(hits),
            "delay_median": float(np.median(delays)) if delays else None,
            "delay_p90": float(np.percentile(delays, 90)) if delays else None,
            "false_alarm_rate": _rate(alarms),
            "false_alarm_ci": _ci(alarms),
            "near_miss_alarm_rate": _rate(near_alarms),
            "near_miss_alarm_ci": _ci(near_alarms),
            "identification_rate": _rate([r.reaction_signal == r.signal and r.named for r in reacted]),
            "unnamed_call_rate": _rate([r.unnamed_call for r in group]),
            "wrongful_call_rate": _rate([r.called_gated_tool for r in events]) if gating else None,
            "wrongful_refusal_rate": _rate([not r.called_gated_tool for r in nominal]) if gating else None,
        }
    return out
