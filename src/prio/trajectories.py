"""Per-episode signal time series.

Internally every signal is handled as a "pressure" p in [0, 1] that rises toward a limit
p_lim (p = value for bad_when="high", p = 1 - value for bad_when="low"). That gives one set
of samplers for all signals; the result is mapped back to the signal's own orientation.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from prio.signals import SignalSpec

SAFE_MARGIN = 0.05
# Signals are reported in steps of 1 / RESOLUTION (a harness says "89%", then "90%"). Each value
# is the center of its step, so a limit on a step boundary is crossed by a whole step. Without
# this a slow counter sits at limit - 0.001 for dozens of tokens and then at limit + 0.001: a
# difference the phase-0 spike showed the model cannot learn to act on.
RESOLUTION = 100
_EPS = 1e-3  # gap kept on each side of a limit so rounding cannot put a value in the wrong step
_MIN_SLOPE = 1 / 32000  # slowest counter: one 32k-token window / budget / deadline
_NEAR_LO, _NEAR_HI = SAFE_MARGIN - 0.005, 0.005  # near-miss peak sits this far below the limit


@dataclass
class Trajectory:
    values: np.ndarray  # float32 [T, N] in [0, 1]
    event_signal: int | None
    t_cross: int | None
    t_recover: int | None


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return lo if hi <= lo else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _ramp_ending_at(end: float, length: int, rng) -> np.ndarray:
    slope = _log_uniform(rng, _MIN_SLOPE, end / max(length - 1, 1))
    return np.clip(end - slope * (length - 1 - np.arange(length)), 0.0, None)


def _ramp_from(start: float, length: int, rng, cap: float) -> np.ndarray:
    slope = _log_uniform(rng, _MIN_SLOPE, 1 / 2000)
    return np.minimum(start + slope * np.arange(length), cap)


def _noisy(length: int, rng, cap: float, dynamics: str) -> np.ndarray:
    """Nominal path for drift_down / walk signals, never above `cap`."""
    level = rng.uniform(0.05, max(0.06, cap - 0.1))
    if dynamics == "drift_down":  # slow monotone trend plus sensor noise
        steps = rng.exponential(1.0, length)
        total = rng.uniform(0.0, min(0.3, cap - level))
        path = level + total * np.cumsum(steps) / steps.sum() + rng.normal(0.0, 0.003, length)
    else:  # walk: mean-reverting AR(1) around `level`, vectorized
        a, t = 0.95, np.arange(length)
        noise = rng.normal(0.0, 0.02, length)
        path = level + a**t * np.cumsum(noise * a ** (-t.astype(np.float64)))
    return np.clip(path, 0.0, cap)


def _counter(length, rng, p_lim, cap, role, t_cross, t_recover) -> np.ndarray:
    t = np.arange(length)
    if role == "nominal":
        return _ramp_ending_at(rng.uniform(0.05, cap), length, rng)
    if role == "near":
        return _ramp_ending_at(rng.uniform(p_lim - _NEAR_LO, p_lim - _NEAR_HI), length, rng)
    if t_cross == 0:  # starts past the limit
        path = np.minimum(rng.uniform(p_lim + _EPS, 1.0) + _log_uniform(rng, _MIN_SLOPE, 1 / 200) * t, 1.0)
    else:  # ramp through the limit exactly at t_cross; slope range decorrelates position from value
        ramp = p_lim + _log_uniform(rng, _MIN_SLOPE, p_lim / t_cross) * (t - t_cross)
        path = np.where(t < t_cross, np.minimum(ramp, p_lim - _EPS), np.maximum(ramp, p_lim + _EPS))
        path = np.clip(path, 0.0, 1.0)
    if t_recover is not None and t_recover < length:
        path[t_recover:] = _ramp_from(rng.uniform(0.05, 0.5), length - t_recover, rng, cap)
    return path


def _noisy_signal(length, rng, p_lim, cap, role, intervals, dynamics) -> np.ndarray:
    path = _noisy(length, rng, cap, dynamics)
    if role == "nominal":
        return path
    if role == "near":  # triangular bump whose peak enters the margin band without passing
        peak = rng.uniform(p_lim - _NEAR_LO, p_lim - _NEAR_HI)
        center, width = rng.integers(0, length), rng.integers(5, 101)
        bump = peak * np.clip(1.0 - np.abs(np.arange(length) - center) / width, 0.0, None)
        return np.maximum(path, bump)
    good_from = 0  # first token of the good stretch before the interval
    for t_cross, t_recover in intervals:
        end = length if t_recover is None else min(t_recover, length)
        if t_cross > good_from and rng.random() < 0.5:  # ramp into the limit; otherwise a sudden jump
            r = int(min(t_cross - good_from, rng.integers(5, 201)))
            path[t_cross - r:t_cross] = np.linspace(path[t_cross - r], p_lim - _EPS, r)
        past = rng.uniform(p_lim + _EPS, min(1.0, p_lim + 0.08)) + rng.normal(0.0, 0.005, max(end - t_cross, 0))
        path[t_cross:end] = np.clip(past, p_lim + _EPS, 1.0)
        good_from = end
    return path


def _checked(intervals, dynamics: str) -> list[tuple[int, int | None]]:
    """Past-limit intervals [a, b) in order, at least one good token between two; only the last may stay open."""
    out = [(int(a), None if b is None else int(b)) for a, b in intervals]
    if not out:
        raise ValueError("an event needs at least one past-limit interval")
    if len(out) > 1 and dynamics in ("counter_up", "counter_down"):
        raise ValueError("a counter passes its limit once: several intervals need a binary, drift or walk signal")
    for k, (a, b) in enumerate(out):
        if a < 0 or (b is not None and b <= a) or (b is None and k < len(out) - 1) or (k and a <= out[k - 1][1]):
            raise ValueError(f"bad past-limit intervals {out}")
    return out


def sample_trajectory(
    specs: Sequence[SignalSpec],
    length: int,
    rng: np.random.Generator,
    *,
    event: int | None = None,
    t_cross: int | None = None,
    t_recover: int | None = None,
    near_miss: bool = False,
    think_end: int | None = None,
    intervals: Sequence[tuple[int, int | None]] | None = None,
) -> Trajectory:
    """Sample one [length, N] trajectory.

    event=None: every signal stays at least SAFE_MARGIN from its limit, except that with
    near_miss=True one or two non-binary signals enter the margin band without passing.
    event=i: signal i is past its limit exactly on [t_cross, t_recover or end); others nominal.
    intervals (instead of t_cross / t_recover; binary, drift and walk signals): signal `event` is
    past its limit exactly on each [a, b), b=None = to the end (P-e1, several flips). The result's
    t_cross is the first a, its t_recover the last b.
    think_end: from that step on, `think` holds its nominal value (thinking is over).
    """
    if intervals is not None:
        if event is None or t_cross is not None or t_recover is not None:
            raise ValueError("intervals need an event and replace t_cross / t_recover")
        intervals = _checked(intervals, specs[event].dynamics)
        t_cross, t_recover = intervals[0][0], intervals[-1][1]
    elif event is not None:
        if t_cross is None:
            raise ValueError("an event needs t_cross")
        intervals = [(t_cross, t_recover)]
    near: set[int] = set()
    if near_miss:
        candidates = [i for i, s in enumerate(specs) if s.dynamics != "binary" and i != event]
        k = min(len(candidates), int(rng.integers(1, 3)))
        near = set(rng.choice(candidates, size=k, replace=False).tolist())

    values = np.empty((length, len(specs)), dtype=np.float32)
    for i, spec in enumerate(specs):
        p_lim = spec.limit if spec.bad_when == "high" else 1.0 - spec.limit
        cap = p_lim - SAFE_MARGIN - _EPS
        role = "event" if i == event else "near" if i in near else "nominal"
        if spec.dynamics == "binary":
            p = np.zeros(length)
            if role == "event":
                for a, b in intervals:
                    p[a:b] = 1.0
        elif spec.dynamics in ("counter_up", "counter_down"):
            p = _counter(length, rng, p_lim, cap, role, t_cross, t_recover)
        else:
            p = _noisy_signal(length, rng, p_lim, cap, role, intervals, spec.dynamics)
        p = np.clip(p, 0.0, 1.0)
        if spec.dynamics != "binary":  # report at 1% resolution, as the center of the bin
            p = (np.minimum(np.floor(p * RESOLUTION), RESOLUTION - 1) + 0.5) / RESOLUTION
        values[:, i] = p if spec.bad_when == "high" else 1.0 - p
        if spec.name == "think" and think_end is not None:
            values[think_end:, i] = spec.nominal

    has_event = event is not None
    return Trajectory(values, event, t_cross if has_event else None, t_recover if has_event else None)
