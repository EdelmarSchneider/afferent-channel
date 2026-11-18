"""Progress curves of background tasks, one value per token position.

A curve is a small JSON-serializable dict, so an episode's metadata can carry it and the evaluation
harness can extend it past the forced prefix. Values are snapped to 1% level centers (the channel's
resolution) and never decrease. The rate is sampled per curve, so the same position maps to
different progress values (against position leakage, as in phase 1).
"""

import math
from functools import lru_cache

import numpy as np

SHAPES = ("smooth", "stepped", "bursty")


def sample_curve(rng, start: int, end_hint: int, near_miss: bool = False, shape: str | None = None) -> dict:
    """A curve that reaches `cap` at `end_hint` and stays there. cap: a level center in
    [0.205, 0.935], or in [0.955, 0.985] for a near miss (close to the end, never finishing)."""
    level = int(rng.integers(95, 99)) if near_miss else int(rng.integers(20, 94))
    shape = shape or SHAPES[int(rng.integers(len(SHAPES)))]
    rate = {"smooth": float(rng.uniform(0.6, 1.6)), "stepped": float(rng.integers(3, 7)),
            "bursty": float(rng.uniform(0.01, 0.04))}[shape]
    return {"start": int(start), "end_hint": int(max(end_hint, start + 1)), "shape": shape, "rate": rate,
            "seed": int(rng.integers(2**31)), "cap": round((level + 0.5) / 100, 3)}


@lru_cache(maxsize=4096)
def _profile(start: int, end_hint: int, shape: str, rate: float, seed: int) -> np.ndarray:
    """Fraction of the way to the cap at each position from start to end_hint, in [0, 1]."""
    n = end_hint - start
    t = np.arange(n + 1) / n
    if shape == "smooth":
        return t ** rate
    if shape == "stepped":
        stages = int(rate)
        return np.floor(t * stages + 1e-9) / stages
    bursts = np.random.default_rng(seed).random(n + 1) < rate  # bursty: jumps at random positions
    steps = np.cumsum(bursts.astype(float))
    return steps / steps[-1] if steps[-1] > 0 else t


def progress_at(curve: dict, position: int) -> float:
    if position < curve["start"]:
        return 0.0
    cap = curve["cap"]
    index = min(position, curve["end_hint"]) - curve["start"]
    raw = cap * float(_profile(curve["start"], curve["end_hint"], curve["shape"], curve["rate"], curve["seed"])[index])
    snapped = (math.floor(raw * 100 + 1e-9) + 0.5) / 100
    return round(min(snapped, cap), 3)
