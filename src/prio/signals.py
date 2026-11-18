"""Signal specifications: what the interoceptive channel carries and when a value is past its limit."""

from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

BadWhen = Literal["high", "low"]
Kind = Literal["interrupt", "gating"]
Dynamics = Literal["counter_up", "counter_down", "drift_down", "walk", "binary"]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    limit: float
    bad_when: BadWhen  # "high": value >= limit is past; "low": value <= limit is past
    kind: Kind  # "interrupt": react now; "gating": react only when about to call the gated tool
    dynamics: Dynamics
    nominal: float  # resting / not-applicable value

    def past_limit(self, value: float) -> bool:
        return value >= self.limit if self.bad_when == "high" else value <= self.limit


CORE_SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec("ctx", 0.9, "high", "interrupt", "counter_up", 0.0),
    SignalSpec("think", 0.1, "low", "interrupt", "counter_down", 1.0),
    SignalSpec("energy", 0.1, "low", "interrupt", "drift_down", 1.0),
    SignalSpec("time", 0.9, "high", "interrupt", "counter_up", 0.0),
    SignalSpec("net", 0.0, "low", "gating", "binary", 1.0),
    SignalSpec("ram", 0.1, "low", "gating", "walk", 1.0),
)


def signal_set(n: int) -> tuple[SignalSpec, ...]:
    """Core signals plus generic `resource_k` signals, n in total."""
    if n < len(CORE_SIGNALS):
        raise ValueError(f"n must be >= {len(CORE_SIGNALS)}, got {n}")
    extra = tuple(
        SignalSpec(f"resource_{k}", 0.1, "low", "interrupt", "walk", 1.0)
        for k in range(n - len(CORE_SIGNALS))
    )
    return CORE_SIGNALS + extra


def with_limits(specs: Sequence[SignalSpec], limits: Mapping[str, float]) -> tuple[SignalSpec, ...]:
    """`specs` with the limits named in `limits` replaced: per-episode limits (R10, limits given in
    the prompt). Signals not named keep their limit; the trajectory samplers derive everything else
    from `limit`."""
    unknown = set(limits) - {s.name for s in specs}
    if unknown:
        raise ValueError(f"no such signals: {sorted(unknown)}")
    return tuple(replace(s, limit=float(limits[s.name])) if s.name in limits else s for s in specs)
