import numpy as np
import pytest

from prio.signals import CORE_SIGNALS
from prio.trajectories import SAFE_MARGIN, sample_trajectory

SPECS = CORE_SIGNALS


def past(spec, col):
    return np.array([spec.past_limit(float(v)) for v in col])


@pytest.mark.parametrize("seed", range(20))
def test_nominal_never_crosses_and_keeps_margin(seed):
    tr = sample_trajectory(SPECS, 400, np.random.default_rng(seed))
    assert tr.values.shape == (400, 6) and tr.values.dtype == np.float32
    assert tr.values.min() >= 0.0 and tr.values.max() <= 1.0
    for i, s in enumerate(SPECS):
        assert not past(s, tr.values[:, i]).any()
        if s.dynamics != "binary":
            assert np.abs(tr.values[:, i] - s.limit).min() >= SAFE_MARGIN


@pytest.mark.parametrize("event", range(6))
@pytest.mark.parametrize("t_cross", [0, 1, 137, 399])
def test_event_crosses_exactly_at_t_cross(event, t_cross):
    tr = sample_trajectory(SPECS, 400, np.random.default_rng(7), event=event, t_cross=t_cross)
    p = past(SPECS[event], tr.values[:, event])
    assert not p[:t_cross].any() and p[t_cross:].all()
    for i, s in enumerate(SPECS):
        if i != event:
            assert not past(s, tr.values[:, i]).any()


@pytest.mark.parametrize("kwargs", [{}, {"event": 0, "t_cross": 137}, {"event": 5, "t_cross": 60}, {"near_miss": True}])
def test_signals_are_reported_at_one_percent_resolution(kwargs):
    # A harness reports "89%" then "90%": every value sits at the center of a 1% bin, so crossing a
    # limit changes the value by a whole bin instead of by a hair (0.899 -> 0.901).
    tr = sample_trajectory(SPECS, 400, np.random.default_rng(11), **kwargs)
    for i, s in enumerate(SPECS):
        col = tr.values[:, i].astype(np.float64)
        if s.dynamics == "binary":
            assert set(np.unique(col)) <= {0.0, 1.0}
        else:
            steps = col * 100 - 0.5
            assert np.abs(steps - np.round(steps)).max() < 1e-3
    if "event" in kwargs:
        e, t = kwargs["event"], kwargs["t_cross"]
        assert abs(float(tr.values[t, e]) - float(tr.values[t - 1, e])) >= 0.01 - 1e-6


def test_recovery_window():
    net = 4
    tr = sample_trajectory(SPECS, 400, np.random.default_rng(1), event=net, t_cross=100, t_recover=180)
    p = past(SPECS[net], tr.values[:, net])
    assert p[100:180].all() and not p[:100].any() and not p[180:].any()


@pytest.mark.parametrize("event", [0, 2, 5])
def test_recovery_window_for_continuous_signals(event):
    tr = sample_trajectory(SPECS, 400, np.random.default_rng(1), event=event, t_cross=100, t_recover=180)
    p = past(SPECS[event], tr.values[:, event])
    assert p[100:180].all() and not p[:100].any() and not p[180:].any()


def test_near_miss_enters_margin_without_passing():
    for seed in range(20):
        tr = sample_trajectory(SPECS, 400, np.random.default_rng(seed), near_miss=True)
        in_band = 0
        for i, s in enumerate(SPECS):
            assert not past(s, tr.values[:, i]).any()
            if s.dynamics != "binary" and np.abs(tr.values[:, i] - s.limit).min() < SAFE_MARGIN:
                in_band += 1
        assert in_band >= 1


def test_think_is_nominal_after_think_end():
    tr = sample_trajectory(SPECS, 400, np.random.default_rng(3), think_end=150)
    assert (tr.values[150:, 1] == 1.0).all()


def test_counters_are_monotonic():
    tr = sample_trajectory(SPECS, 400, np.random.default_rng(4), event=0, t_cross=250)
    assert (np.diff(tr.values[:, 0]) >= 0).all()  # ctx rises
    assert (np.diff(tr.values[:, 1]) <= 0).all()  # think budget falls
    assert (np.diff(tr.values[:, 3]) >= 0).all()  # time rises


def test_same_position_maps_to_different_ctx_values():  # position-leak guard
    starts = [
        sample_trajectory(SPECS, 400, np.random.default_rng(s), event=0, t_cross=200).values[0, 0]
        for s in range(50)
    ]
    assert np.std(starts) > 0.05


def test_deterministic_given_seed():
    a = sample_trajectory(SPECS, 100, np.random.default_rng(5), event=2, t_cross=40)
    b = sample_trajectory(SPECS, 100, np.random.default_rng(5), event=2, t_cross=40)
    assert np.array_equal(a.values, b.values)
    assert (a.event_signal, a.t_cross, a.t_recover) == (2, 40, None)


# --- several past-limit intervals (P-e1, preregistration amendment A2) ---------------------------

@pytest.mark.parametrize("event", [2, 4, 5])  # energy (drift), net (binary), ram (walk)
@pytest.mark.parametrize("intervals", [[(50, 90), (140, None)], [(0, 3), (10, 11), (200, 260)], [(30, 31), (33, 40), (399, None)]])
def test_intervals_are_past_the_limit_exactly_there(event, intervals):
    tr = sample_trajectory(SPECS, 400, np.random.default_rng(3), event=event, intervals=intervals)
    expected = np.zeros(400, dtype=bool)
    for a, b in intervals:
        expected[a:b] = True
    assert np.array_equal(past(SPECS[event], tr.values[:, event]), expected)
    assert (tr.t_cross, tr.t_recover) == (intervals[0][0], intervals[-1][1])
    for i, s in enumerate(SPECS):
        if i != event:
            assert not past(s, tr.values[:, i]).any()


@pytest.mark.parametrize("event", [2, 4, 5])
@pytest.mark.parametrize("t_recover", [180, None])
def test_one_interval_draws_the_same_trajectory_as_cross_and_recover(event, t_recover):
    # existing datasets must rebuild bit for bit
    a = sample_trajectory(SPECS, 400, np.random.default_rng(1), event=event, t_cross=100, t_recover=t_recover)
    b = sample_trajectory(SPECS, 400, np.random.default_rng(1), event=event, intervals=[(100, t_recover)])
    assert np.array_equal(a.values, b.values) and (b.t_cross, b.t_recover) == (100, t_recover)


@pytest.mark.parametrize("event, intervals", [
    (0, [(10, 20), (30, None)]),  # a counter cannot come back twice
    (4, [(50, 40)]),  # empty interval
    (4, [(10, 30), (30, 40)]),  # no good token between two intervals
    (4, [(10, None), (30, 40)]),  # only the last interval may stay open
    (4, []),
])
def test_bad_intervals_are_rejected(event, intervals):
    with pytest.raises(ValueError):
        sample_trajectory(SPECS, 400, np.random.default_rng(0), event=event, intervals=intervals)


def test_intervals_and_t_cross_together_are_rejected():
    with pytest.raises(ValueError):
        sample_trajectory(SPECS, 400, np.random.default_rng(0), event=4, t_cross=5, intervals=[(5, 9)])
