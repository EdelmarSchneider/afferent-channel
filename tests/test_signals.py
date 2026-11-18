import pytest

from prio.signals import CORE_SIGNALS, signal_set


def test_core_signals_match_spec():
    by = {s.name: s for s in CORE_SIGNALS}
    assert [s.name for s in CORE_SIGNALS] == ["ctx", "think", "energy", "time", "net", "ram"]
    assert (by["ctx"].limit, by["ctx"].bad_when, by["ctx"].kind) == (0.9, "high", "interrupt")
    assert (by["think"].limit, by["think"].bad_when, by["think"].kind) == (0.1, "low", "interrupt")
    assert (by["energy"].limit, by["energy"].bad_when, by["energy"].kind) == (0.1, "low", "interrupt")
    assert (by["time"].limit, by["time"].bad_when, by["time"].kind) == (0.9, "high", "interrupt")
    assert (by["net"].limit, by["net"].bad_when, by["net"].kind) == (0.0, "low", "gating")
    assert (by["ram"].limit, by["ram"].bad_when, by["ram"].kind) == (0.1, "low", "gating")
    assert {s.name: s.nominal for s in CORE_SIGNALS} == {
        "ctx": 0.0, "think": 1.0, "energy": 1.0, "time": 0.0, "net": 1.0, "ram": 1.0}


def test_past_limit_is_inclusive():
    by = {s.name: s for s in CORE_SIGNALS}
    assert by["ctx"].past_limit(0.9) and not by["ctx"].past_limit(0.89)
    assert by["ram"].past_limit(0.1) and not by["ram"].past_limit(0.11)
    assert by["net"].past_limit(0.0) and not by["net"].past_limit(1.0)


def test_signal_set_adds_generic_resources():
    s = signal_set(16)
    assert len(s) == 16 and s[:6] == CORE_SIGNALS
    assert s[6].name == "resource_0" and s[15].name == "resource_9"
    assert all((x.limit, x.bad_when, x.kind, x.dynamics) == (0.1, "low", "interrupt", "walk") for x in s[6:])
    with pytest.raises(ValueError):
        signal_set(5)
