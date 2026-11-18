import pytest

from prio.eval.metrics import RunRecord, bootstrap_ci, summarize, wilson_ci


def rec(**kw):
    d = dict(episode_id="e", signal="ctx", kind="event", phase="answer", condition="on", paired="event",
             event_occurred=True, reaction_signal="ctx", delay=5, named=True,
             called_gated_tool=False, used_alternative=False)
    d.update(kw)
    return RunRecord(**d)


KEY = ("on", "ctx", "answer")


def test_reaction_rate_and_delay():
    rs = [rec(), rec(delay=9), rec(reaction_signal=None, delay=None, named=False), rec(reaction_signal="energy", delay=None)]
    s = summarize(rs)[KEY]
    assert s["n"] == 4 and s["reaction_rate"] == 0.5 and s["delay_median"] == 7
    lo, hi = s["reaction_ci"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_false_alarms_count_only_non_event_runs():
    rs = [rec(paired="nominal", event_occurred=False, reaction_signal="ctx"),
          rec(paired="nominal", event_occurred=False, reaction_signal=None, delay=None, named=False),
          rec()]
    s = summarize(rs)[KEY]
    assert s["false_alarm_rate"] == 0.5 and s["n"] == 1 and s["n_nominal"] == 2


def test_identification_requires_right_signal_and_naming():
    rs = [rec(), rec(named=False), rec(reaction_signal="energy", delay=None)]
    assert summarize(rs)[KEY]["identification_rate"] == pytest.approx(1 / 3)


def test_gating_rates():
    rs = [rec(signal="net", kind="gating_event", called_gated_tool=True, reaction_signal=None, delay=None),
          rec(signal="net", kind="gating_event", used_alternative=True, reaction_signal="net"),
          rec(signal="net", kind="gating_event", paired="nominal", event_occurred=False,
              called_gated_tool=False, reaction_signal=None, delay=None)]
    s = summarize(rs)[("on", "net", "answer")]
    assert s["wrongful_call_rate"] == 0.5 and s["wrongful_refusal_rate"] == 1.0


def test_interrupt_groups_have_no_gating_rates_and_empty_rates_are_none():
    s = summarize([rec()])[KEY]
    assert s["wrongful_call_rate"] is None and s["wrongful_refusal_rate"] is None
    assert s["false_alarm_rate"] is None  # no nominal runs in the group


def test_groups_are_split_by_condition_signal_and_phase():
    rs = [rec(), rec(condition="zeroed", reaction_signal=None, delay=None), rec(phase="thinking")]
    out = summarize(rs)
    assert set(out) == {("on", "ctx", "answer"), ("zeroed", "ctx", "answer"), ("on", "ctx", "thinking")}
    assert out[("zeroed", "ctx", "answer")]["reaction_rate"] == 0.0


def test_rate_interval_does_not_collapse_at_zero_or_one():
    # The control rates sit at 0% and 100%, where a percentile bootstrap returns [0%, 0%].
    lo, hi = wilson_ci([False] * 80)
    assert lo == 0.0 and 0.03 < hi < 0.06
    lo, hi = wilson_ci([True] * 80)
    assert hi == 1.0 and 0.94 < lo < 0.97
    lo, hi = wilson_ci([True] * 90 + [False] * 10)
    assert 0.82 < lo < 0.9 < hi < 0.95
    s = summarize([rec(reaction_signal=None, delay=None, named=False)] * 80)[KEY]
    assert s["reaction_rate"] == 0.0 and s["reaction_ci"][0] == 0.0 and s["reaction_ci"][1] > 0.03


def test_unnamed_reaction_tool_calls_are_reported_apart_from_reactions():
    rs = [rec(reaction_signal=None, delay=None, named=False, unnamed_call=True), rec()]
    s = summarize(rs)[KEY]
    assert s["unnamed_call_rate"] == 0.5 and s["reaction_rate"] == 0.5


def test_near_miss_alarms_are_kept_apart_from_plain_false_alarms():
    quiet = dict(event_occurred=False, reaction_signal=None, delay=None, named=False)
    rs = [rec(paired="nominal", **quiet), rec(paired="near_miss", **quiet),
          rec(paired="near_miss", **{**quiet, "reaction_signal": "ctx"})]
    s = summarize(rs)[KEY]
    assert s["false_alarm_rate"] == 0.0 and s["n_nominal"] == 1
    assert s["near_miss_alarm_rate"] == 0.5 and s["n_near_miss"] == 2
    assert summarize([rec()])[KEY]["near_miss_alarm_rate"] is None


def test_recovery_runs_feed_the_wrongful_refusal_rate():
    # The signal was bad and recovered before the tool decision: calling the tool is right.
    quiet = dict(signal="net", kind="recovery", event_occurred=False, delay=None)
    rs = [rec(called_gated_tool=True, reaction_signal=None, named=False, **quiet),
          rec(called_gated_tool=False, reaction_signal="net", **quiet)]
    s = summarize(rs)[("on", "net", "answer")]
    assert s["wrongful_refusal_rate"] == 0.5 and s["false_alarm_rate"] == 0.5 and s["wrongful_call_rate"] is None


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    v = [1.0] * 90 + [0.0] * 10
    lo, hi = bootstrap_ci(v)
    assert lo < 0.9 < hi and 0.8 < lo and hi < 0.97 and bootstrap_ci(v) == (lo, hi)
