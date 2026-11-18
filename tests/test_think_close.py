"""scripts/analysis/think_close.py: where the thinking closes in thinking-budget runs (R24, post hoc)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import think_close as tc  # noqa: E402

CLOSE = 99
BUDGET = 7  # stands for the naming sentence


def _names(ids):
    return BUDGET in ids


def _rec(episode, kind, counted, paired="event"):
    return {"episode_id": episode, "signal": "think", "kind": kind, "condition": "on", "paired": paired,
            "event_occurred": True, "reaction_signal": "think" if counted else None}


def _gen(episode, ids, paired="event"):
    return {"episode_id": episode, "signal": "think", "condition": "on", "paired": paired, "token_ids": ids}


def test_close_of_gives_the_index_of_the_first_close_and_whether_the_span_before_it_names_the_budget():
    ids = [BUDGET] + [1] * 49 + [CLOSE, 1, CLOSE]
    assert tc.close_of(ids, CLOSE, _names, span=60) == (50, True)


def test_close_of_does_not_count_a_naming_sentence_further_back_than_the_span():
    ids = [BUDGET] + [1] * 70 + [CLOSE]
    assert tc.close_of(ids, CLOSE, _names, span=60) == (71, False)


def test_close_of_reports_a_thinking_that_never_closes():
    assert tc.close_of([BUDGET, 1, 1], CLOSE, _names, span=60) == (None, False)


def test_reading_sorts_the_runs_not_counted_into_late_named_closes_and_the_rest():
    records = [
        _rec("a", "event", True),
        _rec("b", "event", False),            # named, closes at 50: late
        _rec("c", "event", False),            # never closes
        _rec("d", "swap", False, "swapped"),  # closes at 50 without the name
        {**_rec("e", "event", False), "condition": "zeroed"},  # channel off: not read
        {**_rec("f", "event", False), "signal": "ctx"},        # another signal: not read
    ]
    generations = [
        _gen("a", [BUDGET, 1, CLOSE]),
        _gen("b", [BUDGET] + [1] * 49 + [CLOSE]),
        _gen("c", [BUDGET, 1, 1]),
        _gen("d", [1] * 50 + [CLOSE], "swapped"),
    ]
    table = tc.reading(records, generations, CLOSE, _names, window=48, span=60)
    assert table["event"] == {"n": 3, "counted": 1, "late": [("b", "event", 50)], "other": [("c", "event", None)]}
    assert table["swap"] == {"n": 1, "counted": 0, "late": [], "other": [("d", "swapped", 50)]}
    assert table["all"]["n"] == 4 and table["all"]["counted"] == 1 and len(table["all"]["late"]) == 1


def test_report_prints_the_counted_rate_and_the_rate_with_the_window_lifted_side_by_side():
    table = {"event": {"n": 22, "counted": 19, "late": [("b", "event", 50)], "other": []},
             "all": {"n": 22, "counted": 19, "late": [("b", "event", 50)], "other": []}}
    text = tc.report(table, "results/x", window=48)
    assert "| event | 22 | 19 / 22 = 86.4% | 1 | 20 / 22 = 90.9% | 0 |" in text
    assert "post hoc" in text
    assert "`b` (event): thinking closes at token 50" in text
