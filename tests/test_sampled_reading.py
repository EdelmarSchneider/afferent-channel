"""scripts/analysis/sampled_reading.py: the draws of one model decoded with sampling, pooled (R24b, A13)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import sampled_reading as sr  # noqa: E402


def _item(signal, naming, seen, kind="event", summary_tokens=None):
    item = {"signal": signal, "naming": naming, "naming_seen": seen, "kind": kind}
    if summary_tokens is not None:
        item["summary_tokens"] = summary_tokens
    return item


def _summary(n, hits, alarms, near):
    return {"groups": {"on|all interrupt|all": {"n": n, "reaction_rate": hits / n, "n_nominal": n, "false_alarm_rate": alarms / n,
                                                 "n_near_miss": n, "near_miss_alarm_rate": near / n}}}


DRAW_A = {"items": [_item("ctx", "A", True), _item("ctx", "B", False), _item("net", "N", True)]}
DRAW_B = {"items": [_item("ctx", "A", True), _item("ctx", "C", True), _item("net", "N", True), _item("net", "M", False)]}


def test_pooled_wording_counts_verbatim_reactions_and_distinct_sentences_per_draw_and_pooled():
    table = sr.pooled_wording([DRAW_A, DRAW_B])
    assert table["all"] == {"n": 7, "seen": 5, "distinct_per_draw": [3, 4], "distinct_pooled": 5, "top": ("A", 2)}
    assert table["ctx"] == {"n": 4, "seen": 3, "distinct_per_draw": [2, 2], "distinct_pooled": 3, "top": ("A", 2)}
    assert table["net"]["distinct_pooled"] == 2 and table["net"]["top"] == ("N", 2)


def test_pooled_wording_keeps_the_same_sentence_of_two_signals_apart():
    table = sr.pooled_wording([{"items": [_item("ctx", "X", True), _item("net", "X", True)]}])
    assert table["all"]["distinct_pooled"] == 2


def test_detection_gives_each_draw_the_pool_and_the_verdict_of_the_prediction():
    draws = [_summary(88, 84, 0, 1), _summary(88, 80, 2, 0), _summary(88, 85, 0, 0)]
    table = sr.detection(draws, greedy=_summary(88, 85, 0, 0))
    assert table["draws"][0] == {"n": 88, "hits": 84, "alarms": 0, "n_quiet": 88, "near": 1, "n_near": 88}
    assert table["pooled"] == {"n": 264, "hits": 249, "alarms": 2, "n_quiet": 264, "near": 1, "n_near": 264}
    assert table["greedy"]["hits"] == 85
    assert table["every_draw_reaction_ok"] is True  # 80 / 88 = 90.9%
    assert table["every_draw_alarms_ok"] is True  # 2 / 88 = 2.3%
    assert round(table["points_under_greedy"], 1) == 2.3  # 96.6 - 94.3
    assert table["held"] is True


def test_detection_fails_the_prediction_when_one_draw_is_under_ninety_percent():
    draws = [_summary(88, 84, 0, 0), _summary(88, 79, 0, 0), _summary(88, 85, 0, 0)]
    table = sr.detection(draws, greedy=_summary(88, 85, 0, 0))
    assert table["every_draw_reaction_ok"] is False and table["held"] is False


def test_detection_fails_the_prediction_when_the_pool_is_more_than_five_points_under_greedy():
    draws = [_summary(100, 90, 0, 0), _summary(100, 90, 0, 0), _summary(100, 91, 0, 0)]
    table = sr.detection(draws, greedy=_summary(100, 96, 0, 0))
    assert table["every_draw_reaction_ok"] is True
    assert round(table["points_under_greedy"], 1) == 5.7 and table["held"] is False


def test_summary_presence_pools_thinking_budget_reactions_by_kind():
    draws = [{"items": [_item("think", "T", True, "event", 20), _item("think", "T", True, "starts_past", 0), _item("ctx", "A", True)]},
             {"items": [_item("think", "U", True, "event", 7)]}]
    assert sr.summary_presence(draws) == {"all": (1, 3), "event": (1, 2), "starts_past": (0, 1)}


def test_report_states_both_predictions_with_their_verdicts():
    wording = sr.pooled_wording([DRAW_A, DRAW_B])
    det = sr.detection([_summary(88, 84, 0, 1), _summary(88, 80, 2, 0)], greedy=_summary(88, 85, 0, 0))
    text = sr.report("varied", ["d0", "d1"], wording, det, {"all": (1, 3)}, verbatim_line=0.70)
    assert "5 / 7 = 71.4%" in text and "prediction <= 70%: failed" in text
    assert "| draw 1 | 80 / 88 = 90.9% |" in text
    assert "S2" in text and "held" in text
