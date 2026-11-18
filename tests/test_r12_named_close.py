"""scripts/analysis/r12_named_close.py: runs whose thinking closes right after text that names the deadline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import r12_named_close as nc  # noqa: E402

CLOSE = 99
DEADLINE = 7  # stands for a deadline keyword


def _names(ids):
    return DEADLINE in ids


def _rec(task, run_type, k=10, hit=False, false_alarm=None, occurred=True):
    return {"task_id": task, "run_type": run_type, "k": k, "hit": hit, "false_alarm": false_alarm, "occurred": occurred}


def _row(task, run_type, tokens, k=10):
    return {"task_id": task, "run_type": run_type, "pre": tokens[:k], "post": tokens[k:]}


def test_named_close_gives_the_index_of_the_close_when_the_window_before_it_names_the_deadline():
    assert nc.named_close([1, DEADLINE, 1, CLOSE, 1], CLOSE, _names, window=47) == 3


def test_named_close_ignores_a_deadline_named_further_back_than_the_window_or_after_the_close():
    assert nc.named_close([DEADLINE] + [1] * 50 + [CLOSE], CLOSE, _names, window=47) is None
    assert nc.named_close([1, CLOSE, DEADLINE], CLOSE, _names, window=47) is None
    assert nc.named_close([1, DEADLINE, 1], CLOSE, _names, window=47) is None


def test_table_counts_event_runs_only_when_the_close_falls_before_token_k_and_says_how_many_are_hits():
    early = [1, DEADLINE, CLOSE] + [1] * 20  # close at 2, k = 10: before the crossing
    late = [1] * 12 + [DEADLINE, CLOSE, 1]   # close at 13: the reaction itself, after the crossing
    records = [_rec("a", "event", hit=True), _rec("b", "event", hit=False), _rec("c", "event", hit=True)]
    rows = [_row("a", "event", early), _row("b", "event", early), _row("c", "event", late)]
    table = nc.table(records, rows, CLOSE, _names)
    assert table["event"] == {"runs": 3, "named_close": 2, "of_them_hits": 1, "tasks": ["a", "b"]}


def test_table_counts_twin_and_off_runs_anywhere_and_says_how_many_are_already_false_alarms():
    anywhere = [1] * 12 + [DEADLINE, CLOSE, 1]
    records = [_rec("a", "twin", false_alarm=True), _rec("b", "twin", false_alarm=False), _rec("c", "off", false_alarm=False)]
    rows = [_row("a", "twin", anywhere), _row("b", "twin", anywhere), _row("c", "off", [1, 1, CLOSE, 1])]
    table = nc.table(records, rows, CLOSE, _names)
    assert table["twin"] == {"runs": 2, "named_close": 2, "of_them_false_alarms": 1, "tasks": ["a", "b"]}
    assert table["off"] == {"runs": 1, "named_close": 0, "of_them_false_alarms": 0, "tasks": []}


def test_report_labels_the_count_as_a_keyword_match_and_not_a_counted_reaction():
    tables = {"results/x": {"event": {"runs": 48, "named_close": 1, "of_them_hits": 1, "tasks": ["a"]},
                            "twin": {"runs": 48, "named_close": 2, "of_them_false_alarms": 1, "tasks": ["a", "b"]},
                            "off": {"runs": 48, "named_close": 0, "of_them_false_alarms": 0, "tasks": []}}}
    text = nc.report(tables)
    assert "keyword match only, not a counted reaction" in text
    assert "| results/x | 1 / 48 | 1 | 2 / 48 | 1 | 0 / 48 | 0 |" in text
