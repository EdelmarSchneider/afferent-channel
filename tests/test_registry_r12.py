"""The index of runs and evaluations lists the R12 arms without a forced prefix (prio.registry):
their folders hold a `stream_summary.json`, not the `summary.json` of a standard evaluation."""

import json

from prio.registry import build_index, to_markdown, write_index
from test_registry import _tree, _write


def _stream(root, folder, arm, model, set_name=None, hits=5, occurred=6):
    event = {"planned": 6, "occurred": occurred, "hits": hits, "not_occurred": {"ended": 6 - occurred, "thinking_over": 0, "no_crossing": 0},
             "reaction_rate": hits / occurred, "reaction_ci": [0.4, 0.97], "premature": 1, "crossing_off_k": 0,
             "by_signal": {"ctx": {"planned": 6, "occurred": occurred, "hits": hits}}, "onset": {"b": {"0": hits}}}
    summary = {"arm": arm, "model": model, "event": event,
               "twin": {"n": 6, "false_alarms": 1, "false_alarm_rate": 1 / 6, "false_alarm_ci": [0.03, 0.56], "same_prefix": 6, "same_prefix_of": 6},
               "off": {"n": 6, "false_alarms": 0, "false_alarm_rate": 0.0, "false_alarm_ci": [0.0, 0.39]}}
    _write(root / "results" / folder / "stream_summary.json", summary)
    if set_name is not None:
        _write(root / "results" / folder / "eval_meta.json", {"invocations": [
            {"run": "runs/n6-wide-levels-s0", "set": set_name, "arm": arm, "args": {"batch_size": 16}, "started": "t1", "finished": "t2"}]})


def test_r12_stream_folders_get_their_own_table_and_disturb_no_other_row(tmp_path):
    _tree(tmp_path)
    before = build_index(tmp_path)
    assert before["r12_stream"] == []
    text_before = to_markdown({**before, "generated": "X"})
    assert "R12" not in text_before  # without such folders the index reads as it always did

    _stream(tmp_path, "r12p-free-wide-levels-s0-b16", "free", "channel", "pilot")
    _stream(tmp_path, "r12-Ls-notice-none-s0-b16", "Ls", "notice", "confirm", hits=3, occurred=4)
    _stream(tmp_path, "r12p-live-wide-levels-s0-b1", "live", "channel")  # no eval_meta.json: still listed
    _write(tmp_path / "results" / "r12p" / "smoke.json", {"passed": True})
    _write(tmp_path / "results" / "q" / "stream_summary.json", {"arm": "free"})  # a standard evaluation stays one
    after = build_index(tmp_path)
    for key in ("runs", "evaluations", "preservation"):
        assert after[key] == before[key]
    assert after["other_results"] == before["other_results"] + ["r12p/"]  # the stream folders are read, not "other files"

    rows = {e["folder"]: e for e in after["r12_stream"]}
    assert list(rows) == ["r12-Ls-notice-none-s0-b16", "r12p-free-wide-levels-s0-b16", "r12p-live-wide-levels-s0-b1"]
    pilot = rows["r12p-free-wide-levels-s0-b16"]
    assert (pilot["set"], pilot["arm"], pilot["model"], pilot["run"], pilot["finished"], pilot["batch_size"]) == (
        "pilot", "free", "channel", "runs/n6-wide-levels-s0", "t2", 16)
    assert pilot["event"] == {"planned": 6, "occurred": 6, "hits": 5, "reaction_rate": 5 / 6, "reaction_ci": [0.4, 0.97],
                              "premature": 1, "crossing_off_k": 0, "not_occurred": {"ended": 0, "thinking_over": 0, "no_crossing": 0}}
    assert pilot["twin"] == {"n": 6, "false_alarms": 1, "false_alarm_rate": 1 / 6} and pilot["off"]["false_alarms"] == 0
    assert rows["r12-Ls-notice-none-s0-b16"]["set"] == "confirm"
    bare = rows["r12p-live-wide-levels-s0-b1"]
    assert (bare["set"], bare["run"], bare["finished"], bare["batch_size"]) == ("pilot", None, None, None)

    text = to_markdown({**after, "generated": "X"})
    assert "## R12 arms without a forced prefix (free, live, Ls)" in text and "never reported as results" in text
    assert ("| r12p-free-wide-levels-s0-b16 | pilot, never reported | free | channel | `runs/n6-wide-levels-s0` | t2 | 6 | 6 | 5 | "
            "83.3% [40.0, 97.0] | 1 | 1 / 6 | 0 / 6 | 0 |") in text
    assert "| r12-Ls-notice-none-s0-b16 | confirmatory | Ls | notice |" in text and "| 4 | 3 | 75.0% [40.0, 97.0] |" in text
    rest = iter(text.split("\n"))
    assert all(line in rest for line in text_before.split("\n"))  # every line of the old index, in order
    json.dumps(after)

    written = write_index(tmp_path)
    assert written["r12_stream"] == after["r12_stream"]
    assert "R12 arms without a forced prefix" in (tmp_path / "results" / "index.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "results" / "index.json").read_text(encoding="utf-8"))["r12_stream"][0]["folder"] == "r12-Ls-notice-none-s0-b16"
