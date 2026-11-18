"""scripts/analysis/r12_read.py: event runs of an R12 arm without a forced prefix, decoded for reading by hand."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import r12_read as rr  # noqa: E402


def _rec(task, run_type="event", **more):
    base = {"task_id": task, "family": "math", "signal": "ctx", "run_type": run_type, "k": 3, "occurred": True,
            "why_not": None, "hit": True, "onset": 0, "premature": False, "reaction_signal": "ctx", "phase": "thinking"}
    return {**base, **more}


def _row(task, run_type="event", pre=(1, 2, 3), post=(4, 5), notice_text=None):
    return {"task_id": task, "run_type": run_type, "pre": list(pre), "post": list(post), "notice_text": notice_text,
            "stop_reason": "eos"}


def _decode(ids):
    return " ".join(f"t{i}" for i in ids)


def test_pick_takes_event_runs_only_at_even_spacing_over_the_file():
    records = [_rec(f"a{i}") for i in range(10)] + [_rec("twin", "twin"), _rec("off", "off")]
    assert [r["task_id"] for r in rr.pick(records, 3)] == ["a0", "a3", "a6"]
    assert len(rr.pick(records, 50)) == 10


def test_render_shows_the_tokens_before_the_crossing_the_mark_and_the_judged_part():
    text = rr.render(_rec("a0"), _row("a0"), _decode, tail=2, head=1)
    assert "a0 | math | ctx | k = 3 | crossing occurred | HIT | onset 0" in text
    assert "... t2 t3" in text and "t1" not in text.split("<<< token k >>>")[0].replace("a0", "")
    assert "<<< token k >>>" in text
    assert text.rstrip().endswith("t4 ...")


def test_render_prints_the_notice_between_the_two_parts_for_the_text_notice_model():
    text = rr.render(_rec("a0"), _row("a0", notice_text=" [status ctx=0.90]"), _decode, tail=5, head=5)
    before, after = text.split("<<< token k >>>")
    assert "[status ctx=0.90]" in after and "[status" not in before


def test_render_says_why_a_crossing_did_not_occur_and_marks_a_premature_reaction():
    text = rr.render(_rec("a0", occurred=False, why_not="thinking_over", hit=False, onset=None, premature=True),
                     _row("a0"), _decode, tail=5, head=5)
    assert "no crossing (thinking_over)" in text and "miss" in text and "PREMATURE" in text
