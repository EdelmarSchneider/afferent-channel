"""scripts/analysis/preservation_compare.py: two trained models of the preservation check on the same items."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import preservation_compare as pc  # noqa: E402


def _item(task, family, condition, ok):
    return {"task_id": task, "family": family, "condition": condition, "correct": ok, "format": ok}


def test_paired_compares_the_trained_rows_of_two_folders_item_by_item():
    a = [_item("m1", "math", "on", True), _item("m2", "math", "on", True), _item("m3", "math", "on", False),
         _item("m4", "math", "on", False), _item("m1", "math", "base", False), _item("e1", "explain", "on", True)]
    b = [_item("m1", "math", "on", True), _item("m2", "math", "on", False), _item("m3", "math", "on", True),
         _item("m4", "math", "on", False), _item("m5", "math", "on", True), _item("e1", "explain", "on", False)]
    out = pc.paired(a, b, "math", "correct")
    assert out == {"n": 4, "both": 1, "only_a": 1, "only_b": 1, "neither": 1, "rate_a": 0.5, "rate_b": 0.5, "p": 1.0}
    out = pc.paired(a, b, "explain", "format")
    assert (out["n"], out["only_a"], out["only_b"]) == (1, 1, 0)


def test_base_rows_of_the_two_folders_must_agree():
    a = [_item("m1", "math", "base", True), _item("m2", "math", "base", False)]
    b = [_item("m1", "math", "base", True), _item("m2", "math", "base", True)]
    assert pc.base_disagreements(a, b, "math", "correct") == 1
