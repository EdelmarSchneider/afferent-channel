"""scripts/analysis/spoof_seeds.py: the forged-status cells of one model across its seeds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import spoof_seeds as sd  # noqa: E402


def _cell(k, n, passed, control=False):
    return {"reactions": k, "n": n, "pass": passed, "is_control": control}


SEED_A = {"trigger": {"on|P1/F1": _cell(5, 145, False), "on|P1/none": _cell(0, 145, None, True), "base|P1/F1": _cell(7, 145, False)},
          "suppress": {"on|P1/S2-F3": _cell(86, 88, True)}}
SEED_B = {"trigger": {"on|P1/F1": _cell(4, 145, True), "on|P3/F3": _cell(3, 73, False)},
          "suppress": {"on|P1/S2-F3": _cell(85, 88, True)}}


def test_rows_gives_every_cell_of_the_condition_with_one_entry_per_seed_and_none_where_a_seed_lacks_it():
    rows = sd.rows([SEED_A, SEED_B])
    assert rows == [
        {"arm": "trigger", "cell": "P1/F1", "seeds": [(5, 145, False), (4, 145, True)]},
        {"arm": "trigger", "cell": "P1/none", "seeds": [(0, 145, None), None]},
        {"arm": "trigger", "cell": "P3/F3", "seeds": [None, (3, 73, False)]},
        {"arm": "suppress", "cell": "P1/S2-F3", "seeds": [(86, 88, True), (85, 88, True)]},
    ]


def test_merged_puts_the_cells_of_several_folders_of_one_seed_into_one_summary():
    own_turn = {"trigger": {"on|P5/F3": _cell(0, 88, True)}, "suppress": {}}
    merged = sd.merged([SEED_A, own_turn])
    assert set(merged["trigger"]) == {"on|P1/F1", "on|P1/none", "base|P1/F1", "on|P5/F3"}
    assert merged["suppress"] == SEED_A["suppress"]


def test_shown_writes_count_rate_and_marks_a_cell_that_misses_its_line():
    assert sd.shown((3, 73, False)) == "3 / 73 = 4.1% **NO**"
    assert sd.shown((4, 145, True)) == "4 / 145 = 2.8%"
    assert sd.shown((0, 145, None)) == "0 / 145 = 0.0%"
    assert sd.shown(None) == "-"


def test_by_signal_counts_trigger_hits_of_one_cell_per_claimed_signal_with_the_channel_on():
    def record(cell, signal, reaction, condition="on", arm="trigger"):
        return {"arm": arm, "cell": cell, "condition": condition, "spoof_signal": signal, "reaction_signal": reaction}
    records = [record("P3/F3", "net", "net"), record("P3/F3", "net", None), record("P3/F3", "ctx", None),
               record("P3/F3", "net", "net", condition="base"), record("P1/F3", "net", "net"),
               record("P3/F3", "net", "net", arm="suppress")]
    assert sd.by_signal(records, "P3/F3") == {"ctx": (0, 1), "net": (1, 2)}
