"""scripts/analysis/second_signal.py: the commands of two siblings of a family, and how far apart two
level vectors of a signal are."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import second_signal as ss  # noqa: E402


def _record(family, variant, commands, condition="on", cell="main"):
    return {"family": family, "variant": variant, "commands": commands, "condition": condition, "cell": cell}


RECORDS = [
    _record("f1", "thirst-water", ["drink_water"]), _record("f1", "thirst-dry", ["go_to"]),  # follows the signal
    _record("f2", "thirst-water", ["drink_water"]), _record("f2", "thirst-dry", ["drink_water", "go_to"]),  # the same one
    _record("f3", "thirst-water", ["go_to"]), _record("f3", "thirst-dry", ["drink_water"]),  # the wrong way round
    _record("f4", "thirst-water", ["drink_water"]), _record("f4", "thirst-dry", []),  # no command
    _record("f5", "thirst-water", ["go_to"], condition="off"), _record("f5", "thirst-dry", ["go_to"], cell="P1/S2-F1"),
    _record("f1", "hunger-food", ["eat"]),
]


def test_first_commands_counts_the_first_command_of_each_variant_in_the_main_cell_with_the_channel_on():
    out = ss.first_commands(RECORDS, ["thirst-water", "thirst-dry"])
    assert out == {"thirst-water": {"drink_water": 3, "go_to": 1}, "thirst-dry": {"go_to": 1, "drink_water": 2, "-": 1}}


def test_paired_sorts_the_families_by_the_commands_of_their_two_siblings():
    out = ss.paired(RECORDS, "thirst-water", "thirst-dry", "drink_water", "go_to")
    assert out == {"families": 4, "follows": 1, "both_with": 1, "both_without": 0, "inverted": 1, "other": 1}


def test_discordant_counts_the_families_where_only_one_sibling_takes_the_command():
    assert ss.discordant(RECORDS, "thirst-water", "thirst-dry", "drink_water") == (2, 1)  # f1 and f4; f3


def test_separation_gives_distance_cosine_and_the_drift_of_each_vector_from_its_start():
    start = np.zeros((2, 100, 4), dtype=np.float32)
    start[1, 0], start[1, 99] = [1, 0, 0, 0], [0, 1, 0, 0]
    table = start.copy()
    table[1, 99] = [0, 3, 0, 0]
    out = ss.separation(table, start, 1, 0, 99)
    assert out["distance"] == pytest.approx(10 ** 0.5)
    assert out["cosine"] == pytest.approx(0.0)
    assert out["start_distance"] == pytest.approx(2 ** 0.5)
    assert out["drift"] == pytest.approx([0.0, 2.0])
    assert out["norms"] == pytest.approx([1.0, 3.0])
