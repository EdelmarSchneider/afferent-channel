import itertools
import json
import re

import numpy as np

from prio.bg.curves import progress_at, sample_curve
from prio.bg.kinds import HELD_OUT_KINDS, TRAIN_KINDS, sample_tasks


def test_kind_counts_and_held_out_kinds_differ_in_verb_and_object():
    assert len(TRAIN_KINDS) == 8 and len(HELD_OUT_KINDS) == 4
    assert not {k.verb for k in TRAIN_KINDS} & {k.verb for k in HELD_OUT_KINDS}
    heads = [h for k in TRAIN_KINDS + HELD_OUT_KINDS for _, h in k.objects]
    assert len(heads) == len(set(heads))
    assert all(len(k.objects) >= 4 for k in TRAIN_KINDS + HELD_OUT_KINDS)


def test_no_digits_in_names_objects_or_end_instructions():
    text = " ".join(f"{k.verb} {k.noun} {p} {h} {e}" for k in TRAIN_KINDS + HELD_OUT_KINDS
                    for p, h in k.objects for e, _ in k.ends)
    assert not re.search(r"\d", text)


def test_sample_tasks_are_distinct_and_in_order():
    tasks = sample_tasks(3, TRAIN_KINDS, np.random.default_rng(0), held_out=False)
    assert [t.task_id for t in tasks] == ["t1", "t2", "t3"]
    assert len({t.kind for t in tasks}) == 3 and len({t.head for t in tasks}) == 3
    assert all(t.detail_value in t.result for t in tasks if t.detail_key)


def test_task_label_and_description():
    t = sample_tasks(1, [k for k in TRAIN_KINDS if k.name == "download"], np.random.default_rng(0), held_out=False)[0]
    assert t.describe() == "download " + t.phrase and t.label() == t.head + " download"
    assert t.question.endswith("yes / no")


def test_progress_is_snapped_monotone_and_below_cap():
    c = sample_curve(np.random.default_rng(1), start=10, end_hint=900)
    values = [progress_at(c, p) for p in range(901)]
    assert values[:10] == [0.0] * 10
    assert all(b >= a for a, b in zip(values, values[1:])) and max(values) <= 0.945
    assert all(abs(v * 100 - 0.5 - round(v * 100 - 0.5)) < 1e-6 for v in values[10:] if v > 0)


def test_near_miss_curve_ends_between_95_and_99():
    c = sample_curve(np.random.default_rng(2), start=0, end_hint=500, near_miss=True)
    assert 0.95 <= progress_at(c, 500) <= 0.99


def test_stepped_curve_has_flat_stretches():
    c = sample_curve(np.random.default_rng(3), start=0, end_hint=800, shape="stepped")
    values = [progress_at(c, p) for p in range(801)]
    assert max(len(list(g)) for _, g in itertools.groupby(values[50:])) >= 30


def test_curve_survives_json():
    c = sample_curve(np.random.default_rng(4), start=5, end_hint=300)
    assert [progress_at(json.loads(json.dumps(c)), p) for p in range(300)] == [progress_at(c, p) for p in range(300)]


def test_same_position_maps_to_different_progress():
    values = {progress_at(sample_curve(np.random.default_rng(s), start=0, end_hint=800), 400) for s in range(20)}
    assert len(values) >= 8
