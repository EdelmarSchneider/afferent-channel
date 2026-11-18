import re

import numpy as np

from prio.bg.kinds import TRAIN_KINDS, sample_tasks
from prio.bg.sentences import (REACTIONS, bridge, closing_line, reaction_sentence, report_line, request_text,
                               waiting_answer_sentence)


def test_every_reaction_bank_has_six_wordings_naming_the_task():
    assert set(REACTIONS) == {"done", "error", "waiting", "stalled"}
    assert all(len(b) >= 6 and all("{task}" in s for s in b) for b in REACTIONS.values())


def test_reaction_sentence_names_the_task_once_with_an_article():
    t = sample_tasks(1, TRAIN_KINDS, np.random.default_rng(0), held_out=False)[0]
    s = reaction_sentence("done", t, np.random.default_rng(1))
    assert t.label() in s and "the the" not in s.lower()


def test_report_lines_carry_the_needed_fact():
    rng = np.random.default_rng(1)
    for seed in range(10):
        t = sample_tasks(1, TRAIN_KINDS, np.random.default_rng(seed), held_out=False)[0]
        if t.detail_key:
            assert t.detail_value in report_line("done", t, rng)
        assert t.reason in report_line("error", t, rng)
        assert "yes" in report_line("waiting", t, rng, answer="yes")


def test_waiting_answer_sentence_gives_the_answer():
    t = sample_tasks(1, TRAIN_KINDS, np.random.default_rng(6), held_out=False)[0]
    assert "no" in waiting_answer_sentence(t, "no", np.random.default_rng(7)).split()[-1] or \
        " no" in waiting_answer_sentence(t, "no", np.random.default_rng(7))


def test_closing_line_names_running_tasks_without_digits():
    tasks = sample_tasks(2, TRAIN_KINDS, np.random.default_rng(2), held_out=False)
    line = closing_line(tasks, np.random.default_rng(3))
    assert line.startswith("The ") and all(t.head in line for t in tasks) and not re.search(r"\d", line)
    one = closing_line(tasks[:1], np.random.default_rng(3))
    assert " is still running" in one and " are still running" in line


def test_bridges_have_no_digits():
    rng = np.random.default_rng(8)
    assert not any(re.search(r"\d", bridge(p, rng)) for p in ("thinking", "answer") for _ in range(20))


def test_request_states_tasks_ends_policy_and_question():
    tasks = sample_tasks(2, TRAIN_KINDS, np.random.default_rng(4), held_out=False)
    text = request_text(tasks, "How do tides work?", "no", np.random.default_rng(5))
    assert all(t.describe() in text.lower() and t.end in text for t in tasks)
    assert "answer no" in text and text.endswith("How do tides work?")
