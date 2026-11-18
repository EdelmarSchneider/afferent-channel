import numpy as np

from helpers import FAKE_FMT, event_plan, fixed_tasks
from prio.bg.assemble import assemble, twin_plan
from prio.bg.curves import sample_curve
from prio.bg.harness import SimHarness
from prio.bg.protocol import slot_columns, state_of
from prio.reactions import ToolCall

tasks = fixed_tasks()[:2]
curves = {t.task_id: sample_curve(np.random.default_rng(i), start=21, end_hint=600) for i, t in enumerate(tasks)}


def call(name, **args):
    return ToolCall(name, args, 0)


def harness_with_event(kind, task_id, position):
    h = SimHarness(tasks, curves, "yes", {"task_id": task_id, "kind": kind, "position": position})
    h.respond([call("watch", task_id="t1"), call("watch", task_id="t2")], 20)
    return h


def test_start_and_watch_follow_the_order():
    h = SimHarness(tasks, curves, "yes", None)
    assert h.respond([call("start_task", description="x"), call("start_task", description="y")], 10) == [
        {"task_id": "t1"}, {"task_id": "t2"}]
    assert h.respond([call("watch", task_id="t2"), call("watch", task_id="t1")], 20) == [{"slot": 1}, {"slot": 2}]


def test_event_row_and_unwatch():
    h = harness_with_event("done", task_id="t1", position=100)
    assert state_of(h.row(99)[1]) == "running" and state_of(h.row(100)[1]) == "done"
    assert h.respond([call("task_result", task_id="t1")], 120) == [{"result": tasks[0].result}]
    h.respond([call("unwatch", task_id="t1")], 130)
    assert state_of(h.row(130)[1]) == "done" and state_of(h.row(131)[1]) == "off"


def test_waiting_question_then_answer_resumes_running():
    h = harness_with_event("waiting", task_id="t1", position=100)
    assert h.respond([call("task_status", task_id="t1")], 110)[0]["question"].endswith("yes / no")
    h.respond([call("answer_task", task_id="t1", answer="yes")], 140)
    assert state_of(h.row(140)[1]) == "waiting" and state_of(h.row(141)[1]) == "running"


def test_status_before_the_event_reports_running():
    h = harness_with_event("error", task_id="t2", position=300)
    assert h.respond([call("task_status", task_id="t2")], 50) == [{"state": "running"}]


def test_unknown_task_and_tool_get_error_payloads():  # Review Focus 5
    h = SimHarness(tasks, curves, "yes", None)
    assert h.respond([call("unwatch", task_id="t9"), call("format_disk")], 5) == [
        {"error": "unknown task"}, {"error": "unknown tool"}]


def test_pending_changes_take_effect_after_the_segment():
    h = harness_with_event("done", task_id="t1", position=100)
    payloads = h.respond([call("unwatch", task_id="t1")])        # length of the segment not known yet
    assert payloads == [{"ok": True}]
    h.commit(150)
    assert state_of(h.row(150)[1]) == "done" and state_of(h.row(151)[1]) == "off"


def test_from_episode_rebuilds_rows_and_swaps_slots(tok, ms):
    ep = assemble(event_plan("done", n_tasks=2, task_index=1), FAKE_FMT, tok, ms)
    h = SimHarness.from_episode(ep)
    p = len(ep.prompt_ids) + ep.t_cross
    np.testing.assert_allclose(np.stack([h.row(i) for i in range(p + 1)]), ep.signals[:p + 1])
    s = SimHarness.from_episode(ep, swap_slot=1)
    assert state_of(s.row(p)[slot_columns(1)[1]]) == "done" and state_of(s.row(p)[slot_columns(2)[1]]) == "running"


def test_from_episode_follows_the_assembled_reaction(tok, ms):
    for kind in ("done", "error", "waiting", "stalled"):
        ep = assemble(event_plan(kind), FAKE_FMT, tok, ms)
        h = SimHarness.from_episode(ep)
        n = len(ep.prompt_ids)
        for (a, b), end in zip(ep.harness_spans[2:], ep.meta["answer_ends"]):
            from prio.reactions import parse_tool_calls
            calls = parse_tool_calls(ep.target_ids[:a], tok, ms)
            round_calls = [c for c in calls if c.token_index >= (ep.t_cross if end == ep.meta["answer_ends"][0]
                                                                   else ep.harness_spans[2][1])]
            h.respond(round_calls, n + b - 1)
        np.testing.assert_allclose(np.stack([h.row(i) for i in range(len(ep.signals))]), ep.signals)


def test_twin_rows_match_including_near_miss(tok, ms):
    tw = assemble(twin_plan(event_plan("done"), near_miss=True), FAKE_FMT, tok, ms)
    h = SimHarness.from_episode(tw)
    np.testing.assert_allclose(np.stack([h.row(i) for i in range(len(tw.signals))]), tw.signals)


def test_opening_probe_ids_start_at_one(tok, ms):  # final review C4
    from helpers import nominal_plan

    plan = nominal_plan(2)
    plan.opening = True
    h = SimHarness.from_episode(assemble(plan, FAKE_FMT, tok, ms))
    assert h.respond([call("start_task", description="a"), call("start_task", description="b")], 5) == [
        {"task_id": "t1"}, {"task_id": "t2"}]


def test_non_string_task_id_gets_an_error_payload():  # final review I11
    h = SimHarness(tasks, curves, "yes", None)
    assert h.respond([ToolCall("unwatch", {"task_id": ["t1"]}, 0)], 5) == [{"error": "unknown task"}]
