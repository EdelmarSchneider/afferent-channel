from helpers import FAKE_FMT, FAKE_MS, FAKE_TOK, event_plan, nominal_plan
from prio.bg.assemble import assemble, twin_plan
from prio.bg.score import BgRun, named_tasks, score_event, score_opening, score_resumption, score_twin
from prio.reactions import parse_tool_calls

ep_done_t2 = assemble(event_plan("done"), FAKE_FMT, FAKE_TOK, FAKE_MS)          # t1 photos backup, t2 report download
ep_twin = assemble(twin_plan(event_plan("done")), FAKE_FMT, FAKE_TOK, FAKE_MS)
ep_math_gold_42 = assemble(event_plan("done", phase="thinking", family="math"), FAKE_FMT, FAKE_TOK, FAKE_MS)
ep_explain = ep_done_t2
_opening_plan = nominal_plan(2)
_opening_plan.opening = True
ep_opening_two_tasks = assemble(_opening_plan, FAKE_FMT, FAKE_TOK, FAKE_MS)

DONE_CALLS = FAKE_FMT.calls_text([("unwatch", {"task_id": "t2"}), ("task_result", {"task_id": "t2"})])


def make_run(*pieces):
    """Alternating model text and harness payload lists -> BgRun (calls indexed within their segment)."""
    ids, segments, rounds = [], [], []
    for piece in pieces:
        start = len(ids)
        if isinstance(piece, str):
            seg = FAKE_FMT.encode(piece)
            rounds.append(parse_tool_calls(seg, FAKE_TOK, FAKE_MS))
            segments.append(("model", start, start + len(seg)))
        else:
            seg = FAKE_FMT.segment(piece)
            segments.append(("harness", start, start + len(seg)))
        ids += seg
    last_model = [p for p in pieces if isinstance(p, str)][-1]
    return BgRun(ids, segments, rounds, "eos" if last_model.rstrip().endswith("<eos>") else "budget")


def run_naming(label, task_id):
    return make_run(f"The {label} has finished , so I will fetch its result . "
                    + FAKE_FMT.calls_text([("unwatch", {"task_id": task_id}), ("task_result", {"task_id": task_id})])
                    + " <eos>", [{"ok": True}, {"result": "x"}], "back . <eos>")


def run_reporting(value):
    return make_run("The report download has finished . " + DONE_CALLS + " <eos>", [{"ok": True}, {"result": "r"}],
                    f"<think> </think> The report download is done : its total is {value} . Back . <eos>")


def run_answering(text):
    return make_run("The report download has finished . </think> " + DONE_CALLS + " <eos>",
                    [{"ok": True}, {"result": "r"}], "<think> Back to the problem . </think> " + text)


def test_named_tasks_in_order_of_first_mention():
    tasks = ep_done_t2.meta["tasks"]
    assert named_tasks("The photos backup and then the report download", tasks) == ["t1", "t2"]
    assert named_tasks("nothing here", tasks) == []


def test_named_reaction_with_right_commands_counts(tok, ms):
    run = make_run("The report download has finished , so I will fetch its result . " + DONE_CALLS + " <eos>",
                   [{"ok": True}, {"result": "report saved: total 4,210"}],
                   "<think> </think> The report download is done : its total is 4,210 . Now , back to your question . <eos>")
    r = score_event(run, ep_done_t2, "on", tok, ms, expected_task="t2")
    assert (r.reacted, r.right_task, r.commands_ok, r.report_ok) == (True, True, True, True)
    assert r.delay == 1 and r.event_kind == "done" and r.n_watched == 2


def test_naming_the_other_task_is_an_identification_miss(tok, ms):
    r = score_event(run_naming("photos backup", "t2"), ep_done_t2, "on", tok, ms, expected_task="t2")
    assert r.reacted and not r.right_task


def test_wrong_commands_are_not_ok(tok, ms):
    run = make_run("The report download has finished . " + FAKE_FMT.calls_text([("task_status", {"task_id": "t2"})])
                   + " <eos>", [{"state": "done"}], "ok . <eos>")
    assert score_event(run, ep_done_t2, "on", tok, ms, "t2").commands_ok is False


def test_command_without_sentence_is_an_unnamed_call(tok, ms):
    run = make_run(FAKE_FMT.calls_text([("unwatch", {"task_id": "t2"})]) + " <eos>", [{"ok": True}], "more . <eos>")
    r = score_event(run, ep_done_t2, "on", tok, ms, expected_task="t2")
    assert not r.reacted and r.unnamed_call


def test_reaction_after_the_window_is_a_miss(tok, ms):
    late = "word " * 60 + "The report download has finished . " + FAKE_FMT.calls_text([("unwatch", {"task_id": "t2"})])
    assert not score_event(make_run(late + " <eos>", [{"ok": True}], "x <eos>"), ep_done_t2, "on", tok, ms, "t2").reacted


def test_no_reaction_at_all(tok, ms):
    r = score_event(make_run("the tides rise and fall . <eos>"), ep_done_t2, "on", tok, ms, "t2")
    assert (r.reacted, r.unnamed_call, r.commands_ok) == (False, False, False)


def test_twin_alarm_on_any_reaction_tool(tok, ms):
    run = make_run("so far so good . " + FAKE_FMT.calls_text([("task_status", {"task_id": "t1"})]) + " <eos>",
                   [{"state": "running"}], "fine . <eos>")
    assert score_twin(run, ep_twin, "on", tok, ms).false_alarm


def test_twin_alarm_on_a_reaction_sentence(tok, ms):
    assert score_twin(make_run("The report download has finished . <eos>"), ep_twin, "on", tok, ms).false_alarm
    assert not score_twin(make_run("the moon pulls the water . <eos>"), ep_twin, "on", tok, ms).false_alarm


def test_report_accepts_number_without_separator(tok, ms):
    assert score_event(run_reporting("4210"), ep_done_t2, "on", tok, ms, "t2").report_ok
    assert not score_event(run_reporting("17"), ep_done_t2, "on", tok, ms, "t2").report_ok


def test_closing_line_must_name_the_tasks_still_running(tok, ms):
    good = make_run("The report download has finished . " + DONE_CALLS + " <eos>", [{"ok": True}, {"result": "r"}],
                    "<think> </think> done . The photos backup is still running ; I will tell you . <eos>")
    bad = make_run("The report download has finished . " + DONE_CALLS + " <eos>", [{"ok": True}, {"result": "r"}],
                   "<think> </think> done . <eos>")
    assert score_event(good, ep_done_t2, "on", tok, ms, "t2").closing_ok is True
    assert score_event(bad, ep_done_t2, "on", tok, ms, "t2").closing_ok is False


def test_math_resumption_uses_the_final_number(tok, ms):
    assert score_resumption(run_answering("so the answer is 42 . <eos>"), ep_math_gold_42, tok) == (None, True)
    assert score_resumption(run_answering("so the answer is 41 . <eos>"), ep_math_gold_42, tok) == (None, False)


def test_explain_resumption_rejects_a_restart(tok, ms):
    before = FAKE_TOK.decode(ep_explain.target_ids[ep_explain.meta["watch_end"]:ep_explain.t_cross + 1])
    run = make_run("The report download has finished . " + DONE_CALLS + " <eos>", [{"ok": True}, {"result": "r"}],
                   "<think> </think> " + before + " " + before + " <eos>")
    resumed, _ = score_resumption(run, ep_explain, tok)
    assert resumed is False


def test_explain_resumption_accepts_new_text(tok, ms):
    run = make_run("The report download has finished . " + DONE_CALLS + " <eos>", [{"ok": True}, {"result": "r"}],
                   "<think> </think> Back . a2 . a3 . a4 . <eos>")
    assert score_resumption(run, ep_explain, tok) == (True, None)


def test_opening_counts_started_and_watched(tok, ms):
    starts = FAKE_FMT.calls_text([("start_task", {"description": "back up the family photos"}),
                                  ("start_task", {"description": "download the quarterly report"})])
    run = make_run("<think> plan </think> " + starts + " <eos>", [{"task_id": "t1"}, {"task_id": "t2"}],
                   "<think> </think> " + FAKE_FMT.calls_text([("watch", {"task_id": "t1"})]) + " <eos>", [{"slot": 1}],
                   "now the question . <eos>")
    r = score_opening(run, ep_opening_two_tasks, tok, ms)
    assert (r.opening_started, r.opening_watched, r.kind) == (2, 1, "opening")


def test_reaction_needs_the_right_command_names(tok, ms):  # final review I6: naming + right commands
    run = make_run("The report download has finished . " + FAKE_FMT.calls_text([("task_status", {"task_id": "t2"})])
                   + " <eos>", [{"state": "done"}], "ok . <eos>")
    r = score_event(run, ep_done_t2, "on", tok, ms, "t2")
    assert r.reacted is False and r.reacted_loose is True


def test_correct_waiting_reaction_has_no_unnamed_call(tok, ms):  # final review I7
    ep = assemble(event_plan("waiting"), FAKE_FMT, FAKE_TOK, FAKE_MS)
    run = make_run("The report download is waiting for an answer , so I will see what it needs . "
                   + FAKE_FMT.calls_text([("task_status", {"task_id": "t2"})]) + " <eos>",
                   [{"state": "waiting", "question": "replace? yes / no"}],
                   "<think> </think> It asks whether to replace the existing files ; you said yes , so I will answer yes . "
                   + FAKE_FMT.calls_text([("answer_task", {"task_id": "t2", "answer": "yes"})]) + " <eos>", [{"ok": True}],
                   "<think> </think> The report download asked whether to replace some files ; I answered yes . back . <eos>")
    r = score_event(run, ep, "on", tok, ms, "t2")
    assert (r.reacted, r.commands_ok, r.unnamed_call) == (True, True, False)


def test_non_string_task_id_does_not_crash_scoring(tok, ms):  # final review I11
    run = make_run("The report download has finished . " + FAKE_FMT.calls_text([("unwatch", {"task_id": ["t2"]})])
                   + " <eos>", [{"error": "unknown task"}], "ok . <eos>")
    r = score_event(run, ep_done_t2, "on", tok, ms, "t2")
    assert r.commands_ok is False and r.closing_ok is not None
