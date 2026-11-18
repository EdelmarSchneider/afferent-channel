import pytest

from helpers import FAKE_FMT, event_plan, nominal_plan
from prio.bg.assemble import assemble, twin_plan
from prio.eval.run import extract_number
from prio.bg.protocol import slot_columns, state_of
from prio.reactions import parse_tool_calls


def test_nominal_episode_opens_with_start_then_watch_and_ends_with_closing(tok, ms):
    ep = assemble(nominal_plan(2), FAKE_FMT, tok, ms)
    calls = parse_tool_calls(ep.target_ids, tok, ms)
    assert [c.name for c in calls] == ["start_task", "start_task", "watch", "watch"]
    assert [c.arguments["task_id"] for c in calls[2:]] == ["t1", "t2"]
    assert tok.decode(ep.target_ids).rstrip().endswith("<eos>")
    assert ep.kind == "bg_nominal" and ep.t_cross is None and ep.t_diverge is None


def test_harness_spans_cover_exactly_the_harness_segments(tok, ms):
    ep = assemble(nominal_plan(3), FAKE_FMT, tok, ms)
    texts = [tok.decode(ep.target_ids[a:b]) for a, b in ep.harness_spans]
    assert len(texts) == 2 and all(t.startswith("<user>") and t.endswith("<assistant>") for t in texts)


def test_base_text_follows_the_opening_unchanged(tok, ms):
    plan = nominal_plan(1)
    ep = assemble(plan, FAKE_FMT, tok, ms)
    base = plan.run.target_ids[:-1]
    s = ep.meta["resume_start"]
    assert ep.target_ids[s:s + len(base)] == base


def test_slots_are_off_until_watched_then_running_in_order(tok, ms):
    ep = assemble(nominal_plan(2), FAKE_FMT, tok, ms)
    first = len(ep.prompt_ids) + ep.meta["watch_end"]
    states = lambda row: [state_of(row[slot_columns(k)[1]]) for k in (1, 2, 3)]
    assert states(ep.signals[first - 1]) == ["off", "off", "off"]
    assert states(ep.signals[first]) == ["running", "running", "off"]


def test_prompt_ids_do_not_depend_on_event(tok, ms):  # Review Focus 4
    plans = [nominal_plan(2, seed=7)] + [event_plan(kind, seed=7) for kind in ("done", "error", "waiting", "stalled")]
    assert len({tuple(assemble(p, FAKE_FMT, tok, ms).prompt_ids) for p in plans}) == 1


def test_closing_line_first_token_is_a_focus_span(tok, ms):
    ep = assemble(nominal_plan(1), FAKE_FMT, tok, ms)
    (a, b), = ep.focus_spans
    assert b - a == 1 and tok.decode(ep.target_ids[a:a + 4]).split()[0] == "The"


def test_meta_describes_tasks_slots_and_curves(tok, ms):
    ep = assemble(nominal_plan(2), FAKE_FMT, tok, ms)
    assert [(t["task_id"], t["slot"]) for t in ep.meta["tasks"]] == [("t1", 1), ("t2", 2)]
    assert all(t["curve"]["start"] == len(ep.prompt_ids) + ep.meta["watch_end"] for t in ep.meta["tasks"])
    assert ep.meta["event"] is None and ep.meta["family"] == "explain" and ep.meta["held_out"] is False


def test_done_event_mid_answer(tok, ms):
    plan = event_plan("done")
    ep = assemble(plan, FAKE_FMT, tok, ms)
    t = plan.tasks[1]
    after = ep.target_ids[ep.t_cross + 1:]
    calls = parse_tool_calls(after, tok, ms)
    assert [(c.name, c.arguments["task_id"]) for c in calls[:2]] == [("unwatch", "t2"), ("task_result", "t2")]
    assert t.head in tok.decode(after[:20])            # the sentence names the task
    assert t.detail_value is None or t.detail_value in tok.decode(ep.target_ids)
    row = ep.signals[len(ep.prompt_ids) + ep.t_cross]
    assert state_of(row[slot_columns(2)[1]]) == "done" and row[slot_columns(2)[0]] == pytest.approx(0.995)
    assert ep.kind == "bg_event" and ep.signal == "done" and ep.t_diverge == ep.t_cross + 1


def test_slot_turns_off_after_unwatch_is_answered(tok, ms):
    ep = assemble(event_plan("done"), FAKE_FMT, tok, ms)
    end_of_answer = len(ep.prompt_ids) + ep.harness_spans[2][1]   # opening has two segments
    assert state_of(ep.signals[end_of_answer][slot_columns(2)[1]]) == "off"
    assert state_of(ep.signals[end_of_answer - 1][slot_columns(2)[1]]) == "done"


def test_thinking_event_reports_at_the_start_of_the_answer(tok, ms):
    ep = assemble(event_plan("done", phase="thinking"), FAKE_FMT, tok, ms)
    a, b = ep.meta["report_span"]
    assert tok.decode(ep.target_ids[a - 1:a]) == "</think>"   # right after the last thinking close


def test_waiting_event_answers_the_policy_and_keeps_watching(tok, ms):
    plan = event_plan("waiting")
    ep = assemble(plan, FAKE_FMT, tok, ms)
    calls = parse_tool_calls(ep.target_ids[ep.t_cross + 1:], tok, ms)
    assert [c.name for c in calls] == ["task_status", "answer_task"]
    assert calls[1].arguments == {"task_id": "t2", "answer": plan.policy}
    assert "unwatch" not in [c.name for c in calls]
    back = len(ep.prompt_ids) + ep.harness_spans[3][1]
    assert state_of(ep.signals[back][slot_columns(2)[1]]) == "running"
    assert state_of(ep.signals[back - 1][slot_columns(2)[1]]) == "waiting"


def test_stalled_event_checks_and_keeps_watching(tok, ms):
    ep = assemble(event_plan("stalled"), FAKE_FMT, tok, ms)
    calls = parse_tool_calls(ep.target_ids, tok, ms)
    assert [c.name for c in calls][-1] == "task_status" and "unwatch" not in [c.name for c in calls]
    p, s = slot_columns(2)
    e = len(ep.prompt_ids) + ep.t_cross
    assert state_of(ep.signals[e][s]) == "stalled"
    assert all(b >= a for a, b in zip(ep.signals[e - 1:, p], ep.signals[e:, p]))  # never goes back


def test_error_event_stops_watching(tok, ms):
    ep = assemble(event_plan("error"), FAKE_FMT, tok, ms)
    calls = parse_tool_calls(ep.target_ids[ep.t_cross + 1:], tok, ms)
    assert [c.name for c in calls] == ["task_status", "unwatch"]
    assert "connection lost" in tok.decode(ep.target_ids)   # the report carries the reason


def test_twin_shares_the_prefix_and_parts_after_the_event(tok, ms):
    plan = event_plan("error")
    ev, tw = assemble(plan, FAKE_FMT, tok, ms), assemble(twin_plan(plan), FAKE_FMT, tok, ms)
    k = ev.t_cross
    assert tw.t_cross == k and tw.target_ids[:k + 1] == ev.target_ids[:k + 1]
    assert tw.target_ids[k + 1] != ev.target_ids[k + 1]
    assert state_of(tw.signals[len(tw.prompt_ids) + k][slot_columns(2)[1]]) == "running"
    assert tw.kind == "bg_twin" and tw.meta["twin_of"] == ev.episode_id
    n = len(ev.prompt_ids) + k
    assert (tw.signals[:n] == ev.signals[:n]).all()          # same channel up to the event row


def test_near_miss_twin_is_close_to_the_end_at_the_event(tok, ms):
    tw = assemble(twin_plan(event_plan("done"), near_miss=True), FAKE_FMT, tok, ms)
    p, s = slot_columns(2)
    e = len(tw.prompt_ids) + tw.t_cross
    assert 0.95 <= tw.signals[e][p] <= 0.99 and state_of(tw.signals[e][s]) == "running"


def test_resumption_continues_the_base_run(tok, ms):
    plan = event_plan("done")
    ep = assemble(plan, FAKE_FMT, tok, ms)
    rest = plan.run.target_ids[plan.event.cut + 1:-1]
    s = ep.meta["resume_start"]
    assert ep.target_ids[s:s + len(rest)] == rest


def test_closing_names_only_tasks_still_running(tok, ms):
    ep = assemble(event_plan("done", n_tasks=2, task_index=1), FAKE_FMT, tok, ms)
    tail = tok.decode(ep.target_ids[-20:])
    assert "photos" in tail and "report" not in tail.split("still running")[0].split(".")[-1]


def test_math_answer_keeps_its_final_number(tok, ms):
    plan = event_plan("done", phase="thinking", family="math")   # fake math run answers "... is 42 .", gold "42"
    ep = assemble(plan, FAKE_FMT, tok, ms)
    assert extract_number(tok.decode(ep.target_ids)) == 42.0


@pytest.mark.slow
def test_line_breaks_are_kept_around_the_reaction_on_the_real_tokenizer():
    from transformers import AutoTokenizer

    from prio.bg.assemble import BgEvent, BgPlan
    from prio.bg.protocol import QwenFormat
    from prio.datagen.build import BaseRun
    from prio.modelspec import resolve_spec
    from prio.tasks import Task
    from helpers import fixed_tasks

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    ms, fmt = resolve_spec("Qwen/Qwen3-1.7B", tok), QwenFormat(tok)
    text = "<think>\nPlan it.\n</think>\n\nThe facts:\n1. one fact here.\n2. second fact here.<|im_end|>"
    ids = tok(text, add_special_tokens=False).input_ids
    cut = next(i for i in range(len(ids)) if tok.decode(ids[i:i + 1]).endswith("\n") and "1." in tok.decode(ids[:i + 1])
               and "here" in tok.decode(ids[:i + 1]).split("1.")[-1])
    run = BaseRun(Task("b", "explain", "Explain.", None, "test"), tok("Explain.").input_ids, ids)
    plan = BgPlan("e", run, "test", fixed_tasks()[:1], "yes", BgEvent("done", 0, cut, "answer"))
    ep = assemble(plan, fmt, tok, ms)
    assert tok.decode(ep.target_ids[ep.t_cross + 1:ep.t_cross + 2]) == "The"     # no stray space after the line break
    assert tok.decode(ep.target_ids[:ep.meta["resume_start"]]).endswith("\n")   # the next line stays a new line
    assert "still running" not in tok.decode(ep.target_ids)                       # done task: no closing line here
    nominal = assemble(BgPlan("n", run, "test", fixed_tasks()[:1], "yes", None), fmt, tok, ms)
    assert "here.\n\nThe photos backup is still running" in tok.decode(nominal.target_ids)


def test_divergence_is_where_event_and_twin_really_part(tok, ms):  # final review I8
    from helpers import fake_run, fixed_tasks
    from prio.bg.assemble import BgEvent, BgPlan, align_divergence

    run = fake_run(answer="a1 . The x . y . z .")      # the base text goes on with "The", like most reactions
    cut = run.target_ids.index(tok.convert_tokens_to_ids("."), run.target_ids.index(ms.think_close))
    for seed in range(12):
        plan = BgPlan("e", run, "train", fixed_tasks()[:1], "yes", BgEvent("done", 0, cut, "answer"), seed=seed)
        ev, tw = assemble(plan, FAKE_FMT, tok, ms), assemble(twin_plan(plan), FAKE_FMT, tok, ms)
        align_divergence(ev, tw)
        d = ev.t_diverge
        assert d == tw.t_diverge and d >= ev.t_cross + 1
        assert ev.target_ids[:d] == tw.target_ids[:d] and ev.target_ids[d] != tw.target_ids[d]
