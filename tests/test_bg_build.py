import json

import numpy as np

from helpers import FAKE_FMT, FAKE_MS, FAKE_TOK, fake_run, fake_runs
from prio.bg.assemble import assemble
from prio.bg.build import cut_candidates, plan_split, sentence_ends
from prio.datagen.episodes import read_jsonl


def test_sentence_ends_find_periods_but_not_the_end_of_turn(tok, ms):
    run = fake_run()
    ends = sentence_ends(run.target_ids, tok)
    assert ends and all(tok.decode(run.target_ids[i:i + 1]) == "." for i in ends)


def test_cut_candidates_respect_margins(tok, ms):  # Review Focus 3
    run = fake_run(thinking=" ".join(f"w{i} ." for i in range(60)), answer=" ".join(f"a{i} ." for i in range(60)))
    cuts = cut_candidates(run, tok, ms, "answer", min_before=10, min_after=20)
    close = run.target_ids.index(ms.think_close)
    assert cuts and all(c - close > 10 and len(run.target_ids) - 1 - c >= 20 for c in cuts)
    thinking = cut_candidates(run, tok, ms, "thinking", min_before=10, min_after=20)
    assert thinking and all(c < close for c in thinking)


def test_math_answer_cuts_come_before_the_last_number(tok, ms):
    run = fake_run(family="math", answer="so we add . the total is 42 .")
    cuts = cut_candidates(run, tok, ms, "answer", 1, 1)
    assert cuts and all(c < run.target_ids.index(tok.convert_tokens_to_ids("42")) for c in cuts)


def test_train_plan_mix(tok, ms):
    plans = plan_split(fake_runs(60), "train", np.random.default_rng(0), tok, ms, n_episodes=200,
                       min_before=1, min_after=1)
    events = [p for p in plans if p.event and not p.twin]
    assert {p.episode_id + "-twin" for p in events} == {p.episode_id for p in plans if p.twin}
    assert 0.08 <= sum(p.event is None for p in plans) / len(plans) <= 0.22
    assert not any(t.held_out for p in plans for t in p.tasks)
    assert all(sum(len(p.tasks) == n for p in plans) / len(plans) >= 0.2 for n in (1, 2, 3))
    assert max(sum(p.run.task.task_id == r for p in plans if not p.twin) for r in {p.run.task.task_id for p in plans}) <= 2


def test_test_plan_balances_events_and_holds_out_half(tok, ms):
    plans = plan_split(fake_runs(60), "test", np.random.default_rng(1), tok, ms, n_events=80, n_openings=10,
                       min_before=1, min_after=1)
    events = [p for p in plans if p.event and not p.twin]
    assert 0.4 <= sum(p.tasks[0].held_out for p in events) / len(events) <= 0.6
    assert all(all(t.held_out == p.tasks[0].held_out for t in p.tasks) for p in events)
    assert all(0.17 <= sum(p.event.kind == k for p in events) / len(events) <= 0.33
               for k in ("done", "error", "waiting", "stalled"))
    assert sum(p.opening for p in plans) == 10


def test_opening_probe_is_prompt_only(tok, ms):
    plan = next(p for p in plan_split(fake_runs(6), "test", np.random.default_rng(2), tok, ms, n_events=2,
                                      n_openings=2, min_before=1, min_after=1) if p.opening)
    ep = assemble(plan, FAKE_FMT, tok, ms)
    assert ep.kind == "bg_opening" and ep.target_ids == [] and len(ep.signals) == len(ep.prompt_ids)
    assert [t["task_id"] for t in ep.meta["tasks"]] == [t.task_id for t in plan.tasks]


def test_build_main_writes_files_and_manifest(tmp_path, monkeypatch):  # Review Focus 2
    import prio.bg.build as build

    monkeypatch.setattr(build, "_load_tokenizer", lambda model: (FAKE_TOK, FAKE_MS))
    monkeypatch.setattr(build, "_make_format", lambda tok: FAKE_FMT)
    from dataclasses import replace

    from prio.datagen.build import BaseRun

    runs = fake_runs(30) + [BaseRun(replace(r.task, split="test"), r.prompt_ids, r.target_ids)
                            for r in (fake_run(task_id=f"x{i}") for i in range(30))]
    monkeypatch.setattr(build, "_load_runs", lambda args, tok, ms: runs)
    build.main(["--base", "unused", "--model", "fake", "--out", str(tmp_path), "--train", "40", "--test-events", "8",
                "--test-openings", "2", "--min-before", "1", "--min-after", "1", "--max-len", "140"])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {"episodes", "event_kinds", "phases", "watched", "held_out_events", "dropped_long"} <= set(manifest)
    assert all(len(e.prompt_ids) + len(e.target_ids) <= 140 for e in read_jsonl(tmp_path / "train.jsonl"))
    assert manifest["dropped_long"]["train"] > 0          # long episodes are counted, not written
    assert manifest["episodes"]["test"] > 0


def test_events_are_balanced_between_thinking_and_answer(tok, ms):
    long_answer = " ".join(f"a{i} ." for i in range(8))
    runs = [fake_run(answer=long_answer, task_id=f"l{i}") if i % 3 == 0 else fake_run(answer="short .", task_id=f"s{i}")
            for i in range(90)]
    plans = plan_split(runs, "train", np.random.default_rng(3), tok, ms, n_episodes=120, min_before=1, min_after=2)
    events = [p for p in plans if p.event and not p.twin]
    assert 0.4 <= sum(p.event.phase == "answer" for p in events) / len(events) <= 0.6


def test_held_out_kinds_are_spread_over_every_event_kind(tok, ms):  # final review C3
    plans = plan_split(fake_runs(80), "test", np.random.default_rng(4), tok, ms, n_events=80, min_before=1, min_after=1)
    events = [p for p in plans if p.event and not p.twin]
    for kind in ("done", "error", "waiting", "stalled"):
        of_kind = [p for p in events if p.event.kind == kind]
        assert 0.3 <= sum(p.tasks[0].held_out for p in of_kind) / len(of_kind) <= 0.7


def test_base_runs_with_tool_calls_are_not_used(tok, ms):  # final review I5
    from prio.bg.build import usable_runs

    clean = fake_run(task_id="c")
    calling = fake_run(answer="<tool_call> {} </tool_call>", task_id="w")
    other = fake_run(task_id="h")
    object.__setattr__(other, "task", __import__("dataclasses").replace(other.task, family="web"))
    assert [r.task.task_id for r in usable_runs([clean, calling, other], ms)] == ["c"]


def _split_pair(slot_twins: bool):
    rng = np.random.default_rng(5)
    runs = fake_runs(60)
    train = plan_split(runs, "train", rng, FAKE_TOK, FAKE_MS, n_episodes=200, min_before=1, min_after=1,
                       slot_twins=slot_twins)
    test = plan_split(runs, "test", rng, FAKE_TOK, FAKE_MS, n_events=40, min_before=1, min_after=1)
    return train, test


def test_slot_twins_move_the_event_to_another_watched_task(tok, ms):
    train, _ = _split_pair(True)
    by_id = {p.episode_id: p for p in train}
    slots = [p for p in train if p.slot_of]
    events = [p for p in train if p.event and not p.twin and not p.slot_of]
    assert {p.episode_id + "-slot" for p in events if len(p.tasks) > 1} == {p.episode_id for p in slots}
    for s in slots:
        o = by_id[s.slot_of]
        assert (s.run, s.tasks, s.policy, s.seed, s.twin) == (o.run, o.tasks, o.policy, o.seed, False)
        assert (s.event.kind, s.event.cut, s.event.phase) == (o.event.kind, o.event.cut, o.event.phase)
        assert s.event.task_index != o.event.task_index and 0 <= s.event.task_index < len(s.tasks)


def test_slot_twins_leave_every_other_plan_and_the_test_split_unchanged(tok, ms):
    key = lambda p: (p.episode_id, p.run.task.task_id, p.event and (p.event.kind, p.event.task_index, p.event.cut),
                     p.seed, p.twin, p.near_miss)
    (train0, test0), (train1, test1) = _split_pair(False), _split_pair(True)
    assert [key(p) for p in train0] == [key(p) for p in train1 if not p.slot_of]
    assert [key(p) for p in test0] == [key(p) for p in test1] and not any(p.slot_of for p in test1)


def test_slot_twin_episode_shares_text_and_channel_up_to_the_event(tok, ms):
    from prio.bg.protocol import STATE_VALUE, slot_columns

    train, _ = _split_pair(True)
    by_id = {p.episode_id: p for p in train}
    s = next(p for p in train if p.slot_of)
    a, b = assemble(by_id[s.slot_of], FAKE_FMT, tok, ms), assemble(s, FAKE_FMT, tok, ms)
    assert b.kind == "bg_event" and b.meta["slot_of"] == a.episode_id and a.meta["slot_of"] is None
    assert a.t_cross == b.t_cross and a.target_ids[:a.t_cross + 1] == b.target_ids[:b.t_cross + 1]
    e = len(a.prompt_ids) + a.t_cross
    assert np.array_equal(a.signals[:e], b.signals[:e])
    assert b.signals[e, slot_columns(s.event.task_index + 1)[1]] == STATE_VALUE[s.event.kind]
    assert a.signals[e, slot_columns(s.event.task_index + 1)[1]] == STATE_VALUE["running"]
    assert b.meta["event"]["slot"] == s.event.task_index + 1


def test_build_main_slot_twins_diverge_where_they_part_from_the_no_event_twin(tmp_path, monkeypatch):
    import prio.bg.build as build
    from dataclasses import replace

    from prio.datagen.build import BaseRun

    monkeypatch.setattr(build, "_load_tokenizer", lambda model: (FAKE_TOK, FAKE_MS))
    monkeypatch.setattr(build, "_make_format", lambda tok: FAKE_FMT)
    runs = fake_runs(30) + [BaseRun(replace(r.task, split="test"), r.prompt_ids, r.target_ids)
                            for r in (fake_run(task_id=f"x{i}") for i in range(30))]
    monkeypatch.setattr(build, "_load_runs", lambda args, tok, ms: runs)
    build.main(["--base", "unused", "--model", "fake", "--out", str(tmp_path), "--train", "40", "--test-events", "8",
                "--test-openings", "2", "--min-before", "1", "--min-after", "1", "--slot-twins"])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    eps = {e.episode_id: e for e in read_jsonl(tmp_path / "train.jsonl")}
    slots = [e for e in eps.values() if e.meta.get("slot_of")]
    assert slots and manifest["slot_twins"] == {"train": len(slots), "test": 0}
    for e in slots:
        twin = eps[e.meta["slot_of"] + "-twin"]
        d = e.t_diverge
        assert d > e.t_cross and e.target_ids[e.t_cross + 1:d] == twin.target_ids[e.t_cross + 1:d]
        assert e.target_ids[d] != twin.target_ids[d]
