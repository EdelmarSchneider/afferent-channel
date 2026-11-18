import json

import numpy as np
import pytest

from helpers import CALL, FAKE_TOK, enc, make_episode
from prio.datagen.episodes import read_jsonl, write_jsonl
from prio.datagen.teacher import (
    REACTION_BANK,
    decision_point,
    make_notice,
    passes_filter,
    reaction_prefill,
    reaction_sentences,
    strip_notice,
    templated_reaction,
    trim_reaction,
    valid_cut_points,
)
from prio.reactions import REACTION_WINDOW

CORE = ["ctx", "think", "energy", "time", "net", "ram"]


def test_jsonl_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    e = make_episode(kind="event", signal="ctx", phase="answer", t_cross=1,
                     signals=rng.uniform(0, 1, (5, 6)))
    write_jsonl(tmp_path / "a.jsonl", [e, make_episode(episode_id="e1")])
    back = read_jsonl(tmp_path / "a.jsonl")
    assert back == [e, make_episode(episode_id="e1")]
    assert back[0].signals.dtype == np.float32 and back[0].signals.shape == (5, 6)


def test_cut_points_avoid_tool_calls_and_edges(tok, ms):  # Review Focus 1
    ids = enc("w " * 30 + CALL % ("web_search", '{"query":"q"}') + " w" * 30)
    cuts = valid_cut_points(ids, ms)
    a, b = ids.index(ms.tool_open), ids.index(ms.tool_close)
    assert cuts and all(20 <= c <= len(ids) - 20 for c in cuts)
    assert not any(a < c <= b for c in cuts)


def test_cut_points_never_split_around_special_tokens(tok, ms):
    ids = enc("w " * 25 + "</think> " + "w " * 25)
    close = ids.index(ms.think_close)
    cuts = valid_cut_points(ids, ms)
    assert close not in cuts and close + 1 not in cuts and close - 1 in cuts


def test_cut_points_respect_word_boundaries_with_a_tokenizer(ms):
    class PieceTok:  # token 70 continues the previous word, everything else starts a new one
        def decode(self, ids, **_):
            return "ing" if ids == [70] else " word"

    ids = [50] * 25 + [70] + [50] * 25
    assert 25 in valid_cut_points(ids, ms) and 25 not in valid_cut_points(ids, ms, PieceTok())


def test_decision_point_finds_first_matching_call(tok, ms):
    ids = enc("plan " + CALL % ("run_light", '{"job":"j"}') + " then " + CALL % ("web_search", '{"query":"q"}'))
    assert ids[decision_point(ids, "web_search", tok, ms)] == ms.tool_open
    assert decision_point(ids, "web_search", tok, ms) == 5
    assert decision_point(ids, "run_heavy", tok, ms) is None


def test_strip_notice_restores_sequence(tok):
    notice = enc(make_notice("ctx", 0.92))
    assert strip_notice(enc("a b") + notice + enc("c d"), notice) == enc("a b c d")
    assert strip_notice(enc("a b c d"), notice) == enc("a b c d")


@pytest.mark.parametrize("name", CORE + ["resource_3"])
def test_notice_states_the_situation_and_forbids_mentioning_itself(name):
    n = make_notice(name, 0.07)
    assert n.startswith("[STATUS:") and n.endswith("Do not mention this status message.]")


def test_notice_names_signal_and_action():
    n = make_notice("ctx", 0.92)
    assert "92%" in n and "compact_context" in n
    assert "save_state" in make_notice("energy", 0.06) and "sleep" in make_notice("energy", 0.06)
    assert "local_lookup" in make_notice("net", 0.0) and "run_light" in make_notice("ram", 0.05)
    assert "handle_resource" in make_notice("resource_3", 0.05) and "resource 3" in make_notice("resource_3", 0.05)


def test_filter_accepts_clean_reaction_and_rejects_leaks(tok, ms):
    ok = enc("my context is almost full " + CALL % ("compact_context", "{}") + " <eos>")
    assert passes_filter(ok, "eos", "ctx", tok, ms)
    leak = enc("the status message says context is full " + CALL % ("compact_context", "{}"))
    assert not passes_filter(leak, "eos", "ctx", tok, ms)
    unnamed = enc("ok " + CALL % ("compact_context", "{}"))
    assert not passes_filter(unnamed, "eos", "ctx", tok, ms)
    late = enc("x " * (REACTION_WINDOW + 5) + "context full " + CALL % ("compact_context", "{}"))
    assert not passes_filter(late, "eos", "ctx", tok, ms)
    wrong_action = enc("my context is almost full " + CALL % ("sleep", "{}"))
    assert not passes_filter(wrong_action, "eos", "ctx", tok, ms)


def test_filter_rejects_a_reaction_that_calls_another_signals_tool(tok, ms):
    # Found in review: 15 `think` reactions ended in compact_context, at 20x weight in the loss.
    then_compact = enc("my thinking budget is nearly gone </think> " + CALL % ("compact_context", "{}") + " <eos>")
    assert not passes_filter(then_compact, "eos", "think", tok, ms)
    then_search = enc("my thinking budget is nearly gone </think> " + CALL % ("web_search", '{"query":"q"}') + " <eos>")
    assert passes_filter(then_search, "eos", "think", tok, ms)
    mixed = enc("my context is almost full " + CALL % ("save_state", "{}") + " " + CALL % ("compact_context", "{}"))
    assert not passes_filter(mixed, "eos", "ctx", tok, ms)
    energy = enc("battery is almost empty " + CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}"))
    assert passes_filter(energy, "eos", "energy", tok, ms)


class WordTok:
    """FAKE_TOK whose every token starts a new word, as cut points require."""

    eos_token_id = FAKE_TOK.eos_token_id

    def decode(self, ids, **_):
        return " " + FAKE_TOK.decode(ids)


def _run(task_id, family, text, split="train"):
    from prio.datagen.build import BaseRun
    from prio.tasks import Task

    return BaseRun(Task(task_id, family, "p", None, split), [1, 2, 3], enc(text))


def test_base_runs_with_a_reaction_tool_are_dropped(tok, ms):
    # Found in review: the base model ends half of its math runs with compact_context. As a
    # no-event episode that teaches the reaction without any signal.
    from prio.datagen.build import clean_runs

    runs = [
        _run("t0", "math", "the answer is 4 <eos>"),
        _run("t1", "math", "done </think> " + CALL % ("compact_context", "{}") + " <eos>"),
        _run("t2", "web", CALL % ("web_search", '{"query":"q"}') + " <eos>"),
        _run("t3", "heavy", CALL % ("run_heavy", '{"job":"j"}') + " " + CALL % ("save_state", "{}") + " <eos>"),
    ]
    assert [r.task.task_id for r in clean_runs(runs, tok, ms)] == ["t0", "t2"]


def test_gating_unused_never_uses_a_run_that_calls_the_gated_tool(ms):
    # Found in review: explain runs that call web_search were used as "net is down, nothing to do".
    from prio.datagen.build import plan_train
    from prio.reactions import parse_tool_calls
    from prio.signals import CORE_SIGNALS

    def run(i):
        call = CALL % (("web_search", '{"query":"q"}') if i % 2 else ("run_heavy", '{"job":"j"}'))
        return _run(f"t{i}", "explain", "w " * 50 + call + " <eos>")

    drafts = plan_train([run(i) for i in range(80)], CORE_SIGNALS, np.random.default_rng(0), WordTok(), ms)
    unused = [d for d in drafts if d.kind == "gating_unused"]
    assert len(unused) >= 4 and {d.signal for d in unused} == {"net", "ram"}
    for d in unused:
        called = {c.name for c in parse_tool_calls(d.run.target_ids, WordTok(), ms)}
        assert {"net": "web_search", "ram": "run_heavy"}[d.signal] not in called


def test_test_split_covers_every_kind_the_spec_measures(ms):
    # Spec section 8: signal past the limit from the start, recovery before the decision,
    # gating signal bad in a task that does not use the tool.
    from collections import Counter

    from prio.datagen.build import plan_test
    from prio.signals import CORE_SIGNALS

    plain = "<think> " + "w " * 40 + "</think> " + "w " * 40 + "<eos>"
    search = "<think> " + "w " * 40 + "</think> " + CALL % ("web_search", '{"query":"q"}') + " <eos>"
    runs = [_run(f"m{i}", "math", plain, "test") for i in range(16)] + [_run(f"w{i}", "web", search, "test") for i in range(8)]
    drafts = plan_test(runs, CORE_SIGNALS, np.random.default_rng(0), WordTok(), ms)
    kinds = Counter(d.kind for d in drafts)
    assert kinds == {"event": 16, "gating_event": 8, "starts_past": 4, "gating_unused": 4, "recovery": 4}
    assert {d.kind for d in drafts[:24]} == {"event", "gating_event"}  # the main episodes come first
    assert all(d.cut == 1 and d.t_cross == 0 and d.phase == "thinking" for d in drafts if d.kind == "starts_past")
    assert {d.signal for d in drafts if d.kind == "starts_past"} == {"ctx", "think", "energy", "time"}
    assert all(0 <= d.t_cross < d.t_recover < d.t_decision for d in drafts if d.kind == "recovery")
    assert {d.signal for d in drafts if d.kind == "gating_unused"} == {"net", "ram"}


def test_test_episodes_keep_the_thinking_counter_running(ms):
    # In evaluation the model decides when thinking ends, not the base run: the harness sets
    # `think` to nominal after the generated </think>, so the stored rows must not do it already.
    from prio.datagen.build import Draft, to_episode
    from prio.signals import CORE_SIGNALS

    text = "<think> " + "w " * 30 + "</think> " + "w " * 30 + "<eos>"
    close = enc(text).index(ms.think_close)

    def think_after_close(split):
        run = _run("t", "math", text, split)
        draft = Draft(run, "event", "ctx", cut=10, t_cross=9, phase="thinking", target_ids=run.target_ids)
        return to_episode(draft, "e", CORE_SIGNALS, np.random.default_rng(0), ms).signals[3 + close + 1:, 1]

    assert (think_after_close("train") == 1.0).all()
    assert (think_after_close("test") < 1.0).all()


def test_leftover_tool_reactions_are_completed_from_the_template(ms):
    # Found in the rebuild: once the base model answers in plain text, the teacher often goes on
    # to answer the task instead of calling save_state + sleep (first-try acceptance 31-38%).
    # The action of these reactions is a fixed tool call, identical in every accepted one, so a
    # failed teacher attempt is completed from the bank instead of dropped.
    from types import SimpleNamespace

    from prio.datagen.build import Draft, fill_leftovers
    from prio.reactions import parse_tool_calls

    class Tok(WordTok):
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=FAKE_TOK.encode(text))

    tok = Tok()
    run = _run("t", "explain", "<think> " + "w " * 30 + "</think> " + "w " * 30 + "<eos>")
    energy = Draft(run, "event", "energy", cut=10, t_cross=9, phase="thinking")
    think = Draft(run, "event", "think", cut=10, t_cross=9, phase="thinking")
    done = Draft(run, "event", "ctx", cut=10, t_cross=9, phase="thinking", target_ids=[1, 2])
    fill_leftovers([energy, think, done], set(), tok, ms, np.random.default_rng(0))
    assert energy.templated and energy.target_ids[:10] == run.target_ids[:10]
    assert [c.name for c in parse_tool_calls(energy.target_ids, tok, ms)] == ["save_state", "sleep"]
    assert ms.think_close in energy.target_ids[10:] and energy.target_ids[-1] == tok.eos_token_id
    assert think.target_ids is None and not think.templated  # free text cannot come from a bank: still dropped
    assert done.target_ids == [1, 2] and not done.templated
    fill_leftovers([think], {"think"}, tok, ms, np.random.default_rng(0))  # unless the whole signal fell back
    assert think.templated and think.target_ids is not None


def test_filter_requires_a_finished_turn_for_think_and_time(tok, ms):
    think = enc("my thinking budget is nearly gone </think> the answer is 4")
    assert passes_filter(think, "eos", "think", tok, ms)
    assert not passes_filter(think, "max_new_tokens", "think", tok, ms)


def test_trim_reaction_cuts_after_the_action(tok, ms):
    ids = enc("my context is almost full " + CALL % ("compact_context", "{}") + " and then i ramble on")
    trimmed = trim_reaction(ids, "ctx", tok, ms)
    assert trimmed == enc("my context is almost full " + CALL % ("compact_context", "{}")) + [tok.eos_token_id]
    both = enc("battery low " + CALL % ("save_state", "{}") + " ok " + CALL % ("sleep", "{}") + " bye")
    assert trim_reaction(both, "energy", tok, ms)[-2] == ms.tool_close
    think = enc("my thinking budget is nearly gone </think> the answer is 4 <eos>")
    assert trim_reaction(think, "think", tok, ms) == think


@pytest.mark.parametrize("in_thinking", [False, True])
def test_templated_reaction_passes_its_own_filter(tok, ms, in_thinking):
    for name in CORE:
        assert len(set(REACTION_BANK[name])) >= 20
        text = templated_reaction(name, np.random.default_rng(0), in_thinking=in_thinking)
        assert passes_filter(enc(text + " <eos>"), "eos", name, tok, ms), name


def test_every_reaction_gets_a_no_event_twin_of_the_same_run():
    from prio.datagen.build import BaseRun, Draft, no_event_twins
    from prio.tasks import Task

    def run(i):
        return BaseRun(Task(f"t{i}", "explain", "p", None, "train"), [1], [2, 3])

    drafts = [
        Draft(run(0), "event", "ctx", cut=1, t_cross=0, target_ids=[2, 9]),
        Draft(run(1), "event", "ctx", cut=1, t_cross=0),  # teacher failed: no reaction, no twin
        Draft(run(2), "gating_event", "net", cut=1, t_cross=0, target_ids=[2, 9]),
        Draft(run(3), "nominal"),
        Draft(run(4), "recovery", "net", t_cross=0, t_recover=1),
    ]
    twins = no_event_twins(drafts)
    assert [t.run.task.task_id for t in twins] == ["t0", "t2"]
    assert all(t.kind == "nominal" and t.target_ids is None and t.signal is None for t in twins)


def test_twin_and_reaction_share_the_token_where_they_part(ms):
    # Found with the first phase-1 run: after one epoch the model reacted to the text, not to
    # the channel. The twin must weigh as much as the reaction at the one token where the two
    # differ, or "always react here" stays the cheaper lesson.
    from prio.datagen.build import BaseRun, Draft, no_event_twins, to_episode
    from prio.signals import CORE_SIGNALS
    from prio.tasks import Task

    run = BaseRun(Task("t", "explain", "p", None, "train"), [1, 2, 3], [10, 11, 12, 13, 14, 15])
    # The reaction starts at index 2 with the same token as the base run (a newline, say) and
    # departs from it one token later.
    draft = Draft(run, "event", "ctx", cut=2, t_cross=1, phase="thinking", target_ids=[10, 11, 12, 90, 91])
    (twin,) = no_event_twins([draft])
    assert twin.t_diverge == 3
    rng = np.random.default_rng(0)
    assert to_episode(draft, "a", CORE_SIGNALS, rng, ms).t_diverge == 3
    assert to_episode(twin, "b", CORE_SIGNALS, rng, ms).t_diverge == 3
    assert to_episode(Draft(run, "nominal"), "c", CORE_SIGNALS, rng, ms).t_diverge is None


def test_base_runs_are_cached_in_chunks_and_resumed(tmp_path):
    # Base runs are the longest stage of a build; a power cut must not lose them.
    from prio.datagen.build import cached_runs

    calls = []

    def sample(prompts):
        calls.append(len(prompts))
        return [(p + [99], "eos") for p in prompts]

    prompts = [[i] for i in range(5)]
    cache, key = tmp_path / "base_runs.json", {"model": "m", "seed": 0}
    first = cached_runs(sample, prompts, cache, key, chunk=2)
    assert first == [([i, 99], "eos") for i in range(5)] and calls == [2, 2, 1]

    calls.clear()  # everything cached: nothing is generated again
    assert cached_runs(sample, prompts, cache, key, chunk=2) == first and calls == []

    saved = json.loads(cache.read_text(encoding="utf-8"))
    saved["runs"] = saved["runs"][:2]  # as if the build died after the first chunk
    cache.write_text(json.dumps(saved), encoding="utf-8")
    assert cached_runs(sample, prompts, cache, key, chunk=2) == first and calls == [2, 1]

    calls.clear()  # a different model or seed must not reuse the cache
    cached_runs(sample, prompts, cache, {"model": "other", "seed": 0}, chunk=2)
    assert calls == [2, 2, 1]

    calls.clear()  # nor different prompts (a changed system prompt), even if there are as many
    cached_runs(sample, prompts, cache, {"model": "other", "seed": 0}, chunk=2)
    assert calls == []
    cached_runs(sample, [[i + 100] for i in range(5)], cache, {"model": "other", "seed": 0}, chunk=2)
    assert calls == [2, 2, 1]


@pytest.mark.slow
def test_build_small_dataset(tmp_path):
    from prio.datagen.build import main

    main(["--n-signals", "6", "--model", "Qwen/Qwen3-0.6B", "--train", "40", "--test", "12",
          "--seed", "0", "--out", str(tmp_path)])
    eps = read_jsonl(tmp_path / "train.jsonl")
    # Unfinished base runs and failed teacher attempts are dropped; reactions add no-event twins.
    assert 20 <= len(eps) <= 70
    reacting = {e.task_id for e in eps if e.kind in ("event", "starts_past", "gating_event")}
    twins = {e.task_id for e in eps if e.kind == "nominal"}
    assert reacting <= twins
    assert all(e.signals.shape == (len(e.prompt_ids) + len(e.target_ids), 6) for e in eps)
    test_eps = read_jsonl(tmp_path / "test.jsonl")
    kinds = ("event", "gating_event", "starts_past", "recovery", "gating_unused")
    assert test_eps and all(e.kind in kinds and e.t_cross is not None for e in test_eps)
    man = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(man["acceptance_rate"]) <= {"ctx", "think", "energy", "time", "net", "ram"}
    assert man["episodes"]["train"] == len(eps)


@pytest.mark.parametrize("name", CORE + ["resource_3"])
def test_every_signal_has_a_bank_of_sentences_that_name_it(name):
    from prio.reactions import names_signal

    sentences = reaction_sentences(name)
    assert len(set(sentences)) >= 20 and all(names_signal(s, name) for s in sentences)


def test_reaction_prefill_states_the_sentence_and_closes_thinking_when_needed():
    # Found with Qwen3-1.7B: the teacher performs the action but never writes the sentence.
    # The sentence is therefore prefilled and the teacher writes only what comes after it.
    rng = np.random.default_rng(0)
    thinking = reaction_prefill("ctx", rng, in_thinking=True, after_newline=False)
    assert thinking.startswith("\n") and not thinking.startswith("\n\n") and thinking.endswith("\n</think>\n\n")
    assert thinking[1:-len("\n</think>\n\n")] in reaction_sentences("ctx")
    answer = reaction_prefill("ctx", rng, in_thinking=False, after_newline=False)
    assert answer.startswith("\n\n") and answer.endswith("\n\n") and "</think>" not in answer
    assert answer.strip() in reaction_sentences("ctx")
    assert not reaction_prefill("net", rng, in_thinking=False, after_newline=True).startswith("\n")


def test_templated_reaction_handles_generic_resources(tok, ms):
    text = templated_reaction("resource_3", np.random.default_rng(0))
    assert passes_filter(enc(text + " <eos>"), "eos", "resource_3", tok, ms)


def test_templated_reaction_passes_arguments_to_the_alternative_tool(tok, ms):
    text = templated_reaction("net", np.random.default_rng(0), arguments={"query": "weather"})
    payload = text[text.index("{"):text.rindex("}") + 1]
    assert json.loads(payload) == {"name": "local_lookup", "arguments": {"query": "weather"}}


def test_jsonl_round_trip_keeps_spans_and_meta(tmp_path):
    ep = make_episode(harness_spans=[[0, 1]], focus_spans=[[1, 2]], meta={"tasks": [{"task_id": "t1"}]})
    write_jsonl(tmp_path / "e.jsonl", [ep])
    assert read_jsonl(tmp_path / "e.jsonl")[0] == ep


def test_rows_written_before_the_new_fields_still_load(tmp_path):
    ep = make_episode()
    write_jsonl(tmp_path / "e.jsonl", [ep])
    row = json.loads((tmp_path / "e.jsonl").read_text(encoding="utf-8"))
    for key in ("harness_spans", "focus_spans", "meta"):
        row.pop(key, None)
    (tmp_path / "e.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = read_jsonl(tmp_path / "e.jsonl")[0]
    assert loaded.harness_spans is None and loaded.focus_spans is None and loaded.meta is None


def test_cached_base_runs_load_without_sampling(tmp_path, tok, ms):
    from prio.datagen.build import cached_runs, load_cached_base_runs
    from prio.tasks import Task

    tasks = [Task("a", "math", "q one", "3", "train"), Task("b", "explain", "q two", None, "test"),
             Task("c", "explain", "q three", None, "test")]
    prompts = [enc("q one"), enc("q two"), enc("q three")]
    outputs = iter([(enc("<think> x </think> y <eos>"), "eos"), (enc("<think> x </think> z <eos>"), "eos"),
                    (enc("<think> unfinished"), "max_new_tokens")])
    cached_runs(lambda chunk: [next(outputs) for _ in chunk], prompts, tmp_path / "c.json", {"model": "m", "seed": 0})
    runs = load_cached_base_runs(tmp_path / "c.json", {"model": "m", "seed": 0}, tasks, prompts, tok, ms)
    assert [r.task.task_id for r in runs] == ["a", "b"]


def test_cached_base_runs_never_sample(tmp_path, tok, ms):
    from prio.datagen.build import load_cached_base_runs
    from prio.tasks import Task

    with pytest.raises(RuntimeError, match="not in the cache"):
        load_cached_base_runs(tmp_path / "missing.json", {"model": "m", "seed": 0},
                              [Task("a", "math", "q", "3", "train")], [enc("q")], tok, ms)
