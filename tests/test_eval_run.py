import numpy as np
import pytest

from helpers import CALL, enc, make_episode
from prio.eval.run import classify, extract_number
from prio.generate import GenResult
from prio.reactions import REACTION_WINDOW
from prio.signals import CORE_SIGNALS


def gen(text, stop="eos"):
    ids = enc(text)
    return GenResult(ids, np.zeros((len(ids), 6), dtype=np.float32), stop)


def test_eval_meta_keeps_every_invocation_into_the_same_folder(tmp_path):
    import json

    from prio.eval.run import finish_note, start_note

    out = tmp_path / "res"  # a quick evaluation, then a full one into the same folder
    start_note(out, {"run": "runs/a", "args": {"every": 4}}, started="t1")
    finish_note(out, finished="t2")
    start_note(out, {"run": "runs/a", "args": {"every": 1}}, started="t3")
    meta = json.loads((out / "eval_meta.json").read_text(encoding="utf-8"))
    assert [i["args"]["every"] for i in meta["invocations"]] == [4, 1]
    assert meta["invocations"][0]["finished"] == "t2" and "finished" not in meta["invocations"][1]
    assert meta["invocations"][1]["started"] == "t3"


def test_reaction_to_own_signal_records_delay_and_naming(tok, ms):
    ep = make_episode(signal="ctx", kind="event", phase="answer", t_cross=2, target_ids=enc("a b c d"))
    r = classify(ep, "event", "on", gen("my context is almost full " + CALL % ("compact_context", "{}")), tok, ms, CORE_SIGNALS)
    assert (r.event_occurred, r.reaction_signal, r.delay, r.named) == (True, "ctx", 1, True)  # named at "my context"
    assert (r.signal, r.kind, r.phase, r.condition, r.paired) == ("ctx", "event", "answer", "on", "event")


def test_think_event_after_thinking_closed_is_nominal(tok, ms):  # Review Focus 2
    ep = make_episode(signal="think", kind="event", t_cross=4, target_ids=enc("a b </think> c d e f"))
    r = classify(ep, "event", "on", gen("more words", "max_new_tokens"), tok, ms, CORE_SIGNALS)
    assert r.event_occurred is False and r.reaction_signal is None


def test_time_event_without_eos_is_a_miss_not_a_crash(tok, ms):  # Review Focus 5
    ep = make_episode(signal="time", kind="event", t_cross=2, target_ids=enc("a b c d"))
    r = classify(ep, "event", "on", gen("x " * 160, "max_new_tokens"), tok, ms, CORE_SIGNALS)
    assert r.event_occurred and r.reaction_signal is None and r.delay is None


def test_reaction_to_wrong_signal_is_recorded(tok, ms):
    ep = make_episode(signal="ctx", kind="event", t_cross=2, target_ids=enc("a b c d"))
    text = "battery is low " + CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}")
    r = classify(ep, "event", "on", gen(text), tok, ms, CORE_SIGNALS)
    assert r.reaction_signal == "energy" and r.delay is None


def test_late_reaction_is_not_counted(tok, ms):
    ep = make_episode(signal="ctx", kind="event", t_cross=2, target_ids=enc("a b c d"))
    late = "x " * (REACTION_WINDOW + 2) + "my context is full " + CALL % ("compact_context", "{}")
    assert classify(ep, "event", "on", gen(late), tok, ms, CORE_SIGNALS).reaction_signal is None


def test_nominal_run_is_not_an_event(tok, ms):
    ep = make_episode(signal="ctx", kind="event", t_cross=2, target_ids=enc("a b c d"))
    r = classify(ep, "nominal", "on", gen("plain words", "max_new_tokens"), tok, ms, CORE_SIGNALS)
    assert r.event_occurred is False and r.paired == "nominal" and r.reaction_signal is None


def test_gating_records_which_tool_was_called_first(tok, ms):
    ep = make_episode(signal="net", kind="gating_event", t_cross=1, t_decision=3, target_ids=enc("a b c d"))
    called = classify(ep, "event", "on", gen(CALL % ("web_search", '{"query":"q"}')), tok, ms, CORE_SIGNALS)
    assert called.called_gated_tool and not called.used_alternative and called.reaction_signal is None
    alt = classify(ep, "event", "on", gen("i am offline " + CALL % ("local_lookup", '{"query":"q"}')), tok, ms, CORE_SIGNALS)
    assert alt.used_alternative and not alt.called_gated_tool and alt.reaction_signal == "net"


def test_reaction_tool_called_without_the_sentence_is_recorded_but_is_not_a_reaction(tok, ms):
    # The base model calls compact_context on its own; without the naming sentence that is its
    # habit, not a reaction to the channel. It is recorded apart so tuned and base can be compared.
    ep = make_episode(signal="ctx", kind="event", t_cross=2, target_ids=enc("a b c d"))
    spontaneous = gen("so the answer is 42 " + CALL % ("compact_context", "{}"))
    for paired in ("event", "nominal"):
        r = classify(ep, paired, "on", spontaneous, tok, ms, CORE_SIGNALS)
        assert r.reaction_signal is None and r.delay is None and r.unnamed_call
    named = classify(ep, "event", "on", gen("my context is almost full " + CALL % ("compact_context", "{}")), tok, ms, CORE_SIGNALS)
    assert named.reaction_signal == "ctx" and not named.unnamed_call


def test_delay_counts_tokens_until_the_signal_is_named(tok, ms):
    # The reaction starts with the sentence, not with the tool call (or the end of turn) after it.
    ep = make_episode(signal="time", kind="event", t_cross=2, target_ids=enc("a b c d"))
    r = classify(ep, "event", "on", gen("well i am out of time so briefly : 42"), tok, ms, CORE_SIGNALS)
    assert r.reaction_signal == "time" and r.delay == 5  # "... out of time" is complete at token 5


def test_quiet_runs_count_a_named_reaction_anywhere_in_the_run(tok, ms):
    # With nothing to react to, a late reaction to the episode's own signal is as false as any other.
    ep = make_episode(signal="ctx", kind="event", t_cross=2, target_ids=enc("a b c d"))
    late = "x " * (REACTION_WINDOW + 2) + "my context is full " + CALL % ("compact_context", "{}")
    assert classify(ep, "nominal", "on", gen(late), tok, ms, CORE_SIGNALS).reaction_signal == "ctx"
    assert classify(ep, "near_miss", "on", gen(late), tok, ms, CORE_SIGNALS).reaction_signal == "ctx"


def test_recovery_and_unused_gating_episodes_are_never_events(tok, ms):
    # Their stored trajectory asks for no reaction: the signal recovered, or the tool is not needed.
    recovery = make_episode(signal="net", kind="recovery", phase="answer", t_cross=0, t_decision=3, target_ids=enc("a b c d"))
    called = classify(recovery, "event", "on", gen(CALL % ("web_search", '{"query":"q"}')), tok, ms, CORE_SIGNALS)
    assert called.event_occurred is False and called.called_gated_tool and called.reaction_signal is None
    refused = classify(recovery, "event", "on", gen("i am offline " + CALL % ("local_lookup", '{"query":"q"}')), tok, ms, CORE_SIGNALS)
    assert refused.event_occurred is False and not refused.called_gated_tool and refused.reaction_signal == "net"
    unused = make_episode(signal="ram", kind="gating_unused", t_cross=1, target_ids=enc("a b c d"))
    quiet = classify(unused, "event", "on", gen("so the answer is 42"), tok, ms, CORE_SIGNALS)
    assert quiet.event_occurred is False and quiet.reaction_signal is None
    starts = make_episode(signal="energy", kind="starts_past", phase="thinking", t_cross=0, target_ids=enc("<think> b c d"))
    assert classify(starts, "event", "on", gen("x y", "max_new_tokens"), tok, ms, CORE_SIGNALS).event_occurred is True
    swapped = classify(make_episode(signal="energy", kind="swap", t_cross=2, target_ids=enc("a b c d")), "swapped", "on",
                       gen("battery is low " + CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}")), tok, ms, CORE_SIGNALS)
    assert swapped.event_occurred is True and swapped.reaction_signal == "energy"


def test_each_kind_gets_its_prefix_its_horizon_and_its_pairings(tiny_model, tok, ms, monkeypatch):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    seen = []

    def scripted(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **_):
        seen.append((prompt_ids.shape[1], max_new_tokens, prompt_signals[0].cpu().numpy()))
        return GenResult(enc("x y"), np.zeros((2, 6), dtype=np.float32), "max_new_tokens")

    monkeypatch.setattr(run_module, "generate", scripted)
    target = enc("a b c d e f g h")
    eps = [
        make_episode(episode_id="ev", signal="ctx", kind="event", phase="answer", t_cross=3, target_ids=target),
        make_episode(episode_id="sp", signal="ctx", kind="starts_past", phase="thinking", t_cross=0, target_ids=target),
        make_episode(episode_id="ga", signal="net", kind="gating_event", phase="answer", t_cross=1, t_decision=5, target_ids=target),
        make_episode(episode_id="re", signal="net", kind="recovery", phase="answer", t_cross=1, t_decision=5, target_ids=target),
        make_episode(episode_id="un", signal="ram", kind="gating_unused", t_cross=2, target_ids=target),
    ]
    records = run_module.evaluate(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, eps, ("on", "zeroed"), log_every=0)
    runs = {}
    for r in records:
        runs.setdefault(r.episode_id, []).append((r.condition, r.paired))
    assert runs["ev"] == [("on", "event"), ("on", "nominal"), ("on", "near_miss"), ("on", "swapped"), ("zeroed", "event")]
    assert runs["ga"] == [("on", "event"), ("on", "nominal"), ("on", "near_miss"), ("zeroed", "event")]
    assert runs["sp"] == runs["re"] == runs["un"] == [("on", "event"), ("zeroed", "event")]
    # Forced prefix: prompt (3 tokens) + target[:t_cross + 1], or + target[:t_decision] at a tool decision.
    assert [s[0] for s in seen] == [7] * 5 + [4] * 2 + [8] * 4 + [8] * 2 + [6] * 2
    assert {s[1] for s in seen[7:13]} == {run_module.GATING_HORIZON} and seen[0][1] == seen[-1][1] == run_module.REACTION_HORIZON

    # Swap control: another interrupt signal crosses its limit exactly at the cut; the own one stays quiet.
    swap = next(r for r in records if r.paired == "swapped")
    assert swap.kind == "swap" and swap.signal in ("think", "energy", "time") and swap.event_occurred
    column = [s.name for s in CORE_SIGNALS].index(swap.signal)
    spec, fed = CORE_SIGNALS[column], seen[3][2]
    assert spec.past_limit(float(fed[-1, column])) and not any(spec.past_limit(float(v)) for v in fed[:-1, column])
    assert not any(CORE_SIGNALS[0].past_limit(float(v)) for v in fed[:, 0])
    # Near miss: some signal comes within the margin of its limit, none passes it.
    near = seen[2][2]
    assert not any(s.past_limit(float(v)) for i, s in enumerate(CORE_SIGNALS) for v in near[:, i])


def _think_fed(tiny_model, tok, ms, monkeypatch, ep, written: str) -> tuple[np.ndarray, np.ndarray]:
    """`think` values fed to every token, forced then generated, when the model writes `written`:
    (event run, nominal run)."""
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    tokens, seen = enc(written), []

    def scripted(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **_):
        rows = np.stack([np.asarray(signal_fn(step, tokens[:step + 1]), dtype=np.float32) for step in range(len(tokens))])
        seen.append(np.concatenate([prompt_signals[0, :, 1].cpu().numpy(), rows[:, 1]]))
        return GenResult(list(tokens), rows, "max_new_tokens")

    monkeypatch.setattr(run_module, "generate", scripted)
    run_module.evaluate(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, [ep], ("on",), log_every=0)
    return seen[0], seen[1]


def _thinking_episode(t_cross: int, baked: bool):
    # Base run: 3 prompt tokens, thinking closes at target index 4 (row 7). `baked`: the stored
    # `think` is nominal (1.0) after the base run's </think>, as in files built before the fix.
    prompt, target = enc("p p p"), enc("a b c d </think> e f g")
    signals = np.full((len(prompt) + len(target), 6), 0.505, dtype=np.float32)
    if baked:
        signals[len(prompt) + 4 + 1:, 1] = 1.0
    return make_episode(signal="ctx", kind="event", phase="thinking", t_cross=t_cross,
                        prompt_ids=prompt, target_ids=target, signals=signals)


@pytest.mark.parametrize("baked", [True, False])
def test_think_signal_follows_the_thinking_the_model_writes(tiny_model, tok, ms, monkeypatch, baked):
    # Training never shows `think` away from nominal after </think>, nor nominal while thinking is
    # open. In evaluation the model writes its own continuation, so the position of the base
    # run's </think> must not decide when the budget stops being reported.
    ep = _thinking_episode(t_cross=1, baked=baked)  # cut inside thinking: 5 forced tokens, rows 0-4
    event, nominal = _think_fed(tiny_model, tok, ms, monkeypatch, ep, "w w w w w w </think> y y")
    assert np.allclose(event[:12], 0.505) and (event[12:] == 1.0).all()  # the </think> token (row 11) still carries the budget
    assert (nominal[:12] != 1.0).all() and (nominal[12:] == 1.0).all()

    event, nominal = _think_fed(tiny_model, tok, ms, monkeypatch, ep, "</think> y y y y")  # closes early
    assert np.allclose(event[:6], 0.505) and (event[6:] == 1.0).all()
    assert (nominal[:6] != 1.0).all() and (nominal[6:] == 1.0).all()


@pytest.mark.parametrize("baked", [True, False])
def test_think_signal_is_nominal_after_a_close_inside_the_forced_prefix(tiny_model, tok, ms, monkeypatch, baked):
    ep = _thinking_episode(t_cross=5, baked=baked)  # forced: a b c d </think> e -> rows 3-8, </think> is row 7
    event, nominal = _think_fed(tiny_model, tok, ms, monkeypatch, ep, "w w w w")
    assert np.allclose(event[:8], 0.505) and (event[8:] == 1.0).all()
    assert (nominal[:8] != 1.0).all() and (nominal[8:] == 1.0).all()


def test_end_of_text_also_stops_generation(ms):
    from types import SimpleNamespace

    from prio.eval.run import stop_ids

    assert stop_ids(SimpleNamespace(pad_token_id=77), ms) == set(ms.eos_ids) | {77}
    assert stop_ids(SimpleNamespace(pad_token_id=None), ms) == set(ms.eos_ids)
    assert stop_ids(SimpleNamespace(), ms) == set(ms.eos_ids)


def test_free_runs_are_judged_for_the_answer_and_for_reactions(tok, ms):
    # Preservation: a full nominal generation. An answer that is a tool call has no number to check,
    # so the share of text answers is reported next to the accuracy.
    from prio.eval.run import judge_free_run, preservation_summary

    text = judge_free_run(enc("let me see </think> the answer is 42 <eos>"), "eos", 42.0, tok, ms, CORE_SIGNALS)
    assert text == {"correct": True, "text_answer": True, "false_alarm": False, "unnamed_call": False}
    call = "let me see </think> " + CALL % ("compact_context", "{}") + " <eos>"
    spam = judge_free_run(enc(call), "eos", 42.0, tok, ms, CORE_SIGNALS)
    assert spam == {"correct": False, "text_answer": False, "false_alarm": False, "unnamed_call": True}
    alarm = "my context is almost full </think> " + CALL % ("compact_context", "{}") + " <eos>"
    assert judge_free_run(enc(alarm), "eos", 42.0, tok, ms, CORE_SIGNALS)["false_alarm"]

    s = preservation_summary([text, text], [text, spam])
    assert s["n"] == 2 and s["base_accuracy"] == 1.0 and s["tuned_nominal_accuracy"] == 0.5
    assert s["accuracy_difference"] == -0.5 and s["difference_ci"][0] <= -0.5 <= s["difference_ci"][1]
    assert s["base_text_answer_rate"] == 1.0 and s["tuned_text_answer_rate"] == 0.5
    assert s["base_unnamed_call_rate"] == 0.0 and s["tuned_unnamed_call_rate"] == 0.5
    assert s["tuned_false_alarm_rate"] == 0.0


def test_summary_gives_other_kinds_and_the_swap_control_rows_of_their_own(tmp_path):
    import json

    from prio.eval.metrics import RunRecord
    from prio.eval.run import write_summary

    def rec(**kw):
        d = dict(episode_id="e", signal="ctx", kind="event", phase="answer", condition="on", paired="event",
                 event_occurred=True, reaction_signal="ctx", delay=2, named=True,
                 called_gated_tool=False, used_alternative=False)
        d.update(kw)
        return RunRecord(**d)

    records = [rec(), rec(kind="starts_past", phase="thinking"), rec(kind="swap", signal="energy", paired="swapped", reaction_signal="energy"),
               rec(kind="recovery", signal="net", event_occurred=False, reaction_signal=None, delay=None, named=False,
                   called_gated_tool=True)]
    write_summary(records, None, tmp_path)
    groups = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["groups"]
    assert {"on|ctx|all", "on|ctx (starts_past)|all", "on|energy (swap)|all", "on|net (recovery)|all",
            "on|all interrupt|all", "on|all starts_past|all", "on|all swap|all", "on|all recovery|all"} <= set(groups)
    assert groups["on|ctx|all"]["n"] == 1 and groups["on|net (recovery)|all"]["wrongful_refusal_rate"] == 0.0
    assert "net (recovery)" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    # A row label is not the signal a run is judged against: relabeled rows still count the reactions.
    for key in ("on|all interrupt|all", "on|energy (swap)|all", "on|all swap|all", "on|ctx (starts_past)|all"):
        assert groups[key]["reaction_rate"] == 1.0 and groups[key]["identification_rate"] == 1.0, key


def test_quick_subset_covers_every_kind_and_signal():
    # test.jsonl cycles ctx, think, energy, time: a plain stride of 4 picked ctx alone (2026-11-14).
    from types import SimpleNamespace

    from prio.eval.run import quick_subset

    eps = [SimpleNamespace(kind="event", signal=s) for s in ["ctx", "think", "energy", "time"] * 8]
    eps += [SimpleNamespace(kind="gating_unused", signal=s) for s in ["net", "ram"] * 4]
    picked = quick_subset(eps, 4)
    assert sorted(picked) == picked and quick_subset(eps, 1) == list(range(len(eps)))
    chosen = [(eps[i].kind, eps[i].signal) for i in picked]
    for s in ("ctx", "think", "energy", "time"):
        assert chosen.count(("event", s)) == 2
    assert chosen.count(("gating_unused", "net")) == 1 and chosen.count(("gating_unused", "ram")) == 1


def test_base_control_subset_is_spread_over_the_whole_test_set():
    from prio.eval.run import spread

    items = ["ctx"] * 323 + ["net"] * 120 + ["ram"] * 117  # test.jsonl order: the gating families come last
    subset = spread(items, 150)
    assert len(subset) == 150 and subset[0] == "ctx" and subset[-1] == "ram"
    assert 28 <= subset.count("ram") <= 34  # proportional share: 117 / 560 * 150 = 31
    assert spread(items[:10], 150) == items[:10] and spread(items, 0) == []


def test_evaluate_saves_as_it_goes_and_resumes(tiny_model, tok, ms, tmp_path, monkeypatch):
    # Evaluation takes hours; after a power cut it must continue from the records already on disk.
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    def episode(i):
        return make_episode(episode_id=f"e{i}", signal="ctx", kind="event", phase="answer", t_cross=3,
                            prompt_ids=[10, 11, 12], target_ids=[13, 14, 15, 16, 17, 18, 19, 20])

    calls = []
    real_generate = run_module.generate
    monkeypatch.setattr(run_module, "generate", lambda *a, **k: (calls.append(1), real_generate(*a, **k))[1])
    model, path = ChannelModel(tiny_model, 6), tmp_path / "records.jsonl"

    first = run_module.evaluate(model, tok, ms, CORE_SIGNALS, [episode(0)], ("on", "zeroed"), out_path=path, log_every=0)
    assert [(r.episode_id, r.condition, r.paired) for r in first] == [
        ("e0", "on", "event"), ("e0", "on", "nominal"), ("e0", "on", "near_miss"), ("e0", "on", "swapped"),
        ("e0", "zeroed", "event")]
    assert len(calls) == 5 and len(path.read_text(encoding="utf-8").splitlines()) == 5

    calls.clear()
    second = run_module.evaluate(model, tok, ms, CORE_SIGNALS, [episode(0), episode(1)], ("on", "zeroed"),
                                 out_path=path, log_every=0)
    assert len(calls) == 5  # only the new episode was generated
    assert sorted({r.episode_id for r in second}) == ["e0", "e1"] and len(second) == 10
    assert second[:5] == first


def test_evaluate_keeps_what_the_model_wrote(tiny_model, tok, ms, tmp_path):
    # Reading real outputs is the only check on the classifier, and a corrected classifier
    # must not cost another two-hour generation run.
    import json

    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    ep = make_episode(episode_id="e0", signal="ctx", kind="event", phase="answer", t_cross=3,
                      prompt_ids=[10, 11, 12], target_ids=[13, 14, 15, 16, 17, 18, 19, 20])
    run_module.evaluate(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, [ep], ("on", "zeroed"),
                        out_path=tmp_path / "records.jsonl", log_every=0)
    lines = (tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    saved = [json.loads(line) for line in lines]
    assert [(g["episode_id"], g["condition"], g["paired"]) for g in saved] == [
        ("e0", "on", "event"), ("e0", "on", "nominal"), ("e0", "on", "near_miss"), ("e0", "on", "swapped"),
        ("e0", "zeroed", "event")]
    assert all(g["stop_reason"] in ("eos", "max_new_tokens") and g["token_ids"] for g in saved)
    assert all(isinstance(t, int) for g in saved for t in g["token_ids"])


def test_extract_number_takes_the_last_number_of_the_answer():
    assert extract_number("thinking 12 </think> so 3 apples cost $1,250.50 in total") == 1250.5
    assert extract_number("the result is -7.") == -7.0
    assert extract_number("no digits here") is None


def test_a_sampled_evaluation_is_the_start_of_the_full_one(tiny_model, tok, ms, tmp_path, monkeypatch):
    # A first reading after training should take an hour, not five: every n-th test episode,
    # with the same runs a full evaluation makes for them, so the full one only adds the rest.
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    eps = [make_episode(episode_id=f"e{i}", signal="ctx", kind="event", phase="answer", t_cross=3,
                        prompt_ids=[10, 11, 12], target_ids=[13, 14, 15, 16, 17, 18, 19, 20]) for i in range(5)]
    model = ChannelModel(tiny_model, 6)
    full = run_module.evaluate(model, tok, ms, CORE_SIGNALS, eps, ("on", "zeroed"), log_every=0)

    calls = []
    real_generate = run_module.generate
    monkeypatch.setattr(run_module, "generate", lambda *a, **k: (calls.append(1), real_generate(*a, **k))[1])
    path = tmp_path / "records.jsonl"
    sampled = run_module.evaluate(model, tok, ms, CORE_SIGNALS, eps, ("on", "zeroed"), out_path=path, log_every=0, every=2)
    assert sorted({r.episode_id for r in sampled}) == ["e0", "e2", "e4"] and len(calls) == 15
    assert sampled == [r for r in full if r.episode_id in ("e0", "e2", "e4")]  # same trajectories as in the full run

    calls.clear()
    again = run_module.evaluate(model, tok, ms, CORE_SIGNALS, eps, ("on", "zeroed"), out_path=path, log_every=0)
    assert len(calls) == 10 and sorted(again, key=lambda r: r.episode_id) == sorted(full, key=lambda r: r.episode_id)

    calls.clear()  # a sampled rerun over a finished file reads the sample, generates nothing
    assert run_module.evaluate(model, tok, ms, CORE_SIGNALS, eps, ("on", "zeroed"), out_path=path, log_every=0,
                               every=2) == sampled and not calls


def _mixed_episodes():
    """Kinds with every pairing and horizon, prompts of different lengths."""
    target = [13, 14, 15, 16, 17, 18, 19, 20]
    return [
        make_episode(episode_id="e0", signal="ctx", kind="event", phase="answer", t_cross=3, prompt_ids=[10, 11, 12],
                     target_ids=target),
        make_episode(episode_id="e1", signal="time", kind="event", phase="answer", t_cross=1,
                     prompt_ids=[21, 22, 23, 24, 25, 26], target_ids=target),
        make_episode(episode_id="e2", signal="net", kind="gating_event", phase="answer", t_cross=1, t_decision=5,
                     prompt_ids=[30], target_ids=target),
        make_episode(episode_id="e3", signal="energy", kind="starts_past", phase="answer", t_cross=0,
                     prompt_ids=[31, 32], target_ids=target),
    ]


def test_batched_evaluation_writes_the_records_of_the_sequential_one(tiny_model, tok, ms, tmp_path, monkeypatch):
    # Batch-1 generation keeps the GPU at a few percent of its capacity; batching must not change a record.
    import prio.eval.run as run_module
    from helpers import arm
    from prio.channel import ChannelModel

    model = ChannelModel(tiny_model, 6, delivery="wide", projector="levels")
    arm(model)
    while len(tok.words) < 100:  # every id of the tiny model must decode
        tok.encode(f"filler{len(tok.words)}")
    eps = _mixed_episodes()
    conditions = ("on", "zeroed")
    one = run_module.evaluate(model, tok, ms, CORE_SIGNALS, eps, conditions, out_path=tmp_path / "one" / "r.jsonl",
                              log_every=0)

    sizes, real_batch = [], run_module.generate_batch
    monkeypatch.setattr(run_module, "generate_batch", lambda m, prompts, *a, **k: (sizes.append(len(prompts)),
                                                                                    real_batch(m, prompts, *a, **k))[1])
    monkeypatch.setattr(run_module, "generate", None)  # batch > 1 never takes the sequential path
    path = tmp_path / "two" / "r.jsonl"
    first = run_module.evaluate(model, tok, ms, CORE_SIGNALS, eps[:2], conditions, out_path=path, log_every=0,
                                batch_size=2)
    assert first == [r for r in one if r.episode_id in ("e0", "e1")] and max(sizes) == 2
    sizes.clear()  # interrupted after two episodes: the rest only
    two = run_module.evaluate(model, tok, ms, CORE_SIGNALS, eps, conditions, out_path=path, log_every=0, batch_size=3)
    assert two == one and sum(sizes) == len(one) - len(first) and max(sizes) == 3

    read = lambda folder, name: sorted((tmp_path / folder / name).read_text(encoding="utf-8").splitlines())
    assert read("one", "r.jsonl") == read("two", "r.jsonl")
    assert read("one", "generations.jsonl") == read("two", "generations.jsonl")


def test_batched_preservation_runs_judge_like_the_sequential_ones(tiny_model, tok, ms):
    import prio.eval.run as run_module
    from helpers import arm
    from prio.channel import ChannelModel

    model = ChannelModel(tiny_model, 6, delivery="lanes", projector="levels")
    arm(model)
    while len(tok.words) < 100:
        tok.encode(f"filler{len(tok.words)}")
    prompts, answers = [[10, 11, 12], [13] * 7, [14, 15], [16, 17, 18, 19]], [1.0, 2.0, 3.0, 4.0]
    one = run_module.tuned_free_runs(model, tok, ms, CORE_SIGNALS, prompts, answers, max_new_tokens=40)
    three = run_module.tuned_free_runs(model, tok, ms, CORE_SIGNALS, prompts, answers, max_new_tokens=40, batch_size=3)
    assert three == one and len(one) == 4
