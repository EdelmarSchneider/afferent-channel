"""Evaluation modes of prio.eval.run beyond the standard one: harness mode (row S4), harness-rule
arms (B1), --horizon override, base model through the channel loop for preservation (R11)."""

import json

import numpy as np
import pytest

from helpers import CALL, enc, make_episode
from prio.generate import GenResult
from prio.signals import CORE_SIGNALS

TARGET = "a b c d e f g h i j"


def _variant(ep, notice: str, **kw):
    """`ep` with `notice` inserted right after its crossing token, as the WP3 variants write it:
    harness span over the notice, t_cross on the last notice token, rows copied."""
    words = enc(notice)
    at = ep.t_cross + 1
    target = ep.target_ids[:at] + words + ep.target_ids[at:]
    row = len(ep.prompt_ids) + at
    signals = np.concatenate([ep.signals[:row], np.repeat(ep.signals[row - 1:row], len(words), axis=0), ep.signals[row:]])
    fields = dict(target_ids=target, signals=signals, t_cross=at + len(words) - 1, harness_spans=[[at, at + len(words)]],
                  t_decision=None if ep.t_decision is None else ep.t_decision + len(words))
    fields.update(kw)
    from dataclasses import replace

    return replace(ep, **fields)


def _originals():
    ev = make_episode(episode_id="ev", signal="ctx", kind="event", phase="answer", t_cross=3, target_ids=enc(TARGET))
    ga = make_episode(episode_id="ga", signal="net", kind="gating_event", phase="answer", t_cross=1, t_decision=6,
                      target_ids=enc(TARGET))
    sp = make_episode(episode_id="sp", signal="energy", kind="starts_past", phase="thinking", t_cross=0,
                      target_ids=enc(TARGET))
    return [ev, ga, sp]


def _files():
    ev, ga, sp = _originals()
    files = {
        "event": [_variant(ev, "[status ctx=0.95]"), _variant(ga, "[status net=0.00]"), _variant(sp, "[status energy=0.04]")],
        "quiet": [ev, ga, sp],
        "near_miss": [_variant(ev, "[status ctx=0.87]"), _variant(ga, "[status net=1.00]")],
        "swapped": [_variant(ev, "[status energy=0.04]", signal="energy")],
    }
    return files


def _scripted(seen):
    """Names the context and compacts it whenever `ctx=0.95]` is in the prompt; plain words otherwise."""
    trigger = enc("ctx=0.95]")[0]
    reaction = enc("my context is almost full " + CALL % ("compact_context", "{}"))

    def generate(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **_):
        prompt = prompt_ids[0].tolist()
        seen.append((prompt, max_new_tokens, prompt_signals[0].cpu().numpy(), model.enabled))
        ids = reaction if trigger in prompt else enc("plain words")
        return GenResult(list(ids), np.zeros((len(ids), 6), dtype=np.float32), "max_new_tokens")

    return generate


# --- harness mode ----------------------------------------------------------------------------

def test_harness_files_follow_the_variant_folder_convention(tmp_path):
    from prio.eval.run import harness_files

    variant, original = tmp_path / "n6-notice", tmp_path / "n6"
    variant.mkdir()
    original.mkdir()
    for name in ("test.jsonl", "test_near_miss.jsonl"):
        (variant / name).write_text("", encoding="utf-8")
    (original / "test.jsonl").write_text("", encoding="utf-8")
    files = harness_files(variant, original)
    assert files == {"event": variant / "test.jsonl", "quiet": original / "test.jsonl",
                     "near_miss": variant / "test_near_miss.jsonl"}  # no swap file: no swapped pairing
    (variant / "test_quiet.jsonl").write_text("", encoding="utf-8")
    (variant / "test_swap.jsonl").write_text("", encoding="utf-8")
    files = harness_files(variant, None)
    assert files["quiet"] == variant / "test_quiet.jsonl" and files["swapped"] == variant / "test_swap.jsonl"
    other = tmp_path / "x.jsonl"
    assert harness_files(variant, original, {"swapped": other})["swapped"] == other


def test_harness_mode_pairs_each_event_with_quiet_near_miss_and_swapped_text(tiny_model, tok, ms, monkeypatch):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    seen = []
    monkeypatch.setattr(run_module, "generate", _scripted(seen))
    records = run_module.evaluate_harness(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, _files(), log_every=0)
    runs = {}
    for r in records:
        runs.setdefault(r.episode_id, []).append((r.condition, r.paired))
    assert runs["ev"] == [("on", "event"), ("on", "quiet"), ("on", "near_miss"), ("on", "swapped")]
    assert runs["ga"] == [("on", "event"), ("on", "quiet"), ("on", "near_miss")]
    assert runs["sp"] == [("on", "event")]  # the other kinds run their stored file only

    p = [1, 2, 3]
    assert seen[0][0] == p + enc("a b c d [status ctx=0.95]")  # crossing token + notice: t_cross moved
    assert seen[1][0] == p + enc("a b c d")  # quiet: same prefix, nothing inserted
    assert seen[2][0] == p + enc("a b c d [status ctx=0.87]")
    assert seen[3][0] == p + enc("a b c d [status energy=0.04]")
    assert seen[4][0] == p + enc("a b [status net=0.00] c d e f")  # tool decision moved with the notice
    assert seen[5][0] == p + enc("a b c d e f")
    assert [s[1] for s in seen[:4]] == [run_module.REACTION_HORIZON] * 4
    assert {s[1] for s in seen[4:7]} == {run_module.GATING_HORIZON}
    # Quiet runs get a fresh nominal trajectory: nothing past its limit.
    assert not any(s.past_limit(float(v)) for i, s in enumerate(CORE_SIGNALS) for v in seen[1][2][:, i])

    by = {(r.episode_id, r.paired): r for r in records}
    assert by[("ev", "event")].event_occurred and by[("ev", "event")].reaction_signal == "ctx"
    assert not by[("ev", "quiet")].event_occurred and by[("ev", "quiet")].reaction_signal is None
    assert not by[("ev", "near_miss")].event_occurred
    swap = by[("ev", "swapped")]
    assert swap.kind == "swap" and swap.signal == "energy" and swap.event_occurred and swap.reaction_signal is None


def test_harness_mode_horizon_override_resume_and_generations(tiny_model, tok, ms, monkeypatch, tmp_path):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    seen = []
    monkeypatch.setattr(run_module, "generate", _scripted(seen))
    model, path = ChannelModel(tiny_model, 6), tmp_path / "records.partial.jsonl"
    first = run_module.evaluate_harness(model, tok, ms, CORE_SIGNALS, _files(), out_path=path, log_every=0, horizon=512)
    assert {s[1] for s in seen} == {512} and len(seen) == 8
    gens = [json.loads(line) for line in (tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(g["episode_id"], g["paired"]) for g in gens] == [(r.episode_id, r.paired) for r in first]
    assert gens[3]["signal"] == "energy"
    seen.clear()
    again = run_module.evaluate_harness(model, tok, ms, CORE_SIGNALS, _files(), out_path=path, log_every=0, horizon=512)
    assert not seen and again == first


def test_fixed_horizon_reaches_the_standard_evaluation_and_is_undone(tiny_model, tok, ms, monkeypatch):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    seen = []
    monkeypatch.setattr(run_module, "generate", _scripted(seen))
    ev = _originals()[0]
    with run_module.fixed_horizon(512):
        run_module.evaluate(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, [ev], ("on",), log_every=0)
    assert {s[1] for s in seen} == {512}
    assert run_module._horizon(True, "ctx") == run_module.GATING_HORIZON
    with run_module.fixed_horizon(None):  # no flag: the per-kind horizons
        assert run_module._horizon(False, "time") == run_module.TIME_HORIZON


def test_new_flags_at_their_defaults_are_not_recorded():
    # eval_meta.json of a standard evaluation must stay as it was before these flags existed.
    import argparse
    from pathlib import Path

    from prio.eval.run import _recorded_args, add_mode_arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    add_mode_arguments(parser)
    assert _recorded_args(parser.parse_args(["--run", "r"])) == {"run": "r"}
    rec = _recorded_args(parser.parse_args(["--run", "r", "--mode", "harness", "--horizon", "512"]))
    assert rec == {"run": "r", "mode": "harness", "horizon": 512}


def _stub_run(monkeypatch, run_module, tiny_model, tok, ms, tmp_path):
    from prio.channel import ChannelModel

    run = tmp_path / "run"
    run.mkdir()
    (run / "train_log.json").write_text('{"model": "fake"}', encoding="utf-8")
    monkeypatch.setattr(run_module, "load_run", lambda model_id, out_dir, n: (ChannelModel(tiny_model, n), tok, ms))
    return run


def test_main_harness_mode_end_to_end(tiny_model, tok, ms, monkeypatch, tmp_path):
    import prio.eval.run as run_module
    from prio.datagen.episodes import write_jsonl

    seen = []
    monkeypatch.setattr(run_module, "generate", _scripted(seen))
    run = _stub_run(monkeypatch, run_module, tiny_model, tok, ms, tmp_path)
    files, variant, original = _files(), tmp_path / "n6-notice", tmp_path / "n6"
    variant.mkdir()
    original.mkdir()
    write_jsonl(variant / "test.jsonl", files["event"])
    write_jsonl(variant / "test_near_miss.jsonl", files["near_miss"])
    write_jsonl(tmp_path / "swap.jsonl", files["swapped"])
    write_jsonl(original / "test.jsonl", files["quiet"])
    out = tmp_path / "res"
    run_module.main(["--run", str(run), "--data", str(variant), "--out", str(out), "--mode", "harness",
                     "--quiet-data", str(original), "--pairing-file", f"swapped={tmp_path / 'swap.jsonl'}"])
    assert len(seen) == 8 and (out / "summary.md").exists()
    meta = json.loads((out / "eval_meta.json").read_text(encoding="utf-8"))["invocations"][-1]
    assert meta["args"]["mode"] == "harness" and "finished" in meta
    assert len((out / "records.partial.jsonl").read_text(encoding="utf-8").splitlines()) == 8


def test_main_harness_mode_base_condition_runs_the_untouched_model(tiny_model, tok, ms, monkeypatch, tmp_path):
    # B3z: the zero-shot arm reads the notices with adapter and channel off (condition "base").
    import prio.eval.run as run_module
    from prio.datagen.episodes import write_jsonl

    seen = []
    monkeypatch.setattr(run_module, "generate", _scripted(seen))
    run = _stub_run(monkeypatch, run_module, tiny_model, tok, ms, tmp_path)
    variant = tmp_path / "n6-notice-policy"
    variant.mkdir()
    write_jsonl(variant / "test.jsonl", _files()["event"])
    out = tmp_path / "res"
    run_module.main(["--run", str(run), "--data", str(variant), "--out", str(out), "--mode", "harness",
                     "--conditions", "base"])
    lines = (out / "records.partial.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and {json.loads(line)["condition"] for line in lines} == {"base"}
    assert seen and not any(enabled for *_, enabled in seen)
    assert "## Condition: base" in (out / "summary.md").read_text(encoding="utf-8")
    meta = json.loads((out / "eval_meta.json").read_text(encoding="utf-8"))["invocations"][-1]
    assert meta["args"]["conditions"] == "base"


def test_main_rules_mode_end_to_end(tiny_model, tok, ms, monkeypatch, tmp_path):
    import prio.eval.run as run_module
    from prio.datagen.episodes import write_jsonl
    from prio.tasks import Task

    def generate(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **_):
        ids = enc("so the total is 42 . <eos>")
        return GenResult(ids, np.zeros((len(ids), 6), dtype=np.float32), "eos")

    monkeypatch.setattr(run_module, "generate", generate)
    monkeypatch.setattr(run_module, "dataset_tasks", lambda data: {"math-1": Task("math-1", "math", "q", "42", "test")})
    run = _stub_run(monkeypatch, run_module, tiny_model, tok, ms, tmp_path)
    data = tmp_path / "n6"
    data.mkdir()
    write_jsonl(data / "test.jsonl", _rule_episodes())
    gens = tmp_path / "generations.jsonl"
    gens.write_text(json.dumps({"episode_id": "th", "condition": "on", "paired": "event", "stop_reason": "eos",
                                "token_ids": enc("budget gone </think> the total is 42 . <eos>")}) + "\n", encoding="utf-8")
    out = tmp_path / "b1"
    run_module.main(["--run", str(run), "--data", str(data), "--out", str(out), "--mode", "rules",
                     "--rule-generations", str(gens)])
    s = json.loads((out / "rules_summary.json").read_text(encoding="utf-8"))
    assert s["groups"]["own|on|think"]["success_rate"] == 1.0 and "think|forced_close|base" in s["comparisons"]


# --- B1 harness rules ------------------------------------------------------------------------

def test_judge_outcome_reads_the_answer_after_thinking(tok, ms):
    from prio.eval.run import judge_outcome

    full = judge_outcome(enc("x y </think> so the total is 42 . <eos>"), "eos", 42.0, tok, ms)
    assert full == {"answer_present": True, "answer_tokens": 6, "cut_mid_sentence": False, "correct": True}
    cut = judge_outcome(enc("x </think> the total is still being"), "max_new_tokens", 42.0, tok, ms)
    assert cut["answer_present"] and cut["cut_mid_sentence"] and cut["correct"] is False
    assert judge_outcome(enc("x y z"), "hard_stop", None, tok, ms) == {
        "answer_present": False, "answer_tokens": 0, "cut_mid_sentence": False, "correct": None}
    capped = judge_outcome(enc("</think> " + "w " * 10 + "end . <eos>"), "eos", None, tok, ms, cap=4)
    assert capped["answer_tokens"] == 4 and capped["cut_mid_sentence"]
    call = judge_outcome(enc("</think> " + CALL % ("compact_context", "{}") + " <eos>"), "eos", None, tok, ms)
    assert not call["answer_present"]  # a tool call is no answer


def _rule_episodes():
    think = make_episode(episode_id="th", task_id="math-1", family="math", signal="think", kind="event", phase="thinking",
                         t_cross=2, target_ids=enc("<think> a b c d </think> e f"))
    time_t = make_episode(episode_id="tt", task_id="ex-1", signal="time", kind="event", phase="thinking", t_cross=2,
                          target_ids=enc("<think> a b c d </think> e f"))
    time_a = make_episode(episode_id="ta", task_id="ex-2", signal="time", kind="event", phase="answer", t_cross=6,
                          target_ids=enc("<think> a b </think> the answer is"))
    late = make_episode(episode_id="tl", task_id="ex-3", signal="think", kind="event", phase="thinking", t_cross=5,
                        target_ids=enc("<think> a b c </think> e f"))  # thinking over in the prefix: no rule
    ctx = make_episode(episode_id="cx", task_id="ex-4", signal="ctx", kind="event", phase="thinking", t_cross=2,
                       target_ids=enc("<think> a b c d </think> e f"))
    return [think, time_t, time_a, late, ctx]


def test_rule_arms_force_the_close_or_stop_and_judge_what_follows(tiny_model, tok, ms, monkeypatch, tmp_path):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    seen = []

    def generate(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **_):
        seen.append((prompt_ids[0].tolist(), max_new_tokens, model.enabled, len(prompt_signals[0])))
        ids = enc("so the total is 42 . <eos>")
        return GenResult(ids, np.zeros((len(ids), 6), dtype=np.float32), "eos")

    monkeypatch.setattr(run_module, "generate", generate)
    path = tmp_path / "rules.partial.jsonl"
    model = ChannelModel(tiny_model, 6)
    records = run_module.evaluate_rules(model, tok, ms, CORE_SIGNALS, _rule_episodes(), {"math-1": 42.0},
                                        out_path=path, log_every=0)
    keys = [(r.episode_id, r.condition, r.rule) for r in records]
    assert keys == [("th", "base", "forced_close"), ("th", "zeroed", "forced_close"),
                    ("tt", "none", "hard_stop"), ("tt", "base", "forced_close"), ("tt", "base", "forced_close_cap"),
                    ("tt", "zeroed", "forced_close"), ("tt", "zeroed", "forced_close_cap"),
                    ("ta", "none", "hard_stop")]
    close = ms.think_close
    assert seen[0][0] == [1, 2, 3] + enc("<think> a b") + [close] and seen[0][1] == run_module.RULE_HORIZON
    assert all(not s[2] for s in seen) and all(s[3] == len(s[0]) for s in seen)  # channel off in both arms
    by = {(r.episode_id, r.condition, r.rule): r for r in records}
    assert by[("th", "base", "forced_close")].correct is True and by[("th", "base", "forced_close")].answer_present
    assert by[("tt", "none", "hard_stop")].answer_present is False  # stopped inside thinking: no answer
    stop_answer = by[("ta", "none", "hard_stop")]
    assert stop_answer.answer_present and stop_answer.cut_mid_sentence  # the answer stops mid-sentence
    assert by[("tt", "base", "forced_close")].correct is None  # not a math task

    seen.clear()
    again = run_module.evaluate_rules(model, tok, ms, CORE_SIGNALS, _rule_episodes(), {"math-1": 42.0},
                                      out_path=path, log_every=0)
    assert not seen and again == records

    # [text] variant: Qwen3's own budget sentence before the forced </think>.
    run_module.evaluate_rules(model, tok, ms, CORE_SIGNALS, _rule_episodes()[:1], {}, conditions=("base",),
                              log_every=0, close_text="time is short so i answer now .")
    assert seen[0][0] == [1, 2, 3] + enc("<think> a b time is short so i answer now .") + [close]


def test_own_outcomes_and_gating_rule_from_saved_generations(tok, ms):
    from prio.eval.run import gating_rule_outcome, gating_rule_rows, own_outcomes

    eps = _rule_episodes()
    gens = [{"episode_id": "th", "condition": "on", "paired": "event", "stop_reason": "eos",
             "token_ids": enc("my thinking budget is gone </think> the total is 42 . <eos>")},
            {"episode_id": "th", "condition": "on", "paired": "nominal", "stop_reason": "eos", "token_ids": enc("x")},
            {"episode_id": "cx", "condition": "on", "paired": "event", "stop_reason": "eos", "token_ids": enc("x")}]
    own = own_outcomes(eps, gens, {"math-1": 42.0}, tok, ms)
    assert [(r.episode_id, r.condition, r.rule, r.correct) for r in own] == [("th", "on", "own", True)]

    promised = enc("i will search the web for it " + CALL % ("web_search", '{"query":"q"}'))
    assert gating_rule_outcome(promised, "net", tok, ms) == {"call": "web_search", "swapped": True, "inconsistent": True}
    quiet = enc(CALL % ("web_search", '{"query":"q"}'))
    assert gating_rule_outcome(quiet, "net", tok, ms) == {"call": "web_search", "swapped": True, "inconsistent": False}
    own_alt = enc("i am offline " + CALL % ("local_lookup", '{"query":"q"}'))
    assert gating_rule_outcome(own_alt, "net", tok, ms) == {"call": "local_lookup", "swapped": False, "inconsistent": False}
    heavy = enc("running the full pipeline now " + CALL % ("run_heavy", '{"job":"j"}'))
    assert gating_rule_outcome(heavy, "ram", tok, ms)["inconsistent"]

    gating = make_episode(episode_id="g", signal="net", kind="gating_event", t_cross=1, t_decision=3, target_ids=enc("a b c d"))
    rows = gating_rule_rows([gating], [{"episode_id": "g", "condition": "base", "paired": "event", "stop_reason": "eos",
                                        "token_ids": promised}], tok, ms)
    assert rows == [{"episode_id": "g", "signal": "net", "condition": "base", "call": "web_search", "swapped": True,
                     "inconsistent": True}]


def test_rules_summary_applies_the_b1_success_line(tmp_path):
    from prio.eval.run import RuleRecord, rules_summary, write_rules_summary

    def rec(i, rule, condition, ok):
        return RuleRecord(f"e{i}", "time", "thinking", "explain", condition, rule, "eos", True, 10, not ok, None)

    records = [rec(i, "own", "on", True) for i in range(40)]
    records += [rec(i, "forced_close", "base", i < 20) for i in range(40)]  # 50% complete
    records += [rec(i, "forced_close", "zeroed", i < 38) for i in range(40)]  # 95% complete
    s = rules_summary(records)
    assert s["groups"]["forced_close|base|time"]["success_rate"] == 0.5
    vs_base, vs_zeroed = s["comparisons"]["time|forced_close|base"], s["comparisons"]["time|forced_close|zeroed"]
    assert vs_base["diff_points"] == pytest.approx(50.0) and vs_base["discordant"] == [20, 0] and vs_base["claim"]
    assert vs_zeroed["diff_points"] == pytest.approx(5.0) and not vs_zeroed["claim"]
    write_rules_summary(records, [{"episode_id": "g", "signal": "net", "condition": "base", "call": "web_search",
                                   "swapped": True, "inconsistent": True}], tmp_path)
    md = (tmp_path / "rules_summary.md").read_text(encoding="utf-8")
    assert "forced_close" in md and "McNemar" in md and "inconsistent" in md.lower()
    assert len((tmp_path / "rules.jsonl").read_text(encoding="utf-8").splitlines()) == len(records)


# --- R11: base model through the channel loop ------------------------------------------------

def test_preservation_loop_runs_base_and_trained_through_the_same_loop(tiny_model, tok, ms, monkeypatch, tmp_path):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel
    from prio.tasks import Task

    seen = []

    def generate(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **_):
        seen.append((model.enabled, max_new_tokens))
        ids = enc("x </think> the answer is 42 <eos>")
        return GenResult(ids, np.zeros((len(ids), 6), dtype=np.float32), "eos")

    monkeypatch.setattr(run_module, "generate", generate)
    monkeypatch.setattr(run_module, "build_prompt", lambda tok, task: enc(task.prompt))
    tasks = [Task(f"m{i}", "math", f"q{i} ?", "42" if i else "7", "test") for i in range(3)]
    items = tmp_path / "preservation_items.jsonl"
    s = run_module.preservation_loop(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, tasks, n=3, items_path=items)
    assert seen == [(False, 1024), (True, 1024)] * 3  # base (channel and adapter off), then trained, per task
    assert s["n"] == 3 and s["base_accuracy"] == pytest.approx(2 / 3) and s["accuracy_difference"] == 0.0
    rows = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines()]
    assert [(r["task_id"], r["condition"], r["correct"]) for r in rows] == [
        ("m0", "base", False), ("m0", "on", False), ("m1", "base", True), ("m1", "on", True),
        ("m2", "base", True), ("m2", "on", True)]
