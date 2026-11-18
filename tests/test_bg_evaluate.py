import json

import numpy as np
import pytest

import prio.bg.evaluate as evaluate_module
from helpers import FAKE_FMT, enc, event_plan
from prio.bg.assemble import assemble, twin_plan
from prio.bg.evaluate import evaluate_bg, run_episode, summarize_bg
from prio.bg.harness import SimHarness
from prio.bg.score import BgRecord
from prio.channel import ChannelModel
from prio.datagen.episodes import Episode
from prio.generate import GenResult


class Script:
    """Stand-in for `generate`: returns the scripted token lists in order, feeding signal_fn as
    `generate` does, and remembers what it was given."""

    def __init__(self, outputs, respect_budget=False, model_log=None):
        self.outputs = list(outputs)
        self.respect_budget = respect_budget
        self.calls = 0
        self.prompt_lengths = []
        self.prompt_rows = []
        self.enabled = []

    def __call__(self, model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **kw):
        self.calls += 1
        self.enabled.append(getattr(model, "enabled", None))
        self.prompt_lengths.append(prompt_ids.shape[1])
        self.prompt_rows.append(prompt_signals[0].cpu().numpy())
        tokens = list(self.outputs.pop(0)) if self.outputs else enc("<eos>")
        if self.respect_budget:
            tokens = tokens[:max_new_tokens]
        rows = [signal_fn(step, tokens[:step + 1]) for step in range(len(tokens))]
        stop = "eos" if tokens and tokens[-1] in eos_ids else "max_new_tokens"
        return GenResult(tokens, np.asarray(rows, dtype=np.float32).reshape(len(tokens), -1), stop)


def scripted(monkeypatch, outputs, **kw):
    script = Script(outputs, **kw)
    monkeypatch.setattr(evaluate_module, "generate", script)
    return script


UNWATCH = FAKE_FMT.encode(FAKE_FMT.calls_text([("unwatch", {"task_id": "t2"})]))


def test_loop_executes_tool_rounds_and_feeds_harness_rows(monkeypatch, tok, ms):
    ep = assemble(event_plan("done"), FAKE_FMT, tok, ms)
    script = scripted(monkeypatch, [enc("The report download has finished . ") + UNWATCH + enc("<eos>"),
                                    enc("back to it . <eos>")])
    run = run_episode(None, FAKE_FMT, tok, ms, ep, SimHarness.from_episode(ep), ep.t_cross + 1, 256, "on")
    assert [s[0] for s in run.segments] == ["model", "harness", "model"]
    a, b = run.segments[1][1:]
    assert run.token_ids[a:b] == FAKE_FMT.segment([{"ok": True}])
    n0 = len(ep.prompt_ids) + ep.t_cross + 1
    fresh = SimHarness.from_episode(ep)
    second = script.prompt_rows[1]
    np.testing.assert_allclose(second[n0 + a:n0 + b], np.stack([fresh.row(n0 + p) for p in range(a, b)]))
    assert run.stop == "eos"


def test_prefix_rows_come_from_the_harness(monkeypatch, tok, ms):
    ep = assemble(event_plan("done"), FAKE_FMT, tok, ms)
    script = scripted(monkeypatch, [enc("fine . <eos>")])
    run_episode(None, FAKE_FMT, tok, ms, ep, SimHarness.from_episode(ep), ep.t_cross + 1, 64, "on")
    n = len(ep.prompt_ids) + ep.t_cross + 1
    np.testing.assert_allclose(script.prompt_rows[0], ep.signals[:n])


def test_loop_stops_at_budget_without_eos(monkeypatch, tok, ms):  # Review Focus 5
    ep = assemble(event_plan("done"), FAKE_FMT, tok, ms)
    scripted(monkeypatch, [enc("word " * 64)], respect_budget=True)
    run = run_episode(None, FAKE_FMT, tok, ms, ep, SimHarness.from_episode(ep), ep.t_cross + 1, 64, "on")
    assert run.stop == "budget" and len(run.token_ids) == 64


def test_budget_counts_only_model_tokens(monkeypatch, tok, ms):
    ep = assemble(event_plan("done"), FAKE_FMT, tok, ms)
    first = enc("The report download has finished . ") + UNWATCH + enc("<eos>")
    scripted(monkeypatch, [first, enc("word " * 100)], respect_budget=True)
    run = run_episode(None, FAKE_FMT, tok, ms, ep, SimHarness.from_episode(ep), ep.t_cross + 1, len(first) + 10, "on")
    model_tokens = sum(b - a for kind, a, b in run.segments if kind == "model")
    assert model_tokens == len(first) + 10 and run.stop == "budget"


def _tiny_channel(tiny_model):
    return ChannelModel(tiny_model, 6, delivery="wide", projector="levels")


def _pair(tok, ms):
    plan = event_plan("done")
    return [assemble(plan, FAKE_FMT, tok, ms), assemble(twin_plan(plan), FAKE_FMT, tok, ms)]


def test_zeroed_condition_turns_the_channel_off(monkeypatch, tmp_path, tok, ms, tiny_model):
    script = scripted(monkeypatch, [enc("fine . <eos>")] * 20)
    records = evaluate_bg(_tiny_channel(tiny_model), FAKE_FMT, tok, ms, _pair(tok, ms), tmp_path, base_n=0)
    conditions = [r.condition for r in records]
    assert conditions.count("zeroed") == 1 and "swap" in conditions
    assert script.enabled[conditions.index("zeroed")] is False
    assert script.enabled[conditions.index("on")] is True
    assert script.enabled[conditions.index("swap")] is True          # final review C2: swap runs with the channel on


def test_evaluate_resumes_from_partial_records(tmp_path, monkeypatch, tok, ms, tiny_model):
    model = _tiny_channel(tiny_model)
    scripted(monkeypatch, [enc("fine . <eos>")] * 20)
    first = evaluate_bg(model, FAKE_FMT, tok, ms, _pair(tok, ms), tmp_path, base_n=0)
    again = scripted(monkeypatch, [])
    second = evaluate_bg(model, FAKE_FMT, tok, ms, _pair(tok, ms), tmp_path, base_n=0)
    assert again.calls == 0 and len(second) == len(first)
    assert len((tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()) == len(first)


def synthetic_records():
    def event(i, reacted, family="explain", correct=None):
        return BgRecord(f"e{i}", "event", "on", event_kind="done", phase="answer", held_out=i % 2 == 1, n_watched=2,
                        near_miss=False, family=family, reacted=reacted, right_task=reacted, commands_ok=reacted,
                        delay=2 if reacted else None, unnamed_call=False, report_ok=True, closing_ok=True,
                        resumed=True if family == "explain" else None, math_correct=correct)

    def twin(i, alarm, family="explain", correct=None, near=False):
        return BgRecord(f"e{i}-twin", "twin", "on", event_kind="done", phase="answer", held_out=i % 2 == 1,
                        n_watched=2, near_miss=near, family=family, false_alarm=alarm, math_correct=correct)

    return [event(0, True), event(1, True), event(2, True, "math", True), event(3, False, "math", False),
            twin(0, False), twin(1, True, near=True), twin(2, False, "math", True), twin(3, False, "math", True)]


def test_summary_keys_and_rates():
    s = summarize_bg(synthetic_records())
    assert s["kind"] == "bg"
    g = s["groups"]["on|all|all"]
    assert g["n"] == 4 and g["reaction_rate"] == 0.75 and len(g["reaction_ci"]) == 2
    assert s["groups"]["on|all|held_out"]["n"] == 2
    assert s["groups"]["on|twin|all"]["false_alarm_rate"] == 0.25
    assert s["groups"]["on|near_miss|all"]["n"] == 1
    m = s["resumption"]["math"]
    assert {"difference", "difference_ci", "n_pairs"} <= set(m) and m["n_pairs"] == 2 and m["difference"] == -0.5


def _small_ids(ep: Episode) -> Episode:
    """The same episode with every non-special id folded under 100 (the tiny model's vocabulary)."""
    fold = lambda ids: [i if i < 6 else 6 + (i - 6) % 94 for i in ids]
    ep.prompt_ids, ep.target_ids = fold(ep.prompt_ids), fold(ep.target_ids)
    return ep


def test_end_to_end_on_tiny_model(tiny_model, tmp_path, tok, ms):
    model = _tiny_channel(tiny_model)
    episodes = [_small_ids(e) for e in _pair(tok, ms)]
    budgets = {k: 6 for k in evaluate_module.BUDGETS}
    records = evaluate_bg(model, FAKE_FMT, tok, ms, episodes, tmp_path, base_n=0, budgets=budgets)
    evaluate_module.write_bg_summary(records, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "bg" and {r.condition for r in records} == {"on", "zeroed", "swap"}
    assert (tmp_path / "summary.md").exists() and (tmp_path / "records.jsonl").exists()


def test_event_budget_covers_the_reference_continuation(tok, ms):  # final review I9
    from helpers import fake_run, fixed_tasks
    from prio.bg.assemble import BgEvent, BgPlan

    run = fake_run(answer=" ".join(f"a{i} ." for i in range(150)))
    cut = run.target_ids.index(tok.convert_tokens_to_ids("."), run.target_ids.index(ms.think_close))
    ep = assemble(BgPlan("e", run, "test", fixed_tasks()[:1], "yes", BgEvent("done", 0, cut, "answer")), FAKE_FMT, tok, ms)
    harness = sum(b - a for a, b in ep.harness_spans if a > ep.t_cross)
    remaining = len(ep.target_ids) - (ep.t_cross + 1) - harness
    budget = evaluate_module._budget(evaluate_module.BUDGETS, "on", ep)
    assert 1.25 * remaining <= budget <= evaluate_module.BUDGETS["cap"]
    long = assemble(BgPlan("f", fake_run(answer=" ".join(f"a{i} ." for i in range(600))), "test", fixed_tasks()[:1], "yes",
                           BgEvent("done", 0, cut, "answer")), FAKE_FMT, tok, ms)
    assert evaluate_module._budget(evaluate_module.BUDGETS, "on", long) == evaluate_module.BUDGETS["cap"]


class TurnModel:
    """Stand-in for `generate` and `generate_batch` that answers from the context alone: a tool call
    while the context holds none of its own calls, then a plain closing turn. Keeps every prompt
    it was given (ids, rows) and the batch sizes."""

    def __init__(self):
        self.prompts, self.sizes = [], []

    def _turn(self, ids, rows, signal_fn, max_new_tokens, eos_ids):
        self.prompts.append((tuple(ids), np.asarray(rows, dtype=np.float32).tobytes()))
        own_call = UNWATCH[1]  # the call's JSON word
        tokens = enc("back to it . <eos>") if own_call in ids else enc("The report download has finished . ") + UNWATCH + enc("<eos>")
        tokens = tokens[:max_new_tokens]
        rows = [signal_fn(step, tokens[:step + 1]) for step in range(len(tokens))]
        stop = "eos" if tokens and tokens[-1] in eos_ids else "max_new_tokens"
        return GenResult(tokens, np.asarray(rows, dtype=np.float32).reshape(len(tokens), -1), stop)

    def generate(self, model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **kw):
        self.sizes.append(1)
        return self._turn(prompt_ids[0].tolist(), prompt_signals[0].cpu().numpy(), signal_fn, max_new_tokens, eos_ids)

    def generate_batch(self, model, prompts, prompt_signals, signal_fns, max_new_tokens, eos_ids, **kw):
        self.sizes.append(len(prompts))
        return [self._turn(list(p), np.asarray(s), f, n, eos_ids)
                for p, s, f, n in zip(prompts, prompt_signals, signal_fns, max_new_tokens)]


def _two_pairs(tok, ms):
    plans = [event_plan("done"), event_plan("error", seed=1, n_tasks=3, task_index=2)]
    return [assemble(p, FAKE_FMT, tok, ms) for plan in plans for p in (plan, twin_plan(plan))]


def test_batched_harness_loop_gives_the_records_of_the_sequential_one(monkeypatch, tmp_path, tok, ms, tiny_model):
    # Several episodes' tool rounds advance together; each keeps its own harness, rows and budget.
    model, episodes = _tiny_channel(tiny_model), _two_pairs(tok, ms)
    one, three = TurnModel(), TurnModel()
    monkeypatch.setattr(evaluate_module, "generate", one.generate)
    first = evaluate_bg(model, FAKE_FMT, tok, ms, episodes, tmp_path / "one", base_n=1)
    assert any(kind == "harness" for line in (tmp_path / "one" / "generations.jsonl").read_text(encoding="utf-8").splitlines()
               for kind, _, _ in json.loads(line)["segments"])  # tool rounds really happened

    monkeypatch.setattr(evaluate_module, "generate", None)  # batch > 1 never takes the sequential path
    monkeypatch.setattr(evaluate_module, "generate_batch", three.generate_batch)
    partial = evaluate_bg(model, FAKE_FMT, tok, ms, episodes[:2], tmp_path / "three", base_n=1, batch_size=2)
    assert partial == [r for r in first if r.episode_id in {episodes[0].episode_id, episodes[1].episode_id}]
    again = evaluate_bg(model, FAKE_FMT, tok, ms, episodes, tmp_path / "three", base_n=1, batch_size=3)
    assert again == first and max(three.sizes) == 3
    assert sorted(three.prompts) == sorted(one.prompts)  # same contexts and rows, round by round
    read = lambda folder, name: sorted((tmp_path / folder / name).read_text(encoding="utf-8").splitlines())
    assert read("one", "generations.jsonl") == read("three", "generations.jsonl")
    assert read("one", "records.partial.jsonl") == read("three", "records.partial.jsonl")


def test_batched_end_to_end_on_tiny_model(tiny_model, tmp_path, tok, ms):
    from helpers import arm

    model = _tiny_channel(tiny_model)
    arm(model)
    episodes = [_small_ids(e) for e in _two_pairs(tok, ms)]
    budgets = {k: 8 for k in evaluate_module.BUDGETS}
    one = evaluate_bg(model, FAKE_FMT, tok, ms, episodes, tmp_path / "one", base_n=0, budgets=budgets)
    two = evaluate_bg(model, FAKE_FMT, tok, ms, episodes, tmp_path / "two", base_n=0, budgets=budgets, batch_size=2)
    assert two == one
    read = lambda folder: sorted((tmp_path / folder / "generations.jsonl").read_text(encoding="utf-8").splitlines())
    assert read("one") == read("two")


def test_identification_is_split_by_number_of_watched_tasks():  # final review I10
    records = synthetic_records()
    records[0].n_watched = 1
    s = summarize_bg(records)
    assert s["groups"]["on|all|single"]["n"] == 1 and s["groups"]["on|all|multi"]["n"] == 3
