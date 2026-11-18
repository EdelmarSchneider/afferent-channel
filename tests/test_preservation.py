"""R11 capability preservation (preregistration R11, reading fixed in amendment A4): prio.eval.preservation."""

import json

import numpy as np
import pytest

from helpers import CALL, FAKE_MS, FAKE_TOK, enc
from prio.eval import preservation as P
from prio.generate import GenResult
from prio.signals import CORE_SIGNALS
from prio.tasks import Task

WEB = CALL % ("web_search", '{"query":"q"}')


def gen(text, stop="eos"):
    ids = enc(text)
    return GenResult(ids, np.zeros((len(ids), 6), dtype=np.float32), stop)


# --- explain-format check ---------------------------------------------------------------------

@pytest.mark.parametrize("text, stop, keeps, answer_tokens", [
    ("<think> a plan </think> a plain answer <eos>", "eos", True, 3),
    ("a plain answer without thinking <eos>", "eos", True, 5),
    ("<think> a plan </think> an answer that runs into the limit", "max_new_tokens", True, 7),  # length is reported apart
    ("<think> a plan </think> <eos>", "eos", False, 0),  # no answer
    ("<think> a plan that never closes", "max_new_tokens", False, 0),
    (f"<think> a plan </think> {WEB} <eos>", "eos", False, 3),
    (f"<think> a plan </think> some text then {WEB} <eos>", "eos", False, 6),
])
def test_format_is_kept_by_a_closed_thought_and_a_text_answer_without_tools(text, stop, keeps, answer_tokens):
    row = P.judge_format(enc(text), stop, FAKE_TOK, FAKE_MS, CORE_SIGNALS)
    assert row["format"] is keeps and row["ended"] is (stop == "eos")
    assert row["answer_tokens"] == answer_tokens and row["tool_call"] is ("<tool_call>" in text)


def test_format_is_lost_when_the_turn_names_a_signal(monkeypatch):
    ids = enc("<think> a plan </think> a plain answer <eos>")
    assert P.judge_format(ids, "eos", FAKE_TOK, FAKE_MS, CORE_SIGNALS)["reaction"] is False
    monkeypatch.setattr(P, "first_reaction", lambda *a, **k: object())
    row = P.judge_format(ids, "eos", FAKE_TOK, FAKE_MS, CORE_SIGNALS)
    assert row["reaction"] is True and row["format"] is False


# --- the pre-registered line --------------------------------------------------------------------

def test_preserved_needs_minus_two_points_and_a_lower_bound_of_minus_five():
    assert P.preserved(-0.02, -0.05) and P.preserved(0.01, -0.03)
    assert not P.preserved(-0.021, -0.03) and not P.preserved(-0.01, -0.051)


def test_paired_gives_both_rates_the_difference_and_the_line():
    base = [True] * 90 + [False] * 10
    tuned = [True] * 88 + [False] * 12
    s = P.paired(base, tuned)
    assert (s["n"], s["base"], s["trained"]) == (100, 0.90, 0.88) and s["difference"] == pytest.approx(-0.02)
    assert s["ci"][0] <= -0.02 <= s["ci"][1] and s["preserved"] is P.preserved(s["difference"], s["ci"][0])
    same = P.paired(base, base)
    assert same["difference"] == 0.0 and same["ci"] == [0.0, 0.0] and same["preserved"] is True
    worse = P.paired([True] * 100, [True] * 80 + [False] * 20)
    assert worse["difference"] == pytest.approx(-0.20) and worse["preserved"] is False


# --- the runs -----------------------------------------------------------------------------------

@pytest.fixture
def fake_loop(monkeypatch):
    calls = []

    def run_batch(model, prompts, values, horizons, eos_ids, signal_fns, harness=None):
        calls.append((model.enabled, [tuple(p) for p in prompts], list(horizons), [v.shape for v in values]))
        return [gen("<think> a plan </think> the answer is 42 <eos>") for _ in prompts]

    monkeypatch.setattr(P, "_run_batch", run_batch)
    monkeypatch.setattr(P, "build_prompt", lambda tok, task: enc(task.prompt))
    return calls


def _tasks():
    math = [Task(f"m{i}", "math", f"question {i} ?", "42" if i else "7", "test") for i in range(5)]
    explain = [Task(f"e{i}", "explain", f"explain topic {i}", None, "test") for i in range(3)]
    return math, explain


def test_run_puts_base_and_trained_through_the_same_batched_loop(tiny_model, tok, ms, fake_loop, tmp_path):
    from prio.channel import ChannelModel

    math, explain = _tasks()
    s = P.run(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, math, explain, tmp_path, batch_size=2)
    # per family: the base model first (channel and adapter off), then the trained one, in batches of two
    assert [c[0] for c in fake_loop] == [False] * 3 + [True] * 3 + [False] * 2 + [True] * 2
    assert all(h == 1024 for c in fake_loop for h in c[2])
    assert [c[1] for c in fake_loop[:3]] == [c[1] for c in fake_loop[3:6]]  # the same prompts for both
    assert fake_loop[0][3][0] == (len(enc(math[0].prompt)) + 1024, 6)  # one nominal trajectory per task
    rows = [json.loads(line) for line in (tmp_path / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 16
    assert [(r["task_id"], r["condition"]) for r in rows[:4]] == [("m0", "base"), ("m1", "base"), ("m2", "base"), ("m3", "base")]
    assert all(r["family"] == ("math" if r["task_id"].startswith("m") else "explain") for r in rows)
    assert [r["correct"] for r in rows if r["family"] == "math" and r["condition"] == "on"] == [False, True, True, True, True]
    assert all("correct" not in r and r["format"] for r in rows if r["family"] == "explain")
    assert s["gsm8k"]["n"] == 5 and s["gsm8k"]["base"] == pytest.approx(0.8) and s["gsm8k"]["difference"] == 0.0
    assert s["explain"]["n"] == 3 and s["explain"]["base"] == 1.0 and s["explain"]["preserved"] is True
    assert s["beside"]["math"]["on"]["ended"] == 1.0 and s["beside"]["explain"]["base"]["median_answer_tokens"] == 4


def test_run_keeps_the_text_of_every_turn_to_read_by_hand(tiny_model, tok, ms, fake_loop, tmp_path):
    from prio.channel import ChannelModel

    math, explain = _tasks()
    P.run(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, math, explain, tmp_path, batch_size=2)
    kept = [json.loads(line) for line in (tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    items = [json.loads(line) for line in (tmp_path / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(g["task_id"], g["condition"]) for g in kept] == [(r["task_id"], r["condition"]) for r in items]  # one per item, same order
    assert all("the answer is 42" in g["text"] and "a plan" in g["text"] for g in kept)


def test_run_resumes_from_its_items_file(tiny_model, tok, ms, fake_loop, tmp_path):
    from prio.channel import ChannelModel

    math, explain = _tasks()
    model = ChannelModel(tiny_model, 6)
    first = P.run(model, tok, ms, CORE_SIGNALS, math, explain, tmp_path, batch_size=2)
    fake_loop.clear()
    assert P.run(model, tok, ms, CORE_SIGNALS, math, explain, tmp_path, batch_size=2) == first and fake_loop == []
    items = tmp_path / "items.jsonl"
    lines = items.read_text(encoding="utf-8").splitlines()
    items.write_text("\n".join(lines[:7]) + "\n", encoding="utf-8")  # interrupted after 5 base + 2 trained math runs
    assert P.run(model, tok, ms, CORE_SIGNALS, math, explain, tmp_path, batch_size=2) == first
    assert sum(len(c[1]) for c in fake_loop) == 9 and len(items.read_text(encoding="utf-8").splitlines()) == 16


def test_summary_files_state_the_line_for_both_checks(tiny_model, tok, ms, fake_loop, tmp_path):
    from prio.channel import ChannelModel

    math, explain = _tasks()
    s = P.run(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, math, explain, tmp_path, batch_size=4)
    text = P.write_summary(s, tmp_path, "runs/some-run")
    assert (tmp_path / "preservation_summary.md").read_text(encoding="utf-8") == text
    assert json.loads((tmp_path / "preservation_summary.json").read_text(encoding="utf-8")) == s
    assert "runs/some-run" in text and "GSM8K" in text and "Explain" in text
    assert text.count("preserved") >= 2 and ">= -2 points" in text and "-5" in text
