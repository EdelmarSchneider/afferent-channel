"""Runner and judge of the R12 arms without a forced prefix (prio.r12.stream; analysis plan
Sections 5 and 6): a crossing on the k-th token the model wrote itself, simulated (`free`) or
counted from the generation (`live`, `Ls`); the onset index."""

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import prio.eval.run as run_module
from helpers import CALL, arm
from prio.channel import ChannelModel
from prio.generate import GenResult
from prio.r12 import live as L
from prio.r12 import stream as S
from prio.signals import CORE_SIGNALS
from prio.trajectories import sample_trajectory
from r12_helpers import MS, PieceTok, R12Tok

NAMES = [s.name for s in CORE_SIGNALS]
N_PROMPT = 12
REACTIONS = {
    "ctx": "my context is almost full " + CALL % ("compact_context", "{}") + " <eos>",
    "think": "my thinking budget is almost gone </think> so the answer is short",
    "energy": "my battery is almost empty " + CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}") + " <eos>",
    "time": "I am almost out of time so here it is <eos>",
}
NOTICE = re.compile(r"\[status (\w+)=(\d\.\d\d)\]$")


@pytest.fixture
def tok():
    return R12Tok()


def prompt_of(tok, i: int) -> list[int]:
    ids = tok.encode(f"sys a b c d e f g user task{i} asks assistant")
    assert len(ids) == N_PROMPT
    return ids


def plan(i: int, signal: str, k: int = 20, arm_name: str = "free", family: str = "math") -> S.Plan:
    return S.Plan(arm_name, i, f"r12p-{family}-{i:05d}", family, signal, k, 160 if signal == "time" else 96)


class Scripted:
    """Stand-in for `prio.eval.run.generate`: a model that opens its thinking, writes plain words and
    reacts once: to the first fed row that is past a limit (channel on), or to a past-limit notice at
    the end of its prompt. Per task (the `task<i>` word of the prompt): `close_at` / `end_at` / `early`
    = position in the turn where it closes its thinking / ends the turn / reacts with no reason;
    `think_at` = position where it opens its thinking (0 unless given); `mute` = it never reacts to the state."""

    def __init__(self, tok, options=None, plain="w", prompts=None):
        self.tok, self.options, self.seen, self.plain = tok, options or {}, [], plain
        # Real task prompts (any length): a run's prompt starts with one of them; its options are keyed by that prompt.
        self.known = sorted((list(p) for p in prompts or []), key=len, reverse=True)

    def _task(self, prompt) -> tuple[int, dict]:
        """Length of the task's own prompt and the options of the task."""
        for known in self.known:
            if prompt[:len(known)] == known:
                return len(known), self.options.get(tuple(known), {})
        for i, opt in self.options.items():
            if self.tok.vocab.get(f"task{i}") in prompt:
                return N_PROMPT, opt
        return N_PROMPT, {}

    def __call__(self, model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, prompt_harness=None, **_):
        prompt = prompt_ids[0].tolist()
        n_prompt, opt = self._task(prompt)
        tail = self.tok.decode(prompt[-2:]).strip()
        notice = NOTICE.search(tail)
        written = len(prompt) - n_prompt - (2 if notice else 0)  # tokens of the turn already in the prompt
        self.seen.append(SimpleNamespace(prompt=prompt, horizon=max_new_tokens, enabled=model.enabled,
                                         harness=None if prompt_harness is None else list(prompt_harness),
                                         rows=prompt_signals[0].cpu().numpy(), written=written))
        tokens, rows, queue, stop = [], [], [], "max_new_tokens"
        reacted = bool(opt.get("mute"))  # `mute`: a model that never reacts to the state (a miss)
        if not reacted and notice and CORE_SIGNALS[NAMES.index(notice.group(1))].past_limit(float(notice.group(2))):
            queue, reacted = self.tok.encode(REACTIONS[notice.group(1)]), True
        while len(tokens) < max_new_tokens:
            position = written + len(tokens)
            if not queue:
                past = [s.name for s, v in zip(CORE_SIGNALS, rows[-1]) if s.past_limit(float(v))] if rows and model.enabled else []
                if past and not reacted:
                    queue, reacted = self.tok.encode(REACTIONS[past[0]]), True
                elif position == opt.get("think_at", 0):
                    queue = [MS.think_open]
                elif opt.get("close_at") == position:
                    queue = [MS.think_close]
                elif opt.get("end_at") == position:
                    queue = self.tok.encode("<eos>")
                elif opt.get("early") and opt["early"][1] == position:
                    queue = self.tok.encode(REACTIONS[opt["early"][0]])
                else:
                    queue = self.tok.encode(self.plain)
            tokens.append(queue.pop(0))
            rows.append(np.asarray(signal_fn(len(tokens) - 1, tokens), dtype=np.float32))
            if tokens[-1] in eos_ids:
                stop = "eos"
                break
        return GenResult(tokens, np.stack(rows), stop)


def run(kind, tmp_path, tok, plans, monkeypatch, tiny_model, options=None, **kw):
    script = Scripted(tok, options)
    monkeypatch.setattr(run_module, "generate", script)
    prompts = {p.task_id: prompt_of(tok, p.index) for p in plans}
    runner = S.run_channel if kind == "channel" else S.run_notice
    records = runner(ChannelModel(tiny_model, 6), tok, MS, CORE_SIGNALS, plans, prompts, 0, tmp_path / "out", batch_size=1, **kw)
    return {(r.task_id, r.run_type): r for r in records}, script


def generations(tmp_path):
    lines = (tmp_path / "out" / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    return {(g["task_id"], g["run_type"]): g for g in map(json.loads, lines)}


# --- arm free, channel model ------------------------------------------------------------------------

def test_event_run_is_split_after_its_kth_token_and_judged_from_the_next(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 20), plan(1, "think", 31), plan(2, "energy", 16), plan(3, "time", 96)]
    records, script = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model)
    gens = generations(tmp_path)
    assert len(records) == 12 and len(script.seen) == 12
    for p in plans:
        event, g = records[(p.task_id, "event")], gens[(p.task_id, "event")]
        assert len(g["pre"]) == p.k and g["pre"][0] == MS.think_open and set(g["pre"][1:]) == set(tok.encode("w"))
        assert g["post"][:len(tok.encode(REACTIONS[p.signal]))] == tok.encode(REACTIONS[p.signal])  # reacts with its very next token
        assert (event.occurred, event.why_not, event.hit, event.reaction_signal, event.named) == (True, None, True, p.signal, True)
        assert (event.onset, event.onset_word, event.onset_first_text) == (0, 0, 0)
        assert event.crossing_at == p.k - 1 and event.premature is False and event.phase == "thinking"
        assert (event.n_pre, event.n_post, event.k, event.signal, event.arm, event.model) == (p.k, len(g["post"]), p.k, p.signal, "free", "channel")
        assert event.family == "math" and event.false_alarm is None
        spec, fed = CORE_SIGNALS[NAMES.index(p.signal)], g["rows_milli"][p.signal]
        assert len(fed) == p.k + len(g["post"])  # the fed column of the signal, one value per generated token
        assert spec.past_limit(fed[p.k - 1] / 1000) and not any(spec.past_limit(v / 1000) for v in fed[:p.k - 1])
    assert records[("r12p-math-00003", "event")].stop_reason == "eos"
    assert records[("r12p-math-00001", "event")].stop_reason == "max_new_tokens" and records[("r12p-math-00001", "event")].n_post == 96
    assert records[("r12p-math-00000", "event")].delay == 1  # "my context": named by the second token
    horizons = {(s.horizon, s.enabled) for s in script.seen}
    assert horizons == {(20 + 96, True), (20 + 96, False), (31 + 96, True), (31 + 96, False), (16 + 96, True), (16 + 96, False),
                        (96 + 160, True), (96 + 160, False)}
    assert all(s.prompt == prompt_of(tok, i) and s.harness is None for i, s in zip([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], script.seen))


def test_twin_and_off_runs_are_quiet_and_judged_over_the_whole_run(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 20), plan(1, "time", 40)]
    records, script = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model)
    gens = generations(tmp_path)
    for p in plans:
        twin, off = records[(p.task_id, "twin")], records[(p.task_id, "off")]
        for r in (twin, off):
            assert (r.false_alarm, r.hit, r.occurred, r.reaction_signal, r.premature) == (False, False, False, None, False)
            assert r.n_pre == p.k and r.n_post == p.horizon and r.onset is None
        assert twin.same_prefix is True and off.same_prefix is None and twin.crossing_at is None
        assert off.crossing_at == p.k - 1  # the rows are computed, the channel does not deliver them
        assert gens[(p.task_id, "twin")]["pre"] == gens[(p.task_id, "event")]["pre"]
    assert [s.enabled for s in script.seen] == [True, True, False] * 2  # off = the channel disabled


def test_off_run_of_the_channel_model_runs_with_the_channel_disabled(tmp_path, tok, monkeypatch, tiny_model):
    records, script = run("channel", tmp_path, tok, [plan(0, "energy", 25)], monkeypatch, tiny_model, run_types=("off",))
    assert [s.enabled for s in script.seen] == [False] and list(records) == [("r12p-math-00000", "off")]


def test_twin_and_off_runs_count_any_reaction_as_a_false_alarm(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 30), plan(1, "ctx", 30)]
    options = {0: {"early": ("energy", 50)}, 1: {"early": ("time", 100)}}  # a reaction with nothing to react to
    records, _ = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model, options, run_types=("twin", "off"))
    for run_type in ("twin", "off"):
        late = records[("r12p-math-00000", run_type)]
        assert (late.false_alarm, late.reaction_signal, late.named, late.hit) == (True, "energy", True, False)
        assert (late.alarm_start, late.alarm_at_k, late.alarm_at_close) == (True, True, False)  # a tool call is found from anywhere
        # The stated limit of the frozen classifier: a deadline sentence that ends the turn 70 tokens after
        # token k is at none of the three anchors. Not a false alarm; counted beside as a keyword match.
        far = records[("r12p-math-00001", run_type)]
        assert far.false_alarm is False and far.deadline_near_end is True and late.deadline_near_end is False


# --- where the classifier is anchored (second note on A12) --------------------------------------------------

THINK_REACTION = "my thinking budget is almost gone </think> so the answer is short <eos>"
TIME_REACTION = "</think> I am almost out of time so here it is <eos>"


def open_thinking(tok, n: int) -> list[int]:
    """`n` tokens of open thinking."""
    return [MS.think_open] + tok.encode("w") * (n - 1)


def quiet_run(tok, signal: str, k: int, reaction: str, run_type: str = "twin", arm_name: str = "live") -> S.StreamRecord:
    """A twin or off run in which the model reacts right after token k although nothing crossed."""
    whole = open_thinking(tok, k) + tok.encode(reaction)
    return S.judge(plan(0, signal, k, arm_name), run_type, "channel", whole[:k], whole[k:], "eos", tok, MS, CORE_SIGNALS)


def test_a_reaction_right_after_token_k_is_a_false_alarm_for_every_k(tok):
    for signal, reaction in (("think", THINK_REACTION), ("time", TIME_REACTION), ("ctx", REACTIONS["ctx"])):
        for run_type in ("twin", "off"):
            records = {k: quiet_run(tok, signal, k, reaction, run_type) for k in range(16, 161)}
            assert all(r.false_alarm and r.reaction_signal == signal and r.named for r in records.values()), (signal, run_type)
            assert all(r.alarm_at_k for r in records.values())  # judged at the crossing position, as the event run is
            from_start = [k for k, r in records.items() if r.alarm_start]
            at_close = [k for k, r in records.items() if r.alarm_at_close]
            if signal == "think":  # from the first token the classifier sees the close of the thinking only within 48 tokens
                assert from_start == list(range(16, 42)) and at_close == list(range(16, 161))
            elif signal == "time":  # ... and a deadline only when it is named in the first 60 tokens of a turn of at most 128
                assert from_start == list(range(16, 54)) and at_close == []
            else:  # a tool call is found from anywhere
                assert from_start == list(range(16, 161)) and at_close == []
    event = S.judge(plan(0, "think", 100, "live"), "event", "channel", open_thinking(tok, 100), tok.encode(THINK_REACTION), "eos",
                    tok, MS, CORE_SIGNALS, crossing_at=99)
    assert event.hit is True and (event.alarm_start, event.alarm_at_k, event.alarm_at_close, event.false_alarm) == (None,) * 4


def test_the_close_of_the_thinking_is_an_anchor_wherever_it_falls(tok):
    w = tok.encode("w")
    budget = tok.encode(THINK_REACTION)[:-1]  # names the budget, closes the thinking, goes on
    whole = open_thinking(tok, 60) + budget + w * 136  # the reaction is neither at the first token nor at token k = 120
    record = S.judge(plan(0, "ctx", 120), "twin", "channel", whole[:120], whole[120:], "max_new_tokens", tok, MS, CORE_SIGNALS)
    assert (record.alarm_start, record.alarm_at_k, record.alarm_at_close) == (False, False, True)
    assert (record.false_alarm, record.reaction_signal, record.named) == (True, "think", True)
    plain = open_thinking(tok, 60) + [MS.think_close] + w * 155  # a thinking that just ends names no budget
    quiet = S.judge(plan(0, "ctx", 120), "twin", "channel", plain[:120], plain[120:], "max_new_tokens", tok, MS, CORE_SIGNALS)
    assert (quiet.false_alarm, quiet.alarm_start, quiet.alarm_at_k, quiet.alarm_at_close) == (False, False, False, False)
    # the budget is named within the 47 tokens before the close, or the classifier does not see the reaction
    named = tok.encode("my thinking budget is almost gone")  # "thinking" is token 31
    far = open_thinking(tok, 30) + named + w * 43 + [MS.think_close] + w * 100
    assert far.index(MS.think_close) - 31 == 48
    assert S.judge(plan(0, "ctx", 120), "twin", "channel", far[:120], far[120:], "max_new_tokens", tok, MS, CORE_SIGNALS).false_alarm is False
    near = open_thinking(tok, 30) + named + w * 42 + [MS.think_close] + w * 100
    assert near.index(MS.think_close) - 31 == 47
    assert S.judge(plan(0, "ctx", 120), "twin", "channel", near[:120], near[120:], "max_new_tokens", tok, MS, CORE_SIGNALS).alarm_at_close
    # a run that ended before token k has no crossing position: the first token and the close are its anchors
    short = open_thinking(tok, 60) + tok.encode(THINK_REACTION)
    ended = S.judge(plan(0, "ctx", 120), "off", "channel", short, [], "eos", tok, MS, CORE_SIGNALS)
    assert (ended.alarm_start, ended.alarm_at_k, ended.alarm_at_close, ended.false_alarm) == (False, False, True, True)


def test_a_thinking_budget_reaction_before_token_k_is_premature_wherever_the_thinking_closes(tok):
    w = tok.encode("w")
    budget = tok.encode(THINK_REACTION)[:-1]
    pre = (open_thinking(tok, 60) + budget + w * 80)[:120]
    own = S.judge(plan(0, "think", 120, "live"), "event", "channel", pre, w * 96, "max_new_tokens", tok, MS, CORE_SIGNALS)
    assert (own.premature, own.occurred, own.why_not) == (True, False, "thinking_over")
    # ... and the target's reaction after token k is then never a hit
    other = S.judge(plan(0, "ctx", 120), "event", "channel", pre, tok.encode(REACTIONS["ctx"]), "eos", tok, MS, CORE_SIGNALS,
                    crossing_at=119)
    assert (other.premature, other.occurred, other.hit, other.reaction_signal) == (True, True, False, "ctx")
    clean = S.judge(plan(0, "ctx", 120), "event", "channel", open_thinking(tok, 120), tok.encode(REACTIONS["ctx"]), "eos", tok, MS,
                    CORE_SIGNALS, crossing_at=119)
    assert (clean.premature, clean.hit) == (False, True)
    closed = (open_thinking(tok, 60) + [MS.think_close] + w * 80)[:120]  # the thinking just ends: nothing premature
    assert S.judge(plan(0, "ctx", 120), "event", "channel", closed, tok.encode(REACTIONS["ctx"]), "eos", tok, MS, CORE_SIGNALS,
                   crossing_at=119).hit is True
    # the stated limit: a deadline reaction that starts at token 70 and ends the turn is not seen as premature
    turn = open_thinking(tok, 70) + tok.encode(TIME_REACTION)
    late = S.judge(plan(0, "time", 150, "live"), "event", "channel", turn, [], "eos", tok, MS, CORE_SIGNALS)
    assert (late.premature, late.occurred, late.why_not, late.deadline_near_end) == (False, False, "ended", True)


def test_deadline_near_the_end_is_a_keyword_count_beside_the_classifier(tok):
    w = tok.encode("w")
    p = plan(0, "ctx", 20)
    pre = open_thinking(tok, 20)
    deadline = tok.encode("so the deadline is close and I stop here <eos>")
    cases = {
        "ends on the deadline, far from every anchor": (w * 150 + deadline, "eos", True),
        "names the deadline and goes on": (w * 150 + deadline[:-1], "max_new_tokens", False),
        "ends without naming it": (w * 150 + tok.encode("<eos>"), "eos", False),
        "named more than 128 tokens before the end": (deadline[:-1] + w * 128 + tok.encode("<eos>"), "eos", False),
        "named 128 tokens before the end": (deadline[:-1] + w * 118 + tok.encode("<eos>"), "eos", True),
    }
    for label, (post, stop, expected) in cases.items():
        for run_type in S.RUN_TYPES:
            record = S.judge(p, run_type, "channel", pre, post, stop, tok, MS, CORE_SIGNALS, crossing_at=19)
            assert record.deadline_near_end is expected, (label, run_type)
    far = S.judge(p, "twin", "channel", pre, cases["ends on the deadline, far from every anchor"][0], "eos", tok, MS, CORE_SIGNALS)
    assert far.false_alarm is False  # a keyword match only: not a counted reaction
    records = [S.judge(p, t, "channel", pre, w * 150 + deadline, "eos", tok, MS, CORE_SIGNALS, crossing_at=19) for t in S.RUN_TYPES]
    hit = S.judge(plan(1, "time", 20), "event", "channel", pre, tok.encode(REACTIONS["time"]), "eos", tok, MS, CORE_SIGNALS, crossing_at=19)
    summary = S.stream_summary(records + [hit])
    assert summary["deadline_near_end"] == {"event": 2, "event_not_hit": 1, "twin": 1, "off": 1}


def test_a_live_crossing_on_another_token_than_the_kth_is_excluded_under_its_own_reason(tok):
    p = plan(3, "think", 50, "live")
    pre, reaction = open_thinking(tok, 50), tok.encode(THINK_REACTION)
    on_k = S.judge(p, "event", "channel", pre, reaction, "eos", tok, MS, CORE_SIGNALS, crossing_at=49)
    assert (on_k.occurred, on_k.why_not, on_k.hit) == (True, None, True)
    for where in (48, 50, 51):  # the first past-limit row is on another token: not judged from token k as if it were there
        late = S.judge(p, "event", "channel", pre, reaction, "eos", tok, MS, CORE_SIGNALS, crossing_at=where)
        assert (late.occurred, late.why_not, late.hit, late.onset, late.crossing_at) == (False, "crossing_off_k", False, None, where)
    none = S.judge(p, "event", "channel", pre, reaction, "eos", tok, MS, CORE_SIGNALS, crossing_at=None)
    assert (none.occurred, none.why_not) == (False, "no_crossing")
    closed = S.judge(p, "event", "channel", pre[:20] + [MS.think_close] + pre[21:], reaction, "eos", tok, MS, CORE_SIGNALS, crossing_at=51)
    assert closed.why_not == "thinking_over"  # the run's own reason comes first
    summary = S.stream_summary([on_k, late, none, closed])
    assert summary["event"]["not_occurred"] == {"ended": 0, "thinking_over": 1, "no_crossing": 1, "crossing_off_k": 1}
    assert (summary["event"]["occurred"], summary["event"]["crossing_off_k"]) == (1, 1)
    assert S.WHY_NOT == ("ended", "thinking_over", "no_crossing", "crossing_off_k")


def test_a_late_opening_of_the_thinking_moves_the_live_crossing_off_token_k(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "think", 40, "live"), plan(1, "think", 40, "Ls"), plan(2, "time", 40, "live")]
    options = {i: {"think_at": 2} for i in range(3)}  # two plain tokens, then <think>
    records, _ = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model, options, run_types=("event",))
    for i in (0, 1):  # the budget counts thinking tokens: it reaches 90% two tokens late
        event = records[(f"r12p-math-{i:05d}", "event")]
        assert (event.occurred, event.why_not, event.crossing_at, event.hit) == (False, "crossing_off_k", 41, False)
    clock = records[("r12p-math-00002", "event")]  # the deadline counts every generated token: on token k as planned
    assert (clock.occurred, clock.crossing_at, clock.hit) == (True, 39, True)


def test_notice_model_counter_past_its_limit_before_token_k_is_off_k_and_gets_no_notice(tmp_path, tok, monkeypatch, tiny_model):
    real = L.draw_budgets

    def early(arm_name, signal, n_prompt, k, horizon, set_code, index):  # a deadline that is reached five tokens early
        budgets = real(arm_name, signal, n_prompt, k, horizon, set_code, index)
        return {name: L.Budgets(b.ctx, b.think, b.time, b.think_start, b.time_start + 5) for name, b in budgets.items()}

    monkeypatch.setattr(L, "draw_budgets", early)
    records, script = run("notice", tmp_path, tok, [plan(0, "time", 40, "live")], monkeypatch, tiny_model)
    gens = generations(tmp_path)
    event = records[("r12p-math-00000", "event")]
    assert (event.occurred, event.why_not, event.crossing_at, event.n_post) == (False, "crossing_off_k", 34, 0)
    assert gens[("r12p-math-00000", "event")]["notice"] is None
    assert records[("r12p-math-00000", "twin")].n_post == 160  # the twin and the off run still run
    monkeypatch.setattr(L, "draw_budgets", lambda *a: {name: L.Budgets(b.ctx, b.think, b.time * 2, b.think_start, b.time_start)
                                                      for name, b in real(*a).items()})
    late, _ = run("notice", tmp_path / "late", tok, [plan(0, "time", 40, "live")], monkeypatch, tiny_model)
    assert late[("r12p-math-00000", "event")].why_not == "no_crossing"  # not past its limit on token k: no crossing


# --- the onset index at the last keyword before the reaction's marker (a second reading beside L2b) -----------

CALL_PIECES = ("<tool_call>", '{"name": "compact_context", "arguments": {}}', "</tool_call>")


def test_onset_last_is_taken_at_the_last_keyword_before_the_marker():
    tok = PieceTok()
    plain = pieces(tok, " My", " context", " is", " almost", " full", ".", "\n", *CALL_PIECES, "<eos>")
    assert S.onset_index(plain, "ctx", tok) == S.onset_last_index(plain, "ctx", "eos", tok, MS) == 0
    # a keyword in ordinary text before the reaction lowers the first reading; the second one finds the reaction
    talk = pieces(tok, " The", " context", " of", " the", " task", " is", " clear", ".", " Two", " plus", " two", ".", "\n\n",
                  "My", " context", " is", " almost", " full", ",", " so", " I", " compact", " the", " context", ".", "\n",
                  *CALL_PIECES, "<eos>")
    assert S.onset_index(talk, "ctx", tok) == 0
    assert S.onset_last_index(talk, "ctx", "eos", tok, MS) == 12  # the blank line before "My context is almost full"
    assert S.onset_last_index(talk, "ctx", "eos", tok, MS, lead=False) == 13
    # the thinking budget: the marker is </think>
    budget = pieces(tok, " Is", " a", " thinking", " budget", " a", " thing", "?", " Yes", ".", "\n", "My", " thinking", " budget",
                    " is", " gone", ".", "</think>", " So", " 42", ".")
    assert (S.onset_index(budget, "think", tok), S.onset_last_index(budget, "think", "max_new_tokens", tok, MS)) == (0, 9)
    # a keyword after the marker does not count: the reaction is what came before it
    after = pieces(tok, " My", " context", " is", " full", ".", *CALL_PIECES, " The", " context", " is", " compact", " now", ".", "<eos>")
    assert S.onset_last_index(after, "ctx", "eos", tok, MS) == 0
    # the deadline has no marker: the second reading is the first
    deadline = pieces(tok, " The", " deadline", " is", " near", ".", " I", " am", " out", " of", " time", ".", "<eos>")
    assert S.onset_last_index(deadline, "time", "eos", tok, MS) == S.onset_index(deadline, "time", tok) == 0
    assert S.onset_last_index(pieces(tok, " nothing", " here"), "ctx", "max_new_tokens", tok, MS) is None


def test_hits_carry_both_onset_readings():
    tok = PieceTok()
    pre = [MS.think_open] + tok.ids(" w") * 19
    talk = pieces(tok, " The", " context", " is", " clear", ".", "\n\n", "My", " context", " is", " almost", " full", ".", "\n",
                  *CALL_PIECES, "<eos>")
    hit = S.judge(plan(0, "ctx", 20), "event", "channel", pre, talk, "eos", tok, MS, CORE_SIGNALS, crossing_at=19)
    assert (hit.hit, hit.onset, hit.onset_last) == (True, 0, 5)
    miss = S.judge(plan(0, "ctx", 20), "event", "channel", pre, tok.ids(" w") * 96, "max_new_tokens", tok, MS, CORE_SIGNALS, crossing_at=19)
    assert (miss.hit, miss.onset, miss.onset_last) == (False, None, None)
    summary = S.stream_summary([hit, miss])
    assert summary["event"]["onset"] == {"b": {"0": 1}, "c": {"0": 1}, "a": {"0": 1}, "b_at_most_1": 1, "last": {"5": 1}, "last_at_most_1": 0}


def test_run_that_ends_within_k_tokens_is_excluded_and_counted(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 30), plan(1, "ctx", 30), plan(2, "ctx", 30)]
    options = {0: {"end_at": 10}, 1: {"end_at": 29}, 2: {"end_at": 30}}
    records, _ = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model, options, run_types=("event", "twin"))
    for i in (0, 1):  # the turn ended on or before token k: no crossing occurred
        event = records[(f"r12p-math-{i:05d}", "event")]
        assert (event.occurred, event.why_not, event.hit, event.n_post) == (False, "ended", False, 0)
        assert event.n_pre == [11, 30][i] and event.stop_reason == "eos"
        assert records[(f"r12p-math-{i:05d}", "twin")].false_alarm is False
    last = records[("r12p-math-00002", "event")]
    assert (last.occurred, last.hit) == (True, True)  # token k is still plain: the crossing occurs and the model reacts
    summary = S.stream_summary(list(records.values()))
    assert summary["event"]["planned"] == 3 and summary["event"]["occurred"] == 1
    assert summary["event"]["not_occurred"] == {"ended": 2, "thinking_over": 0, "no_crossing": 0, "crossing_off_k": 0}
    assert summary["event"]["hits"] == 1 and summary["event"]["reaction_rate"] == 1.0


def test_think_crossing_after_the_thinking_closed_is_excluded(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "think", 30), plan(1, "ctx", 30), plan(2, "think", 30)]
    options = {0: {"close_at": 12}, 1: {"close_at": 12}, 2: {"close_at": 30}}
    records, _ = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model, options, run_types=("event",))
    closed = records[("r12p-math-00000", "event")]
    assert (closed.occurred, closed.why_not, closed.hit, closed.phase) == (False, "thinking_over", False, "answer")
    assert closed.crossing_at is None  # `think` is nominal after the model's own </think>: nothing was fed
    other = records[("r12p-math-00001", "event")]
    assert (other.occurred, other.hit, other.phase) == (True, True, "answer")  # another signal crosses in the answer
    still = records[("r12p-math-00002", "event")]
    assert (still.occurred, still.phase) == (True, "thinking")


def test_reaction_before_the_crossing_is_premature_and_never_a_hit(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 40), plan(1, "ctx", 40), plan(2, "time", 40)]
    options = {0: {"early": ("ctx", 5)},  # names the context and compacts it before anything crossed; the turn ends
               1: {"early": ("think", 5)},  # closes its thinking on a budget sentence, then goes on: still running at token k
               2: {"early": ("think", 5)}}
    records, _ = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model, options, run_types=("event",))
    ended = records[("r12p-math-00000", "event")]
    assert (ended.premature, ended.occurred, ended.why_not, ended.hit) == (True, False, "ended", False)
    going = records[("r12p-math-00001", "event")]
    assert going.premature is True and going.occurred is True and going.hit is False  # counted in the line as a miss
    summary = S.stream_summary(list(records.values()))
    assert summary["event"]["premature"] == 3 and summary["event"]["occurred"] == 2 and summary["event"]["hits"] == 0


def test_plateau_twin_equals_the_event_rows_before_the_crossing_and_never_passes_the_limit():
    for i, signal in enumerate(("ctx", "think", "energy", "time") * 6):
        p = plan(i, signal, 16 + 3 * i)
        rows = S.free_trajectories(CORE_SIGNALS, 600, p, 1)
        event, twin = rows["event"], rows["twin"]
        column, spec, cross = NAMES.index(signal), CORE_SIGNALS[NAMES.index(signal)], 600 + p.k - 1
        expected = sample_trajectory(CORE_SIGNALS, 600 + p.k + p.horizon, np.random.default_rng([12, 1, 2, i, 1]),
                                     event=column, t_cross=cross).values
        assert np.array_equal(event, expected) and event.shape == (600 + p.k + p.horizon, 6)
        assert spec.past_limit(float(event[cross, column])) and not spec.past_limit(float(event[cross - 1, column]))
        assert L.first_past(event, signal, CORE_SIGNALS) == cross  # row P + j goes with generated token j: token k
        assert np.array_equal(twin[:cross], event[:cross])
        assert (twin[cross:, column] == event[cross - 1, column]).all()  # held at its last value before the crossing
        other = [c for c in range(6) if c != column]
        assert np.array_equal(twin[:, other], event[:, other])
        assert not any(s.past_limit(float(v)) for c, s in enumerate(CORE_SIGNALS) for v in twin[:, c])
    counters = S.free_trajectories(CORE_SIGNALS, 600, plan(0, "ctx", 50), 0)
    assert float(counters["twin"][-1, 0]) == pytest.approx(0.895)  # a counter: one step under its limit


# --- arms live and Ls, channel model ------------------------------------------------------------------

def test_live_counters_are_fed_from_the_generation_and_cross_on_token_k(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 40, "live"), plan(1, "think", 50, "live"), plan(2, "time", 60, "live"),
             plan(3, "think", 45, "Ls"), plan(4, "time", 33, "Ls")]
    records, script = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model)
    gens = generations(tmp_path)
    for p in plans:
        event, g = records[(p.task_id, "event")], gens[(p.task_id, "event")]
        budgets = L.draw_budgets(p.arm, p.signal, N_PROMPT, p.k, p.horizon, 0, p.index)
        assert (event.occurred, event.hit, event.crossing_at, event.onset, event.arm) == (True, True, p.k - 1, 0, p.arm)
        assert event.budget == getattr(budgets["event"], p.signal)
        assert event.start == (0 if p.signal == "ctx" else getattr(budgets["event"], f"{p.signal}_start"))
        assert g["budgets"] == {"ctx": budgets["event"].ctx, "think": budgets["event"].think, "time": budgets["event"].time,
                                "think_start": budgets["event"].think_start, "time_start": budgets["event"].time_start}
        # the rows that were fed are the counters recomputed from the tokens
        stored = L.base_rows(p.arm, CORE_SIGNALS, N_PROMPT, p.k, p.horizon, 0, p.index)
        again = L.counter_rows(N_PROMPT, g["pre"] + g["post"], budgets["event"], stored, MS)
        assert g["rows_milli"] == {name: [int(round(float(v) * 1000)) for v in again[:, c]] for name, c in (("ctx", 0), ("think", 1), ("time", 3))}
        assert S.recomputed_rows_match(g, N_PROMPT, p, 0, CORE_SIGNALS, MS)
        twin, tg = records[(p.task_id, "twin")], gens[(p.task_id, "twin")]
        assert (twin.false_alarm, twin.crossing_at, twin.same_prefix) == (False, None, True)
        assert twin.budget == getattr(budgets["twin"], p.signal) > event.budget and twin.start == event.start
        column = {"ctx": "ctx", "think": "think", "time": "time"}[p.signal]
        last = tg["rows_milli"][column][-1]
        assert 845 <= (1000 - last if p.signal == "think" else last) <= 895  # 0.5 to 4.5 points short of the limit (5.5 from zero)
        off = records[(p.task_id, "off")]
        assert (off.false_alarm, off.crossing_at) == (False, p.k - 1)
    # prompt rows: ctx rises over the prompt, think and time at their start
    first = script.seen[0]
    b = L.draw_budgets("live", "ctx", N_PROMPT, 40, 96, 0, 0)["event"]
    assert np.array_equal(first.rows, L.prompt_rows(N_PROMPT, b, L.base_rows("live", CORE_SIGNALS, N_PROMPT, 40, 96, 0, 0)))
    tampered = dict(gens[("r12p-math-00000", "event")])
    tampered["rows_milli"] = {**tampered["rows_milli"], "ctx": [v + 10 for v in tampered["rows_milli"]["ctx"]]}
    assert not S.recomputed_rows_match(tampered, N_PROMPT, plans[0], 0, CORE_SIGNALS, MS)


def test_live_think_crossing_needs_open_thinking_and_a_late_crossing_is_flagged(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "think", 50, "live"), plan(1, "think", 50, "Ls"), plan(2, "time", 50, "live")]
    options = {0: {"close_at": 20}, 1: {"close_at": 20}, 2: {"end_at": 30}}
    records, _ = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model, options, run_types=("event",))
    for i in (0, 1):
        event = records[(f"r12p-math-{i:05d}", "event")]
        assert (event.occurred, event.why_not, event.crossing_at, event.hit) == (False, "thinking_over", None, False)
    ended = records[("r12p-math-00002", "event")]
    assert (ended.occurred, ended.why_not, ended.crossing_at) == (False, "ended", None)
    summary = S.stream_summary(list(records.values()))
    assert summary["event"]["not_occurred"] == {"ended": 1, "thinking_over": 2, "no_crossing": 0, "crossing_off_k": 0}
    assert summary["event"]["crossing_off_k"] == 0


# --- the text-notice model: two passes ------------------------------------------------------------------

def test_notice_model_reads_prompt_plus_its_own_k_tokens_plus_the_notice(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 20), plan(1, "time", 35)]
    records, script = run("notice", tmp_path, tok, plans, monkeypatch, tiny_model)
    gens = generations(tmp_path)
    first = script.seen[:2]  # pass 1: one run per task, k tokens, no notice
    assert [(s.prompt, s.horizon, s.harness) for s in first] == [(prompt_of(tok, 0), 20, None), (prompt_of(tok, 1), 35, None)]
    assert len(script.seen) == 2 + 6
    by_prompt = {tuple(s.prompt): s for s in script.seen[2:]}
    for p in plans:
        base, pre = prompt_of(tok, p.index), gens[(p.task_id, "event")]["pre"]
        assert len(pre) == p.k and all(gens[(p.task_id, t)]["pre"] == pre for t in ("twin", "off"))  # the same k tokens
        event = gens[(p.task_id, "event")]
        column = NAMES.index(p.signal)
        value = S.free_trajectories(CORE_SIGNALS, N_PROMPT, p, 0)["event"][N_PROMPT + p.k - 1, column]
        assert float(value) == pytest.approx(0.905) and event["notice_text"] == f" [status {p.signal}=0.90]"
        assert event["notice"] == tok.encode(event["notice_text"])
        seen = by_prompt[tuple(base + pre + event["notice"])]
        assert seen.horizon == p.horizon and seen.harness == [False] * (N_PROMPT + p.k) + [True] * len(event["notice"])
        twin = gens[(p.task_id, "twin")]
        below = float(re.search(r"=(\d\.\d\d)\]", twin["notice_text"]).group(1))
        assert 0.855 <= below + 0.005 <= 0.895 and twin["notice_text"].startswith(f" [status {p.signal}=")
        assert by_prompt[tuple(base + pre + twin["notice"])].harness[-len(twin["notice"]):] == [True] * len(twin["notice"])
        off = by_prompt[tuple(base + pre)]
        assert off.harness is None and off.horizon == p.horizon and gens[(p.task_id, "off")]["notice"] is None
        e = records[(p.task_id, "event")]
        assert (e.occurred, e.hit, e.onset, e.model, e.n_pre, e.same_prefix) == (True, True, 0, "notice", p.k, None)
        assert gens[(p.task_id, "event")]["post"][:3] == tok.encode(REACTIONS[p.signal])[:3]  # judged from the token after the notice
        for t in ("twin", "off"):
            assert (records[(p.task_id, t)].false_alarm, records[(p.task_id, t)].n_post) == (False, p.horizon)
    rng = np.random.default_rng([12, 0, 2, 0, 2])  # the twin's value: generator [12, set code, 2, i, 2]
    from prio.datagen.variants import below_limit_value, notice_text
    assert gens[("r12p-math-00000", "twin")]["notice_text"] == notice_text(CORE_SIGNALS[0], below_limit_value(CORE_SIGNALS[0], rng))
    pass1 = [json.loads(line) for line in (tmp_path / "out" / "pass1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["task_id"], len(row["pre"]), row["stop_reason"]) for row in pass1] == [
        ("r12p-math-00000", 20, "max_new_tokens"), ("r12p-math-00001", 35, "max_new_tokens")]


def test_notice_model_gets_no_notice_when_no_crossing_occurs(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 30), plan(1, "think", 30)]
    options = {0: {"end_at": 12}, 1: {"close_at": 12}}
    records, script = run("notice", tmp_path, tok, plans, monkeypatch, tiny_model, options)
    gens = generations(tmp_path)
    ended = {t: records[("r12p-math-00000", t)] for t in S.RUN_TYPES}
    assert (ended["event"].occurred, ended["event"].why_not, ended["event"].n_post) == (False, "ended", 0)
    assert all(r.n_pre == 13 and r.n_post == 0 and r.stop_reason == "eos" for r in ended.values())  # nothing follows an ended turn
    assert ended["twin"].false_alarm is False and gens[("r12p-math-00000", "twin")]["notice"] is None
    closed = records[("r12p-math-00001", "event")]
    assert (closed.occurred, closed.why_not, closed.n_post) == (False, "thinking_over", 0)
    assert gens[("r12p-math-00001", "event")]["notice"] is None
    # the twin and the off run of an open turn still run: the same k tokens, then the below-limit notice or nothing
    assert records[("r12p-math-00001", "twin")].n_post == 96 and gens[("r12p-math-00001", "twin")]["notice"] is not None
    assert records[("r12p-math-00001", "off")].n_post == 96
    assert len(script.seen) == 2 + 2


def test_live_notice_shows_the_computed_value_and_is_withheld_without_a_crossing(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 40, "live"), plan(1, "think", 50, "live"), plan(2, "time", 60, "live"), plan(3, "think", 45, "Ls"),
             plan(4, "think", 50, "live"), plan(5, "time", 44, "Ls")]
    options = {4: {"close_at": 20}}
    records, script = run("notice", tmp_path, tok, plans, monkeypatch, tiny_model, options)
    gens = generations(tmp_path)
    shown = {"ctx": "0.90", "think": "0.10", "time": "0.90"}
    for p in plans:
        event, g = records[(p.task_id, "event")], gens[(p.task_id, "event")]
        if p.index == 4:
            assert (event.occurred, event.why_not, event.n_post, g["notice"]) == (False, "thinking_over", 0, None)
            continue
        budgets = L.draw_budgets(p.arm, p.signal, N_PROMPT, p.k, p.horizon, 0, p.index)["event"]
        count = N_PROMPT + p.k if p.signal == "ctx" else getattr(budgets, f"{p.signal}_start") + p.k
        level = L.level(count, getattr(budgets, p.signal))  # 90 on real prompts; a 12-token prompt makes coarser steps
        assert 90 <= level <= 92 and (level == 90 or p.arm == "Ls" or p.signal == "ctx")
        assert g["notice_text"] == f" [status {p.signal}={(100 - level if p.signal == 'think' else level) / 100:.2f}]"
        if level == 90:
            assert g["notice_text"] == f" [status {p.signal}={shown[p.signal]}]"
        assert (event.occurred, event.hit, event.crossing_at, event.onset) == (True, True, p.k - 1, 0)
        assert event.budget == getattr(budgets, p.signal)
        stored = L.base_rows(p.arm, CORE_SIGNALS, N_PROMPT, p.k, p.horizon, 0, p.index)
        counters = L.counter_rows(N_PROMPT, g["pre"], budgets, stored, MS)
        assert g["rows_milli"]["ctx"] == [int(round(float(v) * 1000)) for v in counters[:, 0]]  # computed by the harness from its tokens
        twin = gens[(p.task_id, "twin")]
        value = float(re.search(r"=(\d\.\d\d)\]", twin["notice_text"]).group(1))
        assert (0.11 <= value <= 0.15) if p.signal == "think" else (0.85 <= value <= 0.89)
        assert records[(p.task_id, "twin")].false_alarm is False and records[(p.task_id, "off")].false_alarm is False
    code = {"live": 3, "Ls": 4}
    from prio.datagen.variants import below_limit_value, notice_text
    for p in (plans[0], plans[3]):  # the twin's value: generator [12, set code, arm code, i, 4]
        spec = CORE_SIGNALS[NAMES.index(p.signal)]
        expected = notice_text(spec, below_limit_value(spec, np.random.default_rng([12, 0, code[p.arm], p.index, 4])))
        assert gens[(p.task_id, "twin")]["notice_text"] == expected


# --- the onset index (reading 13) -----------------------------------------------------------------------

def pieces(tok, *texts):
    return tok.ids(*texts)


def test_onset_is_0_when_the_reaction_opens_the_judged_part():
    tok = PieceTok()
    post = pieces(tok, " My", " context", " is", " almost", " full", ".", "<tool_call>")
    assert S.onset_index(post, "ctx", tok) == 0 and S.onset_index(post, "ctx", tok, lead=False) == 0
    assert S.first_text_index(post, tok) == 0


def test_onset_counts_the_line_break_as_the_reaction_start():
    tok = PieceTok()
    post = pieces(tok, "\n", "I", " am", " almost", " out", " of", " time", ".")
    assert S.onset_index(post, "time", tok) == 0  # (b): the whitespace tokens right before the sentence belong to it
    assert S.onset_index(post, "time", tok, lead=False) == 1  # (c): the first word of the naming sentence
    assert S.first_text_index(post, tok) == 1  # (a): the first token that holds text
    blank = pieces(tok, "\n\n", " ", "The", " battery", " is", " low")
    assert (S.onset_index(blank, "energy", tok), S.onset_index(blank, "energy", tok, lead=False)) == (0, 2)


def test_onset_is_1_after_one_more_token():
    tok = PieceTok()
    post = pieces(tok, " so", "\n", "My", " thinking", " budget", " is", " gone", ".")
    assert S.onset_index(post, "think", tok) == 1 and S.onset_index(post, "think", tok, lead=False) == 2
    comma = pieces(tok, " Wait", ",", " my", " context", " is", " full")  # no sentence end before the name: the sentence starts at 0
    assert S.onset_index(comma, "ctx", tok) == 0


def test_onset_grows_when_a_sentence_is_finished_first():
    tok = PieceTok()
    post = pieces(tok, " plus", " two", " is", " four", ".", " My", " battery", " is", " low", ".")
    assert S.onset_index(post, "energy", tok) == 5 and S.onset_index(post, "energy", tok, lead=False) == 5
    spaced = pieces(tok, " four", ".", "\n", "\n", "The", " deadline", " is", " close")
    assert S.onset_index(spaced, "time", tok) == 2 and S.onset_index(spaced, "time", tok, lead=False) == 4
    # a period inside a number or a name is not a sentence end
    number = pieces(tok, " It", " is", " 3", ".", "5", " and", " my", " context", " is", " full")
    assert S.onset_index(number, "ctx", tok) == 0


def test_onset_word_and_first_text_variants():
    tok = PieceTok()
    assert S.onset_index(pieces(tok, " nothing", " here"), "ctx", tok) is None  # the signal is never named
    assert S.first_text_index(pieces(tok, "\n", " ", "\n"), tok) is None and S.first_text_index([], tok) is None
    mid = pieces(tok, " The", " con", "text", " window", " is", " full")  # a keyword split over two tokens
    assert S.onset_index(mid, "ctx", tok) == 0
    second = pieces(tok, " Fine", ".", " Out", " of", " time", ",", " the", " deadline", " is", " here")
    assert S.onset_index(second, "time", tok) == 2  # the first mention decides ("out of time"), any case
    for signal, text in (("net", " offline"), ("ram", " memory"), ("energy", " power")):
        assert S.onset_index(pieces(tok, " ok", ".", " I", " am", text), signal, tok) == 2


def test_onsets_of_a_saved_paired_evaluation_count_the_hits_only(tmp_path):
    tok = PieceTok()
    runs = [  # (episode, kind, signal, condition, paired, event occurred, reaction signal, pieces)
        ("e0", "event", "ctx", "on", "event", True, "ctx", ("\n", "My", " context", " is", " full")),
        ("e1", "event", "time", "on", "event", True, "time", (" ok", ".", " The", " deadline", " is", " here")),
        ("e2", "gating_event", "net", "on", "event", True, "net", (" I", " am", " offline")),
        ("e3", "event", "energy", "on", "event", True, None, (" plain", " words")),  # a miss
        ("e0", "event", "ctx", "on", "nominal", False, None, (" My", " context")),  # a quiet twin
        ("e0", "event", "ctx", "zeroed", "event", True, "ctx", (" My", " context")),  # the channel off
        ("e4", "starts_past", "ctx", "on", "event", True, "ctx", (" My", " context")),  # not a main kind
        ("e5", "swap", "ctx", "on", "swapped", True, "ctx", (" My", " context")),
    ]
    record_rows, generation_rows = [], []
    for episode, kind, signal, condition, paired, occurred, reaction, texts in runs:
        record_rows.append(json.dumps({"episode_id": episode, "signal": signal, "kind": kind, "condition": condition,
                                       "paired": paired, "event_occurred": occurred, "reaction_signal": reaction}))
        generation_rows.append(json.dumps({"episode_id": episode, "condition": condition, "paired": paired, "signal": signal,
                                           "stop_reason": "eos", "token_ids": tok.ids(*texts)}))
    (tmp_path / "records.jsonl").write_text("\n".join(record_rows) + "\n", encoding="utf-8")
    (tmp_path / "generations.jsonl").write_text("\n".join(generation_rows) + "\n", encoding="utf-8")
    counts = S.evaluation_onsets(tmp_path, tok)
    assert counts["hits"] == 3
    assert (counts["b"], counts["c"], counts["a"]) == ({"0": 2, "2": 1}, {"0": 1, "1": 1, "2": 1}, {"0": 2, "1": 1})
    assert (counts["b_at_most_1"], counts["c_at_most_1"], counts["a_at_most_1"]) == (2, 2, 3)
    assert counts["b_by_kind"] == {"event": {"0": 1, "2": 1}, "gating_event": {"0": 1}}
    assert counts["b_by_signal"] == {"ctx": {"0": 1}, "net": {"0": 1}, "time": {"2": 1}}
    assert [row["episode_id"] for row in counts["b_late"]] == ["e1"]


# --- files, resume, summary ------------------------------------------------------------------------------

def test_records_resume_and_generations_are_saved(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 20), plan(1, "time", 24), plan(2, "energy", 30, family="code")]
    first, script = run("channel", tmp_path, tok, plans[:2], monkeypatch, tiny_model, run_types=("event", "twin"))
    out = tmp_path / "out"
    assert len(script.seen) == 4 and len((out / "records.partial.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    again, script = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model)
    assert len(script.seen) == 9 - 4  # only what is missing: two off runs and the third task
    assert {k: again[k] for k in first} == first
    saved = [S.StreamRecord(**json.loads(line)) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r.task_id, r.run_type) for r in saved] == [(p.task_id, t) for p in plans for t in S.RUN_TYPES]  # plan order
    assert saved == list(again.values()) and len(generations(tmp_path)) == 9
    assert b"\r" not in (out / "records.jsonl").read_bytes() and b"\r" not in (out / "generations.jsonl").read_bytes()
    summary = json.loads((out / "stream_summary.json").read_text(encoding="utf-8"))
    assert summary == S.stream_summary(saved)
    assert (summary["arm"], summary["model"], summary["event"]["planned"], summary["event"]["hits"]) == ("free", "channel", 3, 3)
    assert summary["event"]["by_signal"]["ctx"] == {"planned": 1, "occurred": 1, "hits": 1}
    assert summary["event"]["by_family"] == {"math": {"planned": 2, "occurred": 2, "hits": 2}, "code": {"planned": 1, "occurred": 1, "hits": 1}}
    assert summary["event"]["onset"] == {"b": {"0": 3}, "c": {"0": 3}, "a": {"0": 3}, "b_at_most_1": 3, "last": {"0": 3},
                                         "last_at_most_1": 3}
    assert summary["twin"] == {"n": 3, "false_alarms": 0, "false_alarm_rate": 0.0, "false_alarm_ci": summary["twin"]["false_alarm_ci"],
                               "at_start": 0, "at_k": 0, "at_close": 0, "same_prefix": 3, "same_prefix_of": 3}
    assert summary["deadline_near_end"] == {"event": 1, "event_not_hit": 0, "twin": 0, "off": 0}  # the `time` hit names its deadline
    assert summary["off"]["n"] == 3 and summary["off"]["false_alarms"] == 0
    text = (out / "stream_summary.md").read_text(encoding="utf-8")
    assert "not a line" in text and "| event |" in text and not (out / "summary.json").exists()
    third, script = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model)
    assert not script.seen and third == again


def test_notice_runs_resume_from_the_saved_first_pass(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 20), plan(1, "time", 24)]
    first, script = run("notice", tmp_path, tok, plans, monkeypatch, tiny_model, run_types=("event",))
    assert len(script.seen) == 4
    again, script = run("notice", tmp_path, tok, plans, monkeypatch, tiny_model)
    assert len(script.seen) == 4 and all(s.written > 0 for s in script.seen)  # no first pass again: twin and off continue it
    gens = generations(tmp_path)
    assert all(gens[(p.task_id, t)]["pre"] == gens[(p.task_id, "event")]["pre"] for p in plans for t in ("twin", "off"))
    assert {k: again[k] for k in first} == first and len(again) == 6


def test_plans_are_read_from_the_stream_file(tmp_path):
    rows = [{"arm": "free", "index": 0, "task_id": "r12p-math-00016", "family": "math", "signal": "ctx", "k": 45, "horizon": 96},
            {"arm": "Ls", "index": 3, "task_id": "r12p-code-00627", "family": "code", "signal": "time", "k": 99, "horizon": 160}]
    (tmp_path / "stream.json").write_text(json.dumps(rows), encoding="utf-8")
    assert S.load_plans(tmp_path / "stream.json", "Ls") == [S.Plan("Ls", 3, "r12p-code-00627", "code", "time", 99, 160)]
    assert S.load_plans(tmp_path / "stream.json", "free")[0].k == 45 and S.load_plans(tmp_path / "stream.json", "live") == []


def test_time_runs_carry_the_practical_reading_beside_the_strict_one(tok):
    p = plan(0, "time", 20)
    pre = [MS.think_open] + tok.encode("w") * 19
    late = tok.encode("I am almost out of time so I wrap up " + "w " * 130 + "<eos>")  # ends after the 128-token rule
    record = S.judge(p, "event", "channel", pre, late, "eos", tok, MS, CORE_SIGNALS, crossing_at=19)
    assert (record.occurred, record.hit, record.practical) == (True, False, True)
    going = S.judge(p, "event", "channel", pre, late[:-1], "max_new_tokens", tok, MS, CORE_SIGNALS, crossing_at=19)
    assert (going.hit, going.practical) == (False, False)  # named, still writing at the horizon
    silent = S.judge(p, "event", "channel", pre, tok.encode("w " * 30 + "<eos>"), "eos", tok, MS, CORE_SIGNALS, crossing_at=19)
    assert (silent.hit, silent.practical) == (False, False)  # ended without naming the deadline
    strict = S.judge(p, "event", "channel", pre, tok.encode(REACTIONS["time"]), "eos", tok, MS, CORE_SIGNALS, crossing_at=19)
    assert (strict.hit, strict.practical) == (True, True)
    other = S.judge(plan(1, "ctx", 20), "event", "channel", pre, tok.encode(REACTIONS["ctx"]), "eos", tok, MS, CORE_SIGNALS)
    assert other.hit is True and other.practical is None  # only `time` has a second reading
    twin = S.judge(p, "twin", "channel", pre, tok.encode("w") * 160, "max_new_tokens", tok, MS, CORE_SIGNALS)
    assert twin.practical is None
    summary = S.stream_summary([record, going, silent, strict, other])
    assert summary["event"]["time_practical"] == {"occurred": 4, "strict": 1, "practical": 2}


def test_saved_rows_of_the_free_arm_are_the_planned_trajectory(tmp_path, tok, monkeypatch, tiny_model):
    plans = [plan(0, "ctx", 20), plan(1, "think", 31), plan(2, "think", 30)]
    records, _ = run("channel", tmp_path, tok, plans, monkeypatch, tiny_model, {2: {"close_at": 12}})
    gens = generations(tmp_path)
    for p in plans:
        for run_type in S.RUN_TYPES:
            row = gens[(p.task_id, run_type)]
            assert row["n_prompt"] == N_PROMPT and S.recomputed_rows_match(row, N_PROMPT, p, 0, CORE_SIGNALS, MS)
    closed = gens[("r12p-math-00002", "event")]["rows_milli"]["think"]
    assert closed[12] != 1000 and set(closed[13:]) == {1000}  # nominal after the model's own </think>
    tampered = dict(gens[("r12p-math-00000", "event")])
    tampered["rows_milli"] = {"ctx": [v + 10 for v in tampered["rows_milli"]["ctx"]]}
    assert not S.recomputed_rows_match(tampered, N_PROMPT, plans[0], 0, CORE_SIGNALS, MS)
    assert not S.recomputed_rows_match(gens[("r12p-math-00000", "event")], N_PROMPT, plans[0], 1, CORE_SIGNALS, MS)  # another set
    assert S.recomputed_rows_match({**tampered, "rows_milli": None}, N_PROMPT, plans[0], 0, CORE_SIGNALS, MS)  # nothing saved


# --- on a model ---------------------------------------------------------------------------------------------

class SmallTok(R12Tok):
    """R12Tok for the tiny model (100 ids): every word that is not a special token gets an id in 6-99."""

    def encode(self, text):
        import zlib

        return [self.vocab[w] if w in self.SPECIALS else 6 + zlib.crc32(w.encode("utf-8")) % 94 for w in text.split()]

    def decode(self, ids, **_):
        return " " + " ".join(self.SPECIALS[i] if i < 6 else f"x{i}" for i in ids)


def _tiny_setup(n_tasks=6):
    g = torch.Generator().manual_seed(0)
    plans, prompts = [], {}
    for i in range(n_tasks):
        arm_name = ("free", "live", "Ls")[i % 3]
        signal = {"free": ("ctx", "think", "energy", "time")[i % 4], "live": ("ctx", "think", "time")[i % 3], "Ls": ("think", "time")[i % 2]}[arm_name]
        p = S.Plan(arm_name, i, f"r12p-math-{i:05d}", "math", signal, 16 + 5 * i, 160 if signal == "time" else 96)
        plans.append(p)
        prompts[p.task_id] = torch.randint(6, 100, (5 + i,), generator=g).tolist()
    return plans, prompts


def test_twin_shares_its_first_k_tokens_with_the_event_run(tiny_model, tmp_path):
    model = ChannelModel(tiny_model, 6, delivery="wide", lane_width=8, projector="levels")
    arm(model, std=3.0)
    tok = SmallTok()
    plans, prompts = _tiny_setup()
    ms = type(MS)("fake", MS.think_open, MS.think_close, MS.tool_open, MS.tool_close, frozenset())  # nothing ends a turn
    records = S.run_channel(model, tok, ms, CORE_SIGNALS, plans, prompts, 0, tmp_path / "out", batch_size=4)
    gens = generations(tmp_path)
    free = [p for p in plans if p.arm == "free"]
    for p in plans:
        event, twin = gens[(p.task_id, "event")], gens[(p.task_id, "twin")]
        assert len(event["pre"]) == len(twin["pre"]) == p.k and len(event["post"]) == p.horizon
    for p in free:  # greedy: identical rows up to the crossing give identical tokens
        assert gens[(p.task_id, "twin")]["pre"] == gens[(p.task_id, "event")]["pre"]
    assert all(r.same_prefix for r in records if r.run_type == "twin" and r.arm == "free")
    assert all(r.same_prefix in (True, False) for r in records if r.run_type == "twin")  # live twins: other budgets from the start
    # the channel is live: after the crossing the event run and its twin part somewhere
    assert any(gens[(p.task_id, "event")]["post"] != gens[(p.task_id, "twin")]["post"] for p in free)
    for p in plans:
        if p.arm != "free":
            assert S.recomputed_rows_match(gens[(p.task_id, "event")], len(prompts[p.task_id]), p, 0, CORE_SIGNALS, ms)


@pytest.mark.parametrize("delivery", ["wide", "none"])
def test_stream_runner_end_to_end_on_the_tiny_model(tiny_model, tmp_path, delivery):
    extra = {"lane_width": 8, "projector": "levels"} if delivery == "wide" else {}
    model = ChannelModel(tiny_model, 6, delivery=delivery, **extra)
    if delivery == "wide":
        arm(model)
    tok = SmallTok()
    plans, prompts = _tiny_setup()
    runner = S.run_channel if delivery == "wide" else S.run_notice
    records = runner(model, tok, MS, CORE_SIGNALS, plans, prompts, 0, tmp_path / "out", batch_size=4)
    assert [(r.task_id, r.run_type) for r in records] == [(p.task_id, t) for p in plans for t in S.RUN_TYPES]
    assert all(r.model == ("channel" if delivery == "wide" else "notice") for r in records)
    batch_1 = runner(model, tok, MS, CORE_SIGNALS, plans, prompts, 0, tmp_path / "one", batch_size=1)
    assert batch_1 == records  # the same runs one by one
    out = tmp_path / "out"
    for name in ("records.jsonl", "records.partial.jsonl", "generations.jsonl", "stream_summary.json", "stream_summary.md"):
        assert (out / name).exists()
    assert (out / "pass1.jsonl").exists() == (delivery == "none")
    summary = json.loads((out / "stream_summary.json").read_text(encoding="utf-8"))
    assert summary["event"]["planned"] == summary["twin"]["n"] == summary["off"]["n"] == 6


@pytest.mark.slow
def test_real_model_stream_smoke(tmp_path):
    """Four tasks of each free arm on the channel model: the fed rows are the planned ones and the
    crossing lands on token k. Needs the pilot's task list (CPU) and the trained run (GPU)."""
    from prio.r12 import RUNS, build
    from prio.r12 import tasks as R
    from prio.tasks import build_prompt
    from prio.train import load_run

    model, tok, ms = load_run("Qwen/Qwen3-1.7B", RUNS["channel"], 6)
    tok.padding_side = "left"
    arms = R.task_sets()["pilot"]
    rows = build.stream_plans(arms["free"], arms["live"], arms["Ls"], "pilot")
    by_id = {t.task_id: t for tasks in arms.values() for t in tasks}
    for arm_name in ("free", "live", "Ls"):
        plans = [S.Plan(**row) for row in rows if row["arm"] == arm_name][:4]
        prompts = {p.task_id: build_prompt(tok, by_id[p.task_id]) for p in plans}
        records = S.run_channel(model, tok, ms, CORE_SIGNALS, plans, prompts, 0, tmp_path / arm_name, batch_size=4,
                                run_types=("event", "twin"))
        lines = (tmp_path / arm_name / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        gens = {(g["task_id"], g["run_type"]): g for g in map(json.loads, lines)}
        for r in records:
            if r.run_type == "event" and r.occurred:
                assert r.crossing_at == r.k - 1
            if r.run_type == "twin":
                assert r.crossing_at is None
        if arm_name != "free":
            for p in plans:
                assert S.recomputed_rows_match(gens[(p.task_id, "event")], len(prompts[p.task_id]), p, 0, CORE_SIGNALS, ms)
