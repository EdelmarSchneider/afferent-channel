"""R24 (preregistration amendment A9): varied reaction wording and a written summary when the
thinking budget runs out: prio.datagen.varied, scripts/analysis/wording_reading.py."""

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from helpers import CALL, FAKE_MS, FakeTokenizer
from prio.datagen import varied as V
from prio.datagen.episodes import Episode, reaction_start, read_jsonl, write_jsonl
from prio.datagen.teacher import REACTION_BANK
from prio.reactions import REACTION_WINDOW, find_reaction, names_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

SIGNALS = ("ctx", "think", "energy", "time", "net", "ram")


class LineTokenizer(FakeTokenizer):
    """FakeTokenizer that keeps line breaks: every newline is a token of its own."""

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in re.findall(r"\n|[^\s]+", text):
            if word not in self.vocab:
                self.vocab[word] = len(self.words)
                self.words.append(word)
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids, **_) -> str:
        return re.sub(r" ?\n ?", "\n", " ".join(self.words[i] for i in ids))


TOK = LineTokenizer()
MS = FAKE_MS  # same special words, same ids
e = TOK.encode
assert TOK.vocab["</think>"] == MS.think_close and TOK.vocab["<tool_call>"] == MS.tool_open

PROMPT = e("<sys> system prompt <user> do the task <eos> <assistant>")
N = len(PROMPT)
ACTION = {
    "ctx": CALL % ("compact_context", "{}"),
    "energy": CALL % ("save_state", "{}") + "\n" + CALL % ("sleep", "{}"),
    "net": CALL % ("local_lookup", '{"query":"q"}'),
    "ram": CALL % ("run_light", '{"job":"j"}'),
    "think": "the total is 42 .",
    "time": "in short the total is 42 .",
}
# (signal, where the reaction sits): every kind and phase data/n6 holds
COMBOS = ([(s, w) for s in ("ctx", "energy", "time") for w in ("thinking", "answer", "start")]
          + [("think", "thinking"), ("think", "start"), ("net", "gating"), ("ram", "gating")])
GOOD = "the sum of forty and two is 42 , checked once ."  # 12 tokens, one sentence


def rows(length: int) -> np.ndarray:
    return np.arange(length * 6, dtype=np.float32).reshape(length, 6) / 1000


def reaction(signal, where="thinking", i=0, sentence=None, split="train"):
    """A reaction episode as data/n6 stores it and its no-event twin (the base run)."""
    sentence = sentence or REACTION_BANK[signal][i % len(REACTION_BANK[signal])]
    thinking = f"<think>\nwe add forty and two in case{i} . that gives the sum . now check"
    if where == "gating":
        prefix = e("<think>\nplan the call .\n</think>\n\n")
        base = prefix + e(CALL % ("web_search", '{"query":"q"}') + " <eos>")
        target = prefix + e(sentence + "\n\n" + ACTION[signal] + " <eos>")
        kind, start, t_cross, t_decision, phase = "gating_event", len(prefix), 2, len(prefix), "answer"
    elif where == "start":
        prefix = e("<think>")
        base = e("<think>\nokay let me see . it is 42 .\n</think>\n\nthe total is 42 . <eos>")
        target = prefix + e("\n" + sentence + "\n</think>\n\n" + ACTION[signal] + " <eos>")
        kind, start, t_cross, t_decision, phase = "starts_past", 1, 0, None, "thinking"
    elif where == "thinking":
        prefix = e(thinking)
        base = prefix + e(" it again .\n</think>\n\nso the total is 42 . <eos>")
        target = prefix + e("\n" + sentence + "\n</think>\n\n" + ACTION[signal] + " <eos>")
        kind, start, t_cross, t_decision, phase = "event", len(prefix), len(prefix) - 1, None, "thinking"
    else:
        prefix = e(thinking + " it again .\n</think>\n\nso the total")
        base = prefix + e(" is 42 . <eos>")
        target = prefix + e("\n\n" + sentence + "\n\n" + ACTION[signal] + " <eos>")
        kind, start, t_cross, t_decision, phase = "event", len(prefix), len(prefix) - 1, None, "answer"
    diverge = next(k for k, t in enumerate(target) if k >= len(base) or t != base[k])
    ep = Episode(episode_id=f"{split}-{signal}-{where}-{i}", task_id=f"task-{signal}-{where}-{i}", family="math",
                 split=split, kind=kind, signal=signal, phase=phase, prompt_ids=PROMPT, target_ids=target,
                 signals=rows(N + len(target)), t_cross=t_cross, t_decision=t_decision, templated=False, t_diverge=diverge)
    twin = replace(ep, episode_id=ep.episode_id + "t", kind="nominal", signal=None, phase=None, target_ids=base,
                   signals=rows(N + len(base)), t_cross=None, t_decision=None)
    assert reaction_start(ep) == start
    return ep, twin


def rng(seed=0):
    return np.random.default_rng(seed)


def counted(ep) -> bool:
    r = find_reaction(ep.target_ids[reaction_start(ep):], "eos", ep.signal, TOK, MS)
    return r is not None and r.named and r.token_index < REACTION_WINDOW


# --- the generator ----------------------------------------------------------------------------

@pytest.mark.parametrize("signal", SIGNALS)
def test_every_signal_has_at_least_300_distinct_sentences_in_several_shapes(signal):
    sentences = V.sentences(signal)
    assert len(sentences) == len(set(sentences)) >= 300
    shapes = V.BANKS[signal].shapes
    assert len(shapes) >= 6
    assert any(s.startswith("{state}") for s in shapes) and any(s.startswith("{action}") for s in shapes)  # both orders
    lengths = [len(s.split()) for s in sentences]
    assert min(lengths) <= 11 and max(lengths) - min(lengths) >= 8 and max(lengths) <= 26  # short ones and longer ones
    assert sentences == V.BANKS[signal].sentences()  # a fixed order


def test_the_time_sentences_stay_short():
    assert max(len(s.split()) for s in V.sentences("time")) <= 20  # the turn has to end inside a fixed window


def test_thinking_budget_has_a_second_bank_that_ends_in_a_lead_in():
    leads = V.sentences("think", lead_in=True)
    assert len(leads) == len(set(leads)) >= 300 and len(V.THINK_LEAD.shapes) >= 6
    assert all(s.endswith(":") for s in leads) and all(s.endswith(".") for s in V.sentences("think"))
    assert not set(leads) & set(V.sentences("think"))
    assert sum("in short:" in s.lower() for s in leads) >= 30 and len({s.rsplit(" ", 2)[-1] for s in leads}) >= 6
    with pytest.raises(ValueError):
        V.sentences("ctx", lead_in=True)


@pytest.mark.parametrize("signal", SIGNALS)
def test_sentences_are_plain_first_person_and_name_their_own_signal_only(signal):
    for s in V.sentences(signal) + (V.sentences("think", lead_in=True) if signal == "think" else []):
        assert names_signal(s, signal), s
        assert not any(names_signal(s, other) for other in SIGNALS if other != signal), s
        assert not re.search(r"\b(signals?|channels?|status|levels?|sensors?)\b", s, re.IGNORECASE), s
        assert re.search(r"\b(I|[Mm]y|me)\b", s), s  # first person
        assert s[0].isupper() and "  " not in s and s == s.strip() and "{" not in s, s
        assert not re.search(r"[.!?] [a-z]", s), s  # a second sentence starts with a capital


@pytest.mark.parametrize("signal", SIGNALS)
def test_a_drawn_sentence_follows_the_rng(signal):
    draws = [V.draw(signal, rng(seed)) for seed in range(300)]
    assert len(set(draws)) >= 150 and set(draws) <= set(V.sentences(signal))
    assert draws[7] == V.draw(signal, rng(7))
    assert V.draw("think", rng(3), lead_in=True) in V.sentences("think", lead_in=True)


# --- one episode ------------------------------------------------------------------------------

@pytest.mark.parametrize("signal, where", COMBOS)
def test_rewrite_replaces_the_bank_sentence_and_keeps_the_action(signal, where):
    ep, twin = reaction(signal, where, i=3)
    found = V.locate(ep, TOK, MS)
    start = reaction_start(ep)
    assert (found.start, found.sentence, found.in_thinking) == (start, REACTION_BANK[signal][3], where in ("thinking", "start"))
    assert ep.target_ids[start + found.n_old - 1] == e("\n")[0]  # the replaced span ends with the sentence's line
    out = V.rewrite(ep, TOK, MS, rng(0), base_ids=twin.target_ids)
    info = out.meta["varied"]
    a, b = info["span"]
    assert a == start and info["old"] == found.sentence and info["summary"] is None
    assert info["sentence"] in V.sentences(signal) and TOK.decode(out.target_ids[a:b]).strip() == info["sentence"]
    assert out.target_ids[:a] == ep.target_ids[:a]
    assert out.target_ids[b:] == ep.target_ids[start + found.n_old:]  # the action, token for token
    assert found.sentence not in TOK.decode(out.target_ids)
    # signal rows: the new tokens repeat the row at the reaction start, the rest moves with its tokens
    assert out.signals.shape == (N + len(out.target_ids), 6) and out.signals.dtype == np.float32
    assert np.array_equal(out.signals[:N + a], ep.signals[:N + a])
    assert (out.signals[N + a:N + b] == ep.signals[N + a]).all()
    assert np.array_equal(out.signals[N + b:], ep.signals[N + start + found.n_old:])
    # indices: the decision token stays the first one on which the twin differs
    assert (out.t_cross, out.t_decision, out.t_diverge) == (ep.t_cross, ep.t_decision, ep.t_diverge)
    assert reaction_start(out) == start
    d = out.t_diverge
    assert out.target_ids[:d] == twin.target_ids[:d] and out.target_ids[d] != twin.target_ids[d]
    assert counted(out)  # the unchanged classifier still finds the sentence and the action
    assert not any(find_reaction(out.target_ids[start:], "eos", other, TOK, MS) and
                   find_reaction(out.target_ids[start:], "eos", other, TOK, MS).named for other in SIGNALS if other != signal)
    assert replace(out, target_ids=ep.target_ids, signals=ep.signals, meta=None) == ep  # nothing else changed


def test_a_rewritten_episode_goes_through_the_trainer_batch():
    from prio.train import collate

    ep, twin = reaction("energy", "answer")
    out = V.rewrite(ep, TOK, MS, rng(1), base_ids=twin.target_ids)
    batch = collate([out, twin], 0, 6, reaction_weight=20.0)
    start = reaction_start(out)
    assert batch["dropped"] == 0 and (batch["weights"][0, N + start:N + len(out.target_ids)] == 20).all()
    assert batch["decision"][0, N + out.t_diverge] == 1 and batch["decision"][1, N + twin.t_diverge] == 2


def test_rewrite_follows_the_rng():
    ep, twin = reaction("net", "gating")
    one = V.rewrite(ep, TOK, MS, rng(5), base_ids=twin.target_ids)
    assert one == V.rewrite(ep, TOK, MS, rng(5), base_ids=twin.target_ids)
    assert len({V.rewrite(ep, TOK, MS, rng(seed)).meta["varied"]["sentence"] for seed in range(40)}) >= 30


def test_episodes_without_a_reaction_come_back_untouched():
    ep, twin = reaction("ctx", "thinking")
    recovery = replace(twin, kind="recovery", signal="net", t_cross=3, t_decision=9, t_diverge=None)
    unused = replace(twin, kind="gating_unused", signal="ram", t_cross=3, t_diverge=None)
    for quiet in (twin, recovery, unused, replace(twin, t_diverge=None)):
        assert V.locate(quiet, TOK, MS) is None and V.rewrite(quiet, TOK, MS, rng()) is quiet


def test_a_reaction_that_does_not_open_with_a_bank_sentence_is_left_alone():
    ep, _ = reaction("ctx", "thinking", sentence="My context looks rather full, so I compact.")
    assert V.locate(ep, TOK, MS) is None and V.rewrite(ep, TOK, MS, rng()) is ep
    cut_word, _ = reaction("ctx", "thinking")  # the bank sentence is there, but not at the reaction start
    start = reaction_start(cut_word)
    moved = replace(cut_word, target_ids=cut_word.target_ids[:start] + e("again") + cut_word.target_ids[start:],
                    signals=rows(N + len(cut_word.target_ids) + 1))
    assert V.rewrite(moved, TOK, MS, rng()) is moved


def test_indices_and_spans_past_the_sentence_move_with_it():
    ep, _ = reaction("ctx", "answer")
    start, n_old = reaction_start(ep), V.locate(ep, TOK, MS).n_old
    call = ep.target_ids.index(MS.tool_open, start)
    marked = replace(ep, t_decision=call, focus_spans=[[0, 2], [start + n_old, len(ep.target_ids)]], harness_spans=[[1, start]])
    out = next(o for o in (V.rewrite(marked, TOK, MS, rng(seed)) for seed in range(20)) if len(o.target_ids) != len(ep.target_ids))
    shift = len(out.target_ids) - len(ep.target_ids)
    assert out.t_decision == call + shift and out.target_ids[out.t_decision] == MS.tool_open
    assert out.focus_spans == [[0, 2], [start + n_old + shift, len(out.target_ids)]] and out.harness_spans == [[1, start]]
    with pytest.raises(ValueError):
        V.rewrite(replace(ep, focus_spans=[[start + 1, start + n_old + 2]]), TOK, MS, rng())  # a span across the sentence


def test_the_twins_decision_token_stays_where_it_was():
    ep, twin = reaction("time", "start")  # the base run opens with the same newline: the twin parts on token 2
    assert ep.t_diverge == 2
    base = twin.target_ids[:2] + e("I") + twin.target_ids[3:]  # a base run that goes on with "I"
    free = [V.rewrite(ep, TOK, MS, rng(seed)) for seed in range(40)]
    assert any(o.target_ids[2] == base[2] for o in free)  # many sentences open with "I"
    for seed in range(40):
        out = V.rewrite(ep, TOK, MS, rng(seed), base_ids=base)
        assert out.t_diverge == 2 and out.target_ids[:2] == base[:2] and out.target_ids[2] != base[2]


def test_a_time_reaction_stays_inside_the_classifiers_window():
    ep, twin = reaction("time", "answer")
    start = reaction_start(ep)
    slack = 3  # tokens to spare before the turn would end outside the window
    pad = 128 - (len(ep.target_ids) - start) - slack
    long_ep = replace(ep, target_ids=ep.target_ids[:-1] + e("w") * pad + ep.target_ids[-1:], signals=rows(N + len(ep.target_ids) + pad))
    assert counted(long_ep)
    n_old = V.locate(long_ep, TOK, MS).n_old
    for seed in range(40):  # only sentences that keep the reaction countable are drawn
        out = V.rewrite(long_ep, TOK, MS, rng(seed), base_ids=twin.target_ids)
        assert counted(out) and out.meta["varied"]["span"][1] - start <= n_old + slack


# --- the summary ------------------------------------------------------------------------------

def test_pick_summary_takes_one_or_two_sentences_up_to_the_end_of_the_thinking():
    assert V.pick_summary(e(GOOD + "\n</think>\n\nthe total is 42 . <eos>"), TOK, MS) == (GOOD, "accepted")
    two = GOOD + " so the answer is 42 ."
    three = two + " and then some more words follow here ."
    assert V.pick_summary(e(three + " and it goes on without an end"), TOK, MS) == (two, "accepted")
    assert V.pick_summary(e(three), TOK, MS, max_sentences=0) == (three, "accepted")
    assert V.pick_summary(e(GOOD + " " + CALL % ("web_search", '{"query":"q"}')), TOK, MS) == (None, "tool_call")
    assert V.pick_summary(e(GOOD + " <think> and more ."), TOK, MS) == (None, "special_token")
    # a tool call after the thinking has closed is the answer, not the summary
    assert V.pick_summary(e(GOOD + "\n</think>\n\n" + CALL % ("web_search", '{"query":"q"}')), TOK, MS) == (GOOD, "accepted")


def test_pick_summary_rejects_what_is_too_short_or_has_no_sentence_end():
    assert V.pick_summary(e("it is 42 .\n</think>\n\nthe total is 42 ."), TOK, MS) == (None, "too_short")
    assert V.pick_summary(e("\n</think>\n\nthe total is 42 ."), TOK, MS) == (None, "too_short")  # closed at once
    assert V.pick_summary(e("the sum of forty and two is 42 and I keep going on and on"), TOK, MS) == (None, "no_sentence_end")
    assert V.pick_summary(e("1 . 2 . 3 . 4 . 5 . 6 . 7 ."), TOK, MS) == (None, "no_sentence_end")  # list numbers


def test_pick_summary_trims_a_long_text_at_a_sentence_end_inside_the_limit():
    first = " ".join(f"w{k}" for k in range(29)) + " ."  # 30 tokens
    rest = " ".join(f"v{k}" for k in range(66))  # goes on to the end of the 96 generated tokens
    assert V.pick_summary(e(first + " " + rest), TOK, MS) == (first, "accepted")
    second = " ".join(f"u{k}" for k in range(59)) + " ."  # the second sentence ends at token 90
    assert V.pick_summary(e(first + " " + second), TOK, MS) == (first, "accepted")
    long_one = " ".join(f"w{k}" for k in range(85)) + " ."
    assert V.pick_summary(e(long_one), TOK, MS) == (None, "too_long")
    assert V.pick_summary(e(GOOD), TOK, MS, max_tokens=10) == (None, "too_long")
    assert V.pick_summary(e(GOOD), TOK, MS, min_tokens=13) == (None, "too_short")
    assert V.pick_summary(e(first + " " + GOOD), TOK, MS, fits=lambda text: len(text.split()) <= 30) == (first, "accepted")
    assert V.pick_summary(e(GOOD), TOK, MS, fits=lambda text: False) == (None, "too_long")


def test_a_thinking_budget_reaction_gets_the_summary_before_the_thinking_closes():
    ep, twin = reaction("think", "thinking")
    start, n_old = reaction_start(ep), V.locate(ep, TOK, MS).n_old
    out = V.rewrite(ep, TOK, MS, rng(0), base_ids=twin.target_ids, summary=GOOD)
    info = out.meta["varied"]
    a, b = info["span"]
    assert info["sentence"] in V.sentences("think", lead_in=True) and info["summary"] == GOOD and info["summary_tokens"] == 12
    assert TOK.decode(out.target_ids[a:b]) == "\n" + info["sentence"] + "\n" + GOOD + "\n"
    assert out.target_ids[b] == MS.think_close and out.target_ids[b:] == ep.target_ids[start + n_old:]
    assert out.signals.shape == (N + len(out.target_ids), 6) and (out.signals[N + a:N + b] == ep.signals[N + a]).all()
    assert counted(out) == (b - a < REACTION_WINDOW)
    plain = V.rewrite(ep, TOK, MS, rng(0), base_ids=twin.target_ids)  # no summary: the sentence that ends in a period
    assert plain.meta["varied"]["sentence"] in V.sentences("think") and plain.target_ids[plain.meta["varied"]["span"][1]] == MS.think_close
    with pytest.raises(ValueError):
        V.rewrite(reaction("ctx", "thinking")[0], TOK, MS, rng(0), summary=GOOD)  # thinking budget only


# --- the whole file ---------------------------------------------------------------------------

def case_of(prompt) -> int | None:
    found = re.search(r"case(\d+)", TOK.decode(prompt))
    return int(found.group(1)) if found else None


class Sampler:
    """Scripted untrained model: what it writes after the lead-in depends on the episode (`caseN`)."""

    def __init__(self, fail_on_call: int | None = None):
        self.calls, self.seen, self.fail_on_call = [], {}, fail_on_call

    def __call__(self, prompts):
        if self.fail_on_call is not None and len(self.calls) + 1 == self.fail_on_call:
            raise RuntimeError("power cut")
        self.calls.append([list(p) for p in prompts])
        out = []
        for p in prompts:
            case = case_of(p)
            self.seen[case] = self.seen.get(case, 0) + 1
            if case == 20 and self.seen[case] == 1:
                text = "it is 42 .\n</think>\n\nthe total is 42 . <eos>"  # too short, fine on the second try
            elif case == 21:
                text = GOOD + " " + CALL % ("web_search", '{"query":"q"}') + " <eos>"
            elif case == 22:
                text = " ".join(f"w{k}" for k in range(29)) + " . " + " ".join(f"v{k}" for k in range(66))
            elif case == 24:
                text = "the sum of forty and two is 42 and I keep going on and on"
            else:
                text = GOOD + "\n</think>\n\nthe total is 42 . <eos>"
            out.append((e(text), "eos" if text.endswith("<eos>") else "max_new_tokens"))
        return out


def source(tmp_path):
    """A tiny data/n6: every kind of reaction with its twin, six more thinking-budget events (cases
    20-25), a recovery episode and a reaction that does not open with a bank sentence."""
    src = tmp_path / "n6"
    src.mkdir()
    eps = []
    for i, (signal, where) in enumerate(COMBOS):
        eps += reaction(signal, where, i)
    for i in range(20, 26):
        eps += reaction("think", "thinking", i)
    eps.append(replace(reaction("net", "gating", 30)[1], episode_id="train-recovery", kind="recovery", signal="net",
                       t_cross=1, t_decision=4, t_diverge=None))
    eps.append(reaction("ctx", "answer", 31, sentence="My context looks rather full, so I compact.")[0])
    write_jsonl(src / "train.jsonl", eps)
    tests = [reaction("think", "thinking", 40, split="test")[0], reaction("ctx", "thinking", 41, split="test")[0],
             reaction("net", "gating", 42, split="test")[0], reaction("think", "thinking", 43, split="test")[0]]
    write_jsonl(src / "test.jsonl", tests)
    (src / "manifest.json").write_text(json.dumps({
        "model": "fake", "n_signals": 6, "seed": 0, "tasks_requested": {"train": 0, "test": 12},
        "episodes": {"train": len(eps), "test": len(tests)}}), encoding="utf-8")
    return src, eps


def test_build_without_summaries_rewrites_every_reaction_and_copies_the_rest(tmp_path):
    src, eps = source(tmp_path)
    manifest = V.build(src, tmp_path / "n6v", TOK, MS)
    out = read_jsonl(tmp_path / "n6v" / "train.jsonl")
    assert [o.episode_id for o in out] == [x.episode_id for x in eps] and len(out) == manifest["episodes"]["train"] == len(eps)
    old_lines = (src / "train.jsonl").read_text(encoding="utf-8").splitlines()
    new_lines = (tmp_path / "n6v" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    changed = [o.episode_id for o, a, b in zip(out, old_lines, new_lines) if a != b]
    assert changed == [x.episode_id for x in eps if reaction_start(x) is not None and "31" not in x.episode_id]
    assert all((o.meta or {}).get("varied") is not None for o in out if o.episode_id in changed)
    assert (tmp_path / "n6v" / "test.jsonl").read_bytes() == (src / "test.jsonl").read_bytes()
    assert manifest["variant"] == "varied" and manifest["source"] == str(src)
    assert manifest["source_manifest"] == {"model": "fake", "n_signals": 6, "seed": 0, "tasks_requested": {"train": 0, "test": 12}}
    assert manifest["episodes"] == {"train": len(eps), "test": 4}
    assert manifest["reactions"]["ctx"] == {"rewritten": 3, "unchanged": 1, "distinct_sentences": 3}
    assert manifest["reactions"]["think"]["rewritten"] == 8 and manifest["reactions"]["net"] == {
        "rewritten": 1, "unchanged": 0, "distinct_sentences": 1}
    assert manifest["unchanged"] == {"no_bank_sentence": {"ctx": 1}}
    assert manifest["summaries"] is None and manifest["options"]["summary"] is False
    assert manifest["generator"]["sentences"]["time"] == len(V.sentences("time"))
    assert manifest["generator"]["think_lead_in"] == len(V.sentences("think", lead_in=True))
    assert manifest["counted_by_classifier"]["think"] == {"n": 8, "before": 8, "after": 8}
    assert manifest["counted_by_classifier"]["ctx"] == {"n": 4, "before": 4, "after": 4}  # the odd one names its signal too
    assert manifest["dropped_long"] == {"max_len": 2048, "before": 0, "after": 0}
    lengths = manifest["lengths"]["time"]
    assert set(lengths) == {"old_sentence", "new_sentence", "episode"} and lengths["old_sentence"]["n"] == 3
    assert lengths["new_sentence"]["min"] <= lengths["new_sentence"]["mean"] <= lengths["new_sentence"]["max"]
    assert json.loads((tmp_path / "n6v" / "manifest.json").read_text(encoding="utf-8")) == manifest
    V.build(src, tmp_path / "again", TOK, MS)  # deterministic
    assert (tmp_path / "again" / "train.jsonl").read_bytes() == (tmp_path / "n6v" / "train.jsonl").read_bytes()
    assert V.build(src, tmp_path / "s1", TOK, MS, seed=1)["options"]["seed"] == 1
    assert (tmp_path / "s1" / "train.jsonl").read_bytes() != (tmp_path / "n6v" / "train.jsonl").read_bytes()
    with pytest.raises(ValueError):
        V.build(src, src, TOK, MS)


def test_build_counts_long_episodes_against_the_trainers_limit(tmp_path):
    src, eps = source(tmp_path)
    manifest = V.build(src, tmp_path / "n6v", TOK, MS, max_len=N + 30)
    lengths = [N + len(x.target_ids) for x in read_jsonl(tmp_path / "n6v" / "train.jsonl")]
    assert manifest["dropped_long"] == {"max_len": N + 30, "before": sum(N + len(x.target_ids) > N + 30 for x in eps),
                                        "after": sum(n > N + 30 for n in lengths)}
    assert manifest["dropped_long"]["after"] > 0 and len(lengths) == len(eps)  # counted, never removed


def test_build_asks_the_untrained_model_for_a_summary_of_every_thinking_budget_reaction(tmp_path):
    src, eps = source(tmp_path)
    sampler = Sampler()
    manifest = V.build(src, tmp_path / "n6v", TOK, MS, sampler=sampler, chunk=3)
    out = {o.episode_id: o for o in read_jsonl(tmp_path / "n6v" / "train.jsonl")}
    think = [x for x in eps if x.signal == "think" and reaction_start(x) is not None]
    first = [p for call in sampler.calls[:3] for p in call]  # try 1: eight prompts in chunks of three
    assert [len(call) for call in sampler.calls] == [3, 3, 2, 3, 2] and len(think) == 8
    for prompt, x in zip(first, think):  # prompt + the target up to the reaction + the new sentence and its line end
        start = reaction_start(x)
        assert prompt[:N + start] == x.prompt_ids + x.target_ids[:start]
        said = TOK.decode(prompt[N + start:])
        assert said.startswith("\n") and said.endswith(":\n") and said.strip() in V.sentences("think", lead_in=True)
    assert manifest["summaries"]["episodes"] == 8 and manifest["summaries"]["accepted"] == 6
    assert manifest["summaries"]["none"] == {"tool_call": 1, "no_sentence_end": 1}
    assert manifest["summaries"]["tries"] == [
        {"asked": 8, "accepted": 5, "rejected": {"too_short": 1, "tool_call": 1, "no_sentence_end": 1}},
        {"asked": 3, "accepted": 1, "rejected": {"tool_call": 1, "no_sentence_end": 1}},
        {"asked": 2, "accepted": 0, "rejected": {"tool_call": 1, "no_sentence_end": 1}}]
    assert manifest["summaries"]["by_kind"] == {"event": {"accepted": 5, "none": 2}, "starts_past": {"accepted": 1, "none": 0}}
    assert manifest["summaries"]["tokens"] == {"n": 6, "min": 12, "mean": 15.0, "max": 30}
    for x in think:
        o, case = out[x.episode_id], case_of(x.target_ids)
        info = o.meta["varied"]
        a, b = info["span"]
        assert o.target_ids[b] == MS.think_close and o.target_ids[b:] == x.target_ids[a + V.locate(x, TOK, MS).n_old:]
        if case in (21, 24):  # no accepted summary: none is kept, and the sentence closes with a period
            assert info["summary"] is None and info["sentence"] in V.sentences("think")
        else:
            assert info["sentence"] == TOK.decode(first[think.index(x)][N + a:]).strip()  # the sentence the model saw
            assert TOK.decode(o.target_ids[a:b]) == "\n" + info["sentence"] + "\n" + info["summary"] + "\n"
            assert info["summary"] == (" ".join(f"w{k}" for k in range(29)) + " ." if case == 22 else GOOD)
    assert manifest["think"]["with_summary"]["n"] == 6 and manifest["think"]["without_summary"]["n"] == 2
    assert manifest["think"]["with_summary"]["counted"] == sum(
        counted(o) for o in out.values() if o.signal == "think" and o.meta["varied"]["summary"])
    assert manifest["counted_by_classifier"]["think"]["after"] == sum(counted(o) for o in out.values() if o.signal == "think")
    assert out["train-ctx-thinking-0"].meta["varied"]["summary"] is None  # thinking budget only
    assert manifest["options"]["summary"] is True and manifest["options"]["summary_tokens"] == [8, 80]


def test_the_summary_can_be_held_inside_the_classifiers_window(tmp_path):
    src, _ = source(tmp_path)
    manifest = V.build(src, tmp_path / "n6v", TOK, MS, sampler=Sampler(), close_within=40)
    think = [o for o in read_jsonl(tmp_path / "n6v" / "train.jsonl") if o.signal == "think" and o.meta]
    spans = [o.meta["varied"]["span"][1] - o.meta["varied"]["span"][0] for o in think if o.meta["varied"]["summary"]]
    assert spans and max(spans) < 40 and manifest["options"]["close_within"] == 40
    assert manifest["summaries"]["none"].get("too_long", 0) >= 1  # the 30-token summary no longer fits
    assert manifest["think"]["with_summary"]["counted"] == manifest["think"]["with_summary"]["n"]


def test_the_summary_pass_resumes_from_its_cache(tmp_path):
    src, _ = source(tmp_path)
    out = tmp_path / "n6v"
    with pytest.raises(RuntimeError):
        V.build(src, out, TOK, MS, sampler=Sampler(fail_on_call=2), chunk=3)  # cut after the first chunk
    assert len((out / V.CACHE).read_text(encoding="utf-8").splitlines()) == 3 and not (out / "train.jsonl").exists()
    resumed = Sampler()
    V.build(src, out, TOK, MS, sampler=resumed, chunk=3)
    assert [len(call) for call in resumed.calls] == [3, 2, 3, 3]  # the first three are not asked again
    done = (out / "train.jsonl").read_bytes()

    def refuse(prompts):
        raise AssertionError("everything is in the cache")

    manifest = V.build(src, out, TOK, MS, sampler=refuse, chunk=3)
    assert (out / "train.jsonl").read_bytes() == done and manifest["summaries"]["accepted"] == 6
    other = Sampler()
    V.build(src, out, TOK, MS, sampler=other, seed=1)  # other sentences, other prompts: asked again
    assert sum(len(call) for call in other.calls) >= 8


def test_a_mini_run_takes_the_first_thinking_budget_reactions_only(tmp_path):
    src, eps = source(tmp_path)
    sampler = Sampler()
    manifest = V.build(src, tmp_path / "mini", TOK, MS, sampler=sampler, limit=3)
    out = read_jsonl(tmp_path / "mini" / "train.jsonl")
    think = [x.episode_id for x in eps if x.signal == "think" and reaction_start(x) is not None]
    assert [o.episode_id for o in out] == think[:3] and manifest["summaries"]["episodes"] == 3
    assert manifest["options"]["limit"] == 3 and not (tmp_path / "mini" / "test.jsonl").exists()
    full = {o.episode_id: o for o in read_jsonl(_full(tmp_path, src))}
    assert all(o.meta["varied"]["sentence"] == full[o.episode_id].meta["varied"]["sentence"] for o in out)


def _full(tmp_path, src):
    V.build(src, tmp_path / "full", TOK, MS, sampler=Sampler())
    return tmp_path / "full" / "train.jsonl"


# --- command line -----------------------------------------------------------------------------

def test_main_writes_the_folder_an_evaluation_reads_like_the_source(tmp_path, monkeypatch, capsys):
    from prio.eval.run import dataset_tasks

    src, eps = source(tmp_path)
    loads, batches, scripted = [], [], Sampler()

    def sample_batch(model, tok, ms, inputs, max_new_tokens, batch_size, label=""):
        batches.append((model, max_new_tokens, batch_size, len(inputs)))
        return scripted(inputs)

    monkeypatch.setattr(V, "_load_tokenizer", lambda model: (TOK, MS))
    how = []
    monkeypatch.setattr(V, "load_base", lambda model, **kw: (loads.append(model), how.append(kw)) and ("the model", TOK, MS))
    monkeypatch.setattr(V, "sample_batch", sample_batch)
    monkeypatch.setattr(TOK, "apply_chat_template", _chat([]), raising=False)  # the command asks on a separate request
    out = tmp_path / "n6v"
    V.main(["--src", str(src), "--out", str(out), "--model", "fake/model", "--no-summary"])
    printed = json.loads(capsys.readouterr().out)
    assert loads == [] and printed == json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert printed["variant"] == "varied" and printed["options"]["model"] == "fake/model" and printed["summaries"] is None
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "test.jsonl", "train.jsonl"]
    assert (out / "test.jsonl").read_bytes() == (src / "test.jsonl").read_bytes()
    assert len(dataset_tasks(out)) == len(dataset_tasks(src)) == 12  # what the evaluation needs of the manifest

    full = tmp_path / "n6v-full"
    V.main(["--src", str(src), "--out", str(full), "--seed", "0", "--batch-size", "4"])  # the model of the source manifest
    capsys.readouterr()
    printed = json.loads((full / "manifest.json").read_text(encoding="utf-8"))
    assert loads == ["fake"] and batches[0][:3] == ("the model", V.SUMMARY_NEW_TOKENS, 4)
    assert printed["summaries"]["accepted"] == 6 and printed["options"]["batch_size"] == 4
    assert printed["options"]["summary_method"] == "request" and how == [{"device": "cuda"}]
    assert (full / V.CACHE).exists()
    V.main(["--src", str(src), "--out", str(full), "--batch-size", "4"])  # everything cached: no model
    assert loads == ["fake"]

    mini = tmp_path / "mini"
    V.main(["--src", str(src), "--out", str(mini), "--limit", "2", "--show", "2"])
    text = capsys.readouterr().out
    assert len(read_jsonl(mini / "train.jsonl")) == 2 and GOOD in text and "case9" in text  # printed for reading by hand

    only = tmp_path / "stats"
    V.main(["--src", str(src), "--out", str(only), "--no-summary", "--manifest-only"])
    assert [p.name for p in only.iterdir()] == ["manifest.json"]
    with pytest.raises(ValueError):
        V.main(["--src", str(src), "--out", str(src), "--no-summary"])


# --- reading an evaluation --------------------------------------------------------------------

def _record(ep, paired="event", condition="on", signal=None, reaction="own", kind=None):
    signal = signal or ep.signal
    return {"episode_id": ep.episode_id, "signal": signal, "kind": kind or ep.kind, "phase": ep.phase, "condition": condition,
            "paired": paired, "event_occurred": paired in ("event", "swapped"), "reaction_signal": signal if reaction == "own" else reaction,
            "delay": 2, "named": reaction is not None, "called_gated_tool": False, "used_alternative": False, "unnamed_call": False}


def _results(tmp_path, data):
    """A scripted quick evaluation of four test episodes against the varied training file."""
    train = read_jsonl(data / "train.jsonl")
    tests = read_jsonl(data / "test.jsonl")
    think0, ctx1, net2, think3 = tests
    seen = {s: [x.meta["varied"] for x in train if x.signal == s and x.meta and x.meta.get("varied")] for s in SIGNALS}
    with_summary = next(v["sentence"] for v in seen["think"] if v["summary"])
    plain = next(v["sentence"] for v in seen["think"] if not v["summary"])
    answer = "\n</think>\n\nthe total is 42 ."
    runs = [
        # thinking budget: a training sentence and a summary; a new sentence with the summary on the same line;
        # a training sentence and no summary; the cut word finished first, then a training sentence
        (_record(think0), "\n" + with_summary + "\n" + GOOD + answer),
        (_record(ctx1, paired="swapped", signal="think", kind="swap"), "\nMy thinking budget is done, so in short: " + GOOD + answer),
        (_record(think3, kind="starts_past"), "\n" + plain + answer),
        (_record(think3, paired="swapped"), "again\n" + plain + "\n" + GOOD + answer),
        # other signals: seen, not seen
        (_record(ctx1), "\n" + seen["ctx"][0]["sentence"] + "\n</think>\n\n" + ACTION["ctx"]),
        (_record(net2), "I am offline today, so the local store it is.\n\n" + ACTION["net"]),
        # not counted: a miss, the channel off, a quiet pairing
        (_record(net2, paired="swapped", signal="ram", kind="swap", reaction=None), "and so on"),
        (_record(think0, condition="zeroed", reaction=None), "and so on"),
        (_record(think0, paired="nominal", reaction=None), "and so on"),
    ]
    out = tmp_path / "results"
    out.mkdir()
    with open(out / "records.jsonl", "w", encoding="utf-8") as records, open(out / "generations.jsonl", "w", encoding="utf-8") as gens:
        for r, text in runs:
            records.write(json.dumps(r) + "\n")
            gens.write(json.dumps({"episode_id": r["episode_id"], "condition": r["condition"], "paired": r["paired"],
                                   "signal": r["signal"], "stop_reason": "max_new_tokens", "token_ids": e(text)}) + "\n")
    return out, with_summary, plain


def test_wording_reading_counts_sentences_seen_in_training_and_summaries(tmp_path, monkeypatch, capsys):
    import wording_reading as W

    src, _ = source(tmp_path)
    data = tmp_path / "n6v"
    V.build(src, data, TOK, MS, sampler=Sampler())
    results, with_summary, plain = _results(tmp_path, data)
    assert W.naming_sentence("\n" + plain + "\n</think>", "think") == (plain, plain)
    assert W.naming_sentence("again\n" + with_summary + "\n" + GOOD, "think") == ("again", with_summary)
    assert W.naming_sentence("My thinking budget is done, so in short: " + GOOD, "think")[1] == "My thinking budget is done, so in short:"
    assert W.naming_sentence("I will compact now: my context is almost full.\n", "ctx")[1] == "I will compact now: my context is almost full."

    monkeypatch.setattr(W, "_load_tokenizer", lambda model: (TOK, MS))
    W.main([str(results), str(data), "--out", str(tmp_path / "reading.md"), "--examples", "2"])
    text = capsys.readouterr().out
    assert (tmp_path / "reading.md").read_text(encoding="utf-8").strip() == text.strip()
    table = json.loads((tmp_path / "reading.json").read_text(encoding="utf-8"))
    rows_ = {r["signal"]: r for r in table["rows"]}
    assert rows_["think"] == {"signal": "think", "n": 4, "opens_seen": 2, "naming_seen": 3, "distinct": 3,
                              "training_distinct": 8, "training_reactions": 8}
    assert (rows_["ctx"]["n"], rows_["ctx"]["opens_seen"], rows_["net"]["n"], rows_["net"]["naming_seen"]) == (1, 1, 1, 0)
    assert (rows_["all"]["n"], rows_["all"]["opens_seen"], rows_["all"]["naming_seen"], rows_["all"]["distinct"]) == (6, 3, 4, 5)
    assert "ram" not in rows_  # no counted reaction
    by_kind = table["summary"].pop("by_kind")  # second note on A9: the rate among `event` reactions beside the line
    assert table["summary"] == {"n": 4, "with_summary": 3, "rate": 0.75, "line": 0.8, "passes": False, "min_tokens": 8}
    assert by_kind == {"event": {"n": 2, "with_summary": 2, "rate": 1.0}, "starts_past": {"n": 1, "with_summary": 0, "rate": 0.0},
                       "swap": {"n": 1, "with_summary": 1, "rate": 1.0}}
    assert "after some thinking (kind `event`): 2 / 2 = 100.0%" in text and "starts_past 0 / 1 = 0.0%" in text
    assert "| think | 4 | 2 / 4 = 50.0% | 3 / 4 = 75.0% | 3 | 8 of 8 |" in text
    assert "3 / 4 = 75.0%" in text and "not met" in text
    # examples for reading by hand: the thinking before, the sentence, the summary, the answer
    assert text.count("### ") == 2 and "case40" in text and with_summary in text and GOOD in text and "the total is 42 ." in text
    W.main([str(results), str(data), "--examples", "0"])
    assert "### " not in capsys.readouterr().out


def test_wording_reading_of_a_bank_trained_model(tmp_path, monkeypatch, capsys):
    import wording_reading as W

    src, eps = source(tmp_path)  # the source itself: bank sentences, no summary
    tests = read_jsonl(src / "test.jsonl")
    out = tmp_path / "results"
    out.mkdir()
    bank = REACTION_BANK["think"][20 % len(REACTION_BANK["think"])]  # the sentence of training case 20
    runs = [(_record(tests[0]), "\n" + bank + "\n</think>\n\nthe total is 42 ."),
            (_record(tests[3]), "\n" + REACTION_BANK["think"][5] + "\n</think>\n\nthe total is 42 .")]  # in the bank, not in the file
    with open(out / "records.partial.jsonl", "w", encoding="utf-8") as records, open(out / "generations.jsonl", "w", encoding="utf-8") as gens:
        for r, text in runs:
            records.write(json.dumps(r) + "\n")
            gens.write(json.dumps({"episode_id": r["episode_id"], "condition": "on", "paired": "event", "signal": r["signal"],
                                   "stop_reason": "max_new_tokens", "token_ids": e(text)}) + "\n")
    monkeypatch.setattr(W, "_load_tokenizer", lambda model: (TOK, MS))
    W.main([str(out), str(src)])
    text = capsys.readouterr().out
    assert "| think | 2 | 1 / 2 = 50.0% | 1 / 2 = 50.0% | 2 |" in text and "0 / 2 = 0.0%" in text


# --- the real tokenizer (CPU) -----------------------------------------------------------------

@pytest.mark.slow
def test_qwen_tokens_every_sentence_is_found_by_the_unchanged_classifier():
    from transformers import AutoTokenizer

    from prio.datagen.teacher import _call_text
    from prio.modelspec import resolve_spec

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    ms = resolve_spec("Qwen/Qwen3-1.7B", tok)
    enc = lambda text: tok(text, add_special_tokens=False).input_ids
    longest = {}
    for signal in SIGNALS:
        for s in V.sentences(signal) + (V.sentences("think", lead_in=True) if signal == "think" else []):
            assert names_signal(s, signal) and not any(names_signal(s, o) for o in SIGNALS if o != signal), s
            longest[signal] = max(longest.get(signal, 0), len(enc("\n" + s + "\n")))
    assert max(longest.values()) <= 40 and longest["time"] <= 26, longest  # room for the marker inside the window
    one, two = "The price is 3.5 dollars per unit.", "The price is 3.5 dollars per unit. So 12 units cost 42 dollars."
    text = two + "\n1. Then"
    ends = V.sentence_ends(enc(text), tok)  # a decimal point and a list number end no sentence
    assert [tok.decode(enc(text)[:k]).strip() for k in ends] == [one, two]
    assert V.pick_summary(enc(text) + [ms.think_close], tok, ms) == (two, "accepted")

    action = {"ctx": _call_text("compact_context", {}), "energy": _call_text("save_state", {}) + "\n" + _call_text("sleep", {}),
              "net": _call_text("local_lookup", {"query": "air quality in Addis Ababa"}),
              "ram": _call_text("run_light", {"job": "forecast"}), "think": "Each room has 9 pieces, so 80 rooms have 720.",
              "time": "The total is 720 pieces."}
    prompt = enc("<|im_start|>user\nHow many pieces are in 80 rooms?<|im_end|>\n<|im_start|>assistant\n")
    thinking = enc("<think>\nOkay, each room has 2 beds, 1 dresser, 2 pillowcases per bed and so on. Let me add these up")
    answer = thinking + enc(" carefully.\n</think>\n\nEach room has")
    summary = "Each room has 9 pieces, so 80 rooms need 720."
    fitted = 0
    for signal in SIGNALS:
        for where in ("thinking", "answer"):
            if (signal, where) in (("think", "answer"), ("net", "thinking"), ("ram", "thinking")):
                continue
            prefix, lead, tail = (thinking, "\n", "\n</think>\n\n") if where == "thinking" else (answer, "\n\n", "\n\n")
            for k, old in enumerate(REACTION_BANK[signal][::5]):
                target = prefix + enc(lead + old + tail + action[signal]) + [tok.eos_token_id]
                ep = Episode(episode_id=f"x-{signal}-{where}-{k}", task_id="t", family="math", split="train", kind="event",
                             signal=signal, phase=where, prompt_ids=prompt, target_ids=target,
                             signals=rows(len(prompt) + len(target)), t_cross=len(prefix) - 1, t_decision=None,
                             templated=False, t_diverge=len(prefix))
                found = V.locate(ep, tok, ms)
                assert found is not None and found.sentence == old and found.lead == lead
                for seed in range(8):
                    out = V.rewrite(ep, tok, ms, rng([k, seed]), summary=summary if signal == "think" and seed % 2 else None)
                    a, b = out.meta["varied"]["span"]
                    assert out.target_ids[:a] == target[:a] and out.target_ids[b:] == target[a + found.n_old:]
                    said = tok.decode(out.target_ids[a:b])
                    assert said.strip().startswith(out.meta["varied"]["sentence"]) and said.startswith(lead)
                    r = find_reaction(out.target_ids[a:], "eos", signal, tok, ms)
                    if signal == "think" and seed % 2:  # with a summary: counted while the thinking closes inside the window
                        assert said == lead + out.meta["varied"]["sentence"] + "\n" + summary + "\n"
                        assert (r is not None and r.named and r.token_index == b - a) == (b - a < REACTION_WINDOW), said
                        fitted += b - a < REACTION_WINDOW
                    else:
                        assert r is not None and r.named and r.token_index < REACTION_WINDOW, said
                        assert said == lead + out.meta["varied"]["sentence"] + ("\n" if where == "thinking" else "\n\n")
                    assert not any((o := find_reaction(out.target_ids[a:], "eos", other, tok, ms)) and o.named
                                   for other in SIGNALS if other != signal), said
    assert fitted >= 10


# --- the summary asked for on a separate request (note on A9 of 2026-11-17) ----------------------

def _chat(seen):
    def template(messages, add_generation_prompt=False, enable_thinking=True, tokenize=False, **_):
        seen.append({"messages": messages, "thinking": enable_thinking, "generation": add_generation_prompt})
        return " ".join(f"<{m['role']}> {m['content']}" for m in messages) + " <assistant>"
    return template


def test_a_summary_request_holds_the_notes_and_asks_for_one_first_person_sentence(monkeypatch):
    seen = []
    monkeypatch.setattr(TOK, "apply_chat_template", _chat(seen), raising=False)
    ids = V.summary_request(TOK, "  8 minus 3 is 5 fish .\nthen 5 more  ")
    system, user = seen[0]["messages"]
    assert (system["role"], user["role"]) == ("system", "user") and seen[0] == {**seen[0], "thinking": False, "generation": True}
    assert "8 minus 3 is 5 fish .\nthen 5 more" in user["content"] and "  8 minus" not in user["content"]
    assert "one sentence" in user["content"] and "first person" in user["content"]
    assert TOK.decode(ids).endswith("<assistant>") and "8 minus 3 is 5 fish" in TOK.decode(ids)
    V.summary_request(TOK, " \n ")  # a reaction at the very start of the turn: there are no notes yet
    assert V.NO_NOTES in seen[1]["messages"][1]["content"]


def test_build_can_ask_for_the_summary_on_a_separate_request(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(TOK, "apply_chat_template", _chat(seen), raising=False)
    src, eps = source(tmp_path)
    sampler = Sampler()
    manifest = V.build(src, tmp_path / "n6v", TOK, MS, sampler=sampler, chunk=3, method="request")
    think = [x for x in eps if x.signal == "think" and reaction_start(x) is not None]
    first = [p for call in sampler.calls[:3] for p in call]
    for prompt, x in zip(first, think):  # a request of its own: the notes are the thinking so far, without its opener
        notes = TOK.decode([t for t in x.target_ids[:reaction_start(x)] if t != MS.think_open])
        assert prompt == V.summary_request(TOK, notes) and prompt[:N] != x.prompt_ids
    assert manifest["options"]["summary_method"] == "request"
    assert manifest["summaries"]["episodes"] == 8 and manifest["summaries"]["accepted"] == 6  # same answers, same checks
    out = {o.episode_id: o for o in read_jsonl(tmp_path / "n6v" / "train.jsonl")}
    info = out[think[-1].episode_id].meta["varied"]
    a, b = info["span"]
    assert info["summary"] == GOOD and info["sentence"] in V.sentences("think", lead_in=True)
    assert TOK.decode(out[think[-1].episode_id].target_ids[a:b]) == "\n" + info["sentence"] + "\n" + GOOD + "\n"


def test_the_summary_method_of_a_plain_build_is_the_continuation(tmp_path):
    src, _ = source(tmp_path)
    assert V.build(src, tmp_path / "n6v", TOK, MS, sampler=Sampler())["options"]["summary_method"] == "continue"
    with pytest.raises(ValueError):
        V.build(src, tmp_path / "other", TOK, MS, sampler=Sampler(), method="guess")


def test_a_summary_that_repeats_the_request_is_not_a_summary():
    assert V.echoes_request("I was thinking about a task and have to stop thinking now.")
    assert V.echoes_request("My notes so far: Kris was suspended for bullying.")
    assert not V.echoes_request(GOOD) and not V.echoes_request("I have to stop here; the total is 42 so far.")


def test_an_echo_of_the_request_is_refused_and_asked_again(tmp_path, monkeypatch):
    monkeypatch.setattr(TOK, "apply_chat_template", _chat([]), raising=False)
    src, eps = source(tmp_path)

    class Echo(Sampler):
        def __call__(self, prompts):
            out = super().__call__(prompts)
            echo = (e("I was thinking about a task and have to stop thinking now . <eos>"), "eos")
            return [echo if case_of(p) == 23 else row for p, row in zip(prompts, out)]

    manifest = V.build(src, tmp_path / "n6v", TOK, MS, sampler=Echo(), method="request")
    out = {o.episode_id: o for o in read_jsonl(tmp_path / "n6v" / "train.jsonl")}
    echoed = next(o for o in out.values() if case_of(o.target_ids) == 23 and o.signal == "think")
    assert echoed.meta["varied"]["summary"] is None and manifest["summaries"]["accepted"] == 5
    assert manifest["summaries"]["none"] == {"tool_call": 1, "no_sentence_end": 1, "echo": 1}


def test_each_further_try_asks_for_a_shorter_summary(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(TOK, "apply_chat_template", _chat(seen), raising=False)
    assert V.SUMMARY_WORDS == (12, 9, 7)
    for words in V.SUMMARY_WORDS:
        V.summary_request(TOK, "8 minus 3 is 5 .", words=words)
        assert f"at most {words} words" in seen[-1]["messages"][1]["content"]
    # nothing worked out yet (a reaction at the start of the turn): a full sentence that says so, long enough to be kept
    assert "If nothing is worked out yet, write a sentence of eight to twelve words that says so." in seen[-1]["messages"][1]["content"]
    src, eps = source(tmp_path)
    sampler = Sampler()
    V.build(src, tmp_path / "n6v", TOK, MS, sampler=sampler, method="request")
    think = {case_of(x.target_ids): x for x in eps if x.signal == "think" and reaction_start(x) is not None}
    again = sampler.calls[1]  # try 2: the three that had no accepted summary on try 1
    for prompt in again:
        x = think[case_of(prompt)]
        notes = TOK.decode([t for t in x.target_ids[:reaction_start(x)] if t != MS.think_open])
        assert prompt == V.summary_request(TOK, notes, words=9) != V.summary_request(TOK, notes)
