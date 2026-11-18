"""State-table core (preregistration amendment A8, shared protocol): sentence generator, prompt,
timing kinds, sibling families and their loss weights. The domain here is a toy one
(tests/table_helpers.py), so nothing of the messenger is assumed."""

import numpy as np
import pytest

from prio.reactions import parse_tool_calls
from prio.table.core import Bank, StateClass, Variant, build_prompt, draw_positions, family, state_rows, status_text
from table_helpers import DOOR, HEAT, MS, TOK, TOY, TOY_STATES, call, decision, e, toy_policy, toy_run

N = 2  # signals of the toy domain: heat, door


def rng(seed=0):
    return np.random.default_rng(seed)


def table(ep) -> dict:
    return ep.meta["table"]


# --- sentence generator -----------------------------------------------------------------------

BANK = Bank(shapes=("{reason}, so {action}.", "{action}, because {reason}.", "no way out, so {action}."),
            parts={"reason": ("the door is shut", "the {door} door will not open"),
                   "action": ("I call {who}", "I wait", "{who} gets a call")})


def test_bank_lists_every_distinct_sentence_with_the_episode_slots_filled():
    sentences = BANK.sentences(door="back", who="Ana")
    assert len(sentences) == len(set(sentences)) == 2 * 3 + 3 * 2 + 3
    assert "The back door will not open, so I call Ana." in sentences  # a part may hold a slot
    assert "Ana gets a call, because the door is shut." in sentences  # action first; the first letter is capitalized
    assert "No way out, so I wait." in sentences  # a short shape that uses one part only
    assert not any("{" in s for s in sentences)


def test_bank_draw_is_one_of_its_sentences_follows_the_rng_and_spreads_over_the_shapes():
    every = set(BANK.sentences(door="back", who="Ana"))
    drawn = [BANK.draw(rng(seed), door="back", who="Ana") for seed in range(200)]
    assert set(drawn) <= every and len(set(drawn)) >= 12
    assert BANK.draw(rng(3), door="back", who="Ana") == BANK.draw(rng(3), door="back", who="Ana")
    short = sum(s.startswith("No way out") for s in drawn)  # a shape is drawn first: the short one gets a third
    assert 40 <= short <= 95


def test_bank_refuses_a_missing_slot():
    with pytest.raises(KeyError):
        BANK.sentences(who="Ana")  # `{door}` is not given


def test_state_class_checks_its_row():
    with pytest.raises(ValueError):
        StateClass("x", "shout")
    with pytest.raises(ValueError):
        StateClass("x", "act", keywords=(("hot",),), bank=BANK)  # an act class needs its command
    with pytest.raises(ValueError):
        StateClass("x", "mention", bank=BANK)  # naming needs keywords


# --- prompt and status text -------------------------------------------------------------------

def test_prompt_lists_the_domains_tools_and_system_prompt():
    ids = build_prompt(TOK, TOY, "watch room 1")
    text = TOK.decode(ids)
    assert text.startswith("<tools> note open_door alarm </tools>") and "You guard a room." in text
    assert text.endswith("<|im_start|>user watch room 1 <|im_end|> <|im_start|>assistant")
    assert "a card" in TOK.decode(build_prompt(TOK, TOY, "hello", system="a card"))  # a per-task system prompt


def test_status_text_is_the_harness_format_for_the_named_signals():
    assert status_text((HEAT, DOOR), {"heat": 0.905, "door": 0.0}) == "[status heat=0.90 door=0.00]"
    assert status_text((HEAT, DOOR), {"door": 1.0}) == "[status door=1.00]"


# --- timing kinds -----------------------------------------------------------------------------

def test_draw_positions_per_timing_kind():
    for seed in range(50):
        assert draw_positions("from_start", 9, rng(seed)) == (None, None)
        t_on, t_back = draw_positions("onset", 9, rng(seed))
        assert 0 <= t_on <= 8 and t_back is None
        t_on, t_back = draw_positions("back", 9, rng(seed))
        assert 0 <= t_on < t_back <= 8  # the token before the cut carries the old state again
    assert {draw_positions("onset", 4, rng(s))[0] for s in range(80)} == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        draw_positions("back", 2, rng(0))


def test_state_rows_follow_the_timing_kind():
    quiet = np.tile(np.array([0.105, 1.0], dtype=np.float32), (20, 1))
    names, hot = ["heat", "door"], {"heat": 0.905}
    assert np.array_equal(state_rows(quiet, 5, 8, Variant("quiet"), names), quiet)
    start = state_rows(quiet, 5, 8, Variant("hot", hot, "from_start"), names)
    assert (start[:, 0] == np.float32(0.905)).all() and (start[:, 1] == 1.0).all()  # from the first prompt token
    onset = state_rows(quiet, 5, 8, Variant("hot", hot, "onset", t_on=3), names)
    assert (onset[:8, 0] == np.float32(0.105)).all() and (onset[8:, 0] == np.float32(0.905)).all()
    back = state_rows(quiet, 5, 8, Variant("hot", hot, "back", t_on=2, t_back=6), names)
    assert [bool(v > 0.5) for v in back[5:, 0]] == [False, False, True, True, True, True] + [False] * 9
    assert not np.shares_memory(start, quiet)
    for bad in (Variant("hot", hot, "onset", t_on=8), Variant("hot", hot, "back", t_on=2, t_back=8),
                Variant("hot", hot, "back", t_on=3, t_back=3), Variant("hot", hot, "sometimes"),
                Variant("hot", hot), Variant("hot", {"cold": 1.0}, "from_start")):
        with pytest.raises(ValueError):
            state_rows(quiet, 5, 8, bad, names)


# --- sibling families -------------------------------------------------------------------------

def toy_family(run=None, variants=None, seed=0, **kw):
    run = run or toy_run()
    cut = decision(run)
    variants = variants or [
        Variant("quiet"),
        Variant("hot_open", TOY_STATES["hot_open"], "from_start"),
        Variant("hot_shut", TOY_STATES["hot_shut"], "onset", t_on=2),
        Variant("warm", TOY_STATES["warm"], "onset", t_on=cut - 1),
        Variant("hot_open-back", TOY_STATES["hot_open"], "back", t_on=1, t_back=4),
    ]
    return run, cut, family(TOY, run, cut, variants, rng(seed), TOK, MS, family_id=run.task.task_id, **kw)


def test_siblings_share_the_text_up_to_the_cut_and_differ_only_in_the_signal_rows():
    run, cut, eps = toy_family()
    n = len(run.prompt_ids)
    assert [ep.episode_id for ep in eps] == [f"toy-train-00000-{v}" for v in
                                             ("quiet", "hot_open", "hot_shut", "warm", "hot_open-back")]
    for ep in eps:
        assert ep.prompt_ids == run.prompt_ids and ep.target_ids[:cut] == run.target_ids[:cut]
        assert ep.signals.shape == (n + len(ep.target_ids), N) and ep.signals.dtype == np.float32
        assert (ep.t_decision, ep.t_diverge, ep.split, ep.family, ep.task_id) == (cut, cut, "train", "toy", "toy-train-00000")
        assert table(ep)["family"] == "toy-train-00000" and table(ep)["cut"] == cut and table(ep)["domain"] == "toy"
    quiet, hot_open, hot_shut, warm, back = eps
    heat = lambda ep: [round(float(v), 3) for v in ep.signals[:n + cut, 0]]
    assert heat(quiet) == [0.105] * (n + cut)
    assert heat(hot_open) == [0.905] * (n + cut)  # from the first prompt token
    assert heat(hot_shut) == [0.105] * (n + 2) + [0.905] * (cut - 2)  # onset inside the turn
    assert heat(warm) == [0.105] * (n + cut - 1) + [0.605]  # the token before the cut is the first to carry it
    assert heat(back) == [0.105] * (n + 1) + [0.905] * 3 + [0.105] * (cut - 4)  # changed and returned
    assert [float(ep.signals[n + cut - 1, 1]) for ep in eps] == [1.0, 1.0, 0.0, 1.0, 1.0]
    assert (hot_shut.signals[n + 2:, 1] == 0.0).all() and (hot_shut.signals[:n + 2, 1] == 1.0).all()


def test_the_policy_table_read_at_the_decision_gives_class_behavior_and_kind():
    _, _, eps = toy_family()
    assert [(table(ep)["cls"], table(ep)["behavior"], ep.kind, ep.signal) for ep in eps] == [
        ("quiet", "base", "table_base", None), ("hot_open", "act", "table_act", "hot_open"),
        ("hot_shut", "act", "table_act", "hot_shut"), ("warm", "mention", "table_mention", "warm"),
        ("quiet", "base", "table_base", None)]  # back: the state returned before the decision
    assert [(table(ep)["timing"], table(ep)["t_on"], table(ep)["t_back"], ep.t_cross) for ep in eps][-1] == ("back", 1, 4, 1)
    assert [table(ep)["timing"] for ep in eps] == [None, "from_start", "onset", "onset", "back"]
    assert [table(ep)["signals"] for ep in eps] == [[], ["door", "heat"], ["door", "heat"], ["heat"], ["door", "heat"]]
    assert table(eps[2])["state"] == {"heat": pytest.approx(0.905), "door": 0.0}
    assert [ep.templated for ep in eps] == [False, True, True, True, False]
    assert all(ep.phase == "answer" for ep in eps)


def test_base_sibling_is_the_base_run_unchanged():
    run, cut, eps = toy_family()
    for ep in (eps[0], eps[4]):
        assert ep.target_ids == run.target_ids and ep.focus_spans is None
        assert table(ep)["command"] is None and table(ep)["keywords"] == []


def test_act_sibling_is_a_bank_sentence_then_the_command_and_the_turn_ends():
    run, cut, eps = toy_family()
    hot_open, hot_shut = eps[1], eps[2]
    assert TOK.decode(hot_open.target_ids[cut:]) == "It is too hot so I leave a note . " + call("note", text="all quiet here") + " <eos>"
    assert table(hot_open)["command"] == {"name": "note", "arguments": {"text": "all quiet here"}}  # copied from the base run
    said = TOK.decode(hot_shut.target_ids[cut:])
    assert said in {s + " " + call("open_door", which="front") + " <eos>" for s in
                    ("It is too hot and the front door is shut .", "The front door is shut and it is hot .")}
    assert table(hot_shut)["command"] == {"name": "open_door", "arguments": {"which": "front"}}  # fixed by the table
    assert table(hot_shut)["keywords"] == [["hot"], ["door"]] and table(hot_shut)["conditional"] is True
    for ep in (hot_open, hot_shut):
        assert ep.focus_spans == [[cut, len(ep.target_ids)]]  # sentence + command + end of turn
        assert [c.name for c in parse_tool_calls(ep.target_ids, TOK, MS)] == [table(ep)["command"]["name"]]
        assert table(ep)["own_command"] == {"name": "note", "arguments": {"text": "all quiet here"}}
        assert table(ep)["reaction_commands"] == ["alarm", "open_door"]  # the base run's own command is no reaction


def test_a_copied_argument_keeps_its_characters():
    text = "caf\N{LATIN SMALL LETTER E WITH ACUTE} at 6 \N{EM DASH} see you"  # as the base model wrote it
    _, cut, eps = toy_family(toy_run(note=text))
    said = TOK.decode(eps[1].target_ids[cut:])
    assert text in said and "\\u" not in said  # no escape sequences in the training target
    assert table(eps[1])["command"]["arguments"] == {"text": text}


def test_a_fact_of_the_task_changes_the_command_and_is_kept():
    run = toy_run(facts={"role": "cook", "held_out": True})
    _, cut, eps = toy_family(run)
    assert table(eps[2])["command"] == {"name": "open_door", "arguments": {"which": "back"}}
    assert "back door" in TOK.decode(eps[2].target_ids[cut:])  # the class's own slot fills the sentence
    assert table(eps[2])["held_out"] is True and table(eps[2])["facts"]["role"] == "cook"
    assert table(eps[0])["user"] == "watch room 0" and table(eps[0])["system"] is None


def test_mention_sibling_inserts_the_sentence_and_goes_on_with_the_base_run():
    run, cut, eps = toy_family()
    warm = eps[3]
    a, b = warm.focus_spans[0]
    said = TOK.decode(warm.target_ids[a:b])
    assert a == cut and said in {"It is getting warm in here .", "It is getting stuffy in here .",
                                 "Warm in here , friend .", "Stuffy in here , friend ."}  # a fact fills `{who}`
    assert warm.target_ids[b:] == run.target_ids[cut:] and table(warm)["command"] is None
    assert table(warm)["keywords"] == [["warm", "stuffy"]]


def test_a_cut_inside_the_thinking_closes_it_before_the_command():
    run = toy_run()
    eps = family(TOY, run, 3, [Variant("quiet"), Variant("hot_shut", TOY_STATES["hot_shut"], "onset", t_on=2),
                               Variant("warm", TOY_STATES["warm"], "from_start")], rng(0), TOK, MS, family_id="f")
    act, warm = eps[1], eps[2]
    words = TOK.decode(act.target_ids[3:]).split()
    assert words.index("</think>") < words.index("<tool_call>") and words[-1] == "<eos>"
    assert words.index("shut") < words.index("</think>")  # the sentence is part of the thinking
    assert all(ep.phase == "thinking" for ep in eps)
    assert warm.target_ids[warm.focus_spans[0][1]:] == run.target_ids[3:]  # the thinking goes on after the aside


def test_think_signal_is_nominal_after_the_thinking_in_train_and_keeps_running_in_test():
    from dataclasses import replace

    from prio.signals import CORE_SIGNALS
    from prio.trajectories import sample_trajectory

    think = replace(TOY, specs=(HEAT, DOOR, CORE_SIGNALS[1]),
                    background=lambda specs, length, r: np.concatenate(
                        [np.tile(np.array([0.105, 1.0], dtype=np.float32), (length, 1)),
                         sample_trajectory(specs[2:], length, r).values], axis=1))
    for split in ("train", "test"):
        run = toy_run(split=split)
        cut, n = decision(run), len(run.prompt_ids)
        eps = family(think, run, cut, [Variant("quiet"), Variant("hot_open", TOY_STATES["hot_open"], "from_start")],
                     rng(0), TOK, MS, family_id="f")
        close = run.target_ids.index(MS.think_close)
        for ep in eps:
            assert (ep.signals[:n + close + 1, 2] < 1.0).all()  # the budget counts down while thinking
            assert (ep.signals[n + close + 1:, 2] == 1.0).all() == (split == "train")


def test_default_background_keeps_every_signal_away_from_its_limit():
    from dataclasses import replace

    from prio.signals import CORE_SIGNALS

    plain = replace(TOY, specs=CORE_SIGNALS, background=None,
                    policy=lambda state, facts: toy_policy({"heat": 0.0, "door": 1.0}, facts))
    run = toy_run()
    (ep,) = family(plain, run, decision(run), [Variant("quiet")], rng(0), TOK, MS, family_id="f")
    assert ep.signals.shape[1] == 6
    assert not any(s.past_limit(float(v)) for row in ep.signals for s, v in zip(CORE_SIGNALS, row))


def test_family_is_deterministic_stays_whole_and_checks_its_input():
    run, cut, eps = toy_family(seed=5)
    assert toy_family(seed=5)[2] == eps
    assert toy_family(max_len=len(run.prompt_ids) + len(run.target_ids) + 3)[2] == []  # the act siblings are longer
    assert len(toy_family(max_len=len(run.prompt_ids) + len(run.target_ids) + 40)[2]) == 5
    with pytest.raises(ValueError):
        toy_family(variants=[Variant("quiet"), Variant("quiet")])  # names must differ
    with pytest.raises(ValueError):
        family(TOY, run, len(run.target_ids), [Variant("quiet")], rng(0), TOK, MS, family_id="f")
    bare = toy_run()
    bare.target_ids[:] = e("<think> t1 </think> a1 a2 a3 <eos>")  # no command of its own to copy from
    with pytest.raises(ValueError, match="text"):
        family(TOY, bare, 4, [Variant("hot_open", TOY_STATES["hot_open"], "from_start")], rng(0), TOK, MS, family_id="f")


# --- loss weights -----------------------------------------------------------------------------

def test_collate_weights_the_reaction_and_every_siblings_first_token_after_the_cut():
    from prio.train import collate

    run, cut, eps = toy_family()
    batch = collate(eps, pad_id=0, n_signals=N, reaction_weight=20.0)
    n = len(run.prompt_ids)
    for row, ep in enumerate(eps):
        weights = batch["weights"][row, :n + len(ep.target_ids)].tolist()
        heavy = {n + i for a, b in ep.focus_spans or () for i in range(a, b)} | {n + cut}
        assert [i for i, w in enumerate(weights) if w == 20.0] == sorted(heavy)
        assert set(weights) <= {1.0, 20.0}
        marks = batch["decision"][row].tolist()
        assert [i for i, m in enumerate(marks) if m] == [n + cut]
        assert marks[n + cut] == (2 if ep.kind == "table_base" else 1)  # hold where nothing is to be done, react elsewhere
        assert (batch["labels"][row, n:n + len(ep.target_ids)] == batch["input_ids"][row, n:n + len(ep.target_ids)]).all()
    assert batch["signal"] == [None, "hot_open", "hot_shut", "warm", None]


def test_a_tiny_model_trains_on_a_family(tiny_model):
    from prio.channel import ChannelModel
    from prio.train import fit

    _, _, eps = toy_family()
    for ep in eps:  # the tiny model has 100 embedding rows
        ep.prompt_ids[:] = [t % 100 for t in ep.prompt_ids]
        ep.target_ids[:] = [t % 100 for t in ep.target_ids]
    history: dict = {}
    losses = fit(ChannelModel(tiny_model, N), eps, pad_id=0, epochs=1, batch_size=2, grad_accum=1, device="cpu",
                 history=history)
    assert len(losses) == 3 and all(np.isfinite(losses))
    assert len(history["react"]) == 3 and len(history["hold"]) == 2
    assert set(history["react_by_signal"]) == {"hot_open", "hot_shut", "warm"}


# --- extensions for a second domain (R22, prio.npc) -------------------------------------------

def test_whole_values_hold_on_every_row_whatever_the_timing():
    quiet = np.tile(np.array([0.105, 1.0], dtype=np.float32), (20, 1))
    names = ["heat", "door"]
    rows = state_rows(quiet, 5, 8, Variant("hot", {"heat": 0.905}, "onset", t_on=3, whole={"door": 0.0}), names)
    assert (rows[:, 1] == 0.0).all()  # what is around was so all along (the door was shut)
    assert (rows[:8, 0] == np.float32(0.105)).all() and (rows[8:, 0] == np.float32(0.905)).all()
    still = state_rows(quiet, 5, 8, Variant("shut", whole={"door": 0.0}), names)  # no timing: nothing changes inside
    assert (still[:, 1] == 0.0).all() and np.array_equal(still[:, 0], quiet[:, 0]) and (quiet[:, 1] == 1.0).all()
    for bad in (Variant("x", whole={"cold": 1.0}), Variant("x", {"door": 0.0}, "from_start", whole={"door": 1.0})):
        with pytest.raises(ValueError):
            state_rows(quiet, 5, 8, bad, names)


def test_a_whole_value_is_read_by_the_table_and_named_in_the_record():
    run = toy_run()
    cut = decision(run)
    quiet, hot = family(TOY, run, cut, [Variant("quiet"), Variant("hot", {"heat": 0.905}, "onset", t_on=cut - 1,
                                                                  whole={"door": 0.0})], rng(0), TOK, MS, family_id="f")
    assert (table(hot)["cls"], table(hot)["signals"], table(hot)["whole"]) == ("hot_shut", ["heat"], ["door"])
    assert (hot.signals[:, 1] == 0.0).all() and "whole" not in table(quiet)  # a record without one is as it was


def _marked(**marks):
    """TOY with every reacting class of its table marked (`aloud`, `inline`)."""
    from dataclasses import replace

    def policy(state, facts):
        cls = toy_policy(state, facts)
        return cls if cls.behavior == "base" else replace(cls, **marks)

    return replace(TOY, policy=policy)


def test_an_act_marked_aloud_closes_the_thinking_first_and_speaks_its_sentence():
    run = toy_run()
    hot = Variant("hot_shut", TOY_STATES["hot_shut"], "onset", t_on=2)
    (act,) = family(_marked(aloud=True), run, 3, [hot], rng(0), TOK, MS, family_id="f")
    words = TOK.decode(act.target_ids[3:]).split()
    assert words[0] == "</think>" and words.count("</think>") == 1  # the thinking stops where the state arrives
    assert 0 < words.index("shut") < words.index("<tool_call>") and words[-1] == "<eos>"  # said aloud, then the command
    assert act.focus_spans == [[3, len(act.target_ids)]] and act.phase == "thinking"
    cut = decision(run)  # in the answer nothing changes
    assert family(_marked(aloud=True), run, cut, [hot], rng(0), TOK, MS, family_id="f") == family(
        TOY, run, cut, [hot], rng(0), TOK, MS, family_id="f")


def test_inline_joiners_wrap_a_reaction_that_cuts_a_sentence():
    domain = _marked(inline=(" -- ", " --"))
    run = toy_run(answer="a1 a2 . a3 a4")
    close = run.target_ids.index(MS.think_close)
    mid, end = close + 3, close + 4  # after "a2": inside a sentence; after ".": between two sentences
    variants = [Variant("warm", TOY_STATES["warm"], "from_start"), Variant("hot_shut", TOY_STATES["hot_shut"], "from_start")]
    warm, hot = family(domain, run, mid, variants, rng(0), TOK, MS, family_id="f")
    a, b = warm.focus_spans[0]
    aside = TOK.decode(warm.target_ids[a:b]).split()
    assert aside[0] == "--" and aside[-1] == "--" and len(aside) > 4  # the aside interrupts the sentence
    assert warm.target_ids[b:] == run.target_ids[mid:]  # which then goes on
    cut_in = TOK.decode(hot.target_ids[mid:]).split()
    assert cut_in[0] == "--" and cut_in.count("--") == 1 and cut_in[-1] == "<eos>"
    for ep in family(domain, run, end, variants, rng(0), TOK, MS, family_id="f"):
        assert "--" not in TOK.decode(ep.target_ids).split()  # at a sentence end: as without joiners
    assert family(domain, run, end, variants, rng(0), TOK, MS, family_id="f")[0].target_ids == family(
        TOY, run, end, variants, rng(0), TOK, MS, family_id="f")[0].target_ids
    thinking = family(domain, run, 3, variants, rng(0), TOK, MS, family_id="f")[1]  # an act inside the thinking
    words = TOK.decode(thinking.target_ids[3:]).split()
    assert words[0] == "--" and words.index("</think>") < words.index("<tool_call>")
