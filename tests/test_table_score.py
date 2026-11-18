"""The mechanical scorer of the state-table experiments (preregistration amendment A8, "Scoring,
mechanical"): `act` = names the state within 48 generated tokens, first command the table's, with
the arguments the table fixes; `mention` = named, no command, 20 more tokens; `base` = no naming,
no reaction command."""

import numpy as np

from prio.table.core import Variant, family
from prio.table.score import GO_ON, WINDOW, named_at, names, score, spoken
from table_helpers import MS, TOK, TOY, TOY_STATES, call, decision, e, toy_run

FRONT, NOTE = call("open_door", which="front"), call("note", text="all quiet here")


def siblings(**facts):
    run = toy_run(facts=facts)
    cut = decision(run)
    eps = family(TOY, run, cut, [Variant("quiet"), Variant("hot_open", TOY_STATES["hot_open"], "from_start"),
                                 Variant("hot_shut", TOY_STATES["hot_shut"], "from_start"),
                                 Variant("warm", TOY_STATES["warm"], "from_start")],
                 np.random.default_rng(0), TOK, MS, family_id=run.task.task_id)
    return dict(zip(("quiet", "hot_open", "hot_shut", "warm"), eps))


def judge(ep, text, stop="eos", condition="on", cell="main", **kw):
    return score(ep, e(text), stop, condition, cell, TOK, MS, TOY, **kw)


def test_window_and_go_on_are_the_pre_registered_numbers():
    assert (WINDOW, GO_ON) == (48, 20)


def test_spoken_text_leaves_out_what_is_inside_a_command():
    ids = e(f"it is hot {FRONT} more words")
    assert TOK.decode(spoken(ids, MS)) == "it is hot more words"
    assert TOK.decode(spoken(e('fine <tool_call> {"name":"note" never closed'), MS)) == "fine"


def test_naming_needs_a_word_of_every_group_as_whole_words():
    assert names("It is HOT in here", [["hot"]]) and not names("a hotel", [["hot"]])
    assert names("hot, and the door is shut", [["hot"], ["door", "gate"]])
    assert not names("it is hot", [["hot"], ["door", "gate"]])  # both groups are needed
    assert names("no cell service here", [["cell service", "cellular"]]) and not names("anything", [])


def test_named_at_gives_the_token_that_completes_the_naming_inside_the_limit():
    ids = e("well it is hot and the door is shut")
    assert named_at(ids, [["hot"]], TOK, MS, 48) == 3
    assert named_at(ids, [["hot"], ["door"]], TOK, MS, 48) == 6
    assert named_at(ids, [["hot"], ["door"]], TOK, MS, 6) is None  # completed by the 7th token
    assert named_at(e(f"{FRONT} hot"), [["front"]], TOK, MS, 48) is None  # a command's arguments are not speech


def test_act_is_named_then_the_tables_command_with_its_fixed_arguments():
    ep = siblings()["hot_shut"]
    text = f"it is too hot and the door is shut {FRONT} <eos>"
    r = judge(ep, text)
    assert r.correct and r.named and r.command_ok and r.arguments == {"which": True} and r.turn_ended
    assert (r.cls, r.behavior, r.timing, r.variant, r.condition, r.cell) == ("hot_shut", "act", "from_start", "hot_shut", "on", "main")
    assert (r.family, r.conditional, r.held_out, r.signals) == ("toy-train-00000", True, False, ["door", "heat"])
    assert r.commands == ["open_door"] and r.own_command == "note" and r.n_tokens == len(e(text)) and r.stop_reason == "eos"
    assert not judge(ep, f"it is too hot {FRONT} <eos>").correct  # one group of the class not named
    wrong = judge(ep, f"hot and the door is shut {NOTE} <eos>")
    assert not wrong.correct and wrong.named and wrong.command_ok is False and wrong.arguments == {"which": False}
    back = judge(ep, f"hot and the door is shut {call('open_door', which='back')} <eos>")
    assert not back.correct and back.command_ok and back.arguments == {"which": False}  # right command, wrong argument
    assert not judge(ep, "hot and the door is shut <eos>").correct  # no command
    late = judge(ep, f"{FRONT} it is hot and the door is shut <eos>")
    assert not late.correct and not late.named and late.command_ok  # the sentence comes before the command
    on = judge(ep, f"hot and the door is shut {FRONT} and then more <eos>")
    assert on.correct and on.turn_ended is False  # reported beside, not part of the rule


def test_act_naming_must_fit_in_the_first_48_tokens():
    ep = siblings()["hot_shut"]
    filler = " ".join(["um"] * 44)
    assert judge(ep, f"{filler} hot and the door {FRONT} <eos>").correct  # "door" is the 48th token
    assert not judge(ep, f"{filler} um hot and the door {FRONT} <eos>").correct  # the 49th


def test_copied_arguments_are_compared_with_the_base_runs_own_call():
    ep = siblings()["hot_open"]
    assert judge(ep, f"too hot {NOTE} <eos>").correct
    assert judge(ep, f"too hot {call('note', text='all  quiet here ')} <eos>").correct  # white space is not content
    slip = judge(ep, f"too hot {call('note', text='all quiet')} <eos>")
    assert not slip.correct and slip.command_ok and slip.arguments == {"text": False}


def test_mention_is_named_without_any_command_and_goes_on_for_20_tokens():
    ep = siblings()["warm"]
    more = " ".join(["word"] * 20)
    r = judge(ep, f"it is warm {more}", stop="max_new_tokens")
    assert r.correct and r.named and r.went_on and r.command_ok is None and r.arguments is None
    assert not judge(ep, "it is warm " + " ".join(["word"] * 19), stop="max_new_tokens").correct
    assert not judge(ep, f"it is warm {more} {NOTE} <eos>").correct  # a command anywhere in the turn
    assert not judge(ep, f"nothing to say {more}").correct


def test_base_is_no_naming_and_no_reaction_command_anywhere():
    ep = siblings()["quiet"]
    r = judge(ep, f"{NOTE} <eos>")  # the base run's own command is no reaction
    assert r.correct and not r.named_any and not r.reaction_call and r.named is False
    late = " ".join(["word"] * 60)
    said = judge(ep, f"{late} a bit warm today {NOTE} <eos>")
    assert not said.correct and said.named_any  # a state of the table named anywhere, also after the window
    alarm = judge(ep, f"{call('alarm')} <eos>")
    assert not alarm.correct and alarm.reaction_call and not alarm.named_any
    assert judge(ep, f"{call('note', text='it is hot')} <eos>").correct  # inside a command: not speech


def test_forged_cells_keep_the_claimed_state():
    r = judge(siblings()["quiet"], f"{NOTE} <eos>", condition="base", cell="P1/F3-hot_shut", forged="hot_shut")
    assert (r.condition, r.cell, r.forged) == ("base", "P1/F3-hot_shut", "hot_shut")
