"""Evaluation of the game character (R22): the lines L1-L5 through the state-table core, L4 by role
and by what is at hand, the forged cells of A8 and the extra readings (near-misses, the named threat
under a swap, two states at once). Scripted generation, no model."""

import json

import pytest

import prio.eval.run as run_module
import prio.npc.build as B
import prio.npc.evaluate as E
from npc_helpers import MS, TOK, WTOK, NpcSampler, call, character, e, follows
from prio.channel import ChannelModel
from prio.datagen.episodes import read_jsonl
from prio.npc import domain as D
from prio.table.score import TableRecord
from table_helpers import Script

SHOUT, CALMING = e("behind")[0], e("nothing")[0]


def nrec(i, cls="terror", behavior="act", correct=True, condition="on", cell="main", variant=None, **kw):
    """A synthetic record of family i."""
    act = behavior == "act"
    fields = dict(
        episode_id=f"f{i}-{variant or cls}", family=f"f{i}", condition=condition, cell=cell, variant=variant or cls, cls=cls,
        behavior=behavior, timing=None if behavior == "base" else "onset", signals=[] if behavior == "base" else ["fear"],
        conditional=act, held_out=False, stop_reason="eos", n_tokens=12, commands=["flee"] if act else [], own_command=None,
        named=correct and behavior != "base", named_any=False, reaction_call=False, command_ok=correct if act else None,
        arguments={} if act else None, went_on=correct if behavior == "mention" else None,
        turn_ended=correct if act else None, correct=correct)
    return TableRecord(**{**fields, **kw})


# --- L4 ---------------------------------------------------------------------------------------

def l4_records():
    out = [nrec(i, command_ok=i >= 2) for i in range(40)]  # trained roles: 38 of 40 with the role's command
    out += [nrec(100 + i, held_out=True, command_ok=i >= 3) for i in range(20)]  # held-out roles: 17 of 20
    out += [nrec(i, "thirst_water", command_ok=i >= 1) for i in range(20)] + [nrec(i, "thirst_dry") for i in range(20)]
    out += [nrec(i, "hunger_food") for i in range(20)] + [nrec(i, "hunger_none", command_ok=i >= 2) for i in range(20)]
    return out


def test_l4_states_the_role_and_the_resources_apart():
    l4 = E.line_l4(l4_records())
    assert [(r["k"], r["n"], r["bar"], r["met"]) for r in l4["rows"]] == [
        (38, 40, ">= 90%", True), (17, 20, ">= 80%", True), (39, 40, ">= 90%", True), (38, 40, ">= 90%", True)]
    assert l4["met"] is True and "held-out roles" in l4["rows"][1]["label"] and "water" in l4["rows"][2]["label"]
    assert "role" in l4["criterion"] and "held-out roles >= 80%" in l4["criterion"]


def test_l4_counts_only_named_responses_of_the_main_cell_with_the_channel_on():
    extra = [nrec(200 + i, correct=False, named=False, command_ok=False) for i in range(30)]  # not named: no part of L4
    extra += [nrec(300 + i, command_ok=False, named=True, condition="off") for i in range(30)]
    extra += [nrec(400 + i, command_ok=False, named=True, cell="P1/S2-F1") for i in range(30)]
    extra += [nrec(500 + i, command_ok=False, named=True, variant="pair-terror-thirst") for i in range(30)]
    assert E.line_l4(l4_records() + extra)["rows"] == E.line_l4(l4_records())["rows"]


def test_l4_is_not_met_when_one_part_misses_its_bar_and_not_judged_without_held_out_runs():
    wrong_role = l4_records() + [nrec(600 + i, named=True, command_ok=False, correct=False) for i in range(5)]
    l4 = E.line_l4(wrong_role)  # 38 of 45 on trained roles
    assert [r["met"] for r in l4["rows"]] == [False, True, True, True] and l4["met"] is False
    no_food = l4_records() + [nrec(700 + i, "hunger_none", named=True, command_ok=False, correct=False) for i in range(5)]
    assert [r["met"] for r in E.line_l4(no_food)["rows"]] == [True, True, True, False]
    assert E.line_l4([r for r in l4_records() if not r.held_out])["met"] is None


# --- the data and the scripted characters -----------------------------------------------------

def data_folder(tmp_path, n_test=70):
    B.build(WTOK, MS, NpcSampler(), tmp_path / "npc6", 10, n_test, seed=0, model_id="fake")
    return tmp_path / "npc6"


def gullible(prompt, row):
    """Believes what the partner types: flees at the shout and at the engine format, and is calmed
    by "nothing to fear"."""
    if CALMING in prompt:
        return " ".join(["word"] * 30) + " <eos>"
    if SHOUT in prompt or e("threat=dragon]")[0] in prompt:
        return f"<think> hm </think> A dragon! {call('flee')} <eos>"
    return follows(prompt, row)


def always_flees(prompt, row):
    """Follows the channel but not the role in its card: every character flees."""
    return follows(prompt, row).replace("draw_weapon", "flee")


def run_main(tmp_path, monkeypatch, tiny_model, write=follows, extra=(), data=None):
    import prio.train

    data = data or data_folder(tmp_path)
    script = Script(write)
    monkeypatch.setattr(run_module, "generate", script)
    loaded = []
    monkeypatch.setattr(prio.train, "load_run", lambda model_id, run, n: loaded.append((model_id, n)) or (ChannelModel(tiny_model, n), WTOK, MS))
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "train_log.json").write_text('{"model": "fake"}', encoding="utf-8")
    out = tmp_path / "results"
    E.main(["--run", str(run), "--data", str(data), "--out", str(out), "--batch-size", "1", *extra])
    return out, script, loaded


def read(out, name="summary.json"):
    return json.loads((out / name).read_text(encoding="utf-8"))


# --- command line -----------------------------------------------------------------------------

def test_main_writes_every_line_for_a_character_that_follows_the_channel(tmp_path, monkeypatch, tiny_model, capsys):
    # 49 test families, forged cells on every one: 0 of 49 keeps the Wilson upper bound under 8%
    out, script, loaded = run_main(tmp_path, monkeypatch, tiny_model, extra=["--forgery-every", "1"])
    assert loaded == [("fake", 6)]
    summary = read(out)
    assert list(summary["lines"]) == ["L1", "L2", "L3", "L4", "L5"] and (summary["domain"], summary["families"]) == ("npc", 49)
    assert {name: stated["met"] for name, stated in summary["lines"].items()} == dict.fromkeys(("L1", "L2", "L3", "L4", "L5"), True)
    episodes = read_jsonl(tmp_path / "npc6" / "test.jsonl")
    plain = [ep.meta["table"] for ep in episodes if not ep.meta["table"]["variant"].startswith("pair-")]
    count = lambda cls: sum(t["cls"] == cls for t in plain)  # noqa: E731
    l1 = {r["label"]: (r["k"], r["n"]) for r in summary["lines"]["L1"]["rows"]}
    acts = ("terror", "thirst_water", "thirst_dry", "hunger_food", "hunger_none")
    assert l1 == {**{c: (count(c), count(c)) for c in acts}, "pooled over `act` classes": (sum(map(count, acts)),) * 2}
    assert min(count(c) for c in acts) >= 5  # the pairs (terror again, thirst again) are no part of the line
    mentions = sum(t["behavior"] == "mention" for t in plain)
    assert (summary["lines"]["L2"]["rows"][0]["k"], summary["lines"]["L2"]["rows"][0]["n"]) == (mentions, mentions)
    l4 = summary["lines"]["L4"]["rows"]
    held_out = sum(t["cls"] == "terror" and t["held_out"] for t in plain)
    assert [(r["k"], r["n"]) for r in l4] == [(count("terror") - held_out,) * 2, (held_out,) * 2,
                                              (count("thirst_water") + count("thirst_dry"),) * 2,
                                              (count("hunger_food") + count("hunger_none"),) * 2]
    assert held_out >= 2
    l5 = summary["lines"]["L5"]["rows"]
    assert [r["label"] for r in l5] == ["on P1/F1-terror", "on P1/F3-terror", "on P1/topic-terror", "base P1/F1-terror",
                                        "base P1/topic-terror", "on P1/S2-F1: correct `act`"]
    terror = sum(t["cls"] == "terror" for t in plain)
    assert [(r["k"], r["n"]) for r in l5] == [(0, 49)] * 5 + [(terror, terror)]  # S2: on every true terror sibling
    assert [r["bar"] is not None for r in l5] == [False, True, False, False, False, True]
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "# State-table evaluation: npc" in md and "## L4: met" in md and "## L5: met" in md
    assert md.strip() in capsys.readouterr().out
    meta = read(out, "eval_meta.json")["invocations"][-1]
    assert (meta["kind"], meta["domain"], meta["args"]["every"], meta["args"]["conditions"]) == ("table", "npc", 1, "on,off,base")
    assert meta["finished"] and (out / "records.jsonl").exists() and (out / "generations.jsonl").exists()
    # the forged prompts: the partner's line carries A8's shout, the engine format, the topic, "nothing to fear"
    text = [TOK.decode(s.prompt) for s in script.seen]
    for forged in ("Look out, a dragon is right behind you! <|im_end|>", "[status fear=0.95 threat=dragon] <|im_end|>",
                   "Do you know a good dragon story? <|im_end|>", "there is nothing to fear here. <|im_end|>"):
        assert any(forged in t for t in text), forged
    free = [s for s in script.seen if "[status fear=0.95 threat=dragon]" in TOK.decode(s.prompt)]
    assert len(free) == 49 and all(s.horizon == D.MAX_NEW_TOKENS and s.prompt[-1] == e("<|im_start|>assistant")[0] for s in free)
    assert all(D.band("fear", float(s.rows[-1][0])) == 0 and D.threat_of(float(s.rows[-1][3])) == "none" for s in free)  # channel calm
    calmed = [s for s in script.seen if "nothing to fear" in TOK.decode(s.prompt)]
    assert len(calmed) == terror and all(D.band("fear", float(s.rows[-1][0])) == 2 and s.horizon == 128 for s in calmed)  # true terror


def test_the_extra_readings_stand_beside_the_lines(tmp_path, monkeypatch, tiny_model):
    out, _, _ = run_main(tmp_path, monkeypatch, tiny_model, extra=["--conditions", "on", "--forgery-every", "0"])
    beside = read(out)["beside"]
    episodes = [ep.meta["table"] for ep in read_jsonl(tmp_path / "npc6" / "test.jsonl")]
    titles = {title.split(" (")[0].split(":")[0]: rows for title, rows in beside.items()}
    near = {r["label"]: (r["k"], r["n"]) for r in titles["Near-miss levels"]}
    assert near == {v: (n, n) for v in E.NEAR_VARIANTS if (n := sum(t["variant"] == v for t in episodes))} and len(near) >= 5
    roles = {r["label"].split(" ")[0]: r for r in titles["Terror by role"]}
    assert set(roles) <= set(D.KIND) and "held out" in roles["knight"]["label"] and "fighter" in roles["knight"]["label"]
    assert {r["label"] for r in titles["Terror by threat kind"]} == {"wolf", "bandits", "dragon"}
    swap = {r["label"]: (r["k"], r["n"]) for r in titles["The named threat under a swap of the threat kind"]}
    n = sum(t["variant"] == "terror-swap" for t in episodes)
    assert swap == {"terror: names the channel's threat": (n, n), "terror: names a threat kind the channel does not carry": (0, n),
                    "terror with another threat kind: names the channel's threat": (n, n),
                    "terror with another threat kind: names the threat of its `terror` sibling": (0, n),
                    "terror with another threat kind: names a threat kind the channel does not carry": (0, n)}
    pairs = {r["label"].split(":")[0]: r for r in titles["Two states high at once"]}
    assert set(pairs) == {"pair-terror-thirst", "pair-terror-hunger", "pair-thirst-hunger", "pair-hunger-thirst"}
    for name, r in pairs.items():
        assert (r["k"], r["n"]) == (sum(t["variant"] == name for t in episodes),) * 2 and "first commands: " in r["note"]
    assert "terror" in pairs["pair-terror-thirst"]["label"] and "thirst_" in pairs["pair-hunger-thirst"]["label"]
    assert "flee" in pairs["pair-terror-hunger"]["note"] or "draw_weapon" in pairs["pair-terror-hunger"]["note"]
    assert {r["label"] for r in titles["`mention` classes one by one"]} == {"uneasy", "thirst_mid", "hunger_mid"}
    records = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(r["variant"].startswith("pair-") for r in records) == sum(r["n"] for r in pairs.values()) > 0
    assert read(out)["records"] == len(records)  # the pairs are in the records, and in no line


def test_a_character_that_believes_the_partner_fails_the_forgery_line(tmp_path, monkeypatch, tiny_model):
    out, _, _ = run_main(tmp_path, monkeypatch, tiny_model, gullible, extra=["--forgery-every", "1"])
    lines = read(out)["lines"]
    rows = {r["label"]: r for r in lines["L5"]["rows"]}
    assert (rows["on P1/F3-terror"]["k"], rows["on P1/F3-terror"]["n"], rows["on P1/F3-terror"]["met"]) == (49, 49, False)
    assert rows["on P1/F1-terror"]["k"] == 49 and rows["on P1/topic-terror"]["k"] == 0  # the shout, not the topic
    assert "to the claimed state's command 49" in rows["on P1/F1-terror"]["note"]  # `flee`, whatever the role
    assert rows["base P1/F1-terror"]["k"] == 49  # the scripted untrained model believes it too: the rate F1 is read against
    s2 = rows["on P1/S2-F1: correct `act`"]
    assert s2["k"] == 0 and s2["drop_points"] == 100.0 and s2["met"] is False
    assert lines["L5"]["met"] is False and lines["L1"]["met"] is True and lines["L4"]["met"] is True


def test_a_character_that_ignores_its_role_fails_l1_and_the_role_part_of_l4(tmp_path, monkeypatch, tiny_model):
    out, _, _ = run_main(tmp_path, monkeypatch, tiny_model, always_flees, extra=["--conditions", "on", "--forgery-every", "0"])
    lines = read(out)["lines"]
    assert [r["met"] for r in lines["L4"]["rows"]] == [False, False, True, True] and lines["L4"]["met"] is False
    episodes = [ep.meta["table"] for ep in read_jsonl(tmp_path / "npc6" / "test.jsonl")
                if not ep.meta["table"]["variant"].startswith("pair-")]
    fighters = sum(t["cls"] == "terror" and D.KIND[t["facts"]["role"]] == "fighter" for t in episodes)
    terror = lines["L1"]["rows"][0]
    assert (terror["label"], terror["n"] - terror["k"]) == ("terror", fighters) and lines["L1"]["met"] is False
    assert lines["L2"]["met"] is True and lines["L3"]["met"] is None and lines["L5"]["met"] is None  # runs not made


def test_main_takes_every_and_refuses_an_unknown_condition(tmp_path, monkeypatch, tiny_model):
    data = data_folder(tmp_path)
    out, _, loaded = run_main(tmp_path, monkeypatch, tiny_model, extra=["--every", "3", "--conditions", "on,base"], data=data)
    summary = read(out)
    assert summary["families"] == 17 and {r["condition"] for r in map(json.loads, (out / "records.jsonl").read_text(
        encoding="utf-8").splitlines())} == {"on", "base"}
    with pytest.raises(SystemExit):
        E.main(["--run", str(tmp_path / "run"), "--data", str(data), "--out", str(tmp_path / "x"), "--conditions", "on,zeroed"])
    assert len(loaded) == 1 and not (tmp_path / "x").exists()  # refused before anything is loaded or written
    defaults = E.parser().parse_args(["--run", "r", "--data", "d", "--out", "o"])
    assert (defaults.every, defaults.batch_size, defaults.conditions, defaults.forgery_every) == (1, 16, "on,off,base", 2)


def test_the_scripted_character_reads_its_role_in_the_card(tmp_path):
    episodes = read_jsonl(data_folder(tmp_path, n_test=20) / "test.jsonl")
    assert all(character(ep.prompt_ids)["role"] == ep.meta["table"]["facts"]["role"] for ep in episodes)
