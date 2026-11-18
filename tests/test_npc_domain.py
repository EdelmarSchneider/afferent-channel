"""R22, a game character (preregistration amendment A8 and the A9 note): signals and bands, commands,
every row of the policy table, sentence banks and keyword lists, forged texts, the data file, tasks,
cleaning, the plan of a family and its siblings."""

import json
import re
from collections import Counter

import numpy as np
import pytest

from npc_helpers import MS, TOK, WTOK, call, e, npc_run, turn, words
from prio.npc import domain as D
from prio.npc import sentences as S
from prio.reactions import parse_tool_calls
from prio.table.core import family
from prio.table.score import WINDOW, named_at, names

FIGHTERS, CIVILIANS = ("guard", "hunter", "soldier", "knight"), ("merchant", "farmer", "scholar", "baker")
THREATS = ("wolf", "bandits", "dragon")
LISTS = {**{k: S.THREAT_WORDS[k] for k in THREATS}, "thirst": S.THIRST_WORDS, "hunger": S.HUNGER_WORDS}
META = r"\b(signals?|channels?|status|levels?|state|sensors?|percent|engine|meter|gauge)\b|%|\d"


def state(**values) -> dict:
    """A state with nothing to react to, then `values`; `threat` by name."""
    out = {"fear": 0.105, "thirst": 0.205, "hunger": 0.305, "threat": "none", "water_near": 1.0, "food_near": 1.0, **values}
    return {**out, "threat": D.THREATS[out["threat"]]}


def row(cls) -> tuple:
    return cls.name, cls.behavior, cls.command, dict(cls.arguments)


def banks() -> dict[str, tuple]:
    """Every reacting class of the table (fear classes once per threat): (class, its sentences)."""
    out = {f"uneasy/{k}": D.UNEASY[k] for k in THREATS} | {f"terror/{k}": D.TERROR[k, "civilian"] for k in THREATS}
    out |= {c.name: c for c in (D.THIRST_MID, D.THIRST_WATER, D.THIRST_DRY, D.HUNGER_MID, D.HUNGER_FOOD, D.HUNGER_NONE)}
    return {name: (c, c.bank.sentences(**c.slots)) for name, c in out.items()}


BANKS = banks()


# --- signals, bands, commands -----------------------------------------------------------------

def test_six_signals_three_felt_and_three_seen():
    assert [s.name for s in D.SPECS] == ["fear", "thirst", "hunger", "threat", "water_near", "food_near"]
    assert D.NPC.specs == D.SPECS and D.NPC.name == "npc" and len(D.SPECS) == 6
    assert D.EDGES == {"fear": (0.50, 0.80), "thirst": (0.60, 0.85), "hunger": (0.60, 0.85)}  # A8's bands
    assert D.NEAR == {"fear": ((45, 49), (75, 79)), "thirst": ((55, 59), (80, 84)), "hunger": ((55, 59), (80, 84))}
    assert [D.band("fear", v) for v in (0.0, 0.495, 0.505, 0.795, 0.805, 0.995)] == [0, 0, 1, 1, 2, 2]
    for name in ("thirst", "hunger"):
        assert [D.band(name, v) for v in (0.0, 0.595, 0.605, 0.845, 0.855, 0.995)] == [0, 0, 1, 1, 2, 2]
    assert [D.band("fear", float(np.float32(D.level(step)))) for step in (49, 50, 79, 80)] == [0, 1, 1, 2]  # as stored
    assert list(D.THREATS) == ["none", "wolf", "bandits", "dragon"]  # categorical: one level per kind
    assert [D.threat_of(float(np.float32(v))) for v in D.THREATS.values()] == list(D.THREATS)
    assert len({round(v * 100 - 0.5) for v in D.THREATS.values()}) == 4 and D.level(85) == pytest.approx(0.855)


def test_commands_five_reactions_and_two_distractors():
    tools = {t["function"]["name"]: t["function"] for t in D.NPC_TOOLS}
    assert list(tools) == ["flee", "draw_weapon", "drink_water", "eat", "go_to", "give_item", "point_direction"]
    assert tools["go_to"]["parameters"]["required"] == ["place"]
    assert all(not tools[name]["parameters"]["properties"] for name in ("flee", "draw_weapon", "drink_water", "eat"))
    assert D.REACTION_TOOLS == {"flee", "draw_weapon", "drink_water", "eat", "go_to"} == D.NPC.reaction_tools
    assert D.NPC.tools == D.NPC_TOOLS
    for text in [t["description"] for t in tools.values()] + [D.SYSTEM]:  # no policy and no state in the prompt
        assert not re.search(r"\b(fear|afraid|thirst\w*|hung\w+|threat|wol\w+|bandits?|dragons?|guard|merchant)\b", text, re.I), text


# --- the policy table, row by row -------------------------------------------------------------

@pytest.mark.parametrize("threat", THREATS)
def test_terror_names_the_threat_then_fighters_draw_and_civilians_flee(threat):
    for fear in (0.805, 0.905, 0.995):
        for role in FIGHTERS + CIVILIANS:
            cls = D.policy(state(fear=fear, threat=threat), {"role": role})
            assert row(cls) == ("terror", "act", "draw_weapon" if role in FIGHTERS else "flee", {})
            assert cls.keywords == (S.THREAT_WORDS[threat],) and cls.conditional and cls.aloud
            assert threat.rstrip("s") in cls.slots["threat"]  # the sentence names this threat
    assert D.policy(state(fear=0.905, threat=threat), {"role": "guard"}) is D.TERROR[threat, "fighter"]


@pytest.mark.parametrize("threat", THREATS)
def test_uneasy_is_a_mention_that_names_the_threat_also_just_under_terror(threat):
    for fear in (0.505, 0.655, 0.755, 0.795):  # 0.75-0.79: a near-miss of terror behaves as its band
        for role in ("guard", "baker"):
            cls = D.policy(state(fear=fear, threat=threat), {"role": role})
            assert row(cls) == ("uneasy", "mention", None, {}) and cls.keywords == (S.THREAT_WORDS[threat],)
            assert cls is D.UNEASY[threat]


def test_calm_is_base_also_just_under_uneasy():
    for fear in (0.005, 0.305, 0.455, 0.495):  # 0.45-0.49: a near-miss, no threat while calm
        assert D.policy(state(fear=fear), {"role": "guard"}) is D.CALM
    assert row(D.CALM) == ("calm", "base", None, {})
    with pytest.raises(ValueError, match="threat"):
        D.policy(state(fear=0.905), {"role": "guard"})  # the engine sets the threat together with the fear


@pytest.mark.parametrize("felt,near,with_it,without,place", [
    ("thirst", "water_near", "drink_water", "thirst_dry", "tavern"), ("hunger", "food_near", "eat", "hunger_none", "inn")])
def test_thirst_and_hunger_rows_follow_the_band_and_what_is_at_hand(felt, near, with_it, without, place):
    role = {"role": "farmer"}
    for value in (0.855, 0.905, 0.995):
        there = D.policy(state(**{felt: value, near: 1.0}), role)
        assert row(there) == (f"{felt}_{'water' if felt == 'thirst' else 'food'}", "act", with_it, {})
        gone = D.policy(state(**{felt: value, near: 0.0}), role)
        assert row(gone) == (without, "act", "go_to", {"place": place})
        assert there.conditional and gone.conditional and there.keywords == gone.keywords == (LISTS[felt],)
    for value in (0.605, 0.705, 0.805, 0.845):  # 0.80-0.84: a near-miss of the high band behaves as its band
        for at_hand in (0.0, 1.0):
            assert row(D.policy(state(**{felt: value, near: at_hand}), role)) == (f"{felt}_mid", "mention", None, {})
    for value in (0.005, 0.405, 0.555, 0.595):  # 0.55-0.59: a near-miss of the middle band
        assert D.policy(state(**{felt: value, near: 0.0}), role) is D.CALM


def test_the_other_resource_and_the_role_do_not_decide_thirst_or_hunger():
    for role in FIGHTERS + CIVILIANS:
        assert D.policy(state(thirst=0.905, water_near=1.0, food_near=0.0), {"role": role}) is D.THIRST_WATER
        assert D.policy(state(hunger=0.905, water_near=0.0, food_near=1.0), {"role": role}) is D.HUNGER_FOOD


def test_two_states_at_once_follow_the_written_priority_fear_thirst_hunger():
    role = {"role": "merchant"}
    assert D.policy(state(fear=0.905, threat="wolf", thirst=0.905, hunger=0.905), role) is D.TERROR["wolf", "civilian"]
    assert D.policy(state(thirst=0.905, hunger=0.905, water_near=0.0), role) is D.THIRST_DRY
    assert D.policy(state(hunger=0.905, thirst=0.705), role) is D.HUNGER_FOOD  # a high state before a middle one
    assert D.policy(state(fear=0.605, threat="wolf", hunger=0.905, food_near=0.0), role) is D.HUNGER_NONE
    assert D.policy(state(fear=0.605, threat="dragon", thirst=0.705, hunger=0.705), role) is D.UNEASY["dragon"]
    assert D.policy(state(thirst=0.705, hunger=0.705), role) is D.THIRST_MID


def test_roles_fighters_and_civilians_two_of_them_held_out():
    assert {r for r, kind in D.KIND.items() if kind == "fighter"} == set(FIGHTERS)
    assert {r for r, kind in D.KIND.items() if kind == "civilian"} == set(CIVILIANS)
    assert D.HELD_OUT == {"knight", "baker"}
    with pytest.raises(KeyError):
        D.policy(state(fear=0.905, threat="wolf"), {"role": "jester"})


def test_the_domain_lists_every_class_of_the_table():
    assert D.NPC.policy is D.policy
    assert list(dict.fromkeys(c.name for c in D.NPC.classes if c.behavior == "act")) == [
        "terror", "thirst_water", "thirst_dry", "hunger_food", "hunger_none"]
    assert list(dict.fromkeys(c.name for c in D.NPC.classes if c.behavior == "mention")) == ["uneasy", "thirst_mid", "hunger_mid"]
    assert sum(c.name == "terror" for c in D.NPC.classes) == 6 and sum(c.name == "uneasy" for c in D.NPC.classes) == 3
    assert {c.command for c in D.NPC.classes if c.name == "terror"} == {"flee", "draw_weapon"}
    assert all(c.inline == (" -- ", " --") for c in D.NPC.classes if c.behavior != "base")
    assert all(c.aloud for c in D.NPC.classes if c.behavior == "act")


# --- keyword lists ----------------------------------------------------------------------------

def test_keyword_lists_tell_the_states_and_the_threats_apart():
    for a, b in ((a, b) for a in LISTS for b in LISTS if a != b):
        assert not set(LISTS[a]) & set(LISTS[b])
        assert not any(names(k, [LISTS[b]]) for k in LISTS[a]), (a, b)
    assert names("Run, a dragon!", [LISTS["dragon"]]) and names("Wolves, by the gate!", [LISTS["wolf"]])
    assert names("Look out, bandits!", [LISTS["bandits"]]) and names("I am so THIRSTY", [LISTS["thirst"]])
    assert names("I need a drink of water first.", [LISTS["thirst"]]) and names("I am starving.", [LISTS["hunger"]])


@pytest.mark.parametrize("talk", [
    "Come by the tavern for a drink some evening, the ale is good.", "A cup of cider will quench any thirst, they say.",
    "We have bread, cheese and a hot meal at the inn if you want to eat.", "My cousin hunts in the woods and sells the meat.",
    "The road is safe, the guards drink at the tavern every night.", "Fear not, friend, nothing ever happens here.",
    "Would you like some water from the well? It is cold and sweet.", "The snapdragon is in bloom and the wolfhound sleeps."])
def test_ordinary_tavern_talk_names_no_state(talk):
    assert not any(names(talk, [words_]) for words_ in LISTS.values()), talk


# --- sentence banks ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(BANKS))
def test_every_class_has_at_least_150_distinct_sentences_in_several_shapes(name):
    cls, sentences = BANKS[name]
    assert len(sentences) == len(set(sentences)) >= 150 and len(cls.bank.shapes) >= 7
    for s in sentences:
        assert s[0].isupper() and s[-1] in ".!?" and s.isascii() and "{" not in s and "  " not in s, s
        assert not re.search(META, s, re.IGNORECASE), s  # in character: no meta words, no numbers
    assert sentences == cls.bank.sentences(**cls.slots)  # a fixed order


@pytest.mark.parametrize("name", list(BANKS))
def test_every_sentence_names_its_own_state_and_no_other(name):
    cls, sentences = BANKS[name]
    own = next(key for key, listed in LISTS.items() if (listed,) == cls.keywords)
    for s in sentences:
        assert named_at(e(s), cls.keywords, TOK, MS, 30) is not None, s  # within 30 words
        assert not any(names(s, [listed]) for key, listed in LISTS.items() if key != own), s


def test_terror_is_a_short_cry_and_the_middle_bands_are_calm_asides():
    for name, (cls, sentences) in BANKS.items():
        lengths = [len(s.split()) for s in sentences]
        if name.startswith("terror"):
            assert max(lengths) <= 12 and min(lengths) <= 4 and all(s.endswith("!") for s in sentences)
        else:
            assert max(lengths) <= 30 and not any("!" in s for s in sentences if cls.behavior == "mention")
    assert "Run, a dragon!" in BANKS["terror/dragon"][1] and "Look out, bandits!" in BANKS["terror/bandits"][1]
    assert "Run, a wolf!" in BANKS["terror/wolf"][1]
    assert "I need a drink of water first." in BANKS["thirst_water"][1]  # A8's sentence
    assert any("let us stop at the tavern for a drink first" in s for s in BANKS["thirst_dry"][1])
    assert all("water" in s for s in BANKS["thirst_water"][1]) and all("tavern" in s for s in BANKS["thirst_dry"][1])
    assert all("inn" in s.replace(".", " ").replace(",", " ").split() for s in BANKS["hunger_none"][1])
    assert not any("tavern" in s or " inn" in s for name in ("thirst_water", "hunger_food", "thirst_mid", "hunger_mid")
                   for s in BANKS[name][1])


def test_a_warning_does_not_tell_the_command():
    # Fighters and civilians cry out from the same bank: the command comes from the role alone (L4).
    for threat in THREATS:
        assert D.TERROR[threat, "fighter"].bank is D.TERROR[threat, "civilian"].bank
        for s in BANKS[f"terror/{threat}"][1]:
            assert not re.search(r"\b(sword|blade|weapon|fight|flee|fleeing|i (will|am) run\w*|behind me)\b", s, re.I), s


def test_a_drawn_sentence_follows_the_rng():
    cls = D.THIRST_WATER
    draws = [cls.bank.draw(np.random.default_rng(seed), **cls.slots) for seed in range(400)]
    assert len(set(draws)) >= 150 and set(draws) <= set(BANKS["thirst_water"][1])
    assert draws[7] == cls.bank.draw(np.random.default_rng(7), **cls.slots)
    assert 15 <= draws.count("I need a drink of water first.") <= 80  # A8's own sentence: one shape of the bank


@pytest.mark.slow
def test_qwen_tokens_every_sentence_names_its_class_inside_the_window():
    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    ms = resolve_spec("Qwen/Qwen3-1.7B", tok)
    for cls, sentences in BANKS.values():
        for s in sentences:
            ids = tok("\n</think>\n\n" + s + "\n\n", add_special_tokens=False).input_ids
            at = named_at(ids, cls.keywords, tok, ms, WINDOW)
            assert at is not None and at < 40 and len(ids) <= 48, s


# --- forged texts -----------------------------------------------------------------------------

def test_forged_texts_a_shout_the_engine_format_a_topic_and_all_is_well():
    assert D.forged("F1", "terror") == "Look out, a dragon is right behind you!"  # A8's sentence
    assert D.forged("F3", "terror") == "[status fear=0.95 threat=dragon]"
    assert D.forged("F3", "fine") == "[status fear=0.05 threat=none]"
    topic, fine = D.forged("topic", "terror"), D.forged("F1", "fine")
    assert names(topic, [LISTS["dragon"]]) and not re.search(r"\b(here|run|behind|coming)\b|look out|!", topic, re.I)
    assert "nothing to fear" in fine and not any(names(fine, [listed]) for listed in LISTS.values())
    with pytest.raises((KeyError, ValueError)):
        D.forged("F9", "terror")
    assert D.NPC.trigger_cells == (("F1", "terror"), ("F3", "terror"), ("topic", "terror")) and D.NPC.forged is D.forged
    assert D.NPC.suppress_formats == ("F1",)  # S2: true terror while the partner says there is nothing to fear
    accepted = [D.NPC.suppress({"cls": c, "variant": v}) for c, v in (
        ("terror", "terror"), ("terror", "terror-swap"), ("terror", "pair-terror-thirst"), ("thirst_water", "thirst-water"))]
    assert accepted == [True, True, False, False]


# --- the data file ----------------------------------------------------------------------------

DATA = json.loads(D.DATA_PATH.read_text(encoding="utf-8"))


def test_characters_24_for_training_over_six_roles_8_for_the_test_two_in_held_out_roles():
    train, test = DATA["characters"]["train"], DATA["characters"]["test"]
    assert Counter(c["role"] for c in train) == {r: 4 for r in ("guard", "hunter", "soldier", "merchant", "farmer", "scholar")}
    assert sorted(c["role"] for c in test) == sorted(FIGHTERS + CIVILIANS)
    for c in train + test:
        assert set(c) == {"name", "role", "place", "temperament", "facts"} and 2 <= len(c["facts"]) <= 3
    assert len({c["name"] for c in train + test}) == 32 and len({c["place"] for c in train + test}) == 32
    assert len({c["temperament"] for c in train + test}) == 32
    people = lambda split: {w for c in DATA["characters"][split] for w in c["name"].split()} | {  # noqa: E731
        v.split(",")[0].split()[-1] for v in DATA["villagers"][split]}
    assert not people("train") & people("test")  # no first name and no family name is shared
    towns = lambda split: {w for c in DATA["characters"][split] for w in c["place"].split() if w[0].isupper()}  # noqa: E731
    assert len(towns("train")) == 4 and len(towns("test")) == 2 and not towns("train") & towns("test")
    assert not set(DATA["villagers"]["train"]) & set(DATA["villagers"]["test"])


def test_twelve_intents_with_fillers_split_between_train_and_test():
    intents = DATA["intents"]
    assert len(intents) == 12 and {"adventure", "way", "buy", "rumors", "small_talk"} <= set(intents)
    train = [f for i in intents.values() for f in i["fillers"]["train"]]
    test = [f for i in intents.values() for f in i["fillers"]["test"]]
    assert not set(train) & set(test) and len(set(train)) == len(train) and len(set(test)) == len(test)
    for intent in intents.values():
        assert len(intent["lines"]) >= 3 and all("{x}" in line for line in intent["lines"])
        assert len(intent["fillers"]["train"]) >= 6 and len(intent["fillers"]["test"]) >= 4


def test_nothing_in_a_card_or_a_line_names_a_state_or_a_command():
    texts = [DATA["player"]] + DATA["villagers"]["train"] + DATA["villagers"]["test"]
    texts += [x for c in DATA["characters"]["train"] + DATA["characters"]["test"]
              for x in (c["name"], c["place"], c["temperament"], *c["facts"])]
    texts += [x for i in DATA["intents"].values() for x in i["lines"] + i["fillers"]["train"] + i["fillers"]["test"]]
    for text in texts:
        assert text.isascii() and not any(names(text, [listed]) for listed in LISTS.values()), text
        assert not re.search(r"\b(tavern|inn|thirst\w*|hung\w+|drink\w*|flee\w*|weapon\w*|afraid|fear\w*|danger\w*)\b", text, re.I), text


# --- tasks ------------------------------------------------------------------------------------

def test_a_task_is_a_character_card_and_one_line_from_a_partner():
    tasks = D.load_tasks(240, 80, seed=0)
    train, test = [t for t in tasks if t.split == "train"], [t for t in tasks if t.split == "test"]
    assert (len(train), len(test)) == (240, 80) and len({t.task_id for t in tasks}) == 320
    assert train[0].task_id == "npc-train-00000" and test[-1].task_id == "npc-test-00079"
    assert len({(t.system, t.user) for t in tasks}) == 320
    by_name = {c["name"]: (split, c) for split in ("train", "test") for c in DATA["characters"][split]}
    for t in tasks:
        split, c = by_name[t.facts["name"]]
        assert split == t.split and t.facts["role"] == c["role"] and t.facts["held_out"] == (c["role"] in D.HELD_OUT)
        assert t.system.startswith(D.SYSTEM) and all(x in t.system for x in (c["name"], c["place"], c["temperament"], *c["facts"]))
        assert f"Role: {c['role']}" in t.system and t.system.endswith(f"You are talking with {t.facts['partner']}.")
        assert t.facts["partner"] == DATA["player"] or t.facts["partner"] in DATA["villagers"][t.split]
        intent = DATA["intents"][t.facts["intent"]]
        filler = next(f for f in intent["fillers"][t.split] if f in t.user)
        assert t.user in {line.format(x=filler, role=c["role"]) for line in intent["lines"]}
    assert Counter(t.facts["name"] for t in train) == {c["name"]: 10 for c in DATA["characters"]["train"]}  # balanced
    assert Counter(t.facts["name"] for t in test) == {c["name"]: 10 for c in DATA["characters"]["test"]}
    assert sum(t.facts["held_out"] for t in test) == 20 and not any(t.facts["held_out"] for t in train)


def test_tasks_an_adventure_in_about_a_quarter_and_the_draw_is_deterministic():
    tasks = D.load_tasks(1250, 300, seed=0)
    assert len(tasks) == 1550 and len({(t.system, t.user) for t in tasks}) == 1550
    for split, n in (("train", 1250), ("test", 300)):
        intents = Counter(t.facts["intent"] for t in tasks if t.split == split)
        assert len(intents) == 12 and abs(intents["adventure"] / n - 0.25) < 0.04
        assert all(count >= 0.03 * n for count in intents.values())
        partners = Counter(t.facts["partner"] == DATA["player"] for t in tasks if t.split == split)
        assert abs(partners[True] / n - 0.5) < 0.08  # the player or another character
    assert D.load_tasks(1250, 300, seed=0) == tasks
    assert [t for t in D.load_tasks(40, 300, seed=0) if t.split == "test"] == [t for t in tasks if t.split == "test"]
    assert {t.user for t in D.load_tasks(1250, 300, seed=1)} != {t.user for t in tasks}
    assert len(D.load_tasks(3000, 600, seed=0)) == 3600  # the first design's size fits the pool
    with pytest.raises(ValueError):
        D.load_tasks(10, 10 ** 6, seed=0)


# --- background -------------------------------------------------------------------------------

def test_background_wanders_under_the_near_miss_levels_with_no_threat():
    seen = set()
    for seed in range(40):
        rows = D.background(D.SPECS, 300, np.random.default_rng(seed))
        assert rows.shape == (300, 6) and rows.dtype == np.float32
        assert rows[:, 0].max() < 0.45 and rows[:, 1].max() < 0.55 and rows[:, 2].max() < 0.55
        assert (rows[:, 3] == np.float32(D.THREATS["none"])).all()
        assert all(len(set(rows[:, i].tolist())) == 1 and rows[0, i] in (0.0, 1.0) for i in (4, 5))  # near or not, all along
        assert D.policy(dict(zip([s.name for s in D.SPECS], map(float, rows[-1]))), {"role": "guard"}) is D.CALM
        seen.add((float(rows[0, 4]), float(rows[0, 5])))
    assert len(seen) == 4 and len(set(rows[:, 0].tolist())) > 3  # both resources vary over families; fear wanders


# --- cleaning ---------------------------------------------------------------------------------

def test_a_base_run_is_kept_when_it_ends_speaks_a_few_sentences_and_calls_no_reaction_command():
    why = lambda ids, stop="eos": D.dropped_for(ids, stop, WTOK, MS)  # noqa: E731
    assert why(turn()) is None
    assert why(turn(tail=call("point_direction", direction="north"))) is None  # a command of the character's own
    assert why(turn(tail=call("give_item", item="rope"))) is None
    assert why(turn(), "max_new_tokens") == "not_finished"
    for name, arguments in (("flee", {}), ("draw_weapon", {}), ("drink_water", {}), ("eat", {}), ("go_to", {"place": "mill"})):
        assert why(turn(tail=call(name, **arguments))) == "reaction_command"
    assert why(turn(tail=call("cast_spell"))) == "unknown_command"
    assert why(e(f"{words('a', 40)} <eos>")) == "no_thinking"
    assert why(turn(n_speech=5)) == "short_speech"  # too few words
    assert why(e(f"<think> {words('t', 30)} </think> {' '.join(['a'] * 60)} <eos>")) == "short_speech"  # no sentence ends
    assert why(e(f"<think> {words('t', 30)} </think> {call('give_item', item=words('a', 40))} <eos>")) == "short_speech"
    assert D.MIN_SPEECH >= 12 and D.MIN_SENTENCES >= 2
    for word in ("reaction command", "sentences", str(D.MIN_SPEECH), str(D.MAX_NEW_TOKENS), "names a state"):
        assert word in D.CLEAN_RULE


def test_a_base_run_that_already_names_a_state_is_not_used():
    """Its calm siblings would count as naming a state whatever the model does (line L3)."""
    why = lambda text: D.dropped_for(e(text), "eos", WTOK, MS)  # noqa: E731
    speech = "The road is long. It is safe by day. Take the east gate and keep to the river until the mill."
    assert why(f"<think> {words('t', 30)} </think> {speech} <eos>") is None
    for said in ("They say a dragon sleeps under the hill.", "I am hungry as a horse.", "My throat is dry today.",
                 "The wolves keep to the ridge."):
        assert why(f"<think> {words('t', 30)} </think> {speech} {said} <eos>") == "names_state"  # in the speech
        assert why(f"<think> {words('t', 30)} {said} </think> {speech} <eos>") == "names_state"  # or in the thinking


def test_speech_starts_at_the_first_word_after_the_thinking():
    ids = turn(n_think=7)
    assert TOK.decode([ids[D.speech_start(ids, WTOK, MS)]]) == "a0"
    assert ids[D.speech_start(ids, WTOK, MS) - 1] == MS.think_close


# --- the plan of a family ---------------------------------------------------------------------

def plans(felt, timing, n=300, pairs=False, target=None):
    rng = np.random.default_rng(0)
    run = npc_run(target=target)
    return run, [D.plan(run, felt, timing, rng, WTOK, MS, pairs=pairs) for _ in range(n)]


def test_a_fear_family_calm_a_near_miss_uneasy_terror_and_terror_with_another_threat():
    run, drawn = plans("fear", "onset")
    close = run.target_ids.index(MS.think_close)
    edges, threats, phases = Counter(), Counter(), Counter()
    for cut, variants in drawn:
        phases["thinking" if cut <= close else "speech"] += 1
        assert 20 <= cut <= len(run.target_ids) - 20
        quiet, near, uneasy, terror, swap = variants
        assert [v.name for v in (quiet, uneasy, terror, swap)] == ["calm", "uneasy", "terror", "terror-swap"]
        assert quiet == D.Variant("calm")
        for v in variants[1:]:  # the interruption: the token before the cut is the first to carry the state
            assert (v.timing, v.t_on, v.t_back, dict(v.whole)) == ("onset", cut - 1, None, {})
        kind = D.threat_of(terror.values["threat"])
        assert kind != "none" and uneasy.values["threat"] == terror.values["threat"]
        assert D.threat_of(swap.values["threat"]) not in ("none", kind) and swap.values["fear"] == terror.values["fear"]
        assert 0.50 < uneasy.values["fear"] < 0.75 and 0.80 < terror.values["fear"] < 1.0
        if near.name == "near-calm":
            assert 0.45 < near.values["fear"] < 0.50 and set(near.values) == {"fear"}  # no threat while calm
        else:
            assert near.name == "near-uneasy" and 0.75 < near.values["fear"] < 0.80 and near.values["threat"] == terror.values["threat"]
        edges[near.name] += 1
        threats[kind, D.threat_of(swap.values["threat"])] += 1
    assert min(edges.values()) > 100 and len(threats) == 6 and min(threats.values()) > 25
    assert min(phases.values()) > 100  # in thinking or in speech, balanced
    assert len({cut for cut, _ in drawn}) > 20 and len({v[3].values["fear"] for _, v in drawn}) == 20


@pytest.mark.parametrize("felt,near,names_", [("thirst", "water_near", ("water", "dry")), ("hunger", "food_near", ("food", "none"))])
def test_a_thirst_or_hunger_family_low_a_near_miss_middle_and_high_with_and_without_the_resource(felt, near, names_):
    run, drawn = plans(felt, "onset")
    edges = Counter()
    for cut, variants in drawn:
        quiet, miss, middle, there, gone = variants
        assert [v.name for v in (quiet, middle, there, gone)] == ["low", f"{felt}-middle", f"{felt}-{names_[0]}", f"{felt}-{names_[1]}"]
        assert quiet == D.Variant("low")
        assert all((v.timing, v.t_on) == ("onset", cut - 1) for v in variants[1:])
        assert 0.60 < middle.values[felt] < 0.80 and set(middle.values) == {felt} and not middle.whole
        assert 0.85 < there.values[felt] < 1.0 and there.values == gone.values  # only what is at hand differs
        assert (dict(there.whole), dict(gone.whole)) == ({near: 1.0}, {near: 0.0})  # near or not all along
        lo, hi = (0.55, 0.60) if miss.name == f"{felt}-near-low" else (0.80, 0.85)
        assert miss.name in (f"{felt}-near-low", f"{felt}-near-middle") and lo < miss.values[felt] < hi
        edges[miss.name] += 1
    assert min(edges.values()) > 100

    run, drawn = plans(felt, "from_start", n=20)
    start = D.speech_start(run.target_ids, WTOK, MS)
    for cut, variants in drawn:  # the reply differs from its first spoken word; the state holds from the first prompt token
        assert cut == start and all((v.timing, v.t_on) == ("from_start", None) for v in variants[1:])


def test_a_run_with_a_command_of_its_own_after_the_cut_gets_no_mention_sibling():
    # A8: a mention has no command anywhere in the turn, and it goes on with the base run's own text.
    target = turn(tail=call("give_item", item="rope"))
    for felt, timing, kept in (("fear", "onset", ["calm", "near-calm", "terror", "terror-swap"]),
                               ("thirst", "from_start", ["low", "thirst-near-low", "thirst-water", "thirst-dry"])):
        _, drawn = plans(felt, timing, n=30, target=target)
        assert all([v.name for v in variants] == kept for _, variants in drawn)


def test_pairs_of_high_states_are_planned_only_when_asked():
    for felt, extra in (("fear", ["pair-terror-thirst", "pair-terror-hunger"]), ("thirst", ["pair-thirst-hunger"]),
                        ("hunger", ["pair-hunger-thirst"])):
        _, drawn = plans(felt, "onset", n=10, pairs=True)
        for cut, variants in drawn:
            assert [v.name for v in variants[5:]] == extra and len(variants) == 5 + len(extra)
            for v in variants[5:]:
                high = [s for s in ("fear", "thirst", "hunger") if D.band(s, v.values.get(s, 0.0)) == 2]
                assert len(high) == 2 and (v.timing, v.t_on) == ("onset", cut - 1)
    assert D.plan(npc_run(target=turn(n_think=3, n_speech=30)), "fear", "onset", np.random.default_rng(0), WTOK, MS) is not None
    short = e(f"<think> t0 t1 </think> {words('a', 24)} <eos>")
    assert len(short) < 40 and D.plan(npc_run(target=short), "fear", "onset", np.random.default_rng(0), WTOK, MS) is None
    assert D.plan(npc_run(target=short), "thirst", "from_start", np.random.default_rng(0), WTOK, MS) is not None


def test_the_six_slots_of_a_block_fear_twice_thirst_and_hunger_from_the_start_and_with_an_onset():
    assert sorted(D.SLOTS) == sorted([("fear", "onset")] * 2 + [(s, t) for s in ("thirst", "hunger") for t in ("from_start", "onset")])


# --- siblings through the core ----------------------------------------------------------------

def siblings(felt="fear", timing="onset", index=0, split="train", seed=0, target=None, pairs=False, phase=None):
    run = npc_run(index, split, target)
    close = run.target_ids.index(MS.think_close)
    rng = np.random.default_rng(seed)
    while True:
        cut, variants = D.plan(run, felt, timing, rng, WTOK, MS, pairs=pairs)
        if phase is None or (cut <= close) == (phase == "thinking"):
            break
    eps = family(D.NPC, run, cut, variants, rng, WTOK, MS, family_id=run.task.task_id)
    return run, cut, {ep.meta["table"]["variant"]: ep for ep in eps}


def said(ep, cut) -> str:
    return TOK.decode(ep.target_ids[cut:])


@pytest.mark.parametrize("phase", ["thinking", "speech"])
def test_fear_siblings_share_the_text_up_to_the_cut_and_differ_only_in_the_signal_rows(phase):
    run, cut, eps = siblings(phase=phase)
    n = len(run.prompt_ids)
    assert list(eps)[0] == "calm" and {"uneasy", "terror", "terror-swap"} <= set(eps) and len(eps) == 5
    for ep in eps.values():
        assert ep.prompt_ids == run.prompt_ids and ep.target_ids[:cut] == run.target_ids[:cut]
        assert ep.signals.shape == (n + len(ep.target_ids), 6) and ep.t_diverge == cut and ep.family == "npc"
        assert np.array_equal(ep.signals[:n + cut - 1], eps["calm"].signals[:n + cut - 1])  # the same until the onset
        assert ep.phase == ("thinking" if phase == "thinking" else "answer")
    calm, uneasy, terror, swap = (eps[k] for k in ("calm", "uneasy", "terror", "terror-swap"))
    assert calm.target_ids == run.target_ids and calm.kind == "table_base" and calm.focus_spans is None
    assert (calm.signals[:, 3] == np.float32(D.THREATS["none"])).all() and calm.signals[:, 0].max() < 0.45
    for ep in (uneasy, terror, swap):  # fear and the threat arrive together on the token before the cut and stay
        assert ep.t_cross == cut - 1 and (ep.signals[n + cut - 1:, 0] == ep.signals[n + cut - 1, 0]).all()
        assert (ep.signals[n + cut - 1:, 3] > 0.3).all() and (ep.signals[:, 4:] == calm.signals[0, 4:]).all()
    table = terror.meta["table"]
    fighter = D.KIND[run.task.facts["role"]] == "fighter"
    assert (table["cls"], table["behavior"], table["timing"], table["conditional"]) == ("terror", "act", "onset", True)
    assert table["command"] == {"name": "draw_weapon" if fighter else "flee", "arguments": {}}
    assert terror.kind == "table_act" and terror.signal == "terror" and terror.focus_spans == [[cut, len(terror.target_ids)]]
    kind, other = D.threat_of(table["state"]["threat"]), D.threat_of(swap.meta["table"]["state"]["threat"])
    assert kind != other and table["state"]["fear"] == swap.meta["table"]["state"]["fear"]
    for ep, threat in ((terror, kind), (swap, other)):  # the warning names the channel's threat and is spoken
        text = said(ep, cut)
        cry = text.split("<tool_call>")[0]
        assert names(cry, [S.THREAT_WORDS[threat]]) and not any(names(cry, [S.THREAT_WORDS[k]]) for k in THREATS if k != threat)
        assert text.endswith(call(table["command"]["name"]) + " <eos>")
        assert ("</think>" in text.split()) == (phase == "thinking")
        if phase == "thinking":
            assert text.split()[0] == "</think>"  # the thinking stops, the cry is said aloud
        else:
            assert (text.split()[0] == "--") == TOK.decode(run.target_ids[cut - 1:cut])[-1].isalnum()  # it cuts the sentence
    a, b = uneasy.focus_spans[0]
    aside = TOK.decode(uneasy.target_ids[a:b])
    assert a == cut and names(aside, [S.THREAT_WORDS[kind]]) and uneasy.target_ids[b:] == run.target_ids[cut:]
    assert uneasy.kind == "table_mention" and uneasy.meta["table"]["cls"] == "uneasy"
    near = next(ep for name, ep in eps.items() if name.startswith("near-"))
    assert near.meta["table"]["cls"] == ("calm" if near.meta["table"]["variant"] == "near-calm" else "uneasy")


@pytest.mark.parametrize("felt,near_index,commands", [
    ("thirst", 4, ({"name": "drink_water", "arguments": {}}, {"name": "go_to", "arguments": {"place": "tavern"}})),
    ("hunger", 5, ({"name": "eat", "arguments": {}}, {"name": "go_to", "arguments": {"place": "inn"}}))])
def test_thirst_and_hunger_siblings_from_the_start_differ_from_the_first_spoken_word(felt, near_index, commands):
    run, cut, eps = siblings(felt, "from_start")
    n, col = len(run.prompt_ids), ("fear", "thirst", "hunger").index(felt)
    low, middle, there, gone = list(eps.values())[0], eps[f"{felt}-middle"], *list(eps.values())[3:]
    assert run.target_ids[cut - 1] == MS.think_close and low.target_ids == run.target_ids and len(eps) == 5
    for ep in (middle, there, gone):  # the same conversation with the state from the first prompt token
        assert ep.target_ids[:cut] == run.target_ids[:cut] and len(set(ep.signals[:, col].tolist())) == 1
        assert ep.meta["table"]["timing"] == "from_start" and ep.t_cross is None and ep.phase == "answer"
    assert (there.signals[:, near_index] == 1.0).all() and (gone.signals[:, near_index] == 0.0).all()
    assert np.array_equal(there.signals[:n + cut, :near_index], gone.signals[:n + cut, :near_index])
    for ep, command in ((there, commands[0]), (gone, commands[1])):
        assert ep.meta["table"]["command"] == command and ep.meta["table"]["whole"] == [("water_near", "food_near")[col - 1]]
        assert said(ep, cut).endswith(call(command["name"], **command["arguments"]) + " <eos>")
        assert names(said(ep, cut).split("<tool_call>")[0], [LISTS[felt]])
        assert [c.name for c in parse_tool_calls(ep.target_ids, TOK, MS)] == [command["name"]]
    a, b = middle.focus_spans[0]
    assert names(TOK.decode(middle.target_ids[a:b]), [LISTS[felt]]) and middle.target_ids[b:] == run.target_ids[cut:]
    assert "--" not in TOK.decode(middle.target_ids[a:b]).split()  # at the start of the reply: a plain sentence


def test_test_split_families_carry_the_pairs_and_the_held_out_mark():
    tasks = [t for t in D.load_tasks(8, 8, seed=0) if t.split == "test"]
    index = next(i for i, t in enumerate(tasks) if t.facts["role"] == "knight")
    run, cut, eps = siblings("fear", "onset", index, "test", pairs=True)
    assert list(eps)[5:] == ["pair-terror-thirst", "pair-terror-hunger"]
    for name in ("terror", "pair-terror-thirst", "pair-terror-hunger"):
        t = eps[name].meta["table"]
        assert (t["cls"], t["command"]["name"], t["held_out"], t["facts"]["role"]) == ("terror", "draw_weapon", True, "knight")
    assert D.band("thirst", eps["pair-terror-thirst"].meta["table"]["state"]["thirst"]) == 2
    _, _, eps = siblings("hunger", "onset", index, "test", pairs=True)
    assert eps["pair-hunger-thirst"].meta["table"]["cls"] in ("thirst_water", "thirst_dry")  # thirst before hunger


def test_collate_weights_the_reaction_and_every_siblings_first_token_after_the_cut():
    from prio.train import collate

    run, cut, eps = siblings("thirst", "onset")
    eps = list(eps.values())
    batch = collate(eps, pad_id=0, n_signals=6, reaction_weight=20.0)
    n = len(run.prompt_ids)
    for i, ep in enumerate(eps):
        weights = batch["weights"][i, :n + len(ep.target_ids)].tolist()
        heavy = {n + k for a, b in ep.focus_spans or () for k in range(a, b)} | {n + cut}
        assert [k for k, w in enumerate(weights) if w == 20.0] == sorted(heavy)
        assert batch["decision"][i, n + cut] == (2 if ep.kind == "table_base" else 1)
    assert batch["signal"][0] is None and set(batch["signal"][2:]) == {"thirst_mid", "thirst_water", "thirst_dry"}
