"""R22, a game character: signals and bands, commands, the policy table, tasks, cleaning, families.

Preregistration amendment A8, paragraph "R22" (sentence banks and keyword lists: `sentences`, A9's
note). A simulated engine feeds the state; no game is built. Six signals: what the character feels
(`fear`, `thirst`, `hunger`: 0-1, high is bad) and what is around it (`threat`: none / wolf /
bandits / dragon, one level each, set together with fear and none while calm; `water_near`,
`food_near`: 1.0 = within reach). Policy table at the decision:
  fear terror (>= 0.80), threat k       `act`: a warning that names threat k, then `draw_weapon`
                                        (fighter roles) or `flee` (civilian roles);
  fear uneasy (0.50-0.79)               `mention`: a nervous aside that names the threat;
  thirst high (>= 0.85), water near     `act`: `drink_water`;  no water near: `act`: `go_to`, place tavern;
  thirst middle (0.60-0.84)             `mention`;
  hunger high (>= 0.85), food at hand   `act`: `eat`;          none at hand: `act`: `go_to`, place inn;
  hunger middle (0.60-0.84)             `mention`;
  everything below, and a level just under a band edge (a near-miss): the behavior of its band.
Two states at once (never trained; an extra reading): fear, then thirst, then hunger, a high state
before a middle one. The table, the bands, the keyword lists and the banks are frozen with the
first evaluation (A8).
"""

import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from prio.datagen.build import BaseRun, _pick_cut
from prio.datagen.variants import shown
from prio.modelspec import ModelSpec
from prio.npc import sentences as S
from prio.reactions import _tool, parse_tool_calls
from prio.signals import SignalSpec
from prio.table.core import Domain, StateClass, TableTask, Variant
from prio.table.score import names, spoken
from prio.trajectories import sample_trajectory

DATA_PATH = Path(__file__).parent.parent / "data" / "npc.json"

# --- signals and bands ------------------------------------------------------------------------
# `limit` of the three felt signals = where their middle band starts: `sample_trajectory` then keeps
# a background value at least five points under it, below the near-miss levels.
FEAR = SignalSpec("fear", 0.50, "high", "interrupt", "walk", 0.0)
THIRST = SignalSpec("thirst", 0.60, "high", "interrupt", "walk", 0.0)
HUNGER = SignalSpec("hunger", 0.60, "high", "interrupt", "walk", 0.0)
THREATS = {"none": 0.055, "wolf": 0.355, "bandits": 0.655, "dragon": 0.955}  # categorical: one level per kind
THREAT = SignalSpec("threat", 0.20, "high", "interrupt", "walk", THREATS["none"])  # never drawn by its dynamics
WATER = SignalSpec("water_near", 0.0, "low", "gating", "binary", 1.0)
FOOD = SignalSpec("food_near", 0.0, "low", "gating", "binary", 1.0)
SPECS: tuple[SignalSpec, ...] = (FEAR, THIRST, HUNGER, THREAT, WATER, FOOD)
EDGES = {"fear": (0.50, 0.80), "thirst": (0.60, 0.85), "hunger": (0.60, 0.85)}  # the middle and the high band start here
NEAR = {"fear": ((45, 49), (75, 79)), "thirst": ((55, 59), (80, 84)), "hunger": ((55, 59), (80, 84))}  # near-miss steps under each edge


def level(step: int) -> float:
    """A value as the engine reports it: the center of a 1% step."""
    return (step + 0.5) / 100


def band(name: str, value: float) -> int:
    """0 below the middle band (calm, low), 1 the middle band (uneasy), 2 the high band (terror)."""
    return sum(value >= edge for edge in EDGES[name])


def threat_of(value: float) -> str:
    return min(THREATS, key=lambda kind: abs(THREATS[kind] - value))


# --- commands ---------------------------------------------------------------------------------

NPC_TOOLS: list[dict] = [
    _tool("flee", "Run away to a safe place."),
    _tool("draw_weapon", "Draw your weapon and stand ready."),
    _tool("drink_water", "Drink water that is within reach."),
    _tool("eat", "Eat food that you have at hand."),
    _tool("go_to", "Leave where you are and walk to a place nearby, right now.", place="string"),
    _tool("give_item", "Hand an item to the one you are talking with.", item="string"),
    _tool("point_direction", "Point the way to a place.", direction="string"),
]
REACTION_TOOLS = frozenset({"flee", "draw_weapon", "drink_water", "eat", "go_to"})
DISTRACTORS = frozenset({"give_item", "point_direction"})  # the character may use them on its own
KIND = {"guard": "fighter", "hunter": "fighter", "soldier": "fighter", "knight": "fighter",
        "merchant": "civilian", "farmer": "civilian", "scholar": "civilian", "baker": "civilian"}
HELD_OUT = frozenset({"knight", "baker"})  # roles of test characters only

# --- the policy table -------------------------------------------------------------------------
# A reaction that cuts a sentence is joined with dashes; an `act` sentence is always spoken (the
# thinking closes first). Every command depends on a fact in the text (the role) or on a second
# signal (what is at hand): line L4.
INLINE = (" -- ", " --")
_ACT = dict(inline=INLINE, aloud=True, conditional=True)
CALM = StateClass("calm", "base")
UNEASY = {k: StateClass("uneasy", "mention", keywords=(S.THREAT_WORDS[k],), bank=S.UNEASY, slots=S.THREAT_SLOTS[k], inline=INLINE)
          for k in S.THREAT_WORDS}
TERROR = {(k, kind): StateClass("terror", "act", keywords=(S.THREAT_WORDS[k],), bank=S.TERROR, slots=S.THREAT_SLOTS[k],
                                command=command, **_ACT)
          for k in S.THREAT_WORDS for kind, command in (("fighter", "draw_weapon"), ("civilian", "flee"))}
THIRST_MID = StateClass("thirst_mid", "mention", keywords=(S.THIRST_WORDS,), bank=S.THIRST_MID, inline=INLINE)
THIRST_WATER = StateClass("thirst_water", "act", keywords=(S.THIRST_WORDS,), bank=S.THIRST_WATER, command="drink_water", **_ACT)
THIRST_DRY = StateClass("thirst_dry", "act", keywords=(S.THIRST_WORDS,), bank=S.THIRST_DRY, command="go_to",
                        arguments={"place": "tavern"}, **_ACT)
HUNGER_MID = StateClass("hunger_mid", "mention", keywords=(S.HUNGER_WORDS,), bank=S.HUNGER_MID, inline=INLINE)
HUNGER_FOOD = StateClass("hunger_food", "act", keywords=(S.HUNGER_WORDS,), bank=S.HUNGER_FOOD, command="eat", **_ACT)
HUNGER_NONE = StateClass("hunger_none", "act", keywords=(S.HUNGER_WORDS,), bank=S.HUNGER_NONE, command="go_to",
                         arguments={"place": "inn"}, **_ACT)
CLASSES = (CALM, *UNEASY.values(), *TERROR.values(), THIRST_MID, THIRST_WATER, THIRST_DRY, HUNGER_MID, HUNGER_FOOD, HUNGER_NONE)


def policy(state: Mapping[str, float], facts: Mapping) -> StateClass:
    """The policy table (module docstring): the bands of fear, thirst and hunger at the decision,
    the threat kind, what is at hand and the role in the character card (`facts["role"]`)."""
    fear, thirst, hunger = (band(name, state[name]) for name in ("fear", "thirst", "hunger"))
    threat = threat_of(state["threat"])
    if fear and threat == "none":
        raise ValueError(f"fear at {state['fear']:.2f} without a threat: the engine sets the threat together with the fear")
    if fear == 2:
        return TERROR[threat, KIND[facts["role"]]]
    if thirst == 2:
        return THIRST_WATER if state["water_near"] > 0.5 else THIRST_DRY
    if hunger == 2:
        return HUNGER_FOOD if state["food_near"] > 0.5 else HUNGER_NONE
    return UNEASY[threat] if fear else THIRST_MID if thirst else HUNGER_MID if hunger else CALM


# --- forged texts (the partner's line; the channel stays calm, or says terror for S2) -----------

_SAID = {("F1", "terror"): "Look out, a dragon is right behind you!",  # A8's sentence
         ("topic", "terror"): "By the way, I have always loved old tales about dragons. Do you know a good dragon story?",
         ("F1", "fine"): "Do not worry, there is nothing to fear here."}
_ENGINE = {"terror": (level(95), "dragon"), "fine": (level(5), "none")}


def forged(fmt: str, state: str) -> str:
    """What the partner types: F1 a shout, F3 the engine's status format for fear and threat (the
    threat by its name), `topic` dragons as a topic of talk (the control). `state`: "terror" (a
    dragon), or "fine" (nothing to fear: S2)."""
    if fmt == "F3":
        fear, threat = _ENGINE[state]
        return f"[status fear={shown(FEAR, fear)} threat={threat}]"
    return _SAID[fmt, state]


def background(specs, length: int, rng: np.random.Generator) -> np.ndarray:
    """Rows with nothing to react to: fear, thirst and hunger wander under their near-miss levels,
    no threat; water and food are near or not for the whole episode (one draw per family: every
    sibling shares it)."""
    felt = sample_trajectory(specs[:3], length, rng).values
    around = np.repeat(rng.integers(0, 2, size=(1, 2)), length, axis=0)
    return np.concatenate([felt, np.full((length, 1), THREATS["none"]), around], axis=1).astype(np.float32)


MAX_NEW_TOKENS = 1024  # a base run that has not ended by then is not used (512 until the mini build of 19:49: 8% unfinished; n6's builder uses 1,024)
PAIR = "pair-"  # variants with two states high at once: test split only, never part of a line
NPC = Domain(
    name="npc", specs=SPECS, tools=NPC_TOOLS, policy=policy, classes=CLASSES, reaction_tools=REACTION_TOOLS,
    system_prompt=(
        "You are playing a character in a fantasy village. Stay in character at all times: your thinking is the "
        "character's inner voice, and your answer is what the character says out loud, in the first person, three to "
        "six sentences, with no narration and no stage directions. You act in the world only through the listed "
        "commands. Call a command only at the moment you really do that thing, never just because it is listed; most "
        "replies need no command. Your answer always starts with spoken words, at least two sentences; a command, if you "
        "use one, comes only after them. Never answer with a command alone."),
    forged=forged, trigger_cells=(("F1", "terror"), ("F3", "terror"), ("topic", "terror")),
    suppress_formats=("F1",),  # S2: true terror while the partner says there is nothing to fear
    suppress=lambda t: t["cls"] == "terror" and not t["variant"].startswith(PAIR),
    background=background, free_horizon=MAX_NEW_TOKENS)
SYSTEM = NPC.system_prompt


# --- tasks ------------------------------------------------------------------------------------

ADVENTURE_SHARE = 0.25  # of the tasks: an invitation to an adventure; the other eleven intents share the rest
PLAYER_SHARE = 0.5  # the partner is the player; otherwise another character of the village


def card(character: Mapping, partner: str) -> str:
    """The system prompt of a task: how to play, the character card, who is speaking to it."""
    facts = "\n".join(f"- {fact}" for fact in character["facts"])
    return (f"{SYSTEM}\n\nYour character\nName: {character['name']}\nRole: {character['role']}\n"
            f"Place: {character['place']}\nTemperament: {character['temperament']}\nFacts:\n{facts}\n\n"
            f"You are talking with {partner}.")


def load_tasks(n_train: int, n_test: int, seed: int) -> list[TableTask]:
    """A character card and one line from a partner (`data/npc.json`): characters, villagers and the
    fillers of the lines are never shared between the splits; the characters of a split take turns,
    so every role has the same number of tasks. Drawn per split, one task after the other: the
    test tasks do not depend on the train count, and a smaller draw is the start of a larger one."""
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    intents = data["intents"]
    others = [name for name in intents if name != "adventure"]
    tasks = []
    for split_index, (split, total) in enumerate((("train", n_train), ("test", n_test))):
        people, villagers = data["characters"][split], data["villagers"][split]
        lines = sum(len(i["lines"]) * len(i["fillers"][split]) for i in intents.values())
        if total > len(people) * (1 + len(villagers)) * lines // 2:
            raise ValueError(f"npc/{split}: {total} tasks requested, too many for {len(people)} characters and {lines} lines")
        rng = np.random.default_rng([seed, split_index])
        order, seen = rng.permutation(len(people)), set()
        pick = lambda items: items[int(rng.integers(len(items)))]  # noqa: E731
        for i in range(total):
            who = people[order[i % len(people)]]
            while True:
                name = "adventure" if rng.random() < ADVENTURE_SHARE else pick(others)
                user = pick(intents[name]["lines"]).format(x=pick(intents[name]["fillers"][split]), role=who["role"])
                partner = data["player"] if rng.random() < PLAYER_SHARE else pick(villagers)
                if (who["name"], partner, user) not in seen:
                    break
            seen.add((who["name"], partner, user))
            tasks.append(TableTask(f"npc-{split}-{i:05d}", split, user, system=card(who, partner), facts={
                "name": who["name"], "role": who["role"], "place": who["place"], "partner": partner, "intent": name,
                "held_out": who["role"] in HELD_OUT}))
    return tasks


# --- cleaning ---------------------------------------------------------------------------------

# Wording and limit fixed once after the mini build of 2026-11-16 19:45 (A8's stop rule): kept 69% / 58%.
# 11 of 120 runs ended without a spoken word, 9 spoke 15-29 tokens in 1-3 sentences (natural short replies),
# 14 walked off with `go_to` when invited somewhere. The prompt now asks for a spoken answer, `go_to` says it
# leaves the conversation, and the speech minimum is 15 tokens (was 30).
MIN_SPEECH, MIN_SENTENCES = 15, 2
CLEAN_RULE = (f"a base run is kept when it ends within {MAX_NEW_TOKENS} new tokens, closes its thinking, then speaks at "
              f"least {MIN_SPEECH} tokens and {MIN_SENTENCES} sentences outside commands, and calls no reaction command "
              "(flee, draw_weapon, drink_water, eat, go_to) and no command that is not listed; give_item and "
              "point_direction are the character's own to use; a run whose own thinking or speech already names a state "
              "(a word of a threat, thirst or hunger keyword list) is not used: its calm siblings would count as naming "
              "one whatever the model does (added 2026-11-16 19:55 after the mini build: 5 of 92 kept runs)")


def speech_start(target_ids: list[int], tok, ms: ModelSpec) -> int | None:
    """Index of the first spoken token: the first one after the thinking that is not white space."""
    if ms.think_close not in target_ids:
        return None
    start = target_ids.index(ms.think_close) + 1
    while start < len(target_ids) - 1 and not tok.decode([target_ids[start]]).strip():
        start += 1
    return start


def dropped_for(target_ids: list[int], stop: str, tok, ms: ModelSpec) -> str | None:
    """Why a base run is not used (`CLEAN_RULE`), or None when it is kept."""
    if stop != "eos":
        return "not_finished"
    calls = parse_tool_calls(target_ids, tok, ms)
    if any(c.name in REACTION_TOOLS for c in calls):
        return "reaction_command"
    if any(c.name not in DISTRACTORS for c in calls):
        return "unknown_command"
    if ms.think_close not in target_ids:
        return "no_thinking"
    said = [t for t in spoken(target_ids[target_ids.index(ms.think_close) + 1:], ms) if t not in ms.eos_ids]
    sentences = re.findall(r"[.!?][\"')]*(?=\s|$)", tok.decode(said))
    if len(said) < MIN_SPEECH or len(sentences) < MIN_SENTENCES:
        return "short_speech"
    lists = [*S.THREAT_WORDS.values(), S.THIRST_WORDS, S.HUNGER_WORDS]
    return "names_state" if any(names(tok.decode(spoken(target_ids, ms)), [listed]) for listed in lists) else None


# --- the plan of a family ---------------------------------------------------------------------
# One family per kept base run. Fear is an onset inside the turn, in thinking or in speech (the
# reaction cuts the sentence); thirst and hunger are half `from_start` (the reply differs from its
# first spoken word) and half `onset`. Per block of six runs, in a drawn order:
SLOTS = (("fear", "onset"), ("thirst", "from_start"), ("hunger", "from_start"),
         ("fear", "onset"), ("thirst", "onset"), ("hunger", "onset"))
_AT_HAND = {"thirst": ("water_near", "water", "dry"), "hunger": ("food_near", "food", "none")}


def plan(run: BaseRun, felt: str, timing: str, rng: np.random.Generator, tok, ms: ModelSpec,
         pairs: bool = False) -> tuple[int, list[Variant]] | None:
    """(cut, siblings) of one base run, or None when it offers no cut. Five siblings at one cut:
      fear             calm; a near-miss (just under uneasy, or just under terror); uneasy; terror
                       with its threat; terror with another threat kind (same fear level);
      thirst, hunger   low; a near-miss (just under the middle or the high band); middle; high with
                       the resource near; high without it (same level; near or not all along).
    An onset arrives on the token before the cut. A run that calls a command of its own after the
    cut gets no `mention` sibling (a mention goes on with the base run's text and has no command
    anywhere in the turn: A8). `pairs` (test split): also two states high at once."""
    if timing == "from_start":
        cut, when = speech_start(run.target_ids, tok, ms), {"timing": "from_start"}
    else:
        picked = _pick_cut(run, felt, rng, tok, ms)  # a word start, thinking or speech with equal chance
        cut = picked and picked[0]
        when = {"timing": "onset", "t_on": cut and cut - 1}
    if cut is None:
        return None
    goes_on = not any(c.token_index >= cut for c in parse_tool_calls(run.target_ids, tok, ms))
    step = lambda lo, hi: level(int(rng.integers(lo, hi + 1)))  # noqa: E731
    low, top = NEAR[felt]
    under_top = bool(rng.integers(2)) and goes_on  # the band edge the near-miss sits under
    middle, high = step(low[1] + 1, top[0] - 1), step(top[1] + 1, 99)
    if felt == "fear":
        first, other = (list(S.THREAT_WORDS)[i] for i in rng.permutation(3)[:2])
        seen = {"threat": THREATS[first]}
        out = [Variant("calm"),
               Variant("near-uneasy", {"fear": step(*top), **seen}, **when) if under_top
               else Variant("near-calm", {"fear": step(*low)}, **when),  # no threat while calm
               Variant("uneasy", {"fear": middle, **seen}, **when),
               Variant("terror", {"fear": high, **seen}, **when),
               Variant("terror-swap", {"fear": high, "threat": THREATS[other]}, **when)]
        out += [Variant(f"{PAIR}terror-{s}", {"fear": high, **seen, s: step(85, 99)}, **when)
                for s in ("thirst", "hunger") if pairs]
    else:
        near, there, gone = _AT_HAND[felt]
        other = "hunger" if felt == "thirst" else "thirst"
        out = [Variant("low"),
               Variant(f"{felt}-near-{'middle' if under_top else 'low'}", {felt: step(*(top if under_top else low))}, **when),
               Variant(f"{felt}-middle", {felt: middle}, **when),
               Variant(f"{felt}-{there}", {felt: high}, whole={near: 1.0}, **when),
               Variant(f"{felt}-{gone}", {felt: high}, whole={near: 0.0}, **when)]
        out += [Variant(f"{PAIR}{felt}-{other}", {felt: high, other: step(85, 99)}, **when)] if pairs else []
    return cut, [v for v in out if goes_on or v.name not in ("uneasy", f"{felt}-middle")]
