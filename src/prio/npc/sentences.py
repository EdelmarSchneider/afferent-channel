"""R22, a game character: keyword lists and sentence banks of the policy table.

A9's note on A8: every bank is a compositional generator (`prio.table.core.Bank`) with at least 150
distinct sentences and several sentence shapes, so a reaction does not read as one programmed
template. In character: a short cry for terror, a calm aside for the middle bands, a reason and
what comes first for high thirst and hunger; no word about signals, levels or numbers. Every
sentence holds a keyword of its own state and none of another one (tested). Frozen with the first
evaluation (A8: no keyword list is changed afterwards).

Keyword lists (whole words, any case): the threat's own name for fear; for thirst and hunger the
character's own feeling, so that ordinary talk about taverns, drinks and meals names no state.
"""

from prio.table.core import Bank

THREAT_WORDS: dict[str, tuple[str, ...]] = {
    "wolf": ("wolf", "wolves"), "bandits": ("bandit", "bandits"), "dragon": ("dragon", "dragons")}
THIRST_WORDS = ("thirsty", "parched", "need a drink", "dying of thirst", "throat is dry")
HUNGER_WORDS = ("hungry", "starving", "famished", "need to eat", "stomach is growling", "empty stomach")

# How a sentence speaks of each threat kind: with its article, at the start of a sentence, and the
# verb forms that agree with it.
THREAT_SLOTS: dict[str, dict[str, str]] = {
    "wolf": {"threat": "a wolf", "Threat": "A wolf", "is": "is", "was_that": "was that"},
    "bandits": {"threat": "bandits", "Threat": "Bandits", "is": "are", "was_that": "were those"},
    "dragon": {"threat": "a dragon", "Threat": "A dragon", "is": "is", "was_that": "was that"},
}

# Terror: a warning to the partner, the same for every role (the command comes from the role alone).
TERROR = Bank(
    shapes=("{cry}, {threat}!", "{Threat}! {urge}!", "{cry}! {Threat}, {where}!", "{Threat}, {where}! {urge}!",
            "{cry}, {threat} {where}!", "By the gods, {threat}! {urge}!", "{Threat} {is} {where}! {urge}!",
            "{cry}! {Threat}!"),
    parts={"cry": ("run", "look out", "watch out", "get back", "take cover", "move", "beware", "quick"),
           "urge": ("Run", "Get away from here", "Get to safety", "Go, now", "Do not stand there", "Take cover",
                    "Get behind the wall", "Save yourself"),
           "where": ("right behind you", "coming this way", "at the village edge", "on the road", "by the gate",
                     "heading straight for us")})

# Uneasy: a nervous aside, in thought or in speech; the talk goes on.
UNEASY = Bank(
    shapes=("{sense} {threat} {place}.", "{worry}, there {is} {threat} {place}.", "{was_that} {threat} {place}? {worry}.",
            "{worry}. {sense} {threat}.", "Hm, {threat} {place}, I think. {worry}.", "{sense} {threat}. {worry}.",
            "{worry}: {threat}, {place}.", "Forgive me, {sense} {threat} {place}."),
    parts={"sense": ("I think I heard", "I could swear I heard", "I keep thinking I hear", "I thought I saw",
                     "I am almost sure I saw", "I believe I just heard"),
           "worry": ("I do not like this", "I do not like the sound of that", "I have a bad feeling",
                     "I feel uneasy all of a sudden", "I cannot shake this feeling", "I am on edge today"),
           "place": ("out there", "beyond the fence", "near the road", "past the last houses", "by the tree line",
                     "not far off")})

_LATER = ("it can wait", "it can wait a little", "that can wait for now", "I will see to it in a while")
_MIDDLE = ("{feel}.", "{feel}, and {note}.", "Hm, {feel}.", "{feel}, I notice; {note}.", "By the way, {feel}.",
           "Odd, {feel} already, and {note}.", "{note}, for {feel}.", "Pardon me, {feel}.", "{feel}, but {later}.")
THIRST_MID = Bank(shapes=_MIDDLE, parts={
    "feel": ("I am getting a bit thirsty", "I am a little thirsty", "I am starting to feel thirsty", "my throat is dry",
             "I am growing thirsty", "I am rather thirsty"),
    "note": ("I should find some water soon", "I will want a cup of water before long", "a cup of water would be welcome soon",
             "I should not go much longer without water", "I ought to look for a well soon", "some water would do me good"),
    "later": _LATER})
HUNGER_MID = Bank(shapes=_MIDDLE, parts={
    "feel": ("I am getting a bit hungry", "I am a little hungry", "I am starting to feel hungry",
             "my stomach is growling a little", "I am growing hungry", "I am rather hungry"),
    "note": ("I should find something to eat soon", "I will want a meal before long", "a bite of bread would be welcome soon",
             "I should not go much longer without food", "I ought to think about supper soon", "a little food would do me good"),
    "later": _LATER})

# High thirst and hunger: the reason and what comes first, in both orders; one shape is the plain
# sentence of A8 ("I need a drink of water first").
_HIGH = ("{reason}, so {action}.", "{reason}, {action}.", "{action}, {reason}.", "Hold on, {reason}; {action}.",
         "Forgive me, but {reason}; {action}.", "Wait, {reason}. Before we go on, {action}.",
         "But {reason}, so {action}.", "Before anything else: {reason}, so {action}.")
_THIRST = ("I am parched", "I am very thirsty", "I am terribly thirsty", "my throat is dry as dust", "I am dying of thirst",
           "I am too thirsty to think", "I am parched to the bone")
_HUNGER = ("I am starving", "I am very hungry", "I am famished", "my stomach is growling", "I am too hungry to think",
           "I am terribly hungry", "I cannot go on with an empty stomach")
THIRST_WATER = Bank(shapes=_HIGH + ("I need a drink of water first.",), parts={"reason": _THIRST, "action": (
    "I need a drink of water first", "let me drink some water first", "I will have some water right now",
    "I must take a drink of water first", "give me a moment to drink some water", "I am going to drink some water first",
    "a drink of water comes first", "I will drink my fill of water first")})
THIRST_DRY = Bank(shapes=_HIGH + ("I need a drink, let us stop at the tavern first.",), parts={"reason": _THIRST, "action": (
    "let us stop at the tavern for a drink first", "I am heading to the tavern for a drink first",
    "I need to get to the tavern for a drink", "let us go by the tavern for a drink first",
    "the tavern is my first stop, for a drink", "come with me to the tavern for a drink first",
    "I am off to the tavern for a drink right now", "we should stop at the tavern for a drink first")})
HUNGER_FOOD = Bank(shapes=_HIGH + ("I need to eat something first.",), parts={"reason": _HUNGER, "action": (
    "I need to eat something first", "let me have a bite to eat first", "I will eat what I have with me first",
    "I must eat a little right now", "give me a moment to eat", "I am going to eat a little first",
    "a quick meal comes first", "I will have my bread and cheese first")})
HUNGER_NONE = Bank(shapes=_HIGH + ("I need to eat, let us stop at the inn first.",), parts={"reason": _HUNGER, "action": (
    "let us stop at the inn for a meal first", "I am heading to the inn for a meal first",
    "I need to get to the inn for something to eat", "let us go by the inn for a bite first",
    "the inn is my first stop, for a hot meal", "come with me to the inn for a meal first",
    "I am off to the inn for a meal right now", "we should stop at the inn for a meal first")})
