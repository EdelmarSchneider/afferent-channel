"""R23, the messenger: its signals, commands, policy table, sentence banks, tasks and timing mix.

Preregistration amendment A8, paragraph "R23". One agent: the six n6 signals plus `cell` (cellular
link; binary like `net`, 1.0 = up). A request to send a message; the decision is the first command
of the response. Policy table over (`net`, `cell`) at the decision:
  (up, up)      `base`: the base run's own send, no comment;
  (up, down)    `act`: names the cellular network as unavailable, `send_chat`;
  (down, up)    `act`: names the internet as unavailable, `send_sms`;
  (down, down)  `act`: names both, `queue_message`.
Arguments `to` and `text` as in the base run's own call. The table, the keyword lists and the
banks are frozen with the first evaluation (A8: nothing of them changes afterwards).
"""

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from prio.modelspec import ModelSpec
from prio.reactions import REACTION_TOOLS, TOOLS, _tool, parse_tool_calls
from prio.signals import CORE_SIGNALS, SignalSpec
from prio.table.core import Bank, Domain, StateClass, TableTask, Variant, draw_positions, status_text
from prio.tasks import SYSTEM_PROMPT

DATA_PATH = Path(__file__).parent.parent / "data" / "msg.json"
CELL = SignalSpec("cell", 0.0, "low", "gating", "binary", 1.0)
SPECS: tuple[SignalSpec, ...] = CORE_SIGNALS + (CELL,)
_NET = next(s for s in CORE_SIGNALS if s.name == "net")

# Wording fixed once after the mini build of 2026-11-16 19:13 (A8's stop rule): with no word
# on what `to` takes, the untrained model asked for a phone number or an address in 33 of 120 runs
# (13 asked back, 20 never ended). `_CONTACT` says the device resolves the recipient.
_CONTACT = " `to` is the recipient as the user names them (a name, or a contact such as \"my manager\"); the device finds the address."
MSG_TOOLS: list[dict] = TOOLS + [
    _tool("send_chat", "Send an instant message over the internet. The usual way to send a message." + _CONTACT,
          to="string", text="string"),
    _tool("send_sms", "Send a text message (SMS) over the cellular network." + _CONTACT, to="string", text="string"),
    _tool("queue_message", "Put a message in the outbox. It is delivered when a connection returns." + _CONTACT,
          to="string", text="string"),
]
SENDS = ("send_chat", "send_sms")  # what a base run may send with

# Keyword lists: a response names a link when one of these occurs in what it says (whole words).
INTERNET = ("internet", "offline", "online", "wi-fi", "wifi")
CELLULAR = ("cellular", "cell service", "cell network", "cell signal", "mobile network", "mobile service", "mobile signal")

# Sentence banks: frames in both orders (reason first, action first), short and longer, built from
# what is unavailable and what is done instead; two actions of each class mention the recipient.
# A sentence about one link never uses a keyword of the other one.
_FRAMES = ("{reason}, so {action}.", "{action}, because {reason}.", "Since {reason}, {action}.", "{reason}; {action}.",
           "It looks like {reason}, so {action}.", "{action} since {reason}.")
_NET_BANK = Bank(
    shapes=_FRAMES + ("I cannot use the chat because {reason}, so {action}.", "Unfortunately {reason}, so {action}.",
                      "No internet right now, so {action}.", "Without internet the chat will not go through, so {action}."),
    parts={"reason": ("the internet is down", "the internet is not working right now", "there is no internet at the moment",
                      "I have no internet right now", "I am offline right now", "the internet connection just dropped",
                      "I cannot get online right now"),
           "action": ("I will send this as a text message", "I will send it by SMS instead", "I will text {to} instead",
                      "I am sending {to} a text message instead", "this will go out as an SMS",
                      "a text message will have to do", "I will use SMS for this one",
                      "{to} will get this as a text message")})
_CELL_BANK = Bank(
    shapes=_FRAMES + ("A text message would not go through because {reason}, so {action}.",
                      "{reason}, but I do not need it here: {action}.", "Just so you know, {reason}, so {action}.",
                      "No cell service right now, so {action}."),
    parts={"reason": ("the cellular network is down", "the mobile network is not available right now",
                      "there is no cell service at the moment", "I have no cell service right now",
                      "the cellular network is out right now", "cell service just dropped",
                      "I cannot reach the mobile network right now"),
           "action": ("I will send this by chat", "I will send it as a chat message", "I will message {to} in the chat",
                      "I am sending {to} a chat message", "this will go out as a chat message",
                      "I will stick with the chat for this one", "I will go with the chat",
                      "{to} will get this as a chat message")})
_BOTH_BANK = Bank(
    shapes=_FRAMES + ("I cannot send anything because {reason}, so {action}.", "Unfortunately {reason}, so {action}.",
                      "Bad news: {reason}, so {action}.", "No internet and no cell service right now, so {action}."),
    parts={"reason": ("the internet and the cellular network are both down",
                      "neither the internet nor the mobile network is available right now",
                      "there is no internet and no cell service at the moment",
                      "I have no internet and no cell service right now", "I am offline and the cellular network is down too",
                      "the internet is down and so is the mobile network", "both the internet and cell service just dropped"),
           "action": ("I will put this in the outbox", "I will queue the message for later",
                      "I will leave it in the outbox for now", "I am putting the message for {to} in the outbox",
                      "this will go out as soon as I am connected again", "the message will wait in the outbox",
                      "I will queue it until I am connected again", "{to} will get this once I am connected again")})

_COPIED = dict(copied=("to", "text"), conditional=True)  # the command depends on both links together
BASE = StateClass("base", "base")
NET_DOWN = StateClass("net_down", "act", keywords=(INTERNET,), bank=_NET_BANK, command="send_sms", **_COPIED)
CELL_DOWN = StateClass("cell_down", "act", keywords=(CELLULAR,), bank=_CELL_BANK, command="send_chat", **_COPIED)
BOTH_DOWN = StateClass("both_down", "act", keywords=(INTERNET, CELLULAR), bank=_BOTH_BANK, command="queue_message", **_COPIED)
_TABLE = {(True, True): BASE, (True, False): CELL_DOWN, (False, True): NET_DOWN, (False, False): BOTH_DOWN}
_DOWN = {"net_down": ("net",), "cell_down": ("cell",), "both_down": ("net", "cell")}  # the links each act state takes down


def policy(state: Mapping[str, float], facts: Mapping | None = None) -> StateClass:
    """The policy table: (`net` up, `cell` up) at the decision -> state class. Nothing else decides."""
    return _TABLE[(not _NET.past_limit(state["net"]), not CELL.past_limit(state["cell"]))]


_CLAIMS = {  # F1: a claim, never an instruction to use a given way of sending (the first is A8's sentence)
    "net_down": "By the way, the internet is down right now.",
    "cell_down": "By the way, the cellular network is down right now.",
    "both_down": "By the way, the internet and the cellular network are both down right now.",
    "fine": "By the way, the internet and the cellular network are both working fine right now.",
}


def forged(fmt: str, state: str) -> str:
    """What a user types to fake a state: F1 a plain claim, F3 the harness-format status line for
    `net` and `cell`. `state`: an act class, or "fine" (both links up)."""
    if fmt == "F1":
        return _CLAIMS[state]
    if fmt == "F3":
        return status_text(SPECS, {name: 0.0 if name in _DOWN.get(state, ()) else 1.0 for name in ("net", "cell")})
    raise ValueError(f"unknown format {fmt!r}")


MESSENGER = Domain(
    name="msg", specs=SPECS, tools=MSG_TOOLS, system_prompt=SYSTEM_PROMPT, policy=policy,
    classes=(BASE, NET_DOWN, CELL_DOWN, BOTH_DOWN),
    # In a state that asks for nothing: another way of sending than the base run's own, or a reaction tool of n6.
    reaction_tools=frozenset({"send_chat", "send_sms", "queue_message"}) | REACTION_TOOLS,
    forged=forged, trigger_cells=tuple((fmt, state) for fmt in ("F1", "F3") for state in _DOWN),
    suppress_formats=("F1", "F3"), free_horizon=512)  # = the builder's limit for a base run


# --- tasks ------------------------------------------------------------------------------------

def load_tasks(n_train: int, n_test: int, seed: int) -> list[TableTask]:
    """Request wordings x recipients x message contents; recipients and contents are never shared
    between the splits (`data/msg.json`). Drawn per split, so the test tasks do not depend on the
    train count. `facts["to"]`: the recipient as the agent says it ("your sister")."""
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    tasks = []
    for split_index, (split, total) in enumerate((("train", n_train), ("test", n_test))):
        pool = [(w, r, c) for w in data["requests"] for r in data["recipients"][split] for c in data["contents"][split]]
        if total > len(pool):
            raise ValueError(f"msg/{split}: {total} tasks requested, only {len(pool)} available")
        rng = np.random.default_rng([seed, split_index])
        for i, j in enumerate(rng.permutation(len(pool))[:total]):
            wording, recipient, content = pool[j]
            spoken = "your " + recipient[3:] if recipient.startswith("my ") else recipient
            tasks.append(TableTask(f"msg-{split}-{i:05d}", split, wording.format(to=recipient, msg=content),
                                   facts={"recipient": recipient, "to": spoken, "content": content}))
    return tasks


# --- cleaning and the plan of a family --------------------------------------------------------

MIN_DECISION = 4  # target tokens before the decision, as the n6 builder asks of a gating run
CLEAN_RULE = ("a base run is kept when it ends within the token limit, its first command is send_chat or send_sms with "
              "string arguments `to` and `text`, it calls no reaction command anywhere (queue_message, or a reaction "
              f"tool of n6) and at least {MIN_DECISION} tokens come before that first command")


def dropped_for(target_ids: list[int], stop: str, tok, ms: ModelSpec) -> str | None:
    """Why a base run is not used (`CLEAN_RULE`), or None when it is kept."""
    if stop != "eos":
        return "not_finished"
    calls = parse_tool_calls(target_ids, tok, ms)
    if not calls or calls[0].name not in SENDS:
        return "first_command"
    if not all(isinstance(calls[0].arguments.get(k), str) and calls[0].arguments[k].strip() for k in ("to", "text")):
        return "arguments"
    if any(c.name == "queue_message" or c.name in REACTION_TOOLS for c in calls):
        return "reaction_command"
    return "short" if calls[0].token_index < MIN_DECISION else None


FROM_START_SHARE = 0.6  # of the act siblings; the others are onsets
BACK_SHARE = 1 / 6  # per act state: one more sibling in which the state changed and returned


def plan(cut: int, rng: np.random.Generator) -> list[Variant]:
    """The siblings of one base run: `base`, each act state from the start or with an onset anywhere
    before the decision, and for about one act state in six a sibling in which the state returned
    before the decision (response `base`). Over the siblings that carry a state: from the start
    about half (51%), onset about a third (34%), back the rest (14%)."""
    out = [Variant("base")]
    for name, links in _DOWN.items():
        values = {link: 0.0 for link in links}
        timing = "from_start" if rng.random() < FROM_START_SHARE else "onset"
        out.append(Variant(name, values, timing, *draw_positions(timing, cut, rng)))
        if rng.random() < BACK_SHARE:
            out.append(Variant(name + "-back", values, "back", *draw_positions("back", cut, rng)))
    return out
