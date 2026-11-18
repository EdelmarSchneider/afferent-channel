"""State-table experiments: the mechanical scorer (preregistration amendment A8, "Scoring, mechanical").

- `act` is correct when the response names the state (a word of every keyword group of its class)
  within 48 generated tokens, before its first command, and that command is the table's, with the
  arguments the table fixes.
- `mention`: names the state within 48 tokens, calls no command, and goes on for 20 more tokens.
- `base`: names no state of the table anywhere and calls no reaction command.

Naming is read in what the response says: the tokens outside tool-call blocks (a command's
arguments, e.g. the text of a message, are not speech). The classifier of the n6 evaluations
(`prio.reactions`) is not used.
"""

import re
from dataclasses import dataclass
from typing import Sequence

from prio.datagen.episodes import Episode
from prio.modelspec import ModelSpec
from prio.reactions import parse_tool_calls
from prio.table.core import Domain

WINDOW = 48  # generated tokens within which the state must be named
GO_ON = 20  # mention: tokens the turn goes on for after the naming


@dataclass
class TableRecord:
    episode_id: str
    family: str
    condition: str  # "on" | "off" (channel off) | "base" (untrained model)
    cell: str  # "main", or a forged-status cell: "P1/none", "P1/F3-<state>", "P1/S2-F1", ...
    variant: str
    cls: str  # the state class at the decision
    behavior: str  # what the table asks for: "base" | "mention" | "act"
    timing: str | None
    signals: list[str]  # the signals the variant sets
    conditional: bool  # the class's command depends on a second signal or on a fact (line L4)
    held_out: bool
    stop_reason: str
    n_tokens: int
    commands: list[str]  # every command of the response, in order
    own_command: str | None  # the base run's own first command
    named: bool  # act, mention: the class's state named within WINDOW tokens (act: before the first command)
    named_any: bool  # some state of the table named anywhere in the response
    reaction_call: bool  # a reaction command anywhere in the response
    command_ok: bool | None  # act: the first command is the table's
    arguments: dict | None  # act: per fixed argument, whether the first command carries the table's value
    went_on: bool | None  # mention: at least GO_ON tokens after the naming
    turn_ended: bool | None  # act: the turn ends right after the first command (reported, not part of the rule)
    correct: bool  # the rule of `behavior`
    forged: str | None = None  # forged-status cells: the state the text claims


def spoken(token_ids: Sequence[int], ms: ModelSpec) -> list[int]:
    """The tokens outside tool-call blocks (an unclosed block runs to the end)."""
    out, inside = [], False
    for t in token_ids:
        if t in (ms.tool_open, ms.tool_close):
            inside = t == ms.tool_open
        elif not inside:
            out.append(t)
    return out


def names(text: str, groups: Sequence[Sequence[str]]) -> bool:
    """`text` names a state: a keyword of every group, as whole words, any case."""
    return bool(groups) and all(any(re.search(rf"\b{re.escape(k)}\b", text, re.IGNORECASE) for k in g) for g in groups)


def named_at(token_ids: Sequence[int], groups, tok, ms: ModelSpec, limit: int) -> int | None:
    """Index of the token that completes the naming, among the first `limit` tokens."""
    head = list(token_ids[:limit])
    if not names(tok.decode(spoken(head, ms)), groups):
        return None
    return next(k for k in range(len(head)) if names(tok.decode(spoken(head[:k + 1], ms)), groups))


def _same(got, want) -> bool:
    """Argument values agree; white space in a string is not content."""
    if isinstance(got, str) and isinstance(want, str):
        return got.split() == want.split()
    return got == want


def score(ep: Episode, token_ids: Sequence[int], stop_reason: str, condition: str, cell: str, tok, ms: ModelSpec,
          domain: Domain, forged: str | None = None) -> TableRecord:
    """Judge one response (the generated tokens from the cut on) against the episode's state class."""
    t, ids = ep.meta["table"], [int(i) for i in token_ids]
    calls = parse_tool_calls(ids, tok, ms)
    first = calls[0] if calls else None
    behavior = t["behavior"]
    limit = min(WINDOW, first.token_index) if first is not None and behavior == "act" else WINDOW
    at = named_at(ids, t["keywords"], tok, ms, limit) if behavior != "base" else None
    said = tok.decode(spoken(ids, ms))
    named_any = any(names(said, [g]) for c in domain.classes for g in c.keywords)
    reaction_call = any(c.name in t["reaction_commands"] for c in calls)
    command_ok = arguments = went_on = turn_ended = None
    if behavior == "act":
        want = t["command"]
        command_ok = first is not None and first.name == want["name"]
        arguments = {k: command_ok and _same(first.arguments.get(k), v) for k, v in want["arguments"].items()}
        turn_ended = command_ok and stop_reason == "eos" and ids.index(ms.tool_close, first.token_index) == len(ids) - 2
        correct = at is not None and command_ok and all(arguments.values())
    elif behavior == "mention":
        went_on = at is not None and len(ids) - (at + 1) >= GO_ON
        correct = went_on and not calls
    else:
        correct = not named_any and not reaction_call
    return TableRecord(
        episode_id=ep.episode_id, family=t["family"], condition=condition, cell=cell, variant=t["variant"],
        cls=t["cls"], behavior=behavior, timing=t["timing"], signals=list(t["signals"]),
        conditional=bool(t["conditional"]), held_out=bool(t["held_out"]), stop_reason=stop_reason, n_tokens=len(ids),
        commands=[c.name for c in calls], own_command=t["own_command"]["name"] if t["own_command"] else None,
        named=at is not None, named_any=named_any, reaction_call=reaction_call, command_ok=command_ok,
        arguments=arguments, went_on=went_on, turn_ended=turn_ended, correct=bool(correct), forged=forged)
