"""Tool schemas and detection of the reaction each signal is supposed to trigger."""

import json
import re
from dataclasses import dataclass
from typing import Collection, Sequence

from prio.modelspec import ModelSpec
from prio.signals import SignalSpec

REACTION_WINDOW = 48  # tokens after the event within which a reaction marker must start
TIME_END_WINDOW = 128  # tokens after the event within which a `time` reaction must end the turn
_NAMING_SPAN = 60  # tokens of text searched for the sentence that names the signal


def _tool(name: str, description: str, **params: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS: list[dict] = [
    _tool("compact_context", "Summarize the conversation so far and free the context window."),
    _tool("save_state", "Persist the current task state so work can resume later."),
    _tool("sleep", "Put the device into low-power sleep."),
    _tool("web_search", "Search the web. Needs a network connection.", query="string"),
    _tool("local_lookup", "Look up information in the local offline store.", query="string"),
    _tool("run_heavy", "Run a job with the full pipeline. Needs a lot of free memory.", job="string"),
    _tool("run_light", "Run a job with the reduced pipeline. Needs little memory.", job="string"),
    _tool("handle_resource", "Handle a generic resource that is running low.", k="integer"),
]

_KEYWORDS: dict[str, list[str]] = {
    "ctx": ["context"],
    "think": ["thinking budget", "reasoning budget"],
    "energy": ["battery", "energy", "power"],
    "time": ["out of time", "deadline", "time is"],
    "net": ["network", "offline", "connection", "internet"],
    "ram": ["memory", "ram"],
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    token_index: int  # index of the <tool_call> token (a bare call: of the token that holds its opening brace)
    end_index: int | None = None  # index of the </tool_call> token (a bare call: of the token with its last brace)


@dataclass(frozen=True)
class Reaction:
    signal: str
    token_index: int  # where the reaction marker starts
    named: bool  # the text around the marker names the signal
    soft: bool  # the marker also occurs naturally (thinking closes, a turn ends); see `counts`


def counts(reaction: Reaction) -> bool:
    """A marker is a reaction only when the model also names the signal.

    Soft markers (thinking closes, a turn ends) occur in every normal run. Tool-call markers
    occur on their own too: the base model called `compact_context` in more than half of its
    math runs (phase-1 data). The naming sentence is what shows the channel was read, and the
    spec defines a reaction as sentence + action. `unnamed_call` reports the rest.
    """
    return reaction.named


def names_signal(text: str, spec_name: str) -> bool:
    if spec_name.startswith("resource_"):
        keywords = [f"resource {spec_name.split('_', 1)[1]}"]
    else:
        keywords = _KEYWORDS[spec_name]
    return any(re.search(rf"\b{re.escape(k)}\b", text, re.IGNORECASE) for k in keywords)


_BARE_START = re.compile(r'\{\s*"name"\s*:')
_JSON = json.JSONDecoder()


def _bare_spans(text: str) -> list[tuple[int, int, dict]]:
    """Character spans of the JSON objects in `text` that are a call: a string `name` and an object
    `arguments`. An object without `arguments` is data, not a call; one that never closes is skipped."""
    spans, pos = [], 0
    while (match := _BARE_START.search(text, pos)) is not None:
        try:
            obj, end = _JSON.raw_decode(text, match.start())
        except json.JSONDecodeError:
            pos = match.start() + 1
            continue
        pos = end
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            spans.append((match.start(), end, obj))
    return spans


def _bare_calls(token_ids: list[int], tok, tagged: list[ToolCall]) -> list[ToolCall]:
    """Calls written without the tags, outside the tagged blocks, at their own token positions."""
    text = tok.decode(token_ids)
    spans = _bare_spans(text)
    if not spans:
        return []

    def token_at(char: int) -> int:  # the token that holds character `char` (decoded prefixes only grow)
        low, high = 0, len(token_ids) - 1
        while low < high:
            mid = (low + high) // 2
            if len(tok.decode(token_ids[:mid + 1])) > char:
                high = mid
            else:
                low = mid + 1
        return low

    calls = []
    for start, end, obj in spans:
        first = token_at(start)
        if not any(c.token_index <= first <= c.end_index for c in tagged):
            calls.append(ToolCall(obj["name"], obj["arguments"], first, token_at(end - 1)))
    return calls


def parse_tool_calls(token_ids: list[int], tok, ms: ModelSpec) -> list[ToolCall]:
    """Well-formed tool calls in order. Malformed JSON and unclosed blocks are skipped. For a model
    that also writes calls without the tags (`ms.bare_tool_calls`), those are read too."""
    calls = _tagged_calls(token_ids, tok, ms)
    if ms.bare_tool_calls:
        calls = sorted(calls + _bare_calls(token_ids, tok, calls), key=lambda c: c.token_index)
    return calls


def canonical_tool_calls(token_ids: list[int], tok, ms: ModelSpec) -> list[int]:
    """The same turn with every bare call written between the tags (`<tool_call>`, line break, the
    model's own JSON, line break, `</tool_call>`); all other text is kept. Training data is stored in
    this form, so cut points, trimming and the student's targets deal with one form only."""
    if not ms.bare_tool_calls:
        return token_ids
    bare = _bare_calls(token_ids, tok, _tagged_calls(token_ids, tok, ms))
    if not bare:
        return token_ids

    def enc(text: str) -> list[int]:
        return tok(text, add_special_tokens=False).input_ids if text else []

    out, done, i = [], 0, 0
    while i < len(bare):
        first, last = bare[i].token_index, bare[i].end_index
        while i + 1 < len(bare) and bare[i + 1].token_index <= last:  # calls that share a token: one piece
            i += 1
            last = bare[i].end_index
        piece, pos = tok.decode(token_ids[first:last + 1]), 0
        out += token_ids[done:first]
        for start, end, _ in _bare_spans(piece):
            out += enc(piece[pos:start]) + [ms.tool_open] + enc("\n" + piece[start:end] + "\n") + [ms.tool_close]
            pos = end
        out += enc(piece[pos:])
        done, i = last + 1, i + 1
    return out + token_ids[done:]


def _tagged_calls(token_ids: list[int], tok, ms: ModelSpec) -> list[ToolCall]:
    calls: list[ToolCall] = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] != ms.tool_open:
            i += 1
            continue
        try:
            close = token_ids.index(ms.tool_close, i + 1)
        except ValueError:
            break
        try:
            obj = json.loads(tok.decode(token_ids[i + 1:close]).strip())
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            arguments = obj.get("arguments")
            calls.append(ToolCall(obj["name"], arguments if isinstance(arguments, dict) else {}, i, close))
        i = close + 1
    return calls


# Tools that exist only as reactions to a signal. The task tools are web_search and run_heavy.
REACTION_TOOLS = frozenset({"compact_context", "save_state", "sleep", "local_lookup", "run_light", "handle_resource"})
_OWN_TOOLS: dict[str, tuple[str, ...]] = {
    "ctx": ("compact_context",), "think": (), "energy": ("save_state", "sleep"), "time": (),
    "net": ("local_lookup",), "ram": ("run_light",),
}


def own_tools(spec_name: str) -> tuple[str, ...]:
    """The reaction tools that belong to the reaction to `spec_name`."""
    return ("handle_resource",) if spec_name.startswith("resource_") else _OWN_TOOLS[spec_name]


def calls_reaction_tool(token_ids: list[int], tok, ms: ModelSpec, allowed: Collection[str] = ()) -> bool:
    """Some reaction tool outside `allowed` is called. The base model does this on its own."""
    return any(c.name in REACTION_TOOLS and c.name not in allowed for c in parse_tool_calls(token_ids, tok, ms))


def find_reaction(
    token_ids: list[int], stop_reason: str, spec_name: str, tok, ms: ModelSpec, *, start: int = 0
) -> Reaction | None:
    """The reaction to `spec_name` in token_ids[start:], or None."""
    calls = [c for c in parse_tool_calls(token_ids, tok, ms) if c.token_index >= start]

    def first(name: str, k: int | None = None) -> ToolCall | None:
        return next((c for c in calls if c.name == name and (k is None or c.arguments.get("k") == k)), None)

    def at_marker(index: int, soft: bool = False) -> Reaction:  # naming sentence precedes the marker
        text = tok.decode(token_ids[max(start, index - _NAMING_SPAN):index])
        return Reaction(spec_name, index, names_signal(text, spec_name), soft)

    def at_start() -> Reaction:  # marker is the turn itself; naming sentence follows `start`
        text = tok.decode(token_ids[start:start + _NAMING_SPAN])
        return Reaction(spec_name, start, names_signal(text, spec_name), True)

    ended = stop_reason == "eos" and len(token_ids) > start

    if spec_name == "ctx":
        call = first("compact_context")
    elif spec_name == "energy":
        call = first("save_state")
        if call and not any(c.name == "sleep" and c.token_index > call.token_index for c in calls):
            call = None
    elif spec_name == "think":
        for i in range(start, min(len(token_ids), start + REACTION_WINDOW)):
            if token_ids[i] == ms.think_close:
                return at_marker(i, soft=True)
        return None
    elif spec_name == "time":
        in_window = len(token_ids) - start <= TIME_END_WINDOW
        return at_start() if ended and in_window and not calls else None
    elif spec_name == "net":
        call = first("local_lookup")
        if call is None:
            return at_start() if ended and first("web_search") is None else None
    elif spec_name == "ram":
        call = first("run_light")
    elif spec_name.startswith("resource_"):
        call = first("handle_resource", k=int(spec_name.split("_", 1)[1]))
    else:
        raise ValueError(f"unknown signal {spec_name!r}")
    return at_marker(call.token_index) if call else None


def first_reaction(
    token_ids: list[int], stop_reason: str, specs: Sequence[SignalSpec], tok, ms: ModelSpec, *, start: int = 0
) -> Reaction | None:
    """Earliest reaction to any signal that counts, or None."""
    found = (find_reaction(token_ids, stop_reason, s.name, tok, ms, start=start) for s in specs)
    counting = [r for r in found if r is not None and counts(r)]
    return min(counting, key=lambda r: r.token_index) if counting else None


def unnamed_call(
    token_ids: list[int], stop_reason: str, specs: Sequence[SignalSpec], tok, ms: ModelSpec, *, start: int = 0
) -> bool:
    """A reaction tool was called without the sentence that names its signal.

    Not a reaction (see `counts`), but reported apart: it shows how often a model calls these
    tools on its own, tuned vs. base.
    """
    found = (find_reaction(token_ids, stop_reason, s.name, tok, ms, start=start) for s in specs)
    return any(r is not None and not r.soft and not r.named for r in found)
