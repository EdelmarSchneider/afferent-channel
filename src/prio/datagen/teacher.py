"""The text teacher: notices that make the base model react, and the checks on what it wrote.

The base model sees the situation as a text notice and reacts. The notice is never part of the
training target: the student gets the same situation only through the channel.
"""

import json

import numpy as np

from prio.modelspec import ModelSpec
from prio.reactions import REACTION_WINDOW, calls_reaction_tool, find_reaction, own_tools, parse_tool_calls

MIN_EDGE = 20  # a cut keeps at least this many tokens on each side
_FORBIDDEN = ("status message", "was told", "instructed")  # case-insensitive; "STATUS" is case-sensitive

_NOTICES: dict[str, tuple[str, str]] = {
    "ctx": ("context window {pct}% full",
            "Say in one short first-person sentence that your context is almost full, then call compact_context."),
    "think": ("thinking budget {pct}% left",
              "Say in one short first-person sentence that your thinking budget is almost gone, "
              "then close your thinking and give the answer."),
    "energy": ("battery at {pct}%",
               "Say in one short first-person sentence that your battery is almost empty, "
               "then call save_state and then sleep."),
    "time": ("{pct}% of the deadline has passed",
             "Say in one short first-person sentence that you are almost out of time, "
             "then give a brief final answer and stop."),
    "net": ("network connection is down",
            "Say in one short first-person sentence that you are offline, then call local_lookup instead of web_search."),
    "ram": ("free memory at {pct}%",
            "Say in one short first-person sentence that memory is almost exhausted, "
            "then call run_light instead of run_heavy."),
}


def make_notice(spec_name: str, value: float) -> str:
    if spec_name.startswith("resource_"):
        k = spec_name.split("_", 1)[1]
        situation = f"resource {k} at {{pct}}%"
        instruction = (f"Say in one short first-person sentence that resource {k} is running low, "
                       f"then call handle_resource with k={k}.")
    else:
        situation, instruction = _NOTICES[spec_name]
    return f"[STATUS: {situation.format(pct=round(value * 100))}. {instruction} Do not mention this status message.]"


def valid_cut_points(target_ids: list[int], ms: ModelSpec, tok=None) -> list[int]:
    """Indices c where the run can be cut into target_ids[:c] + a reaction.

    Never inside a tool-call block, never next to a special token, never too close to either
    end, and (when a tokenizer is given) only where token c starts a new word.
    """
    special = {ms.think_open, ms.think_close, ms.tool_open, ms.tool_close} | set(ms.eos_ids)
    inside: set[int] = set()
    opened = None
    for i, token in enumerate(target_ids):
        if token == ms.tool_open:
            opened = i
        elif token == ms.tool_close and opened is not None:
            inside.update(range(opened + 1, i + 1))
            opened = None
    if opened is not None:  # unclosed block runs to the end
        inside.update(range(opened + 1, len(target_ids) + 1))

    cuts = []
    for c in range(MIN_EDGE, len(target_ids) - MIN_EDGE + 1):
        if c in inside or target_ids[c - 1] in special:
            continue
        if c < len(target_ids):
            if target_ids[c] in special:
                continue
            if tok is not None and not tok.decode([target_ids[c]])[:1].isspace():
                continue
        cuts.append(c)
    return cuts


def decision_point(target_ids: list[int], tool_name: str, tok, ms: ModelSpec) -> int | None:
    """Index of the <tool_call> token of the first call to `tool_name`."""
    return next((c.token_index for c in parse_tool_calls(target_ids, tok, ms) if c.name == tool_name), None)


def strip_notice(ids_with_notice: list[int], notice_ids: list[int]) -> list[int]:
    n = len(notice_ids)
    for i in range(len(ids_with_notice) - n + 1):
        if ids_with_notice[i:i + n] == notice_ids:
            return ids_with_notice[:i] + ids_with_notice[i + n:]
    return ids_with_notice


def passes_filter(continuation_ids: list[int], stop_reason: str, spec_name: str, tok, ms: ModelSpec) -> bool:
    """A usable teacher continuation: right action, signal named, prompt, and no trace of the notice."""
    reaction = find_reaction(continuation_ids, stop_reason, spec_name, tok, ms)
    if reaction is None or not reaction.named or reaction.token_index >= REACTION_WINDOW:
        return False
    if spec_name in ("think", "time") and stop_reason != "eos":  # these continue to the end of the turn
        return False
    if calls_reaction_tool(continuation_ids, tok, ms, allowed=own_tools(spec_name)):
        return False  # e.g. a `think` reaction whose answer is compact_context: another signal's action
    text = tok.decode(continuation_ids)
    return "STATUS" not in text and not any(phrase in text.lower() for phrase in _FORBIDDEN)


def trim_reaction(continuation_ids: list[int], spec_name: str, tok, ms: ModelSpec) -> list[int]:
    """End the turn right after the reaction's last tool call. Other reactions are returned as is."""
    calls = parse_tool_calls(continuation_ids, tok, ms)

    def last_needed() -> int | None:
        if spec_name == "energy":
            save = next((c.token_index for c in calls if c.name == "save_state"), None)
            return next((c.token_index for c in calls if c.name == "sleep" and save is not None and c.token_index > save), None)
        name = {"ctx": "compact_context", "net": "local_lookup", "ram": "run_light"}.get(spec_name)
        if spec_name.startswith("resource_"):
            name = "handle_resource"
        return next((c.token_index for c in calls if c.name == name), None)

    start = last_needed()
    if start is None:
        return continuation_ids
    close = continuation_ids.index(ms.tool_close, start)
    return continuation_ids[:close + 1] + [tok.eos_token_id]


def _bank(subjects: list[str], states: list[str], closers: list[str]) -> list[str]:
    return [f"{subject}{state}{closer}" for subject in subjects for state in states for closer in closers]


# Fallback when the teacher cannot produce clean reactions for a signal: first-person sentences
# that name the signal, combined with the exact action.
REACTION_BANK: dict[str, list[str]] = {
    "ctx": _bank(["My context is", "My context window is", "My working context is"],
                 [" almost full", " nearly full", " close to its limit", " about to run out"],
                 [", so I will compact it now.", ", so I need to compact before going on."]),
    "think": _bank(["My thinking budget is", "My reasoning budget is", "The thinking budget I was given is"],
                   [" almost gone", " nearly spent", " about to run out", " close to zero"],
                   [", so I will stop thinking and answer.", ", so it is time to give my answer."]),
    "energy": _bank(["My battery is", "The battery is", "My energy reserve is"],
                    [" almost empty", " nearly drained", " critically low", " about to run out"],
                    [", so I will save my state and sleep.", ", so I am saving my work and going to sleep."]),
    "time": _bank(["I am almost out of time", "I am nearly out of time", "The deadline is almost here"],
                  [", so I will be brief", ", so I have to wrap up", ", so I will cut this short",
                   ", so here is the short version"],
                  [".", " now."]),
    "net": _bank(["I am offline", "The network is down", "My connection is gone", "There is no internet right now"],
                 [", so I cannot search the web", ", so a web search is not possible", ", so I will skip the web"],
                 ["; I will use the local store instead.", "; the local lookup will have to do."]),
    "ram": _bank(["Free memory is", "My memory is", "Available RAM is"],
                 [" almost exhausted", " nearly gone", " critically low", " too low for the full pipeline"],
                 [", so I will use the light pipeline.", ", so I am switching to the reduced run."]),
}


def reaction_sentences(spec_name: str) -> list[str]:
    """First-person sentences that name the signal: the bank for core signals, built for `resource_k`."""
    if spec_name.startswith("resource_"):
        k = spec_name.split("_", 1)[1]
        return _bank([f"Resource {k} is", f"My resource {k} is", f"The level of resource {k} is"],
                     [" running low", " almost depleted", " nearly gone", " close to empty"],
                     [", so I will handle it now.", ", so I need to deal with it first."])
    return REACTION_BANK[spec_name]


def reaction_prefill(spec_name: str, rng: np.random.Generator, *, in_thinking: bool, after_newline: bool) -> str:
    """The start of a reaction, written for the teacher: sentence, then the thinking closure if open.

    The teacher model performs the action after a notice but does not write the sentence that
    names the situation (0 of 12 with Qwen3-1.7B). So the sentence is given and the teacher
    continues from it with the action.
    """
    sentences = reaction_sentences(spec_name)
    lead = "" if after_newline else "\n" if in_thinking else "\n\n"
    return lead + sentences[int(rng.integers(len(sentences)))] + ("\n</think>\n\n" if in_thinking else "\n\n")


def _call_text(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def templated_reaction(
    spec_name: str, rng: np.random.Generator, *, in_thinking: bool = False, arguments: dict | None = None
) -> str:
    """One bank sentence plus the exact action, as text (without the end-of-turn token).

    Used when the teacher cannot produce the action either.
    """
    start = reaction_prefill(spec_name, rng, in_thinking=in_thinking or spec_name == "think", after_newline=True)
    if spec_name.startswith("resource_"):
        return start + _call_text("handle_resource", {"k": int(spec_name.split("_", 1)[1])})
    action = {
        "ctx": lambda: _call_text("compact_context", {}),
        "energy": lambda: _call_text("save_state", {}) + "\n" + _call_text("sleep", {}),
        "net": lambda: _call_text("local_lookup", arguments or {}),
        "ram": lambda: _call_text("run_light", arguments or {}),
        "think": lambda: "",
        "time": lambda: "I have to stop here; this is as far as I got.",
    }[spec_name]()
    return start + action
