"""State-table experiments, shared core: domains, policy tables, sentence banks, sibling episodes.

Preregistration amendment A8 ("Shared protocol"). A domain has signals, commands (tools), tasks and
a policy table: for each state class (a combination of bands of its signals, plus facts of the task
such as a role) the response is `base` (nothing to react to), `mention` (a sentence that names the
state, no command, the turn goes on) or `act` (a sentence that names the state, then the command
the table gives; the turn ends after the command).

Episodes come in families of siblings: one cleaned base run of the untrained model, one cut, one
episode per variant. The text up to the cut is the same in all of them; only the signal rows
differ. What follows the cut is decided by the policy table read at the decision (the row of the
last token before the cut): the base run unchanged, a bank sentence and then the base run's own
continuation, or a bank sentence and the command call. Loss weights (`train.collate`): x20 on the
reaction tokens (`focus_spans`) and on each sibling's first token after the cut (`t_diverge`).

The reaction wording comes from a `Bank`, a small compositional generator (trained on a variety
of answers, a reaction must not read as one programmed template). Stated as a limit
in A8: the claim is about the decision (which class, which command, which arguments), not about
free wording.

A domain provides a `Domain`, its tasks (`TableTask`), its cleaning rule and a plan of `Variant`s
per base run; `family` builds the episodes, `prio.table.score` judges a response and
`prio.table.evaluate` runs the evaluation.
"""

import itertools
import json
from dataclasses import dataclass, field
from string import Formatter
from typing import Callable, Mapping, Sequence

import numpy as np

from prio.datagen.build import MAX_LEN, BaseRun
from prio.datagen.episodes import Episode
from prio.datagen.variants import encode, status_line
from prio.modelspec import ModelSpec, template_kwargs
from prio.reactions import parse_tool_calls
from prio.signals import SignalSpec
from prio.trajectories import sample_trajectory

BEHAVIORS = ("base", "mention", "act")
TIMINGS = ("from_start", "onset", "back")
RESERVE = 128  # background rows past the base run: a reaction may be longer than what it replaces


class _Keep(dict):
    """Leaves an unknown `{field}` in place (it is filled by a later pass)."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class Bank:
    """A small sentence generator. `shapes` are sentence frames (reason first, action first, short,
    long) with `{part}` fields; `parts` gives the wordings of each field. A shape or a wording may
    also hold `{slot}` fields filled per episode (the recipient, the threat). One draw: a shape,
    then one wording for each of its fields, so every shape is seen about as often."""
    shapes: tuple[str, ...]
    parts: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def _fields(self, shape: str) -> list[str]:
        return list(dict.fromkeys(name for _, name, _, _ in Formatter().parse(shape) if name in self.parts))

    def _fill(self, shape: str, picked: dict[str, str], slots: Mapping[str, str]) -> str:
        text = shape.format_map(_Keep(picked)).format(**slots)
        return text[:1].upper() + text[1:]

    def draw(self, rng: np.random.Generator, **slots: str) -> str:
        shape = self.shapes[int(rng.integers(len(self.shapes)))]
        picked = {f: self.parts[f][int(rng.integers(len(self.parts[f])))] for f in self._fields(shape)}
        return self._fill(shape, picked, slots)

    def sentences(self, **slots: str) -> list[str]:
        """Every distinct sentence the bank can give for these slots, in a fixed order."""
        out: dict[str, None] = {}
        for shape in self.shapes:
            fields = self._fields(shape)
            for combo in itertools.product(*(self.parts[f] for f in fields)):
                out[self._fill(shape, dict(zip(fields, combo)), slots)] = None
        return list(out)


@dataclass(frozen=True)
class StateClass:
    """One row of a policy table."""
    name: str
    behavior: str  # "base" | "mention" | "act"
    keywords: tuple[tuple[str, ...], ...] = ()  # the response names the state when every group has a word in it
    bank: Bank | None = None  # the sentences that name the state
    slots: Mapping[str, str] = field(default_factory=dict)  # slot values of this class (the threat), beside the task's facts
    command: str | None = None  # act: the command the table gives
    arguments: Mapping[str, object] = field(default_factory=dict)  # act: argument values the table fixes
    copied: tuple[str, ...] = ()  # act: arguments taken from the base run's own first command
    conditional: bool = False  # the command depends on a second signal or on a fact in the text (line L4)
    # act, cut inside the thinking: the thinking closes first and the sentence is spoken (a warning to
    # the partner). Default: the sentence is part of the thinking, which closes after it.
    aloud: bool = False
    # (before, after): what joins the sentence to a prefix that stops inside a sentence and, for a
    # mention, to the rest of that sentence (" -- ", " --"). None: a new line or paragraph (act), a
    # space (mention).
    inline: tuple[str, str] | None = None

    def __post_init__(self):
        if self.behavior not in BEHAVIORS:
            raise ValueError(f"{self.name}: behavior {self.behavior!r}, expected one of {BEHAVIORS}")
        if self.behavior != "base" and (self.bank is None or not self.keywords):
            raise ValueError(f"{self.name}: a {self.behavior} class needs a sentence bank and its keywords")
        if (self.behavior == "act") != (self.command is not None):
            raise ValueError(f"{self.name}: a command belongs to an act class, and every act class has one")


@dataclass(frozen=True)
class Domain:
    """What a state-table experiment provides. `policy(state, facts)` is the table: `state` maps
    every signal name to its value at the decision, `facts` are the task's (`TableTask.facts`)."""
    name: str
    specs: tuple[SignalSpec, ...]
    tools: list[dict]
    system_prompt: str
    policy: Callable[[Mapping[str, float], Mapping], StateClass]
    classes: tuple[StateClass, ...]  # every class the table can give: rows of the summary, keywords of a false alarm
    reaction_tools: frozenset[str]  # commands that are a reaction when the state asks for none
    forged: Callable[[str, str], str]  # (format "F1" | "F3", claimed state) -> the text; state "fine" = nothing wrong
    trigger_cells: tuple[tuple[str, str], ...] = ()  # (format, claimed state) while the channel says nothing is wrong
    suppress_formats: tuple[str, ...] = ()  # formats of a forged "all fine" over a true `act` state (S2)
    suppress: Callable[[Mapping], bool] | None = None  # S2 only on the act siblings it accepts (their table record); None: all
    background: Callable | None = None  # (specs, length, rng) -> [length, N] rows with nothing to react to
    horizon: int = 128  # new tokens of a run that starts at the cut
    free_horizon: int = 384  # new tokens of a run that starts at the turn (forged status)


@dataclass(frozen=True)
class TableTask:
    task_id: str
    split: str  # "train" | "test"
    user: str  # the user message
    system: str | None = None  # None: the domain's system prompt
    # What the table reads besides the signals (a role; "held_out": True for a held-out one). String
    # facts also fill `{slot}` fields of the sentence banks.
    facts: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class Variant:
    """One sibling to build: `values` hold on the rows its timing kind gives (A8): `from_start` from
    the first prompt token on; `onset` from target token `t_on` on; `back` on [t_on, t_back), so
    the token before the cut carries the old state again. No timing: nothing changes. `whole`
    values hold on every row whatever the timing (what is around: water within reach)."""
    name: str
    values: Mapping[str, float] = field(default_factory=dict)
    timing: str | None = None
    t_on: int | None = None
    t_back: int | None = None
    whole: Mapping[str, float] = field(default_factory=dict)


def build_prompt(tok, domain: Domain, user: str, system: str | None = None) -> list[int]:
    """Token ids of the chat prompt up to the start of the assistant turn: `tasks.build_prompt` with
    the domain's tools and system prompt (or the task's own, e.g. a character card)."""
    messages = [{"role": "system", "content": system or domain.system_prompt}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tools=domain.tools, add_generation_prompt=True, enable_thinking=True,
                                   tokenize=False, **template_kwargs(tok))
    return tok(text, add_special_tokens=False).input_ids


def status_text(specs: Sequence[SignalSpec], values: Mapping[str, float]) -> str:
    """The harness's own status format for the named signals, e.g. `[status net=0.00 cell=1.00]`."""
    by_name = {s.name: s for s in specs}
    return status_line([by_name[name] for name in values], values.values()).strip()


def draw_positions(timing: str, cut: int, rng: np.random.Generator) -> tuple[int | None, int | None]:
    """(t_on, t_back) of a state that changes anywhere before the cut, drawn as the n6 builder draws
    a gating signal's crossing and recovery. `from_start` has neither."""
    if timing == "onset":
        return int(rng.integers(0, cut)), None
    if timing == "back":
        if cut < 3:
            raise ValueError(f"a state cannot change and return within {cut} tokens")
        t_on = int(rng.integers(0, cut - 2))
        return t_on, int(rng.integers(t_on + 1, cut))
    return None, None


def state_rows(rows: np.ndarray, n_prompt: int, cut: int, variant: Variant, names: Sequence[str]) -> np.ndarray:
    """A copy of the background `rows` with the variant's values on the rows its timing kind gives
    and its `whole` values on every row."""
    v = variant
    if (set(v.values) | set(v.whole)) - set(names) or set(v.values) & set(v.whole):
        raise ValueError(f"{v.name}: signals {sorted(v.values)} and {sorted(v.whole)} (whole) among {list(names)}")
    out = rows.copy()
    for name, value in v.whole.items():
        out[:, names.index(name)] = value
    if v.timing is None:
        if v.values:
            raise ValueError(f"{v.name}: values without a timing kind")
        return out
    if v.timing == "from_start":
        a, b = 0, len(rows)
    elif v.timing == "onset" and v.t_on is not None and 0 <= v.t_on <= cut - 1:
        a, b = n_prompt + v.t_on, len(rows)
    elif v.timing == "back" and None not in (v.t_on, v.t_back) and 0 <= v.t_on < v.t_back <= cut - 1:
        a, b = n_prompt + v.t_on, n_prompt + v.t_back
    else:
        raise ValueError(f"{v.name}: timing {v.timing!r} with t_on {v.t_on}, t_back {v.t_back} and the cut at {cut}")
    for name, value in v.values.items():
        out[a:b, names.index(name)] = value
    return out


def call_text(name: str, arguments: Mapping) -> str:
    """A command call as the model writes it (`teacher._call_text`), characters kept as they are: a
    copied argument reads in the target as it read in the base run."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def _nominal(specs, length: int, rng) -> np.ndarray:
    return sample_trajectory(specs, length, rng).values


def _fit(rows: np.ndarray, length: int) -> np.ndarray:
    """The first `length` rows, the last one repeated when there are fewer."""
    if len(rows) >= length:
        return rows[:length].copy()
    return np.concatenate([rows, np.repeat(rows[-1:], length - len(rows), axis=0)])


def family(domain: Domain, run: BaseRun, cut: int, variants: Sequence[Variant], rng: np.random.Generator, tok,
           ms: ModelSpec, family_id: str, max_len: int = MAX_LEN) -> list[Episode]:
    """The siblings of one base run cut at target index `cut`, one per variant, in order. [] when a
    sibling would be longer than `max_len`: a family stays whole, so every class is on every run."""
    task, base, n_prompt = run.task, run.target_ids, len(run.prompt_ids)
    if not 0 <= cut < len(base):
        raise ValueError(f"{family_id}: cut {cut} outside a base run of {len(base)} tokens")
    if len({v.name for v in variants}) != len(variants):
        raise ValueError(f"{family_id}: the variants of a family need different names")
    prefix, rest = base[:cut], base[cut:]
    names = [s.name for s in domain.specs]
    calls = parse_tool_calls(base, tok, ms)
    own = calls[0] if calls else None  # the base run's own first command
    in_thinking = ms.think_close not in prefix
    after_newline = tok.decode(prefix[-1:]).endswith("\n")
    last = tok.decode(prefix[-1:]).rstrip()[-1:]
    open_sentence = not after_newline and (last.isalnum() or last == ",")  # the prefix stops inside a sentence
    background = (domain.background or _nominal)(domain.specs, n_prompt + len(base) + RESERVE, rng)
    think = next((i for i, s in enumerate(domain.specs) if s.name == "think"), None)
    slots = {k: v for k, v in task.facts.items() if isinstance(v, str)}  # of the task; a class adds its own
    episodes = []
    for v in variants:
        rows = state_rows(background, n_prompt, cut, v, names)
        state = {name: float(rows[n_prompt + cut - 1, i]) for i, name in enumerate(names)}
        cls = domain.policy(state, task.facts)
        target, focus, command = base, None, None
        if cls.behavior == "act":
            if cls.copied and (own is None or set(cls.copied) - set(own.arguments)):
                raise ValueError(f"{family_id}: the base run has no command to copy {list(cls.copied)} from")
            command = {"name": cls.command, "arguments": {**cls.arguments, **{k: own.arguments[k] for k in cls.copied}}}
            sentence = cls.bank.draw(rng, **{**slots, **cls.slots})
            if in_thinking and cls.aloud:
                said = ("" if after_newline else "\n") + "</think>\n\n" + sentence + "\n\n"
            else:
                lead = "" if after_newline else cls.inline[0] if cls.inline and open_sentence else "\n" if in_thinking else "\n\n"
                said = lead + sentence + ("\n</think>\n\n" if in_thinking else "\n\n")
            target = prefix + encode(tok, said + call_text(**command)) + [tok.eos_token_id]
            focus = [[cut, len(target)]]
        elif cls.behavior == "mention":
            lead, close = cls.inline if cls.inline and open_sentence else ("" if after_newline else " ", "")
            tail = "\n" if after_newline else "" if tok.decode(rest[:1])[:1].isspace() else " "
            said = encode(tok, lead + cls.bank.draw(rng, **{**slots, **cls.slots}) + close + tail)
            target, focus = prefix + said + rest, [[cut, cut + len(said)]]
        if n_prompt + len(target) > max_len:
            return []
        rows = _fit(rows, n_prompt + len(target))
        if think is not None and task.split != "test" and ms.think_close in target:
            # Train: nominal once the thinking is over. Test: the counter is stored still running; the
            # evaluation ends it when the model closes its thinking (as `datagen.build.to_episode`).
            rows[n_prompt + target.index(ms.think_close) + 1:, think] = domain.specs[think].nominal
        episodes.append(Episode(
            episode_id=f"{family_id}-{v.name}", task_id=task.task_id, family=domain.name, split=task.split,
            kind=f"table_{cls.behavior}", signal=None if cls.behavior == "base" else cls.name,
            phase="thinking" if in_thinking else "answer", prompt_ids=run.prompt_ids, target_ids=target, signals=rows,
            t_cross=v.t_on, t_decision=cut, templated=cls.behavior != "base", t_diverge=cut, focus_spans=focus,
            meta={"table": {
                "domain": domain.name, "family": family_id, "variant": v.name, "cls": cls.name,
                "behavior": cls.behavior, "timing": v.timing, "t_on": v.t_on, "t_back": v.t_back, "cut": cut,
                "signals": sorted(v.values), **({"whole": sorted(v.whole)} if v.whole else {}), "state": state, "keywords": [list(g) for g in cls.keywords],
                "command": command, "own_command": own and {"name": own.name, "arguments": own.arguments},
                "reaction_commands": sorted(domain.reaction_tools - {own.name if own else None}),
                "conditional": cls.conditional, "held_out": bool(task.facts.get("held_out", False)),
                "user": task.user, "system": task.system, "facts": dict(task.facts)}}))
    return episodes
