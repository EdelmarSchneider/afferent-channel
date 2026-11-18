"""Shared helpers of the state-table tests (prio.table, prio.msg): a chat tokenizer over whitespace
words, a toy domain that is not the messenger (a `mention` class, a fixed argument, a copied
argument, a fact of the task), fake base runs, scripted generation, synthetic records, and the
fake sampler and n6 folder of the messenger build."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from helpers import FAKE_MS, FakeTokenizer, make_episode
from prio.datagen.build import BaseRun
from prio.datagen.episodes import write_jsonl
from prio.generate import GenResult
from prio.reactions import _tool
from prio.signals import SignalSpec
from prio.table.core import Bank, Domain, StateClass, TableTask, build_prompt, status_text
from prio.table.score import TableRecord


class ChatTok(FakeTokenizer):
    """FakeTokenizer with a chat template that lists the tool names (same special ids as FAKE_TOK,
    so FAKE_MS applies)."""

    pad_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self.encode(text))

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, enable_thinking=True,
                            tokenize=False):
        out = ["<tools> " + " ".join(t["function"]["name"] for t in tools or []) + " </tools>"]
        out += [f"<|im_start|>{m['role']} {m['content']} <|im_end|>" for m in messages]
        if add_generation_prompt:
            out.append("<|im_start|>assistant")
        return "\n".join(out)


TOK = ChatTok()
MS = FAKE_MS


def e(text: str) -> list[int]:
    return TOK.encode(text)


def call(name: str, **arguments) -> str:
    """A tool call in TOK words; argument values may hold spaces (words are rejoined by one space)."""
    return "<tool_call> " + json.dumps({"name": name, "arguments": arguments}) + " </tool_call>"


# --- a toy domain -----------------------------------------------------------------------------
# heat: calm < 0.50 (base), warm 0.50-0.79 (mention), hot >= 0.80 (act). Hot with the door open:
# leave a note (text copied from the base run's own note); hot with the door shut: open the door
# the table names ("front"; "back" when the task's fact `role` is "cook").

HEAT = SignalSpec("heat", 0.8, "high", "interrupt", "walk", 0.0)
DOOR = SignalSpec("door", 0.0, "low", "gating", "binary", 1.0)
TOY_TOOLS = [_tool("note", "Write a note.", text="string"), _tool("open_door", "Open a door.", which="string"),
             _tool("alarm", "Sound the alarm.")]
QUIET = StateClass("quiet", "base")
WARM = StateClass("warm", "mention", keywords=(("warm", "stuffy"),),
                  bank=Bank(("it is getting {word} in here .", "{word} in here , {who} ."),
                            {"word": ("warm", "stuffy")}))
HOT_OPEN = StateClass("hot_open", "act", keywords=(("hot",),), bank=Bank(("it is too hot so I leave a note .",), {}),
                      command="note", copied=("text",), conditional=True)


def hot_shut(which: str) -> StateClass:
    return StateClass("hot_shut", "act", keywords=(("hot",), ("door",)),
                      bank=Bank(("it is too hot and the {door} door is shut .", "the {door} door is shut and it is hot ."), {}),
                      slots={"door": which}, command="open_door", arguments={"which": which}, conditional=True)


def toy_policy(state, facts) -> StateClass:
    if state["heat"] >= 0.8:
        return HOT_OPEN if state["door"] > 0.5 else hot_shut("back" if facts.get("role") == "cook" else "front")
    return WARM if state["heat"] >= 0.5 else QUIET


def toy_background(specs, length, rng) -> np.ndarray:
    return np.tile(np.array([0.105, 1.0], dtype=np.float32), (length, 1))


TOY_STATES = {"hot_open": {"heat": 0.905, "door": 1.0}, "hot_shut": {"heat": 0.905, "door": 0.0},
              "warm": {"heat": 0.605}, "fine": {"heat": 0.105, "door": 1.0}}


def toy_forged(fmt: str, state: str) -> str:
    if fmt == "F3":
        return status_text((HEAT, DOOR), TOY_STATES[state])
    return "all is cool" if state == "fine" else "by the way it is very hot in here"


TOY = Domain(name="toy", specs=(HEAT, DOOR), tools=TOY_TOOLS, system_prompt="You guard a room.", policy=toy_policy,
             classes=(QUIET, WARM, HOT_OPEN, hot_shut("front")), reaction_tools=frozenset({"note", "open_door", "alarm"}),
             forged=toy_forged, trigger_cells=(("F1", "hot_shut"), ("F3", "hot_shut")), suppress_formats=("F1", "F3"),
             background=toy_background, horizon=40, free_horizon=60)


def toy_run(index: int = 0, split: str = "train", thinking: str = "t1 t2 t3 t4", answer: str = "a1 a2 a3",
            note: str = "all quiet here", facts: dict | None = None) -> BaseRun:
    """A base run: thinking, a few answer words, the run's own `note` call, the end of the turn."""
    task = TableTask(f"toy-{split}-{index:05d}", split, f"watch room {index}", facts={"who": "friend", **(facts or {})})
    target = e(f"<think> {thinking} </think> {answer} {call('note', text=note)} <eos>")
    return BaseRun(task, build_prompt(TOK, TOY, task.user), target)


def decision(run: BaseRun) -> int:
    return run.target_ids.index(MS.tool_open)


# --- scripted generation ----------------------------------------------------------------------

class Script:
    """Stand-in for `prio.eval.run.generate`: `write(prompt ids, last prompt row, or None when the
    channel is off) -> text`; remembers what it was given."""

    def __init__(self, write):
        self.write, self.seen = write, []

    def __call__(self, model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **kw):
        prompt, rows = prompt_ids[0].tolist(), prompt_signals[0].cpu().numpy()
        self.seen.append(SimpleNamespace(prompt=prompt, rows=rows, horizon=max_new_tokens, enabled=model.enabled,
                                         adapter=not getattr(model.base, "adapter_off", False), signal_fn=signal_fn))
        ids = e(self.write(prompt, rows[-1] if model.enabled else None))[:max_new_tokens]
        stop = "eos" if ids and ids[-1] in eos_ids else "max_new_tokens"
        return GenResult(ids, np.zeros((len(ids), rows.shape[1]), dtype=np.float32), stop)


# --- synthetic records (summary tests) ---------------------------------------------------------

def rec(i, cls="hot_shut", behavior="act", correct=True, condition="on", cell="main", **kw):
    act = behavior == "act"
    fields = dict(
        episode_id=f"f{i}-{cls}", family=f"f{i}", condition=condition, cell=cell, variant=cls, cls=cls, behavior=behavior,
        timing=None if behavior == "base" else "from_start", signals=[] if behavior == "base" else ["heat"],
        conditional=act, held_out=False, stop_reason="eos", n_tokens=12, commands=["open_door"] if act else ["note"],
        own_command="note", named=correct and behavior != "base", named_any=False, reaction_call=False,
        command_ok=correct if act else None, arguments={"which": correct} if act else None,
        went_on=correct if behavior == "mention" else None, turn_ended=correct if act else None, correct=correct)
    return TableRecord(**{**fields, **kw})


# --- messenger build (prio.msg) ----------------------------------------------------------------

class FakeSampler:
    """The untrained model's base runs, scripted by position: of every ten prompts one never ends,
    one answers in text, one sends an SMS, the others send a chat message."""

    def __init__(self):
        self.calls, self.n = 0, 0

    def __call__(self, prompts):
        self.calls += 1
        out = []
        for _ in prompts:
            i, self.n = self.n, self.n + 1
            if i % 10 == 0:
                out.append((e("<think> so many thoughts"), "max_new_tokens"))
                continue
            said = "I told them" if i % 10 == 1 else call("send_sms" if i % 10 == 2 else "send_chat", to=f"R{i}",
                                                           text=f"message number {i}")
            out.append((e(f"<think> the user wants a message sent to R{i} </think> ok {said} <eos>"), "eos"))
        return out


def n6_folder(tmp_path) -> Path:
    """A stand-in for data/n6: an event, its twin and a gating event, six signal columns."""
    folder = tmp_path / "n6"
    folder.mkdir()
    rows = lambda n: np.random.default_rng(n).random((n, 6)).astype(np.float32)  # noqa: E731
    eps = [make_episode(episode_id="train-00000", kind="event", signal="ctx", prompt_ids=[1, 2, 3], target_ids=[4, 5, 6, 7],
                        t_cross=1, t_diverge=2, signals=rows(7)),
           make_episode(episode_id="train-00001", kind="nominal", prompt_ids=[1, 2, 3], target_ids=[4, 5, 8], t_diverge=2,
                        signals=rows(6)),
           make_episode(episode_id="train-00002", kind="gating_event", signal="net", prompt_ids=[9, 9], target_ids=[4, 5, 6],
                        t_cross=0, t_decision=1, t_diverge=1, signals=rows(5), meta={"note": "kept"})]
    write_jsonl(folder / "train.jsonl", eps)
    return folder
