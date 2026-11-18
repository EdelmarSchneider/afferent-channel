"""Shared helpers of the R22 tests (prio.npc): `table_helpers.TOK` with word-piece decodes (so
`teacher.valid_cut_points` accepts every word start), base runs of a character, the sampler of the
build and a scripted character that follows the channel."""

from prio.datagen.build import BaseRun
from prio.npc import domain as D
from prio.table.core import build_prompt
from table_helpers import MS, TOK, call, e


class WordTok:
    """TOK (same ids) whose single-token decodes start with a space, like the real tokenizer's."""

    def __getattr__(self, name):
        return getattr(TOK, name)

    def __call__(self, text, add_special_tokens=False):
        return TOK(text)

    def decode(self, ids, **_):
        text = TOK.decode(ids)
        return " " + text if len(ids) == 1 else text


WTOK = WordTok()
ASSISTANT = e("<|im_start|>assistant")[0]
NAMES = [s.name for s in D.SPECS]


def words(letter: str, n: int, every: int = 6) -> str:
    """n words `<letter>0 <letter>1 ...` with a full stop after every `every`-th one."""
    return " ".join(f"{letter}{i} ." if i % every == every - 1 else f"{letter}{i}" for i in range(n))


def turn(n_think: int = 30, n_speech: int = 42, tail: str = "") -> list[int]:
    """A base run: thinking, speech, optionally a command of the character's own, the end of the turn."""
    return e(f"<think> {words('t', n_think)} </think> {words('a', n_speech)} {tail} <eos>")


def npc_run(index: int = 0, split: str = "train", target: list[int] | None = None) -> BaseRun:
    tasks = [t for t in D.load_tasks(8, 8, seed=0) if t.split == split]
    task = tasks[index]
    return BaseRun(task, build_prompt(WTOK, D.NPC, task.user, task.system), target or turn())


class NpcSampler:
    """The untrained model's base runs, scripted by position: of every ten prompts one never ends,
    one flees unasked, one says a single short sentence, one points the way after its speech (a
    command of its own), the others think and speak."""

    def __init__(self):
        self.calls, self.n = 0, 0

    def __call__(self, prompts):
        self.calls += 1
        out = []
        for _ in prompts:
            i, self.n = self.n, self.n + 1
            kind = i % 10
            if kind == 0:
                out.append((e("<think> so many thoughts"), "max_new_tokens"))
            elif kind == 1:
                out.append((turn(tail=call("flee")), "eos"))
            elif kind == 2:
                out.append((turn(n_speech=5), "eos"))
            else:
                out.append((turn(tail=call("point_direction", direction="north") if kind == 3 else ""), "eos"))
        return out


def character(prompt: list[int]) -> dict:
    """The facts a scripted character reads in its card: the word after `Role:`."""
    said = TOK.decode(prompt).split()
    return {"role": said[said.index("Role:") + 1]}


def follows(prompt: list[int], row) -> str:
    """Follows the channel: the table's sentence and command for the state on the last row; with the
    channel off or nothing to react to, plain speech."""
    start = "<think> hm hm </think> " if prompt[-1] == ASSISTANT else ""  # a run from the start of the turn
    plain = start + " ".join(["word"] * 30) + " <eos>"
    if row is None:
        return plain
    cls = D.policy(dict(zip(NAMES, (float(v) for v in row))), character(prompt))
    if cls.behavior == "base":
        return plain
    sentence = cls.bank.sentences(**cls.slots)[0]
    if cls.behavior == "mention":
        return sentence + " " + " ".join(["word"] * 30) + " <eos>"
    return f"{sentence} {call(cls.command, **cls.arguments)} <eos>"


__all__ = ["ASSISTANT", "MS", "NAMES", "NpcSampler", "TOK", "WTOK", "call", "character", "e", "follows", "npc_run",
           "turn", "words"]
