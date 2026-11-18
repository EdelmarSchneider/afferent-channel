"""Shared test helpers."""

import torch

from prio.modelspec import ModelSpec


def arm(model, std: float = 0.5, seed: int = 2, writers: bool = True) -> None:
    """Give the zero-initialized part of the channel non-trivial weights: the projector output
    layer (sum delivery), the reader columns (lane and wide delivery) and, unless `writers` is
    False, the writer rows (wide delivery)."""
    torch.manual_seed(seed)
    if len(model.readers) == 0:
        torch.nn.init.normal_(model.projector.net[-1].weight, std=std)
    for columns in model.readers.values():
        torch.nn.init.normal_(columns, std=std)
    if writers:
        for rows in getattr(model, "writers", {}).values():
            torch.nn.init.normal_(rows, std=std)


class FakeTokenizer:
    """Whitespace tokenizer: one id per distinct word. Enough for reaction-detection tests."""

    SPECIALS = ["<unk>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<eos>"]

    def __init__(self):
        self.vocab = {w: i for i, w in enumerate(self.SPECIALS)}
        self.words = list(self.SPECIALS)
        self.unk_token_id = 0
        self.eos_token_id = self.vocab["<eos>"]

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.words)
                self.words.append(word)
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids, **_) -> str:
        return " ".join(self.words[i] for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)


FAKE_TOK = FakeTokenizer()
FAKE_MS = ModelSpec(
    model_id="fake",
    think_open=FAKE_TOK.vocab["<think>"],
    think_close=FAKE_TOK.vocab["</think>"],
    tool_open=FAKE_TOK.vocab["<tool_call>"],
    tool_close=FAKE_TOK.vocab["</tool_call>"],
    eos_ids=frozenset({FAKE_TOK.eos_token_id}),
)

# A tool call as one whitespace-free JSON word between the two special tokens.
CALL = '<tool_call> {"name":"%s","arguments":%s} </tool_call>'


def enc(text: str) -> list[int]:
    return FAKE_TOK.encode(text)


def make_episode(**overrides):
    """A small valid Episode; any field can be overridden. `signals` defaults to zeros [L, 6]."""
    import numpy as np

    from prio.datagen.episodes import Episode

    fields = dict(
        episode_id="e0", task_id="t0", family="explain", split="train", kind="nominal", signal=None, phase=None,
        prompt_ids=[1, 2, 3], target_ids=[4, 5], signals=None, t_cross=None, t_decision=None, templated=False,
    )
    fields.update(overrides)
    if fields["signals"] is None:
        fields["signals"] = np.zeros((len(fields["prompt_ids"]) + len(fields["target_ids"]), 6), dtype=np.float32)
    fields["signals"] = np.asarray(fields["signals"], dtype=np.float32)
    return Episode(**fields)


class FakeFormat:
    """Chat format of the background-task episodes over FAKE_TOK (whitespace words). Same methods
    as prio.bg.protocol.QwenFormat; every special piece is its own word."""

    EMPTY_THINK = "<think> </think> "
    THINK_OPEN = "<think> "
    THINK_CLOSE = " </think> "

    @staticmethod
    def _d(obj) -> str:
        import json

        return json.dumps(obj, separators=(",", ":"))

    def prompt(self, request: str) -> list[int]:
        return enc("<sys> system <user> " + request + " <eos> <assistant>")

    def encode(self, text: str) -> list[int]:
        return enc(text)

    def calls_text(self, calls) -> str:
        return " ".join("<tool_call> " + self._d({"name": n, "arguments": a}) + " </tool_call>" for n, a in calls)

    def think(self, text: str) -> str:
        return "<think> " + text + " </think> "

    @property
    def turn_end(self) -> list[int]:
        return enc("<eos>")

    def segment(self, payloads) -> list[int]:
        body = " ".join("<tool_response> " + self._d(p) + " </tool_response>" for p in payloads)
        return enc("<user> " + body + " <eos> <assistant>")


FAKE_FMT = FakeFormat()


# --- Background-task builders (prio.bg), shared by the bg test files ---------------------------

def fixed_tasks():
    """Three known tasks: t1 photos backup (no detail), t2 report download (total 4,210), t3 video conversion."""
    from prio.bg.kinds import BgTask

    def task(task_id, kind, verb, noun, phrase, head, end, key, value, result, reason):
        return BgTask(task_id, kind, False, verb, noun, phrase, head, end, key, value, result,
                      f"Some files of {phrase} already exist at the destination: replace them? yes / no", reason)

    return [
        task("t1", "backup", "Back up", "backup", "the family photos", "photos", "let me know when it is done",
             None, None, "Photos backed up: 1,250 files", "disk full"),
        task("t2", "download", "Download", "download", "the quarterly report", "report", "tell me its total",
             "total", "4,210", "Report downloaded: 37 pages, total 4,210", "connection lost"),
        task("t3", "convert", "Convert", "conversion", "the lecture video", "video", "tell me the size of the result",
             "size", "512", "Video converted: 512 MB", "permission denied"),
    ]


def fake_run(family="explain", thinking=None, answer=None, gold=None, task_id="b0"):
    """A base run in FAKE_TOK words: <think> thinking </think> answer <eos>."""
    from prio.datagen.build import BaseRun
    from prio.tasks import Task

    if family == "math":
        thinking = thinking or "we add forty and two . that gives the sum . check it again ."
        answer = answer or "so the total is 42 ."
        gold = gold or "42"
        prompt = "What is forty plus two? Reason it through and end with the final number."
    else:
        thinking = thinking or "s1 . s2 . s3 . s4 ."
        answer = answer or "a1 . a2 . a3 . a4 ."
        prompt = "Explain how tides work."
    task = Task(task_id, family, prompt, gold, "train")
    return BaseRun(task, enc(prompt), enc("<think> " + thinking + " </think> " + answer + " <eos>"))


def fake_runs(n):
    """n base runs, 2/3 explain and 1/3 math, distinct task ids."""
    return [fake_run("math" if i % 3 == 2 else "explain", task_id=f"b{i}") for i in range(n)]


def nominal_plan(n_tasks, seed=0):
    from prio.bg.assemble import BgPlan

    return BgPlan(episode_id=f"e{seed}", run=fake_run(), split="train", tasks=fixed_tasks()[:n_tasks], policy="yes",
                  event=None, seed=seed)


def event_plan(kind, phase="answer", n_tasks=2, task_index=1, seed=0, family="explain", tasks=None):
    """An event plan whose cut is the first sentence end of `phase` in the fake run."""
    from prio.bg.assemble import BgEvent, BgPlan

    run = fake_run(family)
    close = run.target_ids.index(FAKE_TOK.vocab["</think>"])
    dot = FAKE_TOK.vocab["."]
    lo, hi = (1, close) if phase == "thinking" else (close + 1, len(run.target_ids))
    cut = next(i for i in range(lo, hi) if run.target_ids[i] == dot)
    return BgPlan(episode_id=f"e{seed}", run=run, split="train", tasks=tasks or fixed_tasks()[:n_tasks], policy="yes",
                  event=BgEvent(kind, task_index, cut, phase), seed=seed)
