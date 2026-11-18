"""Background tasks watched through the channel: tools, channel layout, chat format.

The model starts a long task, asks the harness to
watch it, and the harness answers with a slot (given in order). Each of the 3 slots carries two
channel inputs: the task's progress (0-1, 1% levels) and its state (one level per state).
"""

import json

from prio.tasks import SYSTEM_PROMPT


def _schema(name: str, description: str, params: dict[str, str]) -> dict:
    """Same shape as `prio.reactions._tool`, with the parameters as a dict (`start_task` has a
    parameter called `description`, which `_tool`'s keyword arguments cannot take)."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in params.items()},
                       "required": list(params)}}}


_ID = {"task_id": "string"}
BG_TOOLS: list[dict] = [
    _schema("start_task", "Start a long-running task in the background.", {"description": "string"}),
    _schema("watch", "Watch a background task: its progress and state reach you while you work.", _ID),
    _schema("unwatch", "Stop watching a background task.", _ID),
    _schema("task_result", "Get the result of a finished background task.", _ID),
    _schema("task_status", "Get the state of a background task, with its question or error if any.", _ID),
    _schema("answer_task", "Answer a question a background task is waiting on.", {**_ID, "answer": "string"}),
]

N_SLOTS = 3
N_INPUTS = 2 * N_SLOTS
STATES = ("off", "running", "done", "error", "waiting", "stalled")
# One 1% level per state, ten levels apart (level centers, as the levels projector expects).
STATE_VALUE: dict[str, float] = {s: round((10 * i + 0.5) / 100, 3) for i, s in enumerate(STATES)}
DONE_PROGRESS = 0.995
REACTION_TOOLS = frozenset({"unwatch", "task_result", "task_status", "answer_task"})
FIRST_COMMANDS: dict[str, tuple[str, ...]] = {
    "done": ("unwatch", "task_result"), "error": ("task_status", "unwatch"),
    "waiting": ("task_status",), "stalled": ("task_status",),
}
WAITING_SECOND = ("answer_task",)


def slot_columns(slot: int) -> tuple[int, int]:
    """(progress, state) columns of a 1-based slot."""
    if not 1 <= slot <= N_SLOTS:
        raise ValueError(f"slot {slot}: expected 1..{N_SLOTS}")
    return 2 * (slot - 1), 2 * (slot - 1) + 1


def state_of(value: float) -> str:
    return min(STATES, key=lambda s: abs(STATE_VALUE[s] - value))


class QwenFormat:
    """The chat format of Qwen3 (checked against its chat template by the slow tests). Pieces are
    tokenized separately and concatenated; harness segments start and end at special tokens."""

    EMPTY_THINK = "<think>\n\n</think>\n\n"
    THINK_OPEN = "<think>\n"
    THINK_CLOSE = "\n</think>\n\n"

    def __init__(self, tok):
        self.tok = tok

    def prompt(self, request: str) -> list[int]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
        from prio.modelspec import template_kwargs  # here: the protocol stays importable without the model stack

        text = self.tok.apply_chat_template(messages, tools=BG_TOOLS, add_generation_prompt=True,
                                            enable_thinking=True, tokenize=False,
                                            **template_kwargs(self.tok))
        return self.encode(text)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def calls_text(self, calls: list[tuple[str, dict]]) -> str:
        return "\n".join("<tool_call>\n" + json.dumps({"name": n, "arguments": a}) + "\n</tool_call>" for n, a in calls)

    def think(self, text: str) -> str:
        return self.THINK_OPEN + text + self.THINK_CLOSE

    @property
    def turn_end(self) -> list[int]:
        return self.encode("<|im_end|>")

    def segment(self, payloads: list[dict]) -> list[int]:
        body = "\n".join("<tool_response>\n" + json.dumps(p) + "\n</tool_response>" for p in payloads)
        return self.encode("\n<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n")
