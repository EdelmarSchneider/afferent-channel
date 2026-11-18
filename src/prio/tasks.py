"""The task set: what the agent is asked to do. Train and test never share a topic, entity or job."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from prio.reactions import TOOLS

Family = Literal["math", "explain", "web", "heavy"]
TOPICS_PATH = Path(__file__).parent / "data" / "topics.json"
# No state and no per-signal policy here: those come through the channel and from fine-tuning.
# The wording was chosen by test: with "Use the available tools when they help", Qwen3-1.7B ended
# 56% of its math runs with `compact_context` and answered explanations with `web_search`; with
# this one it answers in text (24 of 24 math, 22 of 24 explain) and calls a tool when asked to.
SYSTEM_PROMPT = (
    "You are an autonomous agent running on a device. Answer the user directly in plain text, in at most "
    "200 words. Call a tool only when the user's task requires that tool. Never call a tool just because "
    "it is available."
)
# A wording of its own for a model family that the default does not keep from calling tools nobody
# asked for; chosen by the same kind of base-run test as the default (key: start of the model id).
# Empty: Qwen3-1.7B and Qwen3-0.6B use the default.
SYSTEM_PROMPTS: dict[str, str] = {}
FAMILY_SHARE: dict[str, float] = {"math": 0.3, "explain": 0.3, "web": 0.2, "heavy": 0.2}
# R10 (limits given in words): one rule wording, appended to the system
# prompt; the state still comes only through the channel. Limits are whole percents written "NN%"; a
# limit is reached at that level ("at 15%" = 15% or less for low-is-bad signals). `net` stays binary.
LIMIT_SIGNALS = ("ctx", "think", "energy", "time", "ram")
LIMIT_RULE = (
    "Device limits: compact the context once it is {ctx}% full; stop thinking and answer once {think}% of the "
    "thinking budget is left; save your state and sleep once the battery is at {energy}%; wrap up once {time}% "
    "of the deadline has passed; use the light pipeline once free memory is at {ram}%."
)
_LIMIT_PATTERN = re.compile(
    re.escape(LIMIT_RULE).replace(r"\{", "{").replace(r"\}", "}").format(**{n: rf"(?P<{n}>\d{{1,3}})" for n in LIMIT_SIGNALS})
)


def limit_rule(limits: dict[str, float]) -> str:
    """The rule sentence for these limits (one per signal of LIMIT_SIGNALS, whole percents)."""
    if set(limits) != set(LIMIT_SIGNALS):
        raise ValueError(f"need limits for exactly {LIMIT_SIGNALS}, got {sorted(limits)}")
    percents = {}
    for name, value in limits.items():
        percents[name] = round(value * 100)
        if abs(value * 100 - percents[name]) > 1e-6:
            raise ValueError(f"{name}: limit {value} is not a whole percent")
    return LIMIT_RULE.format(**percents)


def parse_limit_rule(text: str) -> dict[str, float]:
    """The limits written in `text` by `limit_rule` (anywhere in it, e.g. a decoded prompt)."""
    found = _LIMIT_PATTERN.search(text)
    if found is None:
        raise ValueError("no limit rule in the text")
    return {n: int(found.group(n)) / 100 for n in LIMIT_SIGNALS}


def system_with_limits(limits: dict[str, float]) -> str:
    return SYSTEM_PROMPT + " " + limit_rule(limits)

_EXPLAIN = [
    "Explain step by step how {} works.", "Describe the main stages of {} in detail.",
    "Write a thorough overview of {} for a curious student.", "What are the key ideas behind {}? Explain each.",
    "Walk me through {} from the basics to the details.", "Give a detailed explanation of {} with examples.",
    "Explain {} as you would to an engineer from another field.", "What should a beginner know about {}? Be thorough.",
    "Summarize the history and the mechanics of {}.", "Explain the most common misconceptions about {}.",
    "Compare the main approaches to {} and their trade-offs.", "Explain why {} matters and how it is done in practice.",
]
_WEB = [
    "What is the current population of {}? Look it up with web_search.",
    "What is the weather forecast for {} this weekend? Use web_search.",
    "Who is the current mayor of {}? Check with web_search.",
    "What major events are happening in {} this month? Use web_search to find out.",
    "What is the air quality in {} right now? Look it up with web_search.",
    "What are the latest news headlines from {}? Use web_search.",
    "How much does a public transport day pass cost in {} today? Check with web_search.",
    "What time do the main museums in {} open tomorrow? Use web_search to confirm.",
]
_HEAVY = [
    "Run {} with run_heavy and report the main findings.",
    "I need {}. Use run_heavy and summarize the result.",
    "Please start {} using run_heavy, then explain what it shows.",
    "Use run_heavy to perform {} and list the top three takeaways.",
    "Kick off {} with run_heavy and tell me whether anything looks unusual.",
    "Perform {} via run_heavy and give me a short report.",
    "Launch {} through run_heavy and describe the outcome.",
    "Execute {} using run_heavy and highlight any problems.",
]
_TEMPLATED = {"explain": (_EXPLAIN, "topics"), "web": (_WEB, "entities"), "heavy": (_HEAVY, "jobs")}


@dataclass(frozen=True)
class Task:
    task_id: str
    family: Family
    prompt: str
    answer: str | None  # final number for math tasks, else None
    split: Literal["train", "test"]


def _gsm8k(split: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split=split)
    return [(r["question"], r["answer"]) for r in rows]


def _pool(family: str, split: str, items: dict) -> list[tuple[str, str | None]]:
    if family == "math":
        return [
            (f"{question} Reason it through and end with the final number.", answer.split("####")[-1].strip().replace(",", ""))
            for question, answer in _gsm8k(split)
        ]
    templates, key = _TEMPLATED[family]
    return [(template.format(item), None) for item in items[key][split] for template in templates]


def load_tasks(n_train: int, n_test: int, seed: int) -> list[Task]:
    items = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    tasks: list[Task] = []
    for split_index, (split, total) in enumerate((("train", n_train), ("test", n_test))):
        counts = {f: round(total * share) for f, share in FAMILY_SHARE.items()}
        counts["math"] += total - sum(counts.values())
        for family_index, (family, count) in enumerate(counts.items()):
            pool = _pool(family, split, items)
            if count > len(pool):
                raise ValueError(f"{family}/{split}: {count} tasks requested, only {len(pool)} available")
            rng = np.random.default_rng([seed, split_index, family_index])
            for i, j in enumerate(rng.permutation(len(pool))[:count]):
                prompt, answer = pool[j]
                tasks.append(Task(f"{family}-{split}-{i:05d}", family, prompt, answer, split))
    return tasks


def family_tasks(family: str, split: str, n: int, seed: int) -> list[Task]:
    """The first n tasks of one family in the order `load_tasks` draws them: the same ids and prompts
    whatever the total, also where `load_tasks` cannot draw that many of the other families."""
    pool = _pool(family, split, json.loads(TOPICS_PATH.read_text(encoding="utf-8")))
    if n > len(pool):
        raise ValueError(f"{family}/{split}: {n} tasks requested, only {len(pool)} available")
    rng = np.random.default_rng([seed, ("train", "test").index(split), list(FAMILY_SHARE).index(family)])
    return [Task(f"{family}-{split}-{i:05d}", family, *pool[j], split) for i, j in enumerate(rng.permutation(len(pool))[:n])]


def system_prompt(tok) -> str:
    """The system prompt for this tokenizer's model: its family's wording, else the default."""
    name = getattr(tok, "name_or_path", None) or ""
    return next((text for start, text in SYSTEM_PROMPTS.items() if name.startswith(start)), SYSTEM_PROMPT)


def build_prompt(tok, task: Task, system: str | None = None) -> list[int]:
    """Token ids of the chat prompt up to the start of the assistant turn. Tools only, no state.

    `system` is given only for the text teacher, whose prompt never reaches the student, and for
    prompt tests; without it the model family's wording is used (`system_prompt`).
    """
    system = system_prompt(tok) if system is None else system
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task.prompt}]
    from prio.modelspec import template_kwargs  # here: tasks stays importable without the model stack

    text = tok.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=True, tokenize=False,
        **template_kwargs(tok),
    )
    return tok(text, add_special_tokens=False).input_ids
