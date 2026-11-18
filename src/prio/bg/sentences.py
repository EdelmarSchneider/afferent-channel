"""Sentence banks of the background-task episodes, and the user's request.

Semi-templated like phase 1's reactions: several wordings per situation, the task named by its
label ("report download"). The request has no event parameter: whatever happens later, the text
the model sees up to the event is the same (spec anti-goal: the text must not announce the event).
"""

from prio.bg.kinds import BgTask

REACTIONS: dict[str, list[str]] = {
    "done": [
        "The {task} has finished, so I will fetch its result.",
        "The {task} is complete, so I will get its result now.",
        "The {task} just finished; let me fetch the result.",
        "The {task} is done, so I will collect what it produced.",
        "The {task} has completed, so I will look at its result.",
        "Now the {task} is finished, so I will pick up its result.",
    ],
    "error": [
        "The {task} has failed, so I will check what went wrong.",
        "The {task} stopped with an error, so I will see what happened.",
        "Something went wrong with the {task}; let me check the error.",
        "The {task} did not succeed, so I will look at the failure.",
        "The {task} broke off with an error, so I will find out why.",
        "Now the {task} has failed, so I will check its status.",
    ],
    "waiting": [
        "The {task} is waiting for an answer, so I will see what it needs.",
        "The {task} has paused for a question, so I will check what it asks.",
        "The {task} needs a decision from me; let me see the question.",
        "The {task} is on hold until it gets an answer, so I will look.",
        "The {task} is asking something, so I will read its question.",
        "Now the {task} is waiting on me, so I will check what it wants.",
    ],
    "stalled": [
        "The {task} seems stuck, so I will check on it.",
        "The {task} has stopped making progress, so I will look at it.",
        "The {task} is not moving anymore; let me check its status.",
        "The {task} appears to be stalled, so I will see what is going on.",
        "The {task} has made no progress for a while, so I will check it.",
        "Now the {task} looks stuck, so I will check its state.",
    ],
}

_WAITING_ANSWER = [
    "It asks whether to replace the existing files; you said {answer}, so I will answer {answer}.",
    "It wants to know if it should replace the files; as you asked, I will answer {answer}.",
    "The question is whether to replace the files; following your instruction, the answer is {answer}.",
]

_DETAIL = {
    "total": "its total is {v}", "pages": "it has {v} pages", "files": "{v} files", "minutes": "it took {v} minutes",
    "warnings": "it reports {v} warnings", "version": "it installed version {v}", "size": "the result is {v} MB",
    "items": "{v} items changed", "frames": "it rendered {v} frames", "rows": "it exported {v} rows",
    "saved": "it saved {v} MB", "address": "it is running at {v}",
}

_BRIDGES = {
    "thinking": ["Back to the problem.", "Now, back to the reasoning.", "Returning to where I was.",
                 "Back to the question at hand.", "Let me pick up where I left off."],
    "answer": ["Now, back to your question.", "Back to where I was.", "Returning to the explanation.",
               "Now, to continue.", "Picking up where I left off."],
}

_CONNECTORS = ["While that runs, here is my question:", "Meanwhile, please help me with this:",
               "In the meantime, answer this for me:", "While you wait for that, my question is:"]


def _pick(options: list[str], rng) -> str:
    return options[int(rng.integers(len(options)))]


def reaction_sentence(kind: str, task: BgTask, rng) -> str:
    return _pick(REACTIONS[kind], rng).format(task=task.label())


def waiting_answer_sentence(task: BgTask, answer: str, rng) -> str:
    return _pick(_WAITING_ANSWER, rng).format(answer=answer)


def report_line(kind: str, task: BgTask, rng, answer: str | None = None) -> str:
    name = task.label()
    if kind == "done":
        if task.detail_key:
            return f"The {name} is done: {_DETAIL[task.detail_key].format(v=task.detail_value)}."
        return _pick([f"The {name} is done.", f"The {name} has finished."], rng)
    if kind == "error":
        return f"The {name} failed ({task.reason}), so I stopped watching it."
    if kind == "waiting":
        return f"The {name} asked whether to replace some files; I answered {answer}."
    if kind == "stalled":
        return _pick([f"The {name} seems stuck; I am keeping an eye on it.",
                      f"The {name} has stalled; I will keep watching it."], rng)
    raise ValueError(f"unknown event kind {kind!r}")


def bridge(phase: str, rng) -> str:
    return _pick(_BRIDGES[phase], rng)


def closing_line(tasks: list[BgTask], rng) -> str:
    names = [t.label() for t in tasks]
    joined = names[0] if len(names) == 1 else ", the ".join(names[:-1]) + " and the " + names[-1]
    if len(names) == 1:
        return _pick([f"The {joined} is still running; I will tell you when it finishes.",
                      f"The {joined} is still running; I will let you know when it is done."], rng)
    return _pick([f"The {joined} are still running; I will tell you when they finish.",
                  f"The {joined} are still running; I will let you know when they are done."], rng)


def request_text(tasks: list[BgTask], question: str, policy: str, rng) -> str:
    clauses = [f"{t.describe()[0].upper()}{t.describe()[1:]} and {t.end}." for t in tasks]
    confirm = f"If a task asks whether to replace a file, answer {policy}."
    return " ".join(clauses + [confirm, _pick(_CONNECTORS, rng), question])
