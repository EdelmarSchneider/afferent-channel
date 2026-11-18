"""Fresh task pools and the two task sets of R12 (AP 3).

Fresh = used in no training set and no earlier test set. Every earlier draw is recomputed from its
seed (`earlier_draws`); the pools hold none of those prompts. One permutation per family
(generator [SEED, position of the family]) is cut into consecutive slices: the pilot set and the
confirmatory set, disjoint by construction.
"""

import contextlib
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from prio import tasks as T
from prio.r12 import ARMS, FAMILIES, PREFIX, SEED, SET_CODE, SLICE_ORDER, SLICES, codegen
from prio.tasks import Task

ITEMS_PATH = Path(T.__file__).parent / "data" / "r12_items.json"


@contextlib.contextmanager
def _gsm8k_once():
    """`tasks._gsm8k` answers from memory while the earlier draws are recomputed: every call of
    `load_tasks` or `family_tasks` reads the dataset again. The loader is put back afterwards."""
    original = T._gsm8k
    T._gsm8k = functools.lru_cache(maxsize=None)(original)
    try:
        yield
    finally:
        T._gsm8k = original


def new_items() -> dict[str, list[str]]:
    """Topics, cities and datasets of the analysis plan's Appendix A: in no split of `topics.json`."""
    return json.loads(ITEMS_PATH.read_text(encoding="utf-8"))


def new_jobs() -> list[str]:
    """The 10 existing analyses on each of the 4 new datasets, dataset by dataset."""
    old = json.loads(T.TOPICS_PATH.read_text(encoding="utf-8"))["jobs"]["test"][:10]
    analyses = [job.split(" on the ")[0] for job in old]
    return [f"{analysis} on the {dataset} dataset" for dataset in new_items()["datasets"] for analysis in analyses]


@functools.lru_cache(maxsize=1)
def _earlier() -> dict[str, tuple[Task, ...]]:
    with _gsm8k_once():
        draws = {
            "load_tasks(3000, 600, seed 0)": T.load_tasks(3000, 600, 0),  # data/n6 and everything built from it
            "family_tasks(math, test, 300, seed 1)": T.family_tasks("math", "test", 300, 1),  # R11
            "family_tasks(explain, test, 100, seed 1)": T.family_tasks("explain", "test", 100, 1),  # R11
            "load_tasks(0, 20, seed 3)": T.load_tasks(0, 20, 3),  # slow integration test
            "load_tasks(40, 0, seed 3)": T.load_tasks(40, 0, 3),  # a debug script, training split
        }
    return {label: tuple(tasks) for label, tasks in draws.items()}


def earlier_draws() -> dict[str, list[Task]]:
    """Every draw of tasks made before R12 (AP 3.1), recomputed from its seed."""
    return {label: list(tasks) for label, tasks in _earlier().items()}


@functools.lru_cache(maxsize=1)
def _used() -> frozenset[str]:
    return frozenset(t.prompt for tasks in _earlier().values() for t in tasks)


def used_prompts() -> set[str]:
    return set(_used())


@functools.lru_cache(maxsize=None)
def _pool(family: str) -> tuple[tuple[str, str | None], ...]:
    if family == "math":  # the unused GSM8K test problems, in dataset order
        with _gsm8k_once():
            whole = T._pool("math", "test", {})
        return tuple((p, a) for p, a in whole if p not in _used())
    if family == "code":  # template by template, variants in the generator's order; the wording comes with the draw
        return tuple((item.source, None) for item in codegen.functions())
    items = new_items()
    templates, new = {"explain": (T._EXPLAIN, items["topics"]), "web": (T._WEB, items["entities"]),
                      "heavy": (T._HEAVY, new_jobs())}[family]
    return tuple((template.format(item), None) for item in new for template in templates)  # item by item


def fresh_pool(family: str) -> list[tuple[str, str | None]]:
    """(prompt, answer) of every fresh task of a family, in the fixed order of AP 3.3. The code
    family's entries are (function source, None): the request wording is added at the draw."""
    return list(_pool(family))


@functools.lru_cache(maxsize=1)
def _sets() -> dict[str, dict[str, tuple[Task, ...]]]:
    out: dict[str, dict[str, list[Task]]] = {name: {arm: [] for arm in ARMS} for name in SET_CODE}
    functions = codegen.functions()
    for f, family in enumerate(FAMILIES):
        pool = _pool(family)
        order = np.random.default_rng([SEED, f]).permutation(len(pool))
        if sum(SLICES[family]) > len(pool):
            raise ValueError(f"{family}: {sum(SLICES[family])} tasks planned, {len(pool)} in the pool")
        position = 0
        for (name, arm), count in zip(SLICE_ORDER, SLICES[family], strict=True):
            for _ in range(count):
                j = int(order[position])
                prompt, answer = pool[j]
                if family == "code":
                    prompt = codegen.prompt(functions[j], position % len(codegen.REQUESTS))
                out[name][arm].append(Task(f"{PREFIX[name]}-{family}-{position:05d}", family, prompt, answer, "test"))
                position += 1
    return {name: {arm: tuple(tasks) for arm, tasks in arms.items()} for name, arms in out.items()}


def task_sets() -> dict[str, dict[str, list[Task]]]:
    """{"pilot" | "confirm": {"paired" | "free" | "live" | "Ls": tasks}}; each arm's list goes
    family by family (FAMILIES order), each family in draw order. Task id: the set's prefix, the
    family, the position in the family's permutation."""
    return {name: {arm: list(tasks) for arm, tasks in arms.items()} for name, arms in _sets().items()}


def write_tasks(set_name: str, out: Path) -> dict:
    """<out>/tasks.json: every task of the set (id, family, arm, prompt, answer), arm by arm."""
    arms = task_sets()[set_name]
    rows = [{"task_id": t.task_id, "family": t.family, "arm": arm, "prompt": t.prompt, "answer": t.answer}
            for arm, tasks in arms.items() for t in tasks]
    raw = (json.dumps(rows, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_bytes(raw)
    return {"set": set_name, "tasks": len(rows), "by_arm": {arm: len(tasks) for arm, tasks in arms.items()},
            "sha256": hashlib.sha256(raw).hexdigest()}


def read_tasks(folder: Path) -> dict[str, list[Task]]:
    """The arms of a written set, from <folder>/tasks.json."""
    rows = json.loads((Path(folder) / "tasks.json").read_text(encoding="utf-8"))
    arms: dict[str, list[Task]] = {arm: [] for arm in ARMS}
    for row in rows:
        arms[row["arm"]].append(Task(row["task_id"], row["family"], row["prompt"], row["answer"], "test"))
    return arms
