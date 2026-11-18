"""Fresh pools and the two task sets of R12 (prio.r12.tasks; analysis plan Section 3).

Reads the cached GSM8K (the real test split: the counts of the plan are counts of that file)."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

import prio.tasks as T
from prio import r12
from prio.r12 import codegen
from prio.r12 import tasks as R
from prio.reactions import names_signal
from prio.signals import CORE_SIGNALS

PLAN = Path(__file__).resolve().parents[1] / "docs" / "2026-11-17-r12-analysis-plan.md"


@pytest.fixture(scope="module")
def sets():
    return R.task_sets()


@pytest.fixture(scope="module")
def old_items():
    return json.loads(T.TOPICS_PATH.read_text(encoding="utf-8"))


def _appendix(label: str) -> list[str]:
    if not PLAN.exists():
        pytest.skip("the frozen analysis plan of the confirmatory evaluation is not part of this repository")
    lines = PLAN.read_text(encoding="utf-8").split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(label))
    return [item.strip() for item in lines[start + 1].split(";")]


def test_new_items_are_the_lists_of_the_frozen_plan():
    items = R.new_items()
    assert items["topics"] == _appendix("Topics (86)") and len(items["topics"]) == 86
    assert items["entities"] == _appendix("Cities (48)") and len(items["entities"]) == 48
    assert items["datasets"] == _appendix("Datasets (4)") and len(items["datasets"]) == 4
    for key in ("topics", "entities", "datasets"):
        assert len(set(items[key])) == len(items[key])
    assert "Gdańsk" in items["entities"] and "João Pessoa" in items["entities"]  # UTF-8 kept


def test_new_items_share_nothing_with_the_old_lists(old_items):
    new = R.new_items()
    for key in ("topics", "entities"):
        old = {item.lower() for split in ("train", "test") for item in old_items[key][split]}
        assert not old & {item.lower() for item in new[key]}
    old_jobs = [job.lower() for split in ("train", "test") for job in old_items["jobs"][split]]
    for dataset in new["datasets"]:
        assert not any(dataset.lower() in job for job in old_jobs)
    assert not {job.lower() for job in R.new_jobs()} & set(old_jobs)


def test_no_new_item_holds_a_classifier_keyword():
    new = R.new_items()
    for item in new["topics"] + new["entities"] + new["datasets"] + R.new_jobs():
        assert not any(names_signal(item, s.name) for s in CORE_SIGNALS), item


def test_new_jobs_are_the_ten_old_analyses_on_each_new_dataset(old_items):
    jobs = R.new_jobs()
    assert len(jobs) == 40 and len(set(jobs)) == 40
    assert jobs[0] == "a full regression analysis on the warehouse inventory dataset"
    assert jobs[9] == "a complete statistical summary on the warehouse inventory dataset"
    assert jobs[10] == "a full regression analysis on the library loans dataset"
    assert jobs[-1] == "a complete statistical summary on the flight delays dataset"
    analyses = [job.split(" on the ")[0] for job in old_items["jobs"]["test"][:10]]
    assert [job.split(" on the ")[0] for job in jobs[:10]] == analyses


def test_used_prompts_cover_every_earlier_draw():
    draws = R.earlier_draws()
    assert len(draws) == 5 and [len(tasks) for tasks in draws.values()] == [3600, 300, 100, 20, 40]
    used = R.used_prompts()
    assert used == {t.prompt for tasks in draws.values() for t in tasks}
    items = json.loads(T.TOPICS_PATH.read_text(encoding="utf-8"))
    counted = {}
    for family in ("math", "explain", "web", "heavy"):
        pool = [p for p, _ in T._pool(family, "test", items)]
        counted[family] = (len(pool), sum(p in used for p in pool))
    # AP 3.1: pool and used prompts of the test split
    assert counted == {"math": (1319, 450), "explain": (300, 220), "web": (192, 120), "heavy": (160, 122)}


def test_fresh_pools_hold_no_used_prompt_and_have_the_planned_sizes():
    used = R.used_prompts()
    sizes = {}
    for family in r12.FAMILIES:
        pool = R.fresh_pool(family)
        sizes[family] = len(pool)
        assert len({p for p, _ in pool}) == len(pool), family
        assert not used & {p for p, _ in pool}, family
        assert all((answer is not None) == (family == "math") for _, answer in pool)
    assert {f: sizes[f] for f in ("math", "explain", "web", "heavy")} == {
        "math": 869, "explain": 1032, "web": 384, "heavy": 320}
    assert sizes["code"] == len(codegen.functions()) >= 960


def test_pools_are_in_the_fixed_order_of_the_plan(old_items):
    new = R.new_items()
    explain = R.fresh_pool("explain")
    assert [p for p, _ in explain[:12]] == [w.format(new["topics"][0]) for w in T._EXPLAIN]  # item by item, wordings in order
    assert explain[12][0] == T._EXPLAIN[0].format(new["topics"][1])
    web = R.fresh_pool("web")
    assert [p for p, _ in web[:8]] == [w.format("Tampere") for w in T._WEB]
    heavy = R.fresh_pool("heavy")
    assert heavy[0][0] == "Run a full regression analysis on the warehouse inventory dataset with run_heavy and report the main findings."
    assert heavy[8][0] == T._HEAVY[0].format("a clustering analysis on the warehouse inventory dataset")
    math = R.fresh_pool("math")
    whole = [p for p, _ in T._pool("math", "test", old_items)]
    positions = [whole.index(p) for p, _ in math[:50]]
    assert positions == sorted(positions)  # dataset order
    assert all(p.endswith(" Reason it through and end with the final number.") for p, _ in math)
    code = R.fresh_pool("code")
    assert [source for source, _ in code] == [item.source for item in codegen.functions()]


def test_sets_have_the_planned_counts_per_family_and_arm(sets):
    assert list(sets) == ["pilot", "confirm"] and all(list(arms) == list(r12.ARMS) for arms in sets.values())
    counts = {(name, arm): Counter(t.family for t in tasks) for name, arms in sets.items() for arm, tasks in arms.items()}
    assert counts[("pilot", "paired")] == {"math": 16, "explain": 16, "web": 16, "heavy": 16, "code": 32}
    assert counts[("pilot", "free")] == counts[("pilot", "live")] == {"math": 16, "explain": 16, "code": 16}
    assert counts[("pilot", "Ls")] == {"math": 8, "explain": 8, "code": 8}
    assert counts[("confirm", "paired")] == {"math": 160, "explain": 160, "web": 80, "heavy": 80, "code": 320}
    assert counts[("confirm", "free")] == counts[("confirm", "live")] == {"math": 120, "explain": 120, "code": 120}
    assert counts[("confirm", "Ls")] == {"math": 40, "explain": 40, "code": 40}
    totals = {name: {arm: len(tasks) for arm, tasks in arms.items()} for name, arms in sets.items()}
    assert totals == {"pilot": {"paired": 96, "free": 48, "live": 48, "Ls": 24},
                      "confirm": {"paired": 800, "free": 360, "live": 360, "Ls": 120}}
    assert sum(totals["pilot"].values()) == 216 and sum(totals["confirm"].values()) == 1640
    drawn = Counter(t.family for arms in sets.values() for tasks in arms.values() for t in tasks)
    assert drawn == {"math": 496, "explain": 496, "web": 96, "heavy": 96, "code": 672}
    for family, sizes in r12.SLICES.items():  # the table of AP 3.3
        assert sum(sizes) == drawn[family] <= len(R.fresh_pool(family))


def test_arm_lists_go_family_by_family_in_draw_order(sets):
    for name, arms in sets.items():
        for arm, tasks in arms.items():
            families = [t.family for t in tasks]
            assert families == sorted(families, key=r12.FAMILIES.index)
            for family in set(families):
                positions = [int(t.task_id.rsplit("-", 1)[1]) for t in tasks if t.family == family]
                assert positions == list(range(positions[0], positions[0] + len(positions)))  # one slice of the permutation
    # the slices in the order of AP 3.3: the two Ls slices last, so that no other slice moves
    start = {(name, arm): int(next(t for t in tasks if t.family == "code").task_id[-5:])
             for name, arms in sets.items() for arm, tasks in arms.items()}
    assert start == {("pilot", "paired"): 0, ("pilot", "free"): 32, ("pilot", "live"): 48, ("confirm", "paired"): 64,
                     ("confirm", "free"): 384, ("confirm", "live"): 504, ("pilot", "Ls"): 624, ("confirm", "Ls"): 632}
    math = {(name, arm): int(next(t for t in tasks if t.family == "math").task_id[-5:])
            for name, arms in sets.items() for arm, tasks in arms.items()}
    assert math == {("pilot", "paired"): 0, ("pilot", "free"): 16, ("pilot", "live"): 32, ("confirm", "paired"): 48,
                    ("confirm", "free"): 208, ("confirm", "live"): 328, ("pilot", "Ls"): 448, ("confirm", "Ls"): 456}


def test_task_fields_ids_and_the_code_wordings_in_rotation(sets):
    for name, arms in sets.items():
        for arm, tasks in arms.items():
            for t in tasks:
                assert re.fullmatch(rf"{r12.PREFIX[name]}-{t.family}-\d{{5}}", t.task_id)
                assert t.split == "test" and (t.answer is not None) == (t.family == "math")
                if t.family == "code":
                    position = int(t.task_id[-5:])
                    assert t.prompt.startswith(codegen.REQUESTS[position % 8] + "\n\n```python\ndef ")
                    assert t.prompt.endswith("\n```")
                if t.family == "web":
                    assert "web_search" in t.prompt
                if t.family == "heavy":
                    assert "run_heavy" in t.prompt


def test_pilot_and_confirmatory_sets_are_disjoint(sets):
    pilot = [t for tasks in sets["pilot"].values() for t in tasks]
    confirm = [t for tasks in sets["confirm"].values() for t in tasks]
    assert not {t.prompt for t in pilot} & {t.prompt for t in confirm}
    assert not {t.task_id for t in pilot} & {t.task_id for t in confirm}
    everything = pilot + confirm
    assert len({t.prompt for t in everything}) == len(everything) == 1856
    assert len({t.task_id for t in everything}) == len(everything)
    # no function of the code family is asked about twice, whatever the wording
    sources = [t.prompt.split("```python\n")[1] for t in everything if t.family == "code"]
    assert len(set(sources)) == len(sources) == 672
    assert not R.used_prompts() & {t.prompt for t in everything}


def test_task_sets_are_deterministic_and_pinned(sets):
    again = R.task_sets()
    assert again == sets and again is not sets and again["pilot"]["paired"] is not sets["pilot"]["paired"]
    digest = {name: hashlib.sha256("\n".join(f"{t.task_id}\t{t.prompt}" for tasks in arms.values() for t in tasks)
                                   .encode("utf-8")).hexdigest() for name, arms in sets.items()}
    assert digest == {"pilot": "404750626d95ef6868c985f668a6bb07b2ff2e262739be4560d4e7fd9b77341b", "confirm": "60d708fca97ad0e79576a63c42cd64d87076685a4423e557b2fc57702590a95f"}
    first = sets["pilot"]["paired"]
    assert first[0].task_id == "r12p-math-00000" and first[16].task_id == "r12p-explain-00000"
    assert first[16].prompt == "Explain step by step how seed dispersal works."
    assert sets["confirm"]["Ls"][80].task_id == "r12c-code-00632"


def test_the_family_permutation_is_the_generator_of_the_plan(sets):
    import numpy as np

    for f, family in enumerate(r12.FAMILIES):
        pool = R.fresh_pool(family)
        order = np.random.default_rng([12, f]).permutation(len(pool))
        tasks = [t for arms in sets.values() for arm_tasks in arms.values() for t in arm_tasks if t.family == family]
        for t in tasks[:: max(1, len(tasks) // 25)]:
            position = int(t.task_id[-5:])
            prompt, answer = pool[order[position]]
            if family == "code":
                assert t.prompt == codegen.prompt(codegen.functions()[order[position]], position % 8)
            else:
                assert (t.prompt, t.answer) == (prompt, answer)


def test_earlier_draws_are_untouched():
    assert list(T.FAMILY_SHARE) == ["math", "explain", "web", "heavy"]
    assert T.FAMILY_SHARE == {"math": 0.3, "explain": 0.3, "web": 0.2, "heavy": 0.2}
    first = T.load_tasks(0, 20, 3)
    assert [t.family for t in first] == ["math"] * 6 + ["explain"] * 6 + ["web"] * 4 + ["heavy"] * 4
    R.task_sets()
    assert not hasattr(T._gsm8k, "cache_info")  # the loader is put back after the draws are recomputed


def test_write_tasks_lists_every_task_of_the_set_with_its_arm(tmp_path, sets):
    info = R.write_tasks("pilot", tmp_path / "r12p")
    raw = (tmp_path / "r12p" / "tasks.json").read_bytes()
    assert b"\r" not in raw and info["sha256"] == hashlib.sha256(raw).hexdigest()
    rows = json.loads(raw.decode("utf-8"))
    assert info["tasks"] == len(rows) == 216 and info["by_arm"] == {"paired": 96, "free": 48, "live": 48, "Ls": 24}
    assert list(rows[0]) == ["task_id", "family", "arm", "prompt", "answer"]
    expected = [(t.task_id, t.family, arm, t.prompt, t.answer) for arm, tasks in sets["pilot"].items() for t in tasks]
    assert [tuple(row.values()) for row in rows] == expected
    assert R.read_tasks(tmp_path / "r12p") == sets["pilot"]
    R.write_tasks("pilot", tmp_path / "r12p")  # the same bytes again
    assert (tmp_path / "r12p" / "tasks.json").read_bytes() == raw
    with pytest.raises(KeyError):
        R.write_tasks("other", tmp_path / "x")
