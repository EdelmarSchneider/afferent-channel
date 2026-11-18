import json
import re

import pytest

import prio.tasks as tasks_module
from prio.tasks import SYSTEM_PROMPT, TOPICS_PATH, load_tasks


@pytest.fixture(autouse=True)
def gsm8k_stub(monkeypatch):
    def fake(split):
        return [(f"{split} question {i}: what is {i} + 1?", f"work\n#### {i + 1:,}") for i in range(1200, 1300)]

    monkeypatch.setattr(tasks_module, "_gsm8k", fake)


def test_split_sizes_and_family_mix():
    ts = load_tasks(200, 50, seed=0)
    tr = [t for t in ts if t.split == "train"]
    assert len(tr) == 200 and len(ts) - len(tr) == 50
    frac = lambda f: sum(t.family == f for t in tr) / len(tr)
    assert abs(frac("math") - 0.3) < 0.02 and abs(frac("explain") - 0.3) < 0.02
    assert abs(frac("web") - 0.2) < 0.02 and abs(frac("heavy") - 0.2) < 0.02


def test_no_leak_between_splits():
    ts = load_tasks(200, 50, seed=0)
    tr = {t.prompt for t in ts if t.split == "train"}
    te = {t.prompt for t in ts if t.split == "test"}
    assert not tr & te and len({t.task_id for t in ts}) == len(ts)
    cfg = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    for key in ("topics", "entities", "jobs"):
        assert not set(cfg[key]["train"]) & set(cfg[key]["test"])


def test_test_tasks_use_only_test_items():
    cfg = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    train_items = cfg["topics"]["train"] + cfg["entities"]["train"] + cfg["jobs"]["train"]
    for t in load_tasks(200, 50, seed=0):
        if t.split == "test" and t.family != "math":
            assert not any(item in t.prompt for item in train_items)


def test_math_tasks_carry_answers_and_others_do_not():
    for t in load_tasks(100, 20, seed=0):
        assert (t.answer is not None) == (t.family == "math")
        if t.family == "math":
            assert t.answer.isdigit()  # "1,201" -> "1201"


def test_tool_tasks_name_their_tool():
    for t in load_tasks(100, 20, seed=0):
        if t.family == "web":
            assert "web_search" in t.prompt
        if t.family == "heavy":
            assert "run_heavy" in t.prompt


def test_deterministic():
    assert load_tasks(100, 20, seed=1) == load_tasks(100, 20, seed=1)
    assert load_tasks(100, 20, seed=1) != load_tasks(100, 20, seed=2)


def test_too_many_tasks_requested_raises():
    with pytest.raises(ValueError):
        load_tasks(100_000, 20, seed=0)


@pytest.mark.parametrize("family", ["math", "explain", "web", "heavy"])
@pytest.mark.parametrize("split, sizes", [("test", (0, 60)), ("train", (60, 0))])
def test_family_tasks_are_the_first_tasks_of_that_family_in_load_tasks(family, split, sizes):
    from prio.tasks import family_tasks

    whole = [t for t in load_tasks(*sizes, seed=3) if t.family == family]
    assert len(whole) >= 12 and family_tasks(family, split, 12, seed=3) == whole[:12]
    # the same prefix whatever the total: a list longer than load_tasks could draw with the other families
    assert family_tasks(family, split, 20, seed=3)[:12] == whole[:12]
    with pytest.raises(ValueError):
        family_tasks(family, split, 100_000, seed=3)


def test_system_prompt_has_no_state_words():
    assert not re.search(r"battery|context window|memory|network|deadline|budget", SYSTEM_PROMPT, re.I)


# --- chat templates that write today's date -------------------------------------------------------

class _TemplateTok:
    def __init__(self, template):
        self.chat_template, self.seen = template, []

    def apply_chat_template(self, messages, **kwargs):
        self.seen.append(kwargs)
        return "rendered"

    def __call__(self, text, add_special_tokens=False):
        return type("Encoded", (), {"input_ids": [1, 2, 3]})()


def test_a_template_that_writes_todays_date_gets_a_fixed_one():
    from prio.modelspec import FIXED_DATE, frozen_chat_template

    dated = 'A {%- set today = strftime_now("%d %B %Y") -%} {{- "Today Date: " ~ today }} B'
    frozen = frozen_chat_template(_TemplateTok(dated))
    assert "strftime_now" not in frozen and f'set today = "{FIXED_DATE}"' in frozen
    assert frozen.startswith("A ") and frozen.endswith(" B")
    assert frozen_chat_template(_TemplateTok("{{ messages }}")) is None  # no date: the tokenizer's own template (Qwen3)
    assert frozen_chat_template(_TemplateTok(None)) is None


def test_build_prompt_renders_with_the_frozen_template_only_when_there_is_a_date():
    from prio.modelspec import frozen_chat_template, template_kwargs
    from prio.tasks import Task, build_prompt

    task = Task("t", "explain", "explain x", None, "test")
    plain, dated = _TemplateTok("{{ messages }}"), _TemplateTok('{{ strftime_now("%d %B %Y") }}')
    assert build_prompt(plain, task) == [1, 2, 3] and build_prompt(dated, task) == [1, 2, 3]
    assert "chat_template" not in plain.seen[0]  # the call a tokenizer without a date always got
    assert dated.seen[0]["chat_template"] == frozen_chat_template(dated) != dated.chat_template
    assert template_kwargs(plain) == {} and template_kwargs(dated) == {"chat_template": frozen_chat_template(dated)}


# --- a system-prompt wording per model family ----------------------------------------------------

class _NamedTok(_TemplateTok):
    def __init__(self, name):
        super().__init__("{{ messages }}")
        self.name_or_path, self.messages = name, []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return super().apply_chat_template(messages, **kwargs)


def test_the_system_prompt_is_the_default_unless_the_model_family_has_its_own(monkeypatch):
    import prio.tasks as T

    monkeypatch.setitem(T.SYSTEM_PROMPTS, "Fam/Model", "Wording of the family.")
    assert T.system_prompt(_NamedTok("Fam/Model-X")) == "Wording of the family."
    assert T.system_prompt(_NamedTok("Qwen/Qwen3-1.7B")) == T.SYSTEM_PROMPT
    assert T.system_prompt(_TemplateTok(None)) == T.SYSTEM_PROMPT  # a tokenizer without a name


def test_build_prompt_uses_the_family_wording_and_an_explicit_system_wins(monkeypatch):
    import prio.tasks as T

    monkeypatch.setitem(T.SYSTEM_PROMPTS, "Fam/Model", "Wording of the family.")
    task = T.Task("t", "explain", "explain x", None, "test")
    family, other = _NamedTok("Fam/Model-X"), _NamedTok("Qwen/Qwen3-1.7B")
    T.build_prompt(family, task)
    T.build_prompt(other, task)
    T.build_prompt(family, task, system="Given by the caller.")
    assert [m[0]["content"] for m in family.messages] == ["Wording of the family.", "Given by the caller."]
    assert other.messages[0][0]["content"] == T.SYSTEM_PROMPT


def test_the_teacher_prompt_extends_the_wording_of_the_model_family(monkeypatch):
    import prio.tasks as T
    from prio.datagen.build import TEACHER_SYSTEM, teacher_system

    monkeypatch.setitem(T.SYSTEM_PROMPTS, "Fam/Model", "Wording of the family.")
    assert teacher_system(_NamedTok("Qwen/Qwen3-1.7B")) == TEACHER_SYSTEM
    special = teacher_system(_NamedTok("Fam/Model-X"))
    assert special.startswith("Wording of the family. Sometimes a line starting with [STATUS:")
    assert special[len("Wording of the family."):] == TEACHER_SYSTEM[len(T.SYSTEM_PROMPT):]


def test_no_family_wording_names_a_state():
    import prio.tasks as T

    for wording in T.SYSTEM_PROMPTS.values():
        assert not re.search(r"battery|context window|memory|network|deadline|budget", wording, re.I)


def test_both_qwen3_sizes_use_the_default_wording():
    import prio.tasks as T

    assert T.system_prompt(_NamedTok("Qwen/Qwen3-1.7B")) == T.SYSTEM_PROMPT
    assert T.system_prompt(_NamedTok("Qwen/Qwen3-0.6B")) == T.SYSTEM_PROMPT
