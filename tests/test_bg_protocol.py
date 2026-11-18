import pytest

from helpers import FAKE_FMT
from prio.bg.protocol import BG_TOOLS, STATE_VALUE, STATES, QwenFormat, slot_columns, state_of
from prio.reactions import parse_tool_calls


def test_slot_columns():
    assert slot_columns(1) == (0, 1) and slot_columns(3) == (4, 5)


def test_state_values_are_distinct_level_centers():
    assert [STATE_VALUE[s] for s in STATES] == [0.005, 0.105, 0.205, 0.305, 0.405, 0.505]
    assert all(state_of(STATE_VALUE[s]) == s for s in STATES)


def test_tool_names_in_order():
    assert [t["function"]["name"] for t in BG_TOOLS] == [
        "start_task", "watch", "unwatch", "task_result", "task_status", "answer_task"]


def test_fake_calls_parse_back(tok, ms):
    ids = FAKE_FMT.encode(FAKE_FMT.calls_text([("watch", {"task_id": "t1"}), ("watch", {"task_id": "t2"})]))
    assert [(c.name, c.arguments) for c in parse_tool_calls(ids, tok, ms)] == [
        ("watch", {"task_id": "t1"}), ("watch", {"task_id": "t2"})]


@pytest.fixture(scope="module")
def qwen():
    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    return tok, resolve_spec("Qwen/Qwen3-1.7B", tok), QwenFormat(tok)


def _call(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


@pytest.mark.slow
def test_qwen_segment_matches_chat_template(qwen):
    from prio.tasks import SYSTEM_PROMPT

    tok, _, fmt = qwen
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "REQ"},
                {"role": "assistant", "content": "", "tool_calls": [_call("watch", {"task_id": "t1"}),
                                                                   _call("watch", {"task_id": "t2"})]},
                {"role": "tool", "content": '{"slot": 1}'}, {"role": "tool", "content": '{"slot": 2}'}]
    text = tok.apply_chat_template(messages, tools=BG_TOOLS, add_generation_prompt=True, enable_thinking=True,
                                   tokenize=False)
    after_turn = text[text.rindex("</tool_call><|im_end|>") + len("</tool_call><|im_end|>"):]
    assert fmt.segment([{"slot": 1}, {"slot": 2}]) == tok(after_turn, add_special_tokens=False).input_ids


@pytest.mark.slow
def test_qwen_prompt_matches_chat_template(qwen):
    from prio.tasks import SYSTEM_PROMPT

    tok, _, fmt = qwen
    text = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "REQ"}],
                                   tools=BG_TOOLS, add_generation_prompt=True, enable_thinking=True, tokenize=False)
    assert fmt.prompt("REQ") == tok(text, add_special_tokens=False).input_ids


@pytest.mark.slow
def test_qwen_calls_parse_back(qwen):
    tok, ms, fmt = qwen
    ids = fmt.encode("I will check it. " + fmt.calls_text([("task_status", {"task_id": "t2"})]))
    assert [(c.name, c.arguments) for c in parse_tool_calls(ids, tok, ms)] == [("task_status", {"task_id": "t2"})]
