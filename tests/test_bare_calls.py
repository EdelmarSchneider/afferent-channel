"""Tool calls written without the tags (some model families write most of their calls as a bare JSON object).

A character-level tokenizer keeps line breaks and exact positions; `merged` adds tokens that span a
boundary of the call, as a real tokenizer has ("}" glued to the line breaks after it).
"""
import json
import re
from types import SimpleNamespace

import pytest

from prio.modelspec import ModelSpec, resolve_spec
from prio.reactions import canonical_tool_calls, find_reaction, parse_tool_calls


class CharTok:
    SPECIALS = ["<unk>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<eos>"]

    def __init__(self, merged: tuple[str, ...] = ()):
        self.items = list(self.SPECIALS) + list(merged)
        self.vocab = {s: i for i, s in enumerate(self.items)}
        self.unk_token_id, self.eos_token_id = 0, self.vocab["<eos>"]
        whole = sorted(self.SPECIALS[1:] + list(merged), key=len, reverse=True)
        self._split = re.compile("(" + "|".join(map(re.escape, whole)) + ")")

    def encode(self, text: str) -> list[int]:
        ids = []
        for part in self._split.split(text):
            for piece in ([part] if part in self.vocab else part):
                if piece not in self.vocab:
                    self.vocab[piece] = len(self.items)
                    self.items.append(piece)
                ids.append(self.vocab[piece])
        return ids

    def decode(self, ids, **_) -> str:
        return "".join(self.items[i] for i in ids)

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=self.encode(text))

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)


def spec(tok: CharTok, bare: bool) -> ModelSpec:
    v = tok.vocab
    return ModelSpec(model_id="fake", think_open=v["<think>"], think_close=v["</think>"], tool_open=v["<tool_call>"],
                     tool_close=v["</tool_call>"], eos_ids=frozenset({tok.eos_token_id}), bare_tool_calls=bare)


@pytest.fixture
def ctok():
    return CharTok()


COMPACT = '{"name": "compact_context", "arguments": {}}'
RUN = "so the answer is 5.\n</think>\n\n" + COMPACT + "\n\nThe answer is 5.<eos>"


def test_a_bare_call_is_read_with_its_token_span(ctok):
    ids = ctok.encode(RUN)
    calls = parse_tool_calls(ids, ctok, spec(ctok, bare=True))
    assert [(c.name, c.arguments) for c in calls] == [("compact_context", {})]
    assert ctok.decode(ids[calls[0].token_index:calls[0].end_index + 1]) == COMPACT


def test_a_bare_call_is_not_read_for_a_model_that_always_tags(ctok):
    assert parse_tool_calls(ctok.encode(RUN), ctok, spec(ctok, bare=False)) == []


def test_tagged_and_bare_calls_come_in_order_and_a_tagged_call_counts_once(ctok):
    text = ('My battery is low.\n<tool_call>\n{"name": "save_state", "arguments": {}}\n</tool_call>\n'
            '{"name": "sleep", "arguments": {}}<eos>')
    ids = ctok.encode(text)
    calls = parse_tool_calls(ids, ctok, spec(ctok, bare=True))
    assert [c.name for c in calls] == ["save_state", "sleep"]
    assert calls[0].token_index < calls[0].end_index < calls[1].token_index
    assert ids[calls[0].token_index] == ctok.vocab["<tool_call>"] and ids[calls[0].end_index] == ctok.vocab["</tool_call>"]


def test_arguments_with_nested_objects_and_braces_inside_strings(ctok):
    call = '{"name": "web_search", "arguments": {"query": "sets {a, b} and {c}", "filter": {"lang": "en"}}}'
    ids = ctok.encode("Let me look.\n" + call + "  \nThen I will sum up.<eos>")
    calls = parse_tool_calls(ids, ctok, spec(ctok, bare=True))
    assert [c.arguments for c in calls] == [json.loads(call)["arguments"]]
    assert ctok.decode(ids[calls[0].token_index:calls[0].end_index + 1]) == call


@pytest.mark.parametrize("text", [
    'A record looks like {"name": "Alice", "age": 3} in JSON.<eos>',          # no arguments: data, not a call
    'It starts with {"name": "sleep", "arguments": {<eos>',                   # never closed
    'Wrong types: {"name": 7, "arguments": {}} and {"name": "sleep", "arguments": []}<eos>',
])
def test_json_that_is_not_a_complete_call_is_skipped(ctok, text):
    assert parse_tool_calls(ctok.encode(text), ctok, spec(ctok, bare=True)) == []


def test_a_bare_reaction_is_found_where_its_first_call_starts(ctok):
    text = ('and then 12 times\nMy battery is almost empty, I will sleep.\n\n'
            '{"name": "save_state", "arguments": {}}\n{"name": "sleep", "arguments": {}}<eos>')
    ids = ctok.encode(text)
    reaction = find_reaction(ids, "eos", "energy", ctok, spec(ctok, bare=True))
    assert reaction is not None and reaction.named
    assert ctok.decode(ids[reaction.token_index:]).startswith('{"name": "save_state"')
    assert find_reaction(ids, "eos", "energy", ctok, spec(ctok, bare=False)) is None


def test_canonical_form_puts_a_bare_call_between_the_tags(ctok):
    ms = spec(ctok, bare=True)
    out = canonical_tool_calls(ctok.encode(RUN), ctok, ms)
    assert ctok.decode(out) == RUN.replace(COMPACT, "<tool_call>\n" + COMPACT + "\n</tool_call>")
    tagged = parse_tool_calls(out, ctok, spec(ctok, bare=False))       # readable by the tag-only reader
    assert [c.name for c in tagged] == ["compact_context"] and out[tagged[0].token_index] == ms.tool_open


def test_canonical_form_keeps_the_text_of_tokens_that_span_the_call():
    tok = CharTok(merged=("}\n\n", ".\n{"))
    text = 'I am offline.\n{"name": "local_lookup", "arguments": {"query": "x"}}\n\nDone.<eos>'
    ids = tok.encode(text)
    assert tok.vocab["}\n\n"] in ids and tok.vocab[".\n{"] in ids       # the call's braces sit inside merged tokens
    out = canonical_tool_calls(ids, tok, spec(tok, bare=True))
    assert tok.decode(out) == ('I am offline.\n<tool_call>\n{"name": "local_lookup", "arguments": {"query": "x"}}\n'
                               '</tool_call>\n\nDone.<eos>')
    assert [c.arguments for c in parse_tool_calls(out, tok, spec(tok, bare=False))] == [{"query": "x"}]


def test_canonical_form_rewrites_every_bare_call_and_leaves_tagged_ones(ctok):
    text = ('<tool_call>\n{"name": "save_state", "arguments": {}}\n</tool_call>\n{"name": "sleep", "arguments": {}}\n'
            '{"name": "compact_context", "arguments": {}}<eos>')
    out = canonical_tool_calls(ctok.encode(text), ctok, spec(ctok, bare=True))
    assert [c.name for c in parse_tool_calls(out, ctok, spec(ctok, bare=False))] == ["save_state", "sleep", "compact_context"]
    assert ctok.decode(out).count("<tool_call>") == 3 and ctok.decode(out).endswith("</tool_call><eos>")


def test_canonical_form_changes_nothing_without_bare_calls_or_for_a_tagging_model(ctok):
    plain = ctok.encode("Just text.\n</think>\n\nThe answer is 5.<eos>")
    assert canonical_tool_calls(plain, ctok, spec(ctok, bare=True)) == plain
    with_bare = ctok.encode(RUN)
    assert canonical_tool_calls(with_bare, ctok, spec(ctok, bare=False)) == with_bare


@pytest.mark.parametrize("model_id, bare", [
    ("Fam/Model-X", True), ("Qwen/Qwen3-1.7B", False), ("Qwen/Qwen3-0.6B", False),
])
def test_the_model_spec_knows_which_family_writes_bare_calls(ctok, model_id, bare, monkeypatch):
    import prio.modelspec

    monkeypatch.setattr(prio.modelspec, "BARE_CALL_FAMILIES", ("Fam/Model",))
    ctok.vocab["<|im_end|>"] = ctok.vocab["<eos>"]                         # resolve_spec looks this token up
    assert resolve_spec(model_id, ctok).bare_tool_calls is bare


# --- the data builder stores every sampled turn in the canonical form ---------------------------

class _Writes:
    """Stands in for a model: appends the same continuation to every input."""
    device = "cpu"

    def __init__(self, continuation: list[int]):
        self.continuation = continuation

    def generate(self, input_ids, **_):
        import torch

        tail = torch.tensor([self.continuation] * input_ids.shape[0])
        return torch.cat([input_ids, tail], dim=1)


@pytest.mark.parametrize("bare, expected", [
    (True, 'ok.\n<tool_call>\n{"name": "sleep", "arguments": {}}\n</tool_call><eos>'),
    (False, 'ok.\n{"name": "sleep", "arguments": {}}<eos>'),
])
def test_sampled_turns_are_stored_with_tagged_calls_for_a_model_that_writes_them_bare(ctok, bare, expected):
    from prio.datagen.build import sample_batch

    ctok.pad_token_id = ctok.eos_token_id
    written = ctok.encode('ok.\n{"name": "sleep", "arguments": {}}<eos>')
    prompts = [ctok.encode("ab"), ctok.encode("abc")]
    out = sample_batch(_Writes(written), ctok, spec(ctok, bare), prompts, max_new_tokens=64, batch_size=2)
    assert [(ctok.decode(ids), stop) for ids, stop in out] == [(expected, "eos")] * 2
