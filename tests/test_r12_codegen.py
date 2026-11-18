"""The code-explanation generator of R12 (prio.r12.codegen; analysis plan Section 3.4)."""

import ast
import builtins
import copy
import hashlib
import io
import keyword
import re
import tokenize
from collections import Counter

import pytest

from prio.r12 import codegen as G
from prio.reactions import _KEYWORDS, TOOLS

OPAQUE = ("f", "g", "h", "fn", "step", "helper")


@pytest.fixture(scope="module")
def items():
    return G.functions()


def _function(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    return tree.body[0]


def _call(item):
    scope: dict = {}
    exec(compile(item.source, f"<{item.template}-{item.variant}>", "exec"), scope)
    return scope[_function(item.source).name](*copy.deepcopy(item.sample))


def test_request_wordings_are_the_eight_of_the_analysis_plan():
    assert G.REQUESTS == (
        "What does this Python function do? Explain it step by step.",
        "Explain what the following Python function computes and how it works.",
        "Walk me through this function line by line and say what it returns.",
        "I found this function in a codebase. Explain its purpose and its logic in detail.",
        "Describe what this code does for a programmer who is new to Python.",
        "What does a call to this function return, in general terms? Explain the reasoning.",
        "Explain this function: what goes in, what comes out, and what happens in between.",
        "Give a thorough explanation of the following Python function, including edge cases.",
    )


def test_24_templates_each_with_40_to_60_variants_and_all_sources_distinct(items):
    per_template = Counter(item.template for item in items)
    assert list(per_template) == list(G.TEMPLATE_NAMES) and len(G.TEMPLATE_NAMES) == 24  # template by template
    assert all(40 <= n <= 60 for n in per_template.values()), per_template
    assert len(items) >= 960
    assert len({item.source for item in items}) == len(items)
    assert len({(item.template, item.variant) for item in items}) == len(items)
    for name in G.TEMPLATE_NAMES:  # variants in the generator's (product) order
        variants = [item.variant for item in items if item.template == name]
        assert variants == sorted(variants)


def test_the_four_groups_of_the_plan_have_12_6_3_and_3_templates():
    assert [len(names) for names in G.GROUPS.values()] == [12, 6, 3, 3]
    assert list(G.GROUPS) == ["lists", "strings", "dictionaries", "numbers"]
    assert [name for names in G.GROUPS.values() for name in names] == list(G.TEMPLATE_NAMES)


def test_slots_have_2_to_4_operators_and_5_or_more_constants():
    for name, (n_ops, n_constants) in G.slot_counts().items():
        assert 2 <= n_ops <= 4 and n_constants >= 5, name


def test_every_function_compiles_and_runs_on_its_sample(items):
    for item in items:
        _call(item)  # raises on a syntax error or a failing call


def test_some_functions_compute_what_their_template_says(items):
    gcd = next(item for item in items if item.template == "gcd_lcm" and re.search(r", 60\n", item.source)
               and "//" not in item.source)
    assert _call(gcd) == 12  # gcd(84, 60)
    lcm = next(item for item in items if item.template == "gcd_lcm" and re.search(r", 60\n", item.source)
               and "//" in item.source)
    assert _call(lcm) == 420
    total = next(item for item in items if item.template == "sum_filtered" and "> 10:" in item.source)
    assert _call(total) == 12 + 55 + 101
    left = next(item for item in items if item.template == "rotate" and "= 2 %" in item.source and "[-" not in item.source)
    assert _call(left) == [3, 4, 5, 6, 7, 1, 2]
    right = next(item for item in items if item.template == "rotate" and "= 2 %" in item.source and "[-" in item.source)
    assert _call(right) == [6, 7, 1, 2, 3, 4, 5]
    found = next(item for item in items if item.template == "bisect" and "return -1" in item.source)
    assert _call(found) == 3  # 7 in [1, 3, 5, 7, 9, 11, 13]
    root = next(item for item in items if item.template == "digits" and ">= 10:" in item.source)
    assert _call(root) == 2  # 9875 -> 29 -> 11 -> 2


def test_sources_are_short_and_carry_no_comment_docstring_or_annotation(items):
    for item in items:
        lines = item.source.split("\n")
        assert 4 <= len(lines) <= 12, item.source
        assert not item.source.endswith("\n") and all(line.strip() for line in lines)
        function = _function(item.source)
        assert ast.get_docstring(function) is None and function.returns is None
        arguments = function.args
        assert all(a.annotation is None for a in arguments.args) and not arguments.kwonlyargs and not arguments.defaults
        assert not any(isinstance(node, ast.AnnAssign) for node in ast.walk(function))
        kinds = {t.type for t in tokenize.generate_tokens(io.StringIO(item.source + "\n").readline)}
        assert tokenize.COMMENT not in kinds
        # pure: no import, no global state, no input or output
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)) for node in ast.walk(function))
        called = {node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert not called & {"print", "input", "open", "exec", "eval"}


def _words(text: str) -> str:
    """The text with identifiers split at `_`, lower case."""
    return text.replace("_", " ").lower()


def test_banned_words_cover_the_classifier_keywords_and_the_tool_names():
    keywords = {k for words in _KEYWORDS.values() for k in words}
    tools = {t["function"]["name"] for t in TOOLS}
    assert keywords <= set(G.BANNED) and tools <= set(G.BANNED)
    assert {"sleep", "lookup", "search", "compact"} <= set(G.BANNED)
    assert {"context", "thinking budget", "reasoning budget", "battery", "energy", "power", "out of time", "deadline",
            "time is", "network", "offline", "connection", "internet", "memory", "ram"} == keywords  # the plan's list


def test_no_banned_word_in_any_source_or_request(items):
    texts = [item.source for item in items] + list(G.REQUESTS) + [G.prompt(item, i % 8) for i, item in enumerate(items[:64])]
    for text in texts:
        for word in G.BANNED:
            for form in (text, _words(text)):
                assert not re.search(rf"\b{re.escape(word)}\b", form, re.IGNORECASE), (word, text)
            assert not re.search(rf"\b{re.escape(_words(word))}\b", _words(text)), (word, text)
        assert G.banned_in(text) == []
    assert G.banned_in("def binary_search(a):\n    return a") == ["search"]
    assert G.banned_in("the Battery is low") == ["battery"] and G.banned_in("call web_search now") == ["web_search", "search"]
    assert G.banned_in("time_is = 3") == ["time is"]


def test_names_are_neutral_and_never_shadow_a_builtin_or_another_name(items):
    reserved = set(dir(builtins)) | set(keyword.kwlist)
    for item in items:
        function = _function(item.source)
        stored = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        names = stored | {a.arg for a in function.args.args} | {function.name}
        assert not names & reserved, (names & reserved, item.source)
        assert function.name not in stored | {a.arg for a in function.args.args}


def test_half_of_the_variants_of_every_template_have_an_opaque_function_name(items):
    for name in G.TEMPLATE_NAMES:
        names = [_function(item.source).name for item in items if item.template == name]
        opaque = sum(n in OPAQUE for n in names)
        assert abs(2 * opaque - len(names)) <= 1, (name, opaque, len(names))
        assert all(len(n) > 6 and "_" in n for n in names if n not in OPAQUE), name  # the other half says what it does
        assert len({n for n in names if n in OPAQUE}) >= 4  # the opaque names rotate


def test_more_than_60_variants_are_cut_by_a_seeded_sample_kept_in_product_order():
    assert G.kept_variants(48, 3) == list(range(48)) and G.kept_variants(60, 0) == list(range(60))
    kept = G.kept_variants(96, 5)
    assert len(kept) == 60 and kept == sorted(set(kept)) and max(kept) < 96
    assert kept == G.kept_variants(96, 5) and kept != G.kept_variants(96, 6)  # generator [12, 9, template index]
    assert kept[:8] == [0, 2, 3, 4, 5, 7, 9, 11]  # pinned: default_rng([12, 9, 5]).permutation(96)[:60], sorted


def test_generator_is_deterministic_and_pinned(items):
    again = G.functions()
    assert again == items and again is not items
    assert len(items) == 1088
    digest = hashlib.sha256("\n\n".join(item.source for item in items).encode("utf-8")).hexdigest()
    assert digest == "f3c52fd55e5933e9ea72ebf32c734992f4f5c800adf5e32fd19cfe49d2c60045"
    assert items[0].source == "def sum_above(items):\n    result = 0\n    for x in items:\n        if x > 0:\n            result += x\n    return result"
    assert (items[0].template, items[0].variant, items[0].sample) == ("sum_filtered", 0, ([4, 12, -3, 7, 55, 0, 101],))


def test_prompt_is_the_request_then_the_function_in_a_fenced_python_block(items):
    text = G.prompt(items[0], 2)
    assert text == f"{G.REQUESTS[2]}\n\n```python\n{items[0].source}\n```"
    with pytest.raises(IndexError):
        G.prompt(items[0], 8)
