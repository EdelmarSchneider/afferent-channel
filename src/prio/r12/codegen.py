"""The code-explanation task family of R12 (AP 3.4): Python functions written from templates.

A task = one request wording + one function in a fenced Python block. 24 templates, each a pure
function of 4 to 12 lines without comments, docstring or type hints. Slots of a template: the
operator or direction (2 to 4 choices), a constant (5 or more), and the style of the function
name (descriptive in one half of the variants, opaque in the other half). Argument and local
names come from short neutral lists and rotate over the variants. Variants = all slot
combinations in `itertools.product` order (operator, constant, name style); a template with more
than 60 keeps a seeded sample of 60, in product order.

Never in a source or a request: a keyword of the classifier or the name of a tool (`BANNED`), so
that the family earns no hit and no false alarm through its own vocabulary.
"""

import functools
import itertools
import re
from dataclasses import dataclass
from string import Template

import numpy as np

from prio.r12 import SEED
from prio.reactions import _KEYWORDS, TOOLS

REQUESTS: tuple[str, ...] = (
    "What does this Python function do? Explain it step by step.",
    "Explain what the following Python function computes and how it works.",
    "Walk me through this function line by line and say what it returns.",
    "I found this function in a codebase. Explain its purpose and its logic in detail.",
    "Describe what this code does for a programmer who is new to Python.",
    "What does a call to this function return, in general terms? Explain the reasoning.",
    "Explain this function: what goes in, what comes out, and what happens in between.",
    "Give a thorough explanation of the following Python function, including edge cases.",
)
MAX_VARIANTS = 60
OPAQUE = ("f", "g", "h", "fn", "step", "helper")
BANNED: tuple[str, ...] = tuple(dict.fromkeys(
    [k for words in _KEYWORDS.values() for k in words] + [t["function"]["name"] for t in TOOLS]
    + ["sleep", "lookup", "search", "compact"]))


def banned_in(text: str) -> list[str]:
    """The banned words in `text`: whole words, any case, as written or with identifiers split at `_`."""
    split = text.replace("_", " ")
    return [w for w in BANNED
            if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)
            or re.search(rf"\b{re.escape(w.replace('_', ' '))}\b", split, re.IGNORECASE)]


@dataclass(frozen=True)
class CodeItem:
    template: str  # one of TEMPLATE_NAMES
    variant: int  # index among the template's slot combinations, in product order
    source: str  # the function text, 4-12 lines, no trailing line break
    sample: tuple  # arguments of one call that must run


@dataclass(frozen=True)
class _Op:
    name: str  # the descriptive function name
    body: str  # `string.Template` text: $f function name, $c constant (or the constant's own keys), neutral names
    sample: tuple


@dataclass(frozen=True)
class _Template:
    name: str
    ops: tuple[_Op, ...]
    constants: tuple  # scalars (-> $c) or dicts of substitutions
    names: dict[str, tuple[str, ...]]  # placeholder -> neutral names, rotated over the variants


_LIST = ("items", "values", "numbers", "data", "xs")
_TEXT = ("text", "s", "line", "phrase", "message")
_ACC = ("result", "out", "acc", "res")
_ELEM = ("x", "v", "item", "el")
_INDEX = ("i", "j", "k", "idx")
_CHAR = ("ch", "char", "letter", "symbol")
_SEEN = ("seen", "counts", "tally")
_BEST = ("best", "top", "chosen")
_KEY = ("key", "name", "label")
_FIRST = ("first", "left", "one")
_SECOND = ("second", "right", "other")
_NUMBER = ("n", "number", "value", "num")

_NUMBERS = ([4, 12, -3, 7, 55, 0, 101],)
_DIGITS = ([3, 1, 4, 1, 5, 9, 2, 6],)
_TWELVE = ([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5],)
_SENTENCE = ("Hello World 2026, Example Text!",)


def _ops(body: str, choices: list[tuple[str, str]], sample: tuple) -> tuple[_Op, ...]:
    """One op per (descriptive name, text put where the body says `@`)."""
    return tuple(_Op(name, body.replace("@", text), sample) for name, text in choices)


_SUM_FILTERED = """\
def $f($a):
    $r = 0
    for $x in $a:
        if $x @ $c:
            $r += $x
    return $r"""

_COUNT_DIVISIBLE = """\
def $f($a):
    $r = 0
    for $x in $a:
        if $x % $c @ 0:
            $r += 1
    return $r"""

_RUNNING = """\
def $f($a):
    $r = []
    $t = $c
    for $x in $a:
        $t = @
        $r.append($t)
    return $r"""

_EXTREME = """\
def $f($a):
    $b = $a[0]
    $p = $c
    for $i, $x in enumerate($a, $c):
        if $x @ $b:
            $b, $p = $x, $i
    return $b, $p"""

_NEIGHBORS = """\
def $f($a):
    $r = []
    for $i in range(len($a) - $c):
        $r.append(@)
    return $r"""

_CLAMP_BOTH = """\
def $f($a):
    $r = []
    for $x in $a:
        if $x < $lo:
            $x = $lo
        elif $x > $hi:
            $x = $hi
        $r.append($x)
    return $r"""

_CLAMP_ONE = """\
def $f($a):
    $r = []
    for $x in $a:
        if $x @:
            $x = @@
        $r.append($x)
    return $r"""

_WINDOWS = """\
def $f($a):
    $r = []
    for $i in range(len($a) - $c + 1):
        $r.append(@($a[$i:$i + $c]))
    return $r"""

_SECOND_EXTREME = """\
def $f($a):
    $u = sorted(set($a))
    if len($u) < 2:
        return $c
    return $u[@]"""

_DEDUPE_FIRST = """\
def $f($a):
    $s = {}
    $r = []
    for $x in $a:
        $s[$x] = $s.get($x, 0) + 1
        if $s[$x] <= $c:
            $r.append($x)
    return $r"""

_DEDUPE_LAST = """\
def $f($a):
    $s = {}
    $r = []
    for $x in reversed($a):
        $s[$x] = $s.get($x, 0) + 1
        if $s[$x] <= $c:
            $r.append($x)
    $r.reverse()
    return $r"""

_CHUNKS = """\
def $f($a):
    $r = []
    for $i in range(0, @, $c):
        $r.append($a[$i:$i + $c])
    return $r"""

_CHUNKS_PADDED = """\
def $f($a):
    $r = []
    for $i in range(0, len($a), $c):
        $p = $a[$i:$i + $c]
        $p = $p + [0] * ($c - len($p))
        $r.append($p)
    return $r"""

_ROTATE = """\
def $f($a):
    if not $a:
        return []
    $k = $c % len($a)
    return @"""

_INTERLEAVE = """\
def $f($a, $b):
    $n = min(len($a), len($b))
    $r = []
    for $i in range(0, $n, $c):
        $r += @[$i:min($i + $c, $n)]
        $r += @@[$i:min($i + $c, $n)]
    return $r"""

_INTERLEAVE_REST = """\
def $f($a, $b):
    $n = min(len($a), len($b))
    $r = []
    for $i in range(0, $n, $c):
        $r += @[$i:min($i + $c, $n)]
        $r += @@[$i:min($i + $c, $n)]
    $r += $a[$n:] + $b[$n:]
    return $r"""

_COUNT_CHARS = """\
def $f($a):
    $r = 0
    for $x in $a[:$c]:
        if @:
            $r += 1
    return $r"""

_REVERSE_ORDER = """\
def $f($a):
    $p = $a.split("$c")
    $p.reverse()
    return "$c".join($p)"""

_REVERSE_EACH = """\
def $f($a):
    $p = $a.split("$c")
    $p = [$w[::-1] for $w in $p]
    return "$c".join($p)"""

_PALINDROME = """\
def $f($a):
    $t = @
    if len($t) < $c:
        return False
    return $t == $t[::-1]"""

_SHIFT = """\
def $f($a):
    $r = ""
    for $x in $a:
        if $x.islower():
            $x = chr((ord($x) - 97 @ $c) % 26 + 97)
        $r += $x
    return $r"""

_RUN_LENGTH = """\
def $f($a):
    $r = []
    for $x in $a:
        if $r and $r[-1][0] == $x and $r[-1][1] < $c:
            $r[-1][1] += 1
        else:
            $r.append([$x, 1])
    return @"""

_MOST_FREQUENT = """\
def $f($a):
    $s = {}
    for $x in $a:
        if $x != "$c":
            $s[$x] = $s.get($x, 0) + 1
    $b = None
    for $x in $s:
        if $b is None or $s[$x] @ $s[$b]:
            $b = $x
    return $b"""

_TALLY = """\
def $f(@):
    $s = {}
    for $x in @@:
        $s[@@@] = $s.get(@@@, 0) + 1
    return {$k: $n for $k, $n in $s.items() if $n >= $c}"""

_INVERT = """\
def $f($a):
    $r = {}
    for $k, $v in $a.items():
        if $v >= $c:
            @
    return $r"""

_MERGE = """\
def $f($a, $b):
    $r = {}
    for $k in set($a) | set($b):
        $r[$k] = @
    return $r"""

_GCD = """\
def $f($n):
    $x, $y = $n, $c
    while $y:
        $x, $y = $y, $x % $y
    return @"""

_DIGIT_FOLD = """\
def $f($n):
    $r = @
    while $n > 0:
        $r @@ $n % $c
        $n //= $c
    return $r"""

_DIGITAL_ROOT = """\
def $f($n):
    while $n >= $c:
        $r = 0
        while $n > 0:
            $r += $n % $c
            $n //= $c
        $n = $r
    return $n"""

_BISECT_INDEX = """\
def $f($a, $t):
    $lo, $hi = 0, min(len($a), $c) - 1
    while $lo <= $hi:
        $m = ($lo + $hi) // 2
        if $a[$m] == $t:
            return $m
        if $a[$m] < $t:
            $lo = $m + 1
        else:
            $hi = $m - 1
    return -1"""

_BISECT_POINT = """\
def $f($a, $t):
    $lo, $hi = 0, min(len($a), $c)
    while $lo < $hi:
        $m = ($lo + $hi) // 2
        if $a[$m] < $t:
            $lo = $m + 1
        else:
            $hi = $m
    return $lo"""


def _swap(body: str, first: str, second: str) -> str:
    return body.replace("@@", second).replace("@", first)


_WORDS = ("the cat and the dog ran to the big red barn",)
_MAPPING = ({"a": 1, "b": 2, "c": 1, "d": 30, "e": 200},)
_TWO_LISTS = ([1, 2, 3, 4, 5, 6, 7], ["a", "b", "c", "d"])
_TWO_TALLIES = ({"a": 2, "b": 5}, {"b": 1, "c": 7})
_SORTED = ([1, 3, 5, 7, 9, 11, 13], 7)

# AP 3.4, in its order: lists of numbers (12), strings (6), dictionaries (3), numbers and control flow (3).
_TEMPLATES: tuple[_Template, ...] = (
    # --- lists of numbers ---------------------------------------------------------------------
    _Template("sum_filtered",
              _ops(_SUM_FILTERED, [("sum_above", ">"), ("sum_below", "<"), ("sum_at_least", ">="), ("sum_at_most", "<=")],
                   _NUMBERS),
              (0, 3, 5, 10, 20, 50, 100), {"a": _LIST, "r": _ACC, "x": _ELEM}),
    _Template("count_divisible",
              _ops(_COUNT_DIVISIBLE, [("count_multiples", "=="), ("count_non_multiples", "!=")], _NUMBERS),
              tuple(range(2, 12)), {"a": _LIST, "r": ("count", "total", "found", "hits"), "x": _ELEM}),
    _Template("running",
              _ops(_RUNNING, [("running_totals", "$t + $x"), ("running_products", "$t * $x"),
                              ("running_maxima", "max($t, $x)")], _DIGITS),
              (0, 1, 2, 3, 5, 10, 50, 100), {"a": _LIST, "r": _ACC, "x": _ELEM, "t": ("current", "running", "so_far")}),
    _Template("extreme_with_index",
              _ops(_EXTREME, [("largest_with_index", ">"), ("smallest_with_index", "<"),
                              ("last_largest_with_index", ">="), ("last_smallest_with_index", "<=")], _DIGITS),
              (0, 1, 2, 5, 10, 100), {"a": _LIST, "b": _BEST, "p": ("pos", "where", "at"), "i": _INDEX, "x": _ELEM}),
    _Template("neighbors",
              _ops(_NEIGHBORS, [("neighbor_differences", "$a[$i + $c] - $a[$i]"), ("neighbor_sums", "$a[$i] + $a[$i + $c]"),
                                ("neighbor_products", "$a[$i] * $a[$i + $c]")], _TWELVE),
              tuple(range(1, 9)), {"a": _LIST, "r": _ACC, "i": _INDEX}),
    _Template("clamp",
              (_Op("clamp_values", _CLAMP_BOTH, ([-20, -3, 0, 4, 9, 15, 60, 200],)),
               _Op("raise_to_minimum", _swap(_CLAMP_ONE, "< $lo", "$lo"), ([-20, -3, 0, 4, 9, 15, 60, 200],)),
               _Op("cap_at_maximum", _swap(_CLAMP_ONE, "> $hi", "$hi"), ([-20, -3, 0, 4, 9, 15, 60, 200],))),
              tuple({"lo": lo, "hi": hi} for lo, hi in
                    ((0, 10), (1, 9), (-5, 5), (-1, 1), (10, 20), (2, 8), (-10, 100), (5, 50))),
              {"a": _LIST, "r": _ACC, "x": _ELEM}),
    _Template("windows",
              _ops(_WINDOWS, [("window_sums", "sum"), ("window_maxima", "max"), ("window_minima", "min")], _TWELVE),
              tuple(range(2, 10)), {"a": _LIST, "r": _ACC, "i": _INDEX}),
    _Template("second_extreme",
              _ops(_SECOND_EXTREME, [("second_largest", "-2"), ("second_smallest", "1")], ([5, 3, 9, 3, 7, 9, 1],)),
              (-1, 0, 1, 10, -10, 50, 100, -100, 999, -999), {"a": _LIST, "u": ("distinct", "unique", "ordered")}),
    _Template("dedupe",
              (_Op("keep_first_copies", _DEDUPE_FIRST, ([1, 2, 1, 3, 2, 1, 4, 1],)),
               _Op("keep_last_copies", _DEDUPE_LAST, ([1, 2, 1, 3, 2, 1, 4, 1],))),
              tuple(range(1, 11)), {"a": _LIST, "s": _SEEN, "r": _ACC, "x": _ELEM}),
    _Template("chunks",
              (_Op("split_into_chunks", _CHUNKS.replace("@", "len($a)"), ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],)),
               _Op("split_into_full_chunks", _CHUNKS.replace("@", "len($a) - $c + 1"), ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],)),
               _Op("split_into_padded_chunks", _CHUNKS_PADDED, ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],))),
              tuple(range(2, 10)), {"a": _LIST, "r": _ACC, "i": _INDEX, "p": ("part", "piece", "chunk")}),
    _Template("rotate",
              _ops(_ROTATE, [("rotate_left", "$a[$k:] + $a[:$k]"), ("rotate_right", "$a[-$k:] + $a[:-$k]")],
                   ([1, 2, 3, 4, 5, 6, 7],)),
              tuple(range(1, 11)), {"a": _LIST, "k": ("shift", "offset", "amount")}),
    _Template("interleave",
              (_Op("interleave_lists", _swap(_INTERLEAVE, "$a", "$b"), _TWO_LISTS),
               _Op("interleave_second_first", _swap(_INTERLEAVE, "$b", "$a"), _TWO_LISTS),
               _Op("interleave_with_rest", _swap(_INTERLEAVE_REST, "$a", "$b"), _TWO_LISTS),
               _Op("interleave_second_first_with_rest", _swap(_INTERLEAVE_REST, "$b", "$a"), _TWO_LISTS)),
              tuple(range(1, 6)),
              {"a": _FIRST, "b": _SECOND, "n": ("shared", "common", "limit"), "r": _ACC, "i": _INDEX}),
    # --- strings ------------------------------------------------------------------------------
    _Template("count_chars",
              _ops(_COUNT_CHARS, [("count_vowels", '$x.lower() in "aeiou"'),
                                  ("count_consonants", '$x.isalpha() and $x.lower() not in "aeiou"'),
                                  ("count_digits", "$x.isdigit()"), ("count_capitals", "$x.isupper()")], _SENTENCE),
              (5, 10, 20, 30, 50, 100), {"a": _TEXT, "r": ("count", "total", "found", "hits"), "x": _CHAR}),
    _Template("reverse_words",
              (_Op("reverse_word_order", _REVERSE_ORDER, ("one two,three-four_five/six;seven:eight|nine.ten+eleven",)),
               _Op("reverse_each_word", _REVERSE_EACH, ("one two,three-four_five/six;seven:eight|nine.ten+eleven",))),
              (" ", ",", "-", "_", "/", ";", ":", "|", ".", "+"),
              {"a": _TEXT, "p": ("parts", "pieces", "words"), "w": ("w", "token", "unit")}),
    _Template("palindrome",
              _ops(_PALINDROME, [("is_palindrome_ignoring_case", "$a.lower()"),
                                 ("is_palindrome_of_letters", '"".join($x for $x in $a if $x.isalpha())'),
                                 ("is_palindrome_loosely", '"".join($x.lower() for $x in $a if $x.isalpha())')],
                   ("A man, a plan, a canal: Panama",)),
              tuple(range(1, 9)), {"a": _TEXT, "t": ("cleaned", "plain", "kept"), "x": _CHAR}),
    _Template("shift_letters",
              _ops(_SHIFT, [("shift_letters_forward", "+"), ("shift_letters_back", "-")], ("hello, world",)),
              tuple(range(1, 13)), {"a": _TEXT, "r": _ACC, "x": _CHAR}),
    _Template("run_length",
              _ops(_RUN_LENGTH, [("encode_repeats", "$r"),
                                 ("encode_repeats_as_text", '"".join($x + str($n) for $x, $n in $r)'),
                                 ("encode_counts_first", '"".join(str($n) + $x for $x, $n in $r)')], ("aaabccddddde",)),
              (2, 3, 4, 5, 8, 9, 10, 99), {"a": _TEXT, "r": ("runs", "groups", "blocks"), "x": _CHAR, "n": ("n", "count", "size")}),
    _Template("most_frequent_char",
              _ops(_MOST_FREQUENT, [("most_frequent_char", ">"), ("most_frequent_char_last", ">="),
                                    ("least_frequent_char", "<"), ("least_frequent_char_last", "<=")],
                   ("the quick brown fox. jumps, over-the_lazy dog!",)),
              (" ", ".", ",", "-", "_", "!"), {"a": _TEXT, "s": _SEEN, "x": _CHAR, "b": _BEST}),
    # --- dictionaries -------------------------------------------------------------------------
    _Template("tally",
              (_Op("tally_items", _TALLY.replace("@@@", "$x").replace("@@", "$a").replace("@", "$a"),
                   (["a", "b", "a", "c", "a", "b"],)),
               _Op("tally_word_lengths", _TALLY.replace("@@@", "len($x)").replace("@@", "$t.split()").replace("@", "$t"),
                   _WORDS),
               _Op("tally_first_letters", _TALLY.replace("@@@", "$x[0]").replace("@@", "$t.split()").replace("@", "$t"),
                   _WORDS)),
              tuple(range(2, 10)),
              {"a": _LIST, "t": _TEXT, "s": _SEEN, "x": ("x", "w", "item", "el"), "k": _KEY, "n": ("n", "count", "times")}),
    _Template("invert",
              _ops(_INVERT, [("invert_mapping", "$r[$v] = $k"), ("group_keys_by_value", "$r.setdefault($v, []).append($k)")],
                   _MAPPING),
              (0, 1, 2, 3, 4, 5, 10, 20, 50, 100),
              {"a": ("mapping", "table", "pairs", "source"), "r": _ACC, "k": _KEY, "v": ("value", "val", "number")}),
    _Template("merge_tallies",
              _ops(_MERGE, [("merge_by_sum", "$a.get($k, $c) + $b.get($k, $c)"),
                            ("merge_by_maximum", "max($a.get($k, $c), $b.get($k, $c))"),
                            ("merge_by_minimum", "min($a.get($k, $c), $b.get($k, $c))")], _TWO_TALLIES),
              tuple(range(0, 8)), {"a": _FIRST, "b": _SECOND, "r": _ACC, "k": _KEY}),
    # --- numbers and control flow -------------------------------------------------------------
    _Template("gcd_lcm",
              _ops(_GCD, [("gcd_with", "$x"), ("lcm_with", "$n * $c // $x")], (84,)),
              (12, 18, 24, 30, 36, 48, 60, 72, 84, 90), {"n": _NUMBER, "x": ("x", "a", "p"), "y": ("y", "b", "q")}),
    _Template("digits",
              (_Op("digit_sum", _swap(_DIGIT_FOLD, "0", "+="), (9875,)),
               _Op("digit_product", _swap(_DIGIT_FOLD, "1", "*="), (9875,)),
               _Op("digital_root", _DIGITAL_ROOT, (9875,))),
              (2, 3, 5, 7, 8, 9, 10, 16), {"n": _NUMBER, "r": ("total", "result", "acc", "res")}),
    _Template("bisect",
              (_Op("find_sorted_index", _BISECT_INDEX, _SORTED), _Op("find_insertion_point", _BISECT_POINT, _SORTED)),
              (8, 10, 16, 20, 32, 50, 64, 100, 128, 256),
              {"a": _LIST, "t": ("target", "wanted", "goal"), "lo": ("lo", "low", "start"), "hi": ("hi", "high", "end"),
               "m": ("mid", "m", "middle")}),
)

TEMPLATE_NAMES: tuple[str, ...] = tuple(t.name for t in _TEMPLATES)
GROUPS: dict[str, tuple[str, ...]] = {"lists": TEMPLATE_NAMES[:12], "strings": TEMPLATE_NAMES[12:18],
                                      "dictionaries": TEMPLATE_NAMES[18:21], "numbers": TEMPLATE_NAMES[21:]}


def slot_counts() -> dict[str, tuple[int, int]]:
    """Per template: (operator or direction choices, constants)."""
    return {t.name: (len(t.ops), len(t.constants)) for t in _TEMPLATES}


def kept_variants(n: int, template_index: int) -> list[int]:
    """The variants of a template that enter the pool: all `n`, or a seeded sample of MAX_VARIANTS
    (generator [SEED, 9, template index]), in product order."""
    if n <= MAX_VARIANTS:
        return list(range(n))
    order = np.random.default_rng([SEED, 9, template_index]).permutation(n)[:MAX_VARIANTS]
    return sorted(int(v) for v in order)


def _render(template: _Template, variant: int, op: _Op, constant, opaque: bool) -> str:
    body_index = variant // 2  # the two name styles of one (operator, constant) pair are neighbors
    fields = dict(constant) if isinstance(constant, dict) else {"c": constant}
    fields["f"] = OPAQUE[body_index % len(OPAQUE)] if opaque else op.name
    for placeholder, choices in template.names.items():
        fields[placeholder] = choices[variant % len(choices)]
    return Template(op.body).substitute(fields)


@functools.lru_cache(maxsize=1)
def _functions() -> tuple[CodeItem, ...]:
    out = []
    for index, template in enumerate(_TEMPLATES):
        combos = list(itertools.product(template.ops, template.constants, (False, True)))
        for variant in kept_variants(len(combos), index):
            op, constant, opaque = combos[variant]
            out.append(CodeItem(template.name, variant, _render(template, variant, op, constant, opaque), op.sample))
    return tuple(out)


def functions() -> list[CodeItem]:
    """The pool: template by template (AP 3.4 order), variants in product order."""
    return list(_functions())


def prompt(item: CodeItem, request: int) -> str:
    """The user message of a code task: request wording `request` (0-7), then the function."""
    return f"{REQUESTS[request]}\n\n```python\n{item.source}\n```"
