"""Shared helpers of the R12 tests: a chat tokenizer over whitespace words whose single tokens
decode with a leading space (so `teacher.valid_cut_points` accepts every word start), a scripted
sampler of base runs, and a tokenizer over text pieces for the onset index."""

from types import SimpleNamespace

from helpers import CALL, FAKE_MS, FakeTokenizer

MS = FAKE_MS


class R12Tok(FakeTokenizer):
    """FakeTokenizer (same special ids as FAKE_TOK, so FAKE_MS applies) with a chat template."""

    pad_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self.encode(text))

    def decode(self, ids, **_):
        return " " + super().decode(ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, enable_thinking=True,
                            tokenize=False):
        return (f"<|im_start|>system {messages[0]['content']} <|im_end|> "
                f"<|im_start|>user {messages[1]['content']} <|im_end|> <|im_start|>assistant")


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


class ScriptedRuns:
    """Stand-in for the base-run sampler: one run per prompt, its shape decided by the prompt alone
    (`style_of`), so a resumed build gets the same runs. Styles:
    "ok": 40 thinking words, 40 answer words, for a tool task then the call of its tool;
    "unfinished": the turn does not end; "reaction_tool": the model calls compact_context on its own;
    "no_position": too short for a cut (tool tasks: the tool is never called); "too_long": over 2,048 tokens.
    """

    STYLES = ("unfinished", "reaction_tool", "no_position", "too_long")

    def __init__(self, tok, every: int = 12):
        self.tok, self.every, self.calls, self.styles = tok, every, 0, {}

    def family(self, prompt: list[int]) -> str:
        user = self.tok.decode(prompt).split("<|im_start|>user")[1]
        return "web" if "web_search" in user else "heavy" if "run_heavy" in user else "text"

    def style_of(self, prompt: list[int]) -> str:
        slot = sum(prompt) % self.every
        return self.STYLES[slot] if slot < len(self.STYLES) else "ok"

    def run(self, prompt: list[int]) -> tuple[list[int], str]:
        family, style = self.family(prompt), self.style_of(prompt)
        self.styles[tuple(prompt)] = style
        call = {"web": CALL % ("web_search", '{"query":"q"}'), "heavy": CALL % ("run_heavy", '{"job":"j"}'), "text": ""}[family]
        if style == "unfinished":
            return self.tok.encode("<think> " + words("t", 40)), "max_new_tokens"
        if style == "reaction_tool":
            text = f"<think> {words('t', 40)} </think> {words('a', 40)} {CALL % ('compact_context', '{}')} <eos>"
        elif style == "no_position":
            text = f"<think> {words('t', 40)} </think> {words('a', 40)} <eos>" if call else "<think> t0 t1 </think> a0 a1 <eos>"
        elif style == "too_long":
            text = f"<think> {words('t', 40)} </think> {words('a', 2100)} {call} <eos>"
        else:
            text = f"<think> {words('t', 40)} </think> {words('a', 40)} {call} <eos>"
        return self.tok.encode(text), "eos"

    def __call__(self, prompts: list[list[int]]) -> list[tuple[list[int], str]]:
        self.calls += 1
        return [self.run(p) for p in prompts]


class PieceTok:
    """A tokenizer over given text pieces: `ids(...)` registers pieces, `decode` joins them. For
    tests that need line breaks and whitespace-only tokens as the real tokenizer has them."""

    eos_token_id = FAKE_MS.eos_ids and next(iter(FAKE_MS.eos_ids))
    pad_token_id = None

    def __init__(self):
        self.pieces = ["<unk>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<eos>"]
        self.index = {p: i for i, p in enumerate(self.pieces)}

    def ids(self, *pieces: str) -> list[int]:
        out = []
        for piece in pieces:
            if piece not in self.index:
                self.index[piece] = len(self.pieces)
                self.pieces.append(piece)
            out.append(self.index[piece])
        return out

    def decode(self, ids, **_) -> str:
        return "".join(self.pieces[i] for i in ids)
