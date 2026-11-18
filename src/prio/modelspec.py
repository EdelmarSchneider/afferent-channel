"""Load a base model and resolve the special token ids the rest of the code relies on."""

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FIXED_DATE = "16 November 2026"  # any constant does
# Model families (start of the model id) that write most of their tool calls as a bare JSON object,
# without the <tool_call> tags their own template asks for. `reactions.parse_tool_calls` reads both
# forms for them. Empty: Qwen3 always writes the tags.
BARE_CALL_FAMILIES: tuple[str, ...] = ()


def frozen_chat_template(tok) -> str | None:
    """The tokenizer's chat template with a fixed date where it writes today's (some templates put the
    date in the system block: the same prompt would tokenize differently from day to day, and stored
    prompt ids would stop matching rebuilt ones). None for a template without a date: the tokenizer's
    own (Qwen3)."""
    template = getattr(tok, "chat_template", None)
    if not isinstance(template, str) or "strftime_now" not in template:
        return None
    return re.sub(r"strftime_now\([^)]*\)", f'"{FIXED_DATE}"', template)


def template_kwargs(tok) -> dict:
    """Extra arguments for `apply_chat_template`: the frozen template when there is one, else nothing
    (the call a tokenizer without a date always got)."""
    frozen = frozen_chat_template(tok)
    return {"chat_template": frozen} if frozen else {}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    think_open: int
    think_close: int
    tool_open: int
    tool_close: int
    eos_ids: frozenset[int]
    bare_tool_calls: bool = False  # the model also writes calls without the tags


def resolve_spec(model_id: str, tok) -> ModelSpec:
    def one(token: str) -> int:
        ids = tok.convert_tokens_to_ids(token)
        if ids is None or ids == tok.unk_token_id:
            raise ValueError(f"{model_id}: {token!r} is not a single known token")
        return ids

    eos = {tok.eos_token_id, one("<|im_end|>")}
    return ModelSpec(
        model_id=model_id,
        think_open=one("<think>"),
        think_close=one("</think>"),
        tool_open=one("<tool_call>"),
        tool_close=one("</tool_call>"),
        eos_ids=frozenset(i for i in eos if i is not None),
        bare_tool_calls=model_id.startswith(BARE_CALL_FAMILIES),
    )


def load_base(model_id: str, dtype=torch.bfloat16, device: str = "cuda"):
    """Return (model, tokenizer, ModelSpec). The model is in eval mode with frozen weights."""
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, resolve_spec(model_id, tok)
