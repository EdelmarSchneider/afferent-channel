"""Generation loop that feeds the current signal vector with every token, using the KV cache.

`generate`: one sequence. `generate_batch`: several sequences at once (left padding, attention
mask, position ids), each with its own signal_fn, budget and stop; same tokens as `generate`
one by one (greedy), at a fraction of the GPU time per sequence.
"""

from dataclasses import dataclass
from typing import Callable, Collection, Literal, Sequence

import numpy as np
import torch

from prio.channel import ChannelModel

# (step, generated ids so far, including the token of this step) -> the N signal values
# that token receives when it enters the model.
SignalFn = Callable[[int, list[int]], Sequence[float]]
# (step, generated ids so far) -> token ids the harness writes into the stream now, or None.
# `step` = index in the generated sequence where the block would start (= len(generated)).
InjectFn = Callable[[int, list[int]], Sequence[int] | None]


@dataclass
class GenResult:
    token_ids: list[int]
    signals: np.ndarray  # float32 [len(token_ids), N]
    stop_reason: Literal["eos", "max_new_tokens"]
    harness: list[bool] | None = None  # per token: written by `inject`, not sampled; None without `inject`


def _pick(logits: torch.Tensor, temperature: float, top_p: float, rng: torch.Generator) -> int:
    if temperature <= 0.0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float().cpu() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(descending=True)
        keep = sorted_probs.cumsum(0) - sorted_probs < top_p  # always keeps the top token
        probs = torch.zeros_like(probs).scatter(0, order[keep], sorted_probs[keep])
    return int(torch.multinomial(probs, 1, generator=rng))


def _banned(model, banned_ids: Collection[int], device) -> torch.Tensor | None:
    """The caller's banned ids plus the model's INSERT_ID (delivery "insert"): an embedding row only
    the harness writes, never a token the model may produce."""
    ids = set(banned_ids)
    if getattr(model, "insert_id", None) is not None:
        ids.add(int(model.insert_id))
    return torch.tensor(sorted(ids), dtype=torch.long, device=device) if ids else None


def _mask(logits: torch.Tensor, banned: torch.Tensor | None) -> torch.Tensor:
    """Logits [..., V] with the banned ids at -inf: never picked, greedy or sampled."""
    if banned is None:
        return logits
    logits = logits.clone()
    logits[..., banned] = float("-inf")
    return logits


@torch.inference_mode()
def generate(
    model: ChannelModel,
    prompt_ids: torch.Tensor,  # [1, T0]; a forced prefix is simply part of the prompt
    prompt_signals: torch.Tensor,  # [1, T0, N]
    signal_fn: SignalFn,
    max_new_tokens: int,
    eos_ids: Collection[int],
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    banned_ids: Collection[int] = (),
    inject: InjectFn | None = None,
    prompt_harness: Sequence[bool] | torch.Tensor | None = None,  # [T0]: the prompt tokens the harness wrote
) -> GenResult:
    """`banned_ids`: never produced (their logits are masked); the model's `insert_id`, if any, is
    always banned. `inject`: called before every sampled token; the ids it returns go through the
    model as one block into the KV cache (no sampling), each with the current row (the last
    token's row, or the last prompt row), and are marked in `GenResult.harness`. Injected tokens
    count in `step` and in `generated` but not against `max_new_tokens` and never stop the run.

    `harness_mask` of the model's passes (read by the segment delivery): `prompt_harness` in the
    prompt pass (None: no harness token), true on every injected block, false on sampled tokens."""
    device = prompt_ids.device
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    banned = _banned(model, banned_ids, device)

    marked = {}
    if prompt_harness is not None:
        mask = torch.as_tensor(prompt_harness, dtype=torch.bool).reshape(1, -1).to(device)
        if mask.shape != prompt_ids.shape:
            raise ValueError(f"prompt_harness of {mask.shape[1]} tokens for a prompt of {prompt_ids.shape[1]}")
        marked = {"harness_mask": mask}
    out = model(prompt_ids, prompt_signals, use_cache=True, **marked)
    tokens: list[int] = []
    rows: list[list[float]] = []
    harness: list[bool] | None = None if inject is None else []
    stop: Literal["eos", "max_new_tokens"] = "max_new_tokens"
    for step in range(max_new_tokens):  # step: sampled tokens so far
        if inject is not None:
            block = inject(len(tokens), tokens)
            if block is not None and len(block) > 0:
                block = [int(t) for t in block]
                row = rows[-1] if rows else [float(v) for v in prompt_signals[0, -1]]
                out = model(
                    torch.tensor([block], device=device),
                    torch.tensor([[row] * len(block)], dtype=torch.float32, device=device),
                    past_key_values=out.past_key_values,
                    use_cache=True,
                    harness_mask=torch.ones(1, len(block), dtype=torch.bool, device=device),
                )
                tokens += block
                rows += [row] * len(block)
                harness += [True] * len(block)
        token = _pick(_mask(out.logits[0, -1], banned), temperature, top_p, rng)
        tokens.append(token)
        if harness is not None:
            harness.append(False)
        rows.append([float(v) for v in signal_fn(len(tokens) - 1, tokens)])
        if token in eos_ids:
            stop = "eos"
            break
        if step + 1 < max_new_tokens:
            out = model(
                torch.tensor([[token]], device=device),
                torch.tensor([[rows[-1]]], dtype=torch.float32, device=device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    n_signals = prompt_signals.shape[-1]
    return GenResult(tokens, np.asarray(rows, dtype=np.float32).reshape(len(tokens), n_signals), stop, harness)


@torch.inference_mode()
def generate_batch(
    model: ChannelModel,
    prompts: Sequence[Sequence[int]],  # B prompts (lists or 1-D tensors), any lengths
    prompt_signals: Sequence,  # B arrays / tensors [T_i, N]
    signal_fns: Sequence[SignalFn],
    max_new_tokens: int | Sequence[int],  # one for all, or one per sequence
    eos_ids: Collection[int],
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | Sequence[int | None] | None = None,  # one per sequence: same draws as `generate` with it
    banned_ids: Collection[int] = (),
    prompt_harness: Sequence[Sequence[bool] | None] | None = None,  # per prompt, as in `generate`
) -> list[GenResult]:
    """`generate` for several sequences at once: left padding, attention mask and position ids
    from the mask (GPT-2's positions are absolute), one shared KV cache. Each sequence gets its
    own signal_fn rows and stops on eos or at its own max_new_tokens; a finished sequence leaves
    the batch (its cache rows are dropped). The prompt pass computes the last position's logits
    only (full-vocabulary logits of B long prompts would not fit). `prompt_harness`: one mask per
    prompt (or None for a prompt without harness tokens), in the prompt pass only; padding and
    sampled tokens are never marked."""
    n = len(prompts)
    if not len(prompt_signals) == len(signal_fns) == n:
        raise ValueError(f"{n} prompts, {len(prompt_signals)} signal blocks, {len(signal_fns)} signal_fns")
    if prompt_harness is not None and len(prompt_harness) != n:
        raise ValueError(f"{n} prompts, {len(prompt_harness)} prompt_harness masks")
    if n == 0:
        return []
    budgets = [max_new_tokens] * n if isinstance(max_new_tokens, int) else [int(b) for b in max_new_tokens]
    seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed] * n
    if len(budgets) != n or len(seeds) != n:
        raise ValueError(f"{n} prompts, {len(budgets)} budgets, {len(seeds)} seeds")
    device = model.base.get_input_embeddings().weight.device
    prompts = [[int(t) for t in p] for p in prompts]
    # The caller's dtype is kept, as `generate` keeps it: the channel casts to its own.
    blocks = [torch.as_tensor(s).cpu() for s in prompt_signals]
    n_signals = blocks[0].shape[-1]
    for p, s in zip(prompts, blocks):
        if len(p) == 0 or s.shape != (len(p), n_signals):
            raise ValueError(f"prompt of {len(p)} tokens with signals {tuple(s.shape)}")
    rngs = []
    for k in seeds:
        rngs.append(torch.Generator())
        if k is not None:
            rngs[-1].manual_seed(k)
    banned = _banned(model, banned_ids, device)
    tokens: list[list[int]] = [[] for _ in range(n)]
    rows: list[list[list[float]]] = [[] for _ in range(n)]
    stops: list[Literal["eos", "max_new_tokens"]] = ["max_new_tokens"] * n

    active = [i for i in range(n) if budgets[i] > 0]  # batch row r holds sequence active[r]
    if active:
        width = max(len(prompts[i]) for i in active)
        ids = torch.zeros(len(active), width, dtype=torch.long)
        mask = torch.zeros(len(active), width, dtype=torch.long)
        signals = torch.zeros(len(active), width, n_signals, dtype=blocks[active[0]].dtype)
        for r, i in enumerate(active):
            length = len(prompts[i])
            ids[r, width - length:] = torch.tensor(prompts[i])
            mask[r, width - length:] = 1
            signals[r, width - length:] = blocks[i].to(signals.dtype)
        marked = {}
        if prompt_harness is not None and any(prompt_harness[i] is not None for i in active):
            harness = torch.zeros(len(active), width, dtype=torch.bool)
            for r, i in enumerate(active):
                if prompt_harness[i] is not None:
                    row = torch.as_tensor(prompt_harness[i], dtype=torch.bool).reshape(-1)
                    if len(row) != len(prompts[i]):
                        raise ValueError(f"prompt_harness of {len(row)} tokens for a prompt of {len(prompts[i])}")
                    harness[r, width - len(row):] = row
            marked = {"harness_mask": harness.to(device)}
        ids, mask, signals = ids.to(device), mask.to(device), signals.to(device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        out = model(ids, signals, attention_mask=mask, position_ids=positions, use_cache=True, logits_to_keep=1,
                    **marked)
        position = positions[:, -1:]
    while active:
        logits = _mask(out.logits[:, -1], banned)
        greedy = logits.argmax(-1).tolist() if temperature <= 0.0 else None
        keep = []
        for r, i in enumerate(active):
            token = greedy[r] if greedy is not None else _pick(logits[r], temperature, top_p, rngs[i])
            tokens[i].append(token)
            rows[i].append([float(v) for v in signal_fns[i](len(tokens[i]) - 1, tokens[i])])
            if token in eos_ids:
                stops[i] = "eos"
            elif len(tokens[i]) < budgets[i]:
                keep.append(r)
        if not keep:
            break
        if len(keep) < len(active):
            index = torch.tensor(keep, device=device)
            out.past_key_values.batch_select_indices(index)
            mask, position = mask[index], position[index]
            active = [active[r] for r in keep]
        mask = torch.cat([mask, mask.new_ones(len(active), 1)], dim=-1)
        position = position + 1
        out = model(
            torch.tensor([[tokens[i][-1]] for i in active], device=device),
            torch.tensor([[rows[i][-1]] for i in active], dtype=torch.float32, device=device),
            attention_mask=mask,
            position_ids=position,
            past_key_values=out.past_key_values,
            use_cache=True,
        )
    return [GenResult(tokens[i], np.asarray(rows[i], dtype=np.float32).reshape(len(tokens[i]), n_signals), stops[i])
            for i in range(n)]
