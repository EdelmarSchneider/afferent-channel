import numpy as np
import pytest
import torch

from helpers import arm
from prio.channel import ChannelModel
from prio.generate import generate, generate_batch


@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_cached_loop_matches_full_recompute(tiny_model, delivery):
    m = ChannelModel(tiny_model, 6, delivery=delivery, lane_width=8)
    arm(m)
    g = torch.Generator().manual_seed(0)
    p = torch.randint(0, 100, (1, 5), generator=g)
    ps = torch.rand(1, 5, 6, generator=g)
    table = torch.rand(20, 6, generator=g)
    r = generate(m, p, ps, lambda t, _: table[t].tolist(), 20, eos_ids=set())
    ids, sigs = p.clone(), ps.clone()  # reference: full forward at every step, no cache
    for t in range(20):
        nxt = m(ids, sigs).logits[0, -1].argmax().item()
        assert nxt == r.token_ids[t]
        ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
        sigs = torch.cat([sigs, table[t].view(1, 1, 6)], 1)
    assert np.allclose(r.signals, table.numpy()) and r.stop_reason == "max_new_tokens"


def test_signals_steer_the_cached_loop(tiny_model):
    m = ChannelModel(tiny_model, 6)
    arm(m, std=3.0)
    p, ps = torch.randint(0, 100, (1, 5), generator=torch.Generator().manual_seed(0)), torch.zeros(1, 5, 6)
    low = generate(m, p, ps, lambda t, _: [0.0] * 6, 20, eos_ids=set())
    high = generate(m, p, ps, lambda t, _: [1.0] * 6, 20, eos_ids=set())
    assert low.token_ids != high.token_ids


def test_stops_on_eos(tiny_model):
    m = ChannelModel(tiny_model, 6)
    p, ps = torch.randint(0, 100, (1, 5), generator=torch.Generator().manual_seed(0)), torch.zeros(1, 5, 6)
    first = generate(m, p, ps, lambda t, _: [0.0] * 6, 5, eos_ids=set()).token_ids[0]
    r = generate(m, p, ps, lambda t, _: [0.0] * 6, 5, eos_ids={first})
    assert r.token_ids == [first] and r.stop_reason == "eos" and r.signals.shape == (1, 6)


def test_signal_fn_sees_generated_ids(tiny_model):
    m = ChannelModel(tiny_model, 6)
    seen = []

    def fn(t, generated):
        seen.append((t, len(generated)))
        return [0.0] * 6

    generate(m, torch.randint(0, 100, (1, 3)), torch.zeros(1, 3, 6), fn, 4, eos_ids=set())
    assert seen == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_sampling_is_seeded(tiny_model):
    m = ChannelModel(tiny_model, 6)
    p, ps = torch.randint(0, 100, (1, 3), generator=torch.Generator().manual_seed(0)), torch.zeros(1, 3, 6)
    f = lambda t, g: [0.0] * 6
    a = generate(m, p, ps, f, 10, eos_ids=set(), temperature=1.0, seed=3)
    b = generate(m, p, ps, f, 10, eos_ids=set(), temperature=1.0, seed=3)
    c = generate(m, p, ps, f, 10, eos_ids=set(), temperature=1.0, seed=4)
    assert a.token_ids == b.token_ids and a.token_ids != c.token_ids


def _three_prompts(seed: int = 0):
    """Prompts of different lengths (left padding up to 7 positions), their signal rows, and a
    signal_fn per sequence that also reads the sequence's own last token."""
    g = torch.Generator().manual_seed(seed)
    lengths = [5, 9, 2]
    prompts = [torch.randint(0, 100, (n,), generator=g).tolist() for n in lengths]
    signals = [torch.rand(n, 6, generator=g) for n in lengths]
    tables = [torch.rand(24, 6, generator=g) for _ in lengths]
    fns = [lambda t, ids, tb=tb: (tb[t] * 0.5 + (ids[-1] % 5) * 0.1).tolist() for tb in tables]
    return prompts, signals, fns


def _sequential(m, prompts, signals, fns, horizons, eos_ids, **kw):
    return [generate(m, torch.tensor([p]), s[None], f, h, eos_ids, **kw)
            for p, s, f, h in zip(prompts, signals, fns, horizons)]


@pytest.mark.parametrize("projector", ["mlp", "levels"])
@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_batch_matches_sequential_generation(tiny_model, delivery, projector):
    # GPT-2 has absolute positions: left padding without the right position ids shifts every token.
    m = ChannelModel(tiny_model, 6, delivery=delivery, lane_width=8, projector=projector)
    arm(m)
    prompts, signals, fns = _three_prompts()
    horizons = [12, 7, 16]
    seq = _sequential(m, prompts, signals, fns, horizons, set())
    bat = generate_batch(m, prompts, signals, fns, horizons, set())
    for a, b in zip(seq, bat):
        assert a.token_ids == b.token_ids and a.stop_reason == b.stop_reason
        assert np.allclose(a.signals, b.signals) and b.harness is None
    assert [len(r.token_ids) for r in bat] == horizons  # each sequence stops at its own max_new_tokens

    # Per-sequence stop on eos: a token one sequence writes early ends that sequence only.
    eos = {seq[1].token_ids[2]}
    seq = _sequential(m, prompts, signals, fns, horizons, eos)
    bat = generate_batch(m, prompts, signals, fns, horizons, eos)
    assert [(r.token_ids, r.stop_reason) for r in seq] == [(r.token_ids, r.stop_reason) for r in bat]
    assert bat[1].stop_reason == "eos" and len(bat[1].token_ids) <= 3
    assert all(np.allclose(a.signals, b.signals) for a, b in zip(seq, bat))


def test_batch_takes_one_max_new_tokens_for_all_and_per_sequence_seeds(tiny_model):
    m = ChannelModel(tiny_model, 6)
    prompts, signals, fns = _three_prompts(1)
    bat = generate_batch(m, prompts, signals, fns, 6, set(), temperature=1.0, seed=[3, 4, 5])
    seq = [generate(m, torch.tensor([p]), s[None], f, 6, set(), temperature=1.0, seed=k)
           for p, s, f, k in zip(prompts, signals, fns, [3, 4, 5])]
    assert [r.token_ids for r in bat] == [r.token_ids for r in seq]
    assert generate_batch(m, [], [], [], 6, set()) == []


def test_banned_ids_are_never_written(tiny_model):
    # Row S2: the insert delivery's INSERT_ID must never come out of the model.
    m = ChannelModel(tiny_model, 6)
    prompts, signals, fns = _three_prompts(2)
    free = _sequential(m, prompts, signals, fns, [20] * 3, set())
    banned = {r.token_ids[0] for r in free} | set(free[0].token_ids[1:4])
    greedy = _sequential(m, prompts, signals, fns, [20] * 3, set(), banned_ids=banned)
    sampled = _sequential(m, prompts, signals, fns, [20] * 3, set(), banned_ids=banned, temperature=1.0, seed=0)
    batched = generate_batch(m, prompts, signals, fns, 20, set(), banned_ids=banned)
    for r in greedy + sampled + batched:
        assert not banned & set(r.token_ids)
    assert [r.token_ids for r in greedy] == [r.token_ids for r in batched]
    assert all(a.token_ids[0] != b.token_ids[0] for a, b in zip(free, greedy))


def test_the_insert_delivery_never_writes_its_insert_id(tiny_model):
    # The insert delivery's INSERT_ID is an embedding row the harness writes; generation bans it
    # by itself, whoever calls it. The output head is pushed hard toward that row.
    insert = 99
    tiny_model.lm_head.register_forward_hook(
        lambda module, args, out: out.index_fill(-1, torch.tensor([insert], device=out.device), 1e4))
    plain = ChannelModel(tiny_model, 6)
    m = ChannelModel(tiny_model, 6, delivery="insert", insert_id=insert)
    prompts, signals, fns = _three_prompts(3)
    assert set(_sequential(plain, prompts, signals, fns, [5] * 3, set())[0].token_ids) == {insert}  # the push works
    runs = (_sequential(m, prompts, signals, fns, [8] * 3, set())
            + _sequential(m, prompts, signals, fns, [8] * 3, set(), temperature=1.0, seed=0)
            + _sequential(m, prompts, signals, fns, [8] * 3, set(), banned_ids={5})
            + generate_batch(m, prompts, signals, fns, 8, set()))
    assert all(insert not in r.token_ids for r in runs)
    assert all(5 not in r.token_ids for r in runs[6:9])  # the caller's own bans still hold


def _segment_model(tiny_model):
    """A segment-delivery model whose segment vector (100 everywhere) stands out of any embedding,
    and the list that receives, per forward pass of the base, which positions carry it."""
    m = ChannelModel(tiny_model, 6, delivery="segment")
    with torch.no_grad():
        m.segment.fill_(100.0)
    passes = []
    tiny_model.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: passes.append((kwargs["inputs_embeds"].abs().amax(-1) > 50).tolist()),
        with_kwargs=True)
    return m, passes


def test_harness_tokens_of_the_prompt_and_injected_blocks_get_the_segment_vector(tiny_model):
    # Program R6: the segment delivery marks what the harness wrote, at evaluation as in training.
    m, passes = _segment_model(tiny_model)
    p, ps = torch.randint(0, 100, (1, 6), generator=torch.Generator().manual_seed(0)), torch.zeros(1, 6, 6)
    f = lambda t, _: [0.0] * 6
    mask = [False, False, True, True, False, False]
    r = generate(m, p, ps, f, 4, set(), prompt_harness=mask, inject=lambda step, g: [7, 8] if step == 2 else None)
    # Prompt pass: the mask; sampled tokens: never; the injected block: every token.
    assert passes == [[mask], [[False]], [[False]], [[True, True]], [[False]]]
    assert r.harness == [False, False, True, True, False, False]
    passes.clear()
    generate(m, p, ps, f, 4, set())
    assert passes[0] == [[False] * 6] and not any(any(row) for one in passes for row in one)
    with pytest.raises(ValueError, match="harness"):
        generate(m, p, ps, f, 4, set(), prompt_harness=mask[:5])
    # Other deliveries never read the mask.
    s = ChannelModel(tiny_model, 6)
    assert generate(s, p, ps, f, 6, set(), prompt_harness=mask).token_ids == generate(s, p, ps, f, 6, set()).token_ids


def test_batched_generation_marks_each_prompts_harness_tokens(tiny_model):
    m, passes = _segment_model(tiny_model)
    prompts, signals, fns = _three_prompts(4)  # lengths 5, 9, 2
    masks = [[False, True, False, False, True], None, [True, False]]
    bat = generate_batch(m, prompts, signals, fns, [4, 3, 5], set(), prompt_harness=masks)
    assert passes[0] == [[False] * 4 + masks[0], [False] * 9, [False] * 7 + masks[2]]  # left padding: never marked
    assert not any(any(row) for one in passes[1:] for row in one)
    seq = [generate(m, torch.tensor([p]), s[None], f, h, set(), prompt_harness=k)
           for p, s, f, h, k in zip(prompts, signals, fns, [4, 3, 5], masks)]
    assert [r.token_ids for r in bat] == [r.token_ids for r in seq]
    with pytest.raises(ValueError, match="harness"):
        generate_batch(m, prompts, signals, fns, 3, set(), prompt_harness=masks[:2])


class _Recorder:
    """Calls the model and keeps (input length, logits at the last position) of every pass."""

    def __init__(self, model):
        self.model, self.seen = model, []

    def __call__(self, ids, signals, **kw):
        out = self.model(ids, signals, **kw)
        self.seen.append((ids.shape[1], out.logits[0, -1].clone()))
        return out


@pytest.mark.parametrize("delivery", ["sum", "wide"])
def test_injected_block_equals_the_same_tokens_in_the_prompt(tiny_model, delivery):
    # Row S1: the harness writes a status line during generation; it goes into the KV cache
    # exactly as if it had been part of the prompt.
    m = ChannelModel(tiny_model, 6, delivery=delivery, lane_width=8, projector="levels")
    arm(m)
    g = torch.Generator().manual_seed(0)
    p, ps, table = torch.randint(0, 100, (1, 5), generator=g), torch.rand(1, 5, 6, generator=g), torch.rand(20, 6, generator=g)
    fn = lambda t, _: table[t].tolist()
    block, asked = [7, 8, 9], []

    def inject(step, generated):
        asked.append((step, len(generated)))
        return block if step == 3 else None

    # The block's ids are banned for the model (as INSERT_ID will be) and one is an eos id:
    # injected tokens never stop the run.
    rec, quiet = _Recorder(m), dict(eos_ids={block[0]}, banned_ids=set(block))
    r = generate(rec, p, ps, fn, 10, inject=inject, **quiet)
    assert len(r.token_ids) == 13 and r.token_ids[3:6] == block  # max_new_tokens counts sampled tokens only
    assert r.harness == [False] * 3 + [True] * 3 + [False] * 7 and r.stop_reason == "max_new_tokens"
    assert np.allclose(r.signals[3:6], r.signals[2]) and np.allclose(r.signals[6], table[6].numpy())
    assert asked[:5] == [(0, 0), (1, 1), (2, 2), (3, 3), (7, 7)]

    ids = torch.cat([p, torch.tensor([r.token_ids[:6]])], 1)
    sigs = torch.cat([ps, torch.from_numpy(r.signals[:6])[None]], 1)
    after = next(logits for length, logits in rec.seen if length == len(block))
    assert torch.allclose(after, m(ids, sigs).logits[0, -1], atol=1e-5)
    rest = generate(m, ids, sigs, lambda t, _: table[t + 6].tolist(), 7, **quiet)
    assert r.token_ids[6:] == rest.token_ids


def test_without_inject_harness_is_none_and_an_unused_inject_marks_nothing(tiny_model):
    m = ChannelModel(tiny_model, 6)
    p, ps = torch.randint(0, 100, (1, 4), generator=torch.Generator().manual_seed(0)), torch.zeros(1, 4, 6)
    f = lambda t, _: [0.0] * 6
    plain = generate(m, p, ps, f, 5, eos_ids=set())
    quiet = generate(m, p, ps, f, 5, eos_ids=set(), inject=lambda step, generated: None)
    assert plain.harness is None and quiet.harness == [False] * 5 and quiet.token_ids == plain.token_ids
    first = generate(m, p, ps, f, 3, eos_ids=set(), inject=lambda step, generated: [11, 12] if step == 0 else None)
    assert first.token_ids[:2] == [11, 12] and first.harness[:2] == [True, True]
    assert np.allclose(first.signals[:2], 0.0)  # before any sampled token: the last prompt row
