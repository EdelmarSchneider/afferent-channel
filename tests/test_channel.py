import copy

import pytest
import torch

import prio.channel as channel_module
from helpers import arm
from prio.channel import ChannelModel, SignalProjector


def ids(T=12):
    return torch.randint(0, 100, (2, T), generator=torch.Generator().manual_seed(0))


def sig(T=12, N=6):
    return torch.rand(2, T, N, generator=torch.Generator().manual_seed(1))


def test_projector_shape_and_zero_init():
    out = SignalProjector(6, 32)(sig())
    assert out.shape == (2, 12, 32) and torch.count_nonzero(out) == 0


def test_encoding_turns_one_percent_steps_into_a_thermometer_code():
    p = SignalProjector(1, 32)
    code = lambda x: p.encode(torch.tensor([[[x]]]))[0, 0]
    below, above = code(0.895), code(0.905)  # the bin centers on either side of a 0.90 limit
    assert below.shape == (100,)
    assert torch.allclose(below, torch.tensor([1.0] * 90 + [0.0] * 10), atol=1e-3)
    assert torch.allclose(above, torch.tensor([1.0] * 91 + [0.0] * 9), atol=1e-3)  # exactly one feature flips
    assert torch.allclose(code(0.005), torch.tensor([1.0] + [0.0] * 99), atol=1e-3)
    assert torch.allclose(code(0.995), torch.ones(100), atol=1e-3)
    assert torch.isclose(code(0.9)[90], torch.tensor(0.5), atol=1e-3)  # between centers it interpolates
    assert not torch.allclose(code(0.0), code(1.0))  # the binary signal keeps two distinct codes


def test_raw_scalar_encoding_is_available_for_the_ablation():
    p = SignalProjector(6, 32, bins=0)
    assert p.encode(sig()).shape == (2, 12, 6) and p(sig()).shape == (2, 12, 32)


def test_zero_init_leaves_model_unchanged(tiny_model):
    m = ChannelModel(tiny_model, 6)
    assert torch.allclose(m(ids(), sig()).logits, tiny_model(ids()).logits, atol=1e-5)


def test_signals_change_logits_when_armed(tiny_model):
    m = ChannelModel(tiny_model, 6)
    arm(m)
    assert not torch.allclose(m(ids(), sig()).logits, tiny_model(ids()).logits, atol=1e-3)


def test_disabled_channel_equals_base(tiny_model):
    m = ChannelModel(tiny_model, 6)
    arm(m)
    m.enabled = False
    assert torch.allclose(m(ids(), sig()).logits, tiny_model(ids()).logits, atol=1e-5)


def test_signal_at_position_j_only_affects_positions_from_j(tiny_model):
    m = ChannelModel(tiny_model, 6)
    arm(m)
    a = sig()
    b = a.clone()
    b[:, 7, :] = 1.0 - b[:, 7, :]
    la, lb = m(ids(), a).logits, m(ids(), b).logits
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)
    assert not torch.allclose(la[:, 7:], lb[:, 7:], atol=1e-3)


def test_length_mismatch_raises(tiny_model):
    m = ChannelModel(tiny_model, 6)
    with pytest.raises(ValueError):
        m(ids(12), sig(11))
    with pytest.raises(ValueError):
        m(ids(12), sig(12, N=5))


def test_save_load_roundtrip(tiny_model, tmp_path):
    m = ChannelModel(tiny_model, 6)
    arm(m)
    m.save_channel(tmp_path / "c.pt")
    m2 = ChannelModel(tiny_model, 6)
    m2.load_channel(tmp_path / "c.pt")
    assert torch.allclose(m(ids(), sig()).logits, m2(ids(), sig()).logits, atol=1e-6)


def test_bf16_base_gives_finite_logits(tiny_model):
    m = ChannelModel(tiny_model.to(torch.bfloat16), 6)
    arm(m)
    assert torch.isfinite(m(ids(), sig()).logits.float()).all()


# Delivery "lanes": the signals travel in dimensions of their own, next to the residual stream.
# Every matrix that reads the residual stream gets extra input columns for them.

def lanes(base):
    return ChannelModel(base, 6, delivery="lanes", lane_width=8)


def test_untrained_lanes_leave_the_model_unchanged(tiny_model):
    base = tiny_model(ids()).logits
    m = lanes(tiny_model)
    assert m.projector(sig()).shape == (2, 12, 8) and torch.count_nonzero(m.projector(sig())) > 0
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-5)  # the lanes carry values, nobody reads them yet


def test_every_matrix_that_reads_the_residual_stream_gets_lane_columns(tiny_model):
    m = lanes(tiny_model)  # tiny GPT-2: 2 layers x (c_attn: 32 -> 96, c_fc: 32 -> 128)
    assert sorted(tuple(p.shape) for p in m.readers.values()) == [(96, 8), (96, 8), (128, 8), (128, 8)]
    assert all(torch.count_nonzero(p) == 0 and p.requires_grad for p in m.readers.values())


def test_sum_delivery_has_no_readers(tiny_model):
    m = ChannelModel(tiny_model, 6)
    assert m.delivery == "sum" and len(m.readers) == 0


def test_armed_lanes_change_the_logits(tiny_model):
    base = tiny_model(ids()).logits
    m = lanes(tiny_model)
    arm(m)
    assert not torch.allclose(m(ids(), sig()).logits, base, atol=1e-3)


def test_disabled_lanes_equal_base(tiny_model):
    base = tiny_model(ids()).logits
    m = lanes(tiny_model)
    arm(m)
    m.enabled = False
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-5)


def test_lanes_are_gone_after_a_forward_pass_without_gradients(tiny_model):
    base = tiny_model(ids()).logits
    m = lanes(tiny_model)
    arm(m)
    with torch.no_grad():
        m(ids(), sig())
    assert torch.allclose(tiny_model(ids()[:, :9]).logits, base[:, :9], atol=1e-5)  # the bare model runs as before


def test_lane_signal_at_position_j_only_affects_positions_from_j(tiny_model):
    m = lanes(tiny_model)
    arm(m)
    a = sig()
    b = a.clone()
    b[:, 7, :] = 1.0 - b[:, 7, :]
    la, lb = m(ids(), a).logits, m(ids(), b).logits
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)
    assert not torch.allclose(la[:, 7:], lb[:, 7:], atol=1e-3)


def test_lanes_save_load_roundtrip(tiny_model, tmp_path):
    base = tiny_model(ids()).logits
    m = lanes(copy.deepcopy(tiny_model))
    arm(m)
    m.save_channel(tmp_path / "c.pt")
    m2 = lanes(copy.deepcopy(tiny_model))  # its lane network starts with other random weights
    m2.load_channel(tmp_path / "c.pt")
    assert torch.allclose(m(ids(), sig()).logits, m2(ids(), sig()).logits, atol=1e-6)
    assert not torch.allclose(m2(ids(), sig()).logits, base, atol=1e-3)


def test_lanes_with_a_bf16_base_give_finite_logits(tiny_model):
    m = lanes(tiny_model.to(torch.bfloat16))
    arm(m)
    assert torch.isfinite(m(ids(), sig()).logits.float()).all()


def test_unknown_delivery_raises(tiny_model):
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="widen")


def test_lanes_need_a_matrix_to_read_them(tiny_model, monkeypatch):
    monkeypatch.setattr(channel_module, "READERS", ("no_such_module",))
    with pytest.raises(ValueError):
        lanes(tiny_model)


# Delivery "wide": the vector that travels between the layers has d + k dimensions. The k lanes
# start as the harness wrote them; every layer reads them (readers, as above) and can add to
# them (writers: extra output rows of the attention and feed-forward output matrices).

def wide(base):
    torch.manual_seed(5)  # the same lane network as `lanes_like_wide`
    return ChannelModel(base, 6, delivery="wide", lane_width=8)


def lanes_like_wide(base):
    torch.manual_seed(5)
    return ChannelModel(base, 6, delivery="lanes", lane_width=8)


def test_an_untrained_wide_model_equals_the_base(tiny_model):
    base = tiny_model(ids()).logits
    m = wide(tiny_model)
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-5)


def test_matrices_that_write_the_residual_stream_get_lane_rows(tiny_model):
    m = wide(tiny_model)  # tiny GPT-2: 2 layers x (attention output: 32 -> 32, feed-forward output: 128 -> 32)
    assert sorted(tuple(p.shape) for p in m.writers.values()) == [(8, 32), (8, 32), (8, 128), (8, 128)]
    assert all(torch.count_nonzero(p) == 0 and p.requires_grad for p in m.writers.values())
    assert sorted(tuple(p.shape) for p in m.readers.values()) == [(96, 8), (96, 8), (128, 8), (128, 8)]


def test_only_the_wide_delivery_has_writers(tiny_model):
    assert len(ChannelModel(copy.deepcopy(tiny_model), 6).writers) == 0
    assert len(lanes(tiny_model).writers) == 0


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


def lanes_entering_the_last_norm(m):
    seen = []
    m.base.transformer.ln_f.register_forward_pre_hook(lambda module, args: seen.append(args[0][..., 32:]), prepend=True)
    m(ids(), sig())
    return seen[0]


def test_silent_writers_leave_the_lanes_as_the_harness_wrote_them(tiny_model):
    m = wide(tiny_model)
    arm(m, writers=False)
    assert torch.allclose(lanes_entering_the_last_norm(m), unit(m.projector(sig())), atol=2e-3)  # at unit size


def test_armed_writers_add_to_the_lanes(tiny_model):
    m = wide(tiny_model)
    arm(m)
    assert not torch.allclose(lanes_entering_the_last_norm(m), unit(m.projector(sig())), atol=1e-2)


def test_wide_with_silent_writers_is_the_lane_delivery_at_unit_size(tiny_model):
    # The lanes are read at unit size (their own normalization). Lanes of the "lanes" delivery
    # that already have that size give the same model.
    a = lanes_like_wide(copy.deepcopy(tiny_model))
    b = wide(copy.deepcopy(tiny_model))
    arm(a)
    arm(b, writers=False)
    out = b(ids(), sig()).logits
    real_projector = a.projector.forward
    a.projector.forward = lambda signals: unit(real_projector(signals))
    assert torch.allclose(out, a(ids(), sig()).logits, atol=1e-5)
    assert not torch.allclose(out, tiny_model(ids()).logits, atol=1e-3)


def test_the_size_of_the_harness_lanes_does_not_matter(tiny_model):
    m = wide(tiny_model)
    arm(m)
    before = m(ids(), sig()).logits
    with torch.no_grad():
        m.projector.net[-1].weight *= 10
        m.projector.net[-1].bias *= 10
    assert torch.allclose(m(ids(), sig()).logits, before, atol=1e-4)


def test_a_writer_reads_its_input_at_unit_size():
    # In the real model a few tokens carry activations thousands of times larger than the rest.
    # Unscaled, those tokens alone would decide what the writers learn and what the lanes hold.
    from prio.channel import _Lanes, _write_lanes

    torch.manual_seed(0)
    writers = torch.nn.ParameterDict({"w": torch.nn.Parameter(torch.randn(8, 16))})
    activations = torch.randn(2, 5, 16)
    state = _Lanes()
    state.values = torch.zeros(2, 5, 8)
    _write_lanes(state, writers, "w", None, (activations,), None)
    small = state.written
    _write_lanes(state, writers, "w", None, (activations * 5000,), None)
    assert torch.allclose(small, state.written, atol=1e-4)
    assert torch.allclose(small, torch.nn.functional.linear(unit(activations), writers["w"]), atol=1e-5)


def test_armed_writers_change_what_later_layers_read(tiny_model):
    silent = wide(copy.deepcopy(tiny_model))
    arm(silent, writers=False)
    writing = wide(copy.deepcopy(tiny_model))
    arm(writing)
    assert not torch.allclose(writing(ids(), sig()).logits, silent(ids(), sig()).logits, atol=1e-3)


def test_writers_alone_change_nothing(tiny_model):
    base = tiny_model(ids()).logits
    m = wide(tiny_model)
    for rows in m.writers.values():
        torch.nn.init.normal_(rows, std=0.5)
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-5)  # the lanes change, nobody reads them


def test_disabled_wide_equals_base(tiny_model):
    base = tiny_model(ids()).logits
    m = wide(tiny_model)
    arm(m)
    m.enabled = False
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-5)


def test_the_bare_model_still_runs_after_a_wide_forward_pass(tiny_model):
    base = tiny_model(ids()).logits
    m = wide(tiny_model)
    arm(m)
    m(ids(), sig()).logits.sum().backward()  # a pass with gradients, then one without
    with torch.no_grad():
        m(ids(), sig())
    assert torch.allclose(tiny_model(ids()[:, :9]).logits, base[:, :9], atol=1e-5)


def test_wide_signal_at_position_j_only_affects_positions_from_j(tiny_model):
    m = wide(tiny_model)
    arm(m)
    a = sig()
    b = a.clone()
    b[:, 7, :] = 1.0 - b[:, 7, :]
    la, lb = m(ids(), a).logits, m(ids(), b).logits
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)
    assert not torch.allclose(la[:, 7:], lb[:, 7:], atol=1e-3)


def test_wide_save_load_roundtrip(tiny_model, tmp_path):
    m = wide(copy.deepcopy(tiny_model))
    arm(m)
    m.save_channel(tmp_path / "c.pt")
    torch.manual_seed(9)  # its lane network starts with other random weights
    m2 = ChannelModel(copy.deepcopy(tiny_model), 6, delivery="wide", lane_width=8)
    m2.load_channel(tmp_path / "c.pt")
    assert all(torch.count_nonzero(rows) > 0 for rows in m2.writers.values())
    assert torch.allclose(m(ids(), sig()).logits, m2(ids(), sig()).logits, atol=1e-6)


def test_wide_with_a_bf16_base_gives_finite_logits(tiny_model):
    m = wide(tiny_model.to(torch.bfloat16))
    arm(m)
    assert torch.isfinite(m(ids(), sig()).logits.float()).all()


def test_wide_needs_matrices_to_write_the_lanes(tiny_model, monkeypatch):
    monkeypatch.setattr(channel_module, "WRITERS", ("no_such_module",))
    with pytest.raises(ValueError):
        wide(tiny_model)


# Projector "levels": no hidden layers and no shared features. Every 1% level of every signal has
# its own learned vector (a lookup table, like token embeddings) and a token's channel vector is
# the sum of the vectors of its signals' current levels. Found in phase 1: after four epochs the
# hidden-layer projector answered a binary signal going down with a move of 0.226 and a limit
# crossing (one input feature of 600 flips) with 0.005: the limit was not learned. Here a
# crossing swaps one whole vector for another, and each is trained directly.

def test_levels_projector_sums_one_vector_per_signal():
    p = SignalProjector(2, 32, kind="levels", zero_init=False)
    assert p.encode(torch.tensor([[[0.905, 0.5]]])).shape == (1, 1, 200)
    at = lambda a, b: p(torch.tensor([[[a, b]]]))[0, 0]
    level = lambda signal, k: p.net[0].weight[:, signal * 100 + k]
    assert torch.allclose(at(0.905, 0.305), p.net[0].bias + level(0, 90) + level(1, 30), atol=1e-5)
    assert torch.allclose(at(0.905, 0.5) - at(0.895, 0.5), level(0, 90) - level(0, 89), atol=1e-5)  # rising past 0.90
    assert torch.allclose(at(0.3, 0.095) - at(0.3, 0.105), level(1, 9) - level(1, 10), atol=1e-5)  # falling below 0.10


def test_levels_code_is_one_level_per_signal():
    p = SignalProjector(1, 4, kind="levels")
    code = lambda x: p.encode(torch.tensor([[[x]]]))[0, 0]
    assert code(0.905).round(decimals=3).tolist() == [0.0] * 90 + [1.0] + [0.0] * 9
    assert code(0.0).tolist() == [1.0] + [0.0] * 99 and code(1.0).tolist() == [0.0] * 99 + [1.0]  # the binary signal
    between = code(0.90)  # halfway between the centers of levels 89 and 90
    assert torch.allclose(between[89:91], torch.tensor([0.5, 0.5]), atol=1e-3) and torch.isclose(between.sum(), torch.tensor(1.0))


def test_only_the_vectors_in_use_get_a_gradient():
    # With features shared by many levels (a thermometer code read by one linear layer), hundreds
    # of vectors move together at every step and the input embedding is swamped: the first GPU
    # try of that diverged within 100 batches.
    p = SignalProjector(6, 4, kind="levels", zero_init=False)
    p(sig()[:1, :1].round(decimals=2) + 0.005).sum().backward()
    assert torch.count_nonzero(p.net[0].weight.grad[0]) == 6


def test_the_hidden_layer_projector_keeps_its_thermometer_code():
    code = SignalProjector(1, 32).encode(torch.tensor([[[0.905]]]))[0, 0]
    assert code.shape == (100,) and torch.allclose(code, torch.tensor([1.0] * 91 + [0.0] * 9), atol=1e-3)


def test_levels_projector_starts_at_zero_when_asked():
    out = SignalProjector(6, 32, kind="levels")(sig())
    assert out.shape == (2, 12, 32) and torch.count_nonzero(out) == 0
    assert torch.count_nonzero(SignalProjector(6, 32, kind="levels", zero_init=False)(sig())) > 0


def test_unknown_projector_kind_raises(tiny_model):
    with pytest.raises(ValueError):
        SignalProjector(6, 32, kind="table")
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, projector="table")


@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_an_untrained_channel_with_the_levels_projector_equals_the_base(tiny_model, delivery):
    base = tiny_model(ids()).logits
    m = ChannelModel(tiny_model, 6, delivery=delivery, lane_width=8, projector="levels")
    assert m.projector_kind == "levels" and len(m.projector.net) == 1
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-5)
    arm(m)
    assert not torch.allclose(m(ids(), sig()).logits, base, atol=1e-3)


@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_levels_projector_save_load_roundtrip(tiny_model, tmp_path, delivery):
    m = ChannelModel(copy.deepcopy(tiny_model), 6, delivery=delivery, lane_width=8, projector="levels")
    arm(m)
    m.save_channel(tmp_path / "c.pt")
    torch.manual_seed(9)
    m2 = ChannelModel(copy.deepcopy(tiny_model), 6, delivery=delivery, lane_width=8, projector="levels")
    m2.load_channel(tmp_path / "c.pt")
    assert torch.allclose(m(ids(), sig()).logits, m2(ids(), sig()).logits, atol=1e-6)


def test_the_default_projector_has_hidden_layers(tiny_model):
    m = ChannelModel(tiny_model, 6)  # runs saved before the levels projector existed load like this
    assert m.projector_kind == "mlp" and len(m.projector.net) == 5


# Baseline deliveries (paper baselines S2, B6a, B6b; program R6, R8, R10).

def embeds_seen_by_the_base(m, *args, **kwargs):
    """The input embeddings the wrapped model receives in one forward pass."""
    seen = []
    handle = m.base.register_forward_pre_hook(lambda module, a, kw: seen.append(kw["inputs_embeds"]), with_kwargs=True)
    try:
        m(*args, **kwargs)
    finally:
        handle.remove()
    return seen[0]


@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_the_older_deliveries_save_only_their_old_parts(tiny_model, tmp_path, delivery):
    # Runs saved before the baseline deliveries existed must keep loading: no new key appears.
    m = ChannelModel(tiny_model, 6, delivery=delivery, lane_width=8)
    m.save_channel(tmp_path / "c.pt")
    keys = set(torch.load(tmp_path / "c.pt"))
    assert all(k.startswith(("net.", "readers.", "writers.")) for k in keys)


# Delivery "none": the text baselines. The channel is never applied and nothing of it trains.

def test_no_channel_leaves_the_base_unchanged(tiny_model):
    base = tiny_model(ids()).logits
    m = ChannelModel(tiny_model, 6, delivery="none")
    arm(m)
    assert m.channel_parameters() == []
    assert not any(p.requires_grad for p in m.projector.parameters())
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-6)
    assert m.projector.n_signals == 6  # callers still read the signal count and the device from it


def test_no_channel_rejects_signals_of_the_wrong_shape(tiny_model):
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="none")(ids(12), sig(11))


# Delivery "insert" (B6b, soft notice): the harness inserts one position with the id INSERT_ID;
# that position gets the projection of the signals instead of a token embedding.

def with_inserts(T=12, at=((0, 4), (0, 8), (1, 6)), insert=99):
    x = ids(T) % 90
    for row, col in at:
        x[row, col] = insert
    return x


def test_insert_replaces_exactly_the_insert_positions(tiny_model):
    x, s = with_inserts(), sig()
    m = ChannelModel(tiny_model, 6, delivery="insert", insert_id=99, projector="levels")
    seen = embeds_seen_by_the_base(m, x, s)
    plain = tiny_model.transformer.wte(x)
    where = x == 99
    assert torch.allclose(seen[~where], plain[~where])
    assert torch.allclose(seen[where], m.projector(s)[where], atol=1e-6)
    assert not torch.allclose(seen[where], plain[where], atol=1e-3)


def test_insert_changes_the_logits_from_the_insert_on_only(tiny_model):
    x, s = with_inserts(at=((0, 4), (1, 6))), sig()
    base = tiny_model(x).logits
    m = ChannelModel(tiny_model, 6, delivery="insert", insert_id=99)
    out = m(x, s).logits
    assert torch.allclose(out[0, :4], base[0, :4], atol=1e-5) and torch.allclose(out[1, :6], base[1, :6], atol=1e-5)
    assert not torch.allclose(out[0, 4:], base[0, 4:], atol=1e-3)
    assert torch.allclose(m(ids() % 90, s).logits, tiny_model(ids() % 90).logits, atol=1e-5)  # no insert: the base


def test_insert_reads_the_signals_at_the_insert_positions_only(tiny_model):
    x, s = with_inserts(at=((0, 4), (1, 4))), sig()
    m = ChannelModel(tiny_model, 6, delivery="insert", insert_id=99)
    elsewhere, there = s.clone(), s.clone()
    elsewhere[:, 7] = 1.0 - elsewhere[:, 7]
    there[:, 4] = 1.0 - there[:, 4]
    assert torch.allclose(m(x, elsewhere).logits, m(x, s).logits, atol=1e-6)
    assert not torch.allclose(m(x, there).logits[:, 4:], m(x, s).logits[:, 4:], atol=1e-3)


def test_disabled_insert_feeds_the_plain_row(tiny_model):
    x = with_inserts()
    m = ChannelModel(tiny_model, 6, delivery="insert", insert_id=99)
    m.enabled = False
    assert torch.allclose(m(x, sig()).logits, tiny_model(x).logits, atol=1e-5)


def test_insert_trains_the_projector_only(tiny_model):
    m = ChannelModel(tiny_model, 6, delivery="insert", insert_id=99, projector="levels")
    assert m.channel_parameters() == list(m.projector.parameters())
    m(with_inserts(), sig()).logits.sum().backward()
    assert torch.count_nonzero(m.projector.net[0].weight.grad) > 0


def test_insert_needs_an_unused_row(tiny_model):
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="insert")  # no id given
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="insert", insert_id=100)  # the tiny model has rows 0-99
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="sum", insert_id=99)  # only the insert delivery has one


def test_insert_id_is_the_first_row_past_the_tokenizer(tiny_model):
    from prio.channel import insert_id

    class Tok:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    assert insert_id(tiny_model, Tok(90)) == 90  # Qwen3-1.7B: 151,669 of 151,936 rows
    with pytest.raises(ValueError):
        insert_id(tiny_model, Tok(100))  # every row is a token: no free row


# Delivery "segment" (R6, ISE-style role marking): one learned vector added to the embeddings of
# the tokens the harness inserted. A look-alike text written by anyone else has no vector.

def test_segment_vector_only_inside_harness_spans(tiny_model):
    x, s = ids(), sig()
    base = tiny_model(x).logits
    m = ChannelModel(tiny_model, 6, delivery="segment")
    harness = torch.zeros(2, 12, dtype=torch.bool)
    harness[0, 5:8] = True
    assert torch.allclose(m(x, s, harness_mask=harness).logits, base, atol=1e-6)  # starts at zero
    torch.nn.init.normal_(m.segment, std=0.5)
    seen = embeds_seen_by_the_base(m, x, s, harness_mask=harness)
    plain = tiny_model.transformer.wte(x)
    assert torch.allclose(seen[~harness], plain[~harness])
    assert torch.allclose(seen[harness], plain[harness] + m.segment, atol=1e-6)
    out = m(x, s, harness_mask=harness).logits
    assert torch.allclose(out[1], base[1], atol=1e-5) and torch.allclose(out[0, :5], base[0, :5], atol=1e-5)
    assert not torch.allclose(out[0, 5:], base[0, 5:], atol=1e-3)
    assert torch.allclose(m(x, s).logits, base, atol=1e-5)  # the same tokens outside a harness span: no vector


def test_segment_ignores_the_signals_and_trains_its_vector(tiny_model):
    m = ChannelModel(tiny_model, 6, delivery="segment")
    torch.nn.init.normal_(m.segment, std=0.5)
    harness = torch.zeros(2, 12, dtype=torch.bool)
    harness[:, 3:6] = True
    assert torch.allclose(m(ids(), sig(), harness_mask=harness).logits,
                          m(ids(), 1.0 - sig(), harness_mask=harness).logits, atol=1e-6)
    assert m.channel_parameters() == [m.segment] and m.segment.shape == (32,)
    m(ids(), sig(), harness_mask=harness).logits.sum().backward()
    assert torch.count_nonzero(m.segment.grad) > 0


def test_disabled_segment_equals_base(tiny_model):
    m = ChannelModel(tiny_model, 6, delivery="segment")
    torch.nn.init.normal_(m.segment, std=0.5)
    m.enabled = False
    assert torch.allclose(m(ids(), sig(), harness_mask=torch.ones(2, 12, dtype=torch.bool)).logits,
                          tiny_model(ids()).logits, atol=1e-5)


def test_a_harness_mask_of_the_wrong_shape_raises(tiny_model):
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="segment")(ids(), sig(), harness_mask=torch.ones(2, 11, dtype=torch.bool))


# Delivery "steer" (B6a, training-free): a fixed vector per signal is added to the output of one
# block while that signal is past its limit.

def block_output(model_or_base, block, *args, **kwargs):
    seen = []
    handle = block.register_forward_hook(lambda module, a, out: seen.append(out[0] if isinstance(out, tuple) else out))
    try:
        model_or_base(*args, **kwargs)
    finally:
        handle.remove()
    return seen[0]


def nominal(T=12):
    s = torch.zeros(2, T, 6)
    s[..., [1, 2, 4, 5]] = 1.0  # ctx and time at 0, the others at 1: nothing past its limit
    return s


def test_steer_adds_the_vectors_of_the_signals_past_their_limits(tiny_model):
    from prio.channel import blocks

    x = ids()
    base_out = block_output(tiny_model, blocks(tiny_model)[0], x)
    m = ChannelModel(tiny_model, 6, delivery="steer", steer_layer=0)
    torch.manual_seed(3)
    vectors = torch.randn(6, 32)
    m.set_steering(vectors, alpha=2.0)
    s = nominal()
    s[0, 6:, 0] = 0.905  # ctx past 0.90 from position 6
    s[1, 3:, 1] = 0.095  # think below 0.10 from position 3
    s[1, 9:, 4] = 0.0  # network down from position 9
    out = block_output(m, blocks(m.base)[0], x, s)
    expected = base_out.clone()
    expected[0, 6:] += 2.0 * vectors[0]
    expected[1, 3:] += 2.0 * vectors[1]
    expected[1, 9:] += 2.0 * vectors[4]
    assert torch.allclose(out, expected, atol=1e-5)


def test_steer_changes_nothing_while_every_signal_is_within_its_limit(tiny_model):
    base = tiny_model(ids()).logits
    m = ChannelModel(tiny_model, 6, delivery="steer", steer_layer=1)
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-6)  # no vectors yet
    m.set_steering(torch.randn(6, 32), alpha=4.0)
    s = nominal()
    s[..., 0], s[..., 1] = 0.895, 0.105  # one level short of the limit
    assert torch.allclose(m(ids(), s).logits, base, atol=1e-6)
    s[:, 7:, 0] = 0.905
    out = m(ids(), s).logits
    assert torch.allclose(out[:, :7], base[:, :7], atol=1e-5) and not torch.allclose(out[:, 7:], base[:, 7:], atol=1e-3)
    m.enabled = False
    assert torch.allclose(m(ids(), s).logits, base, atol=1e-6)


def test_steer_is_training_free(tiny_model):
    m = ChannelModel(tiny_model, 6, delivery="steer", steer_layer=0)
    assert m.channel_parameters() == []
    assert not m.steer_vectors.requires_grad and m.steer_vectors.shape == (6, 32)


def test_steer_needs_a_block_that_exists(tiny_model):
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="steer")
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="steer", steer_layer=2)  # the tiny model has blocks 0 and 1
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="steer", steer_layer=0).set_steering(torch.zeros(5, 32), 1.0)


def test_the_bare_model_is_not_steered_after_a_steered_pass(tiny_model):
    base = tiny_model(ids()).logits
    m = ChannelModel(tiny_model, 6, delivery="steer", steer_layer=0)
    m.set_steering(torch.randn(6, 32), alpha=4.0)
    s = nominal()
    s[..., 0] = 0.95
    m(ids(), s)
    assert torch.allclose(tiny_model(ids()).logits, base, atol=1e-6)


# R8: scale-controlled addition. The sum delivery with the channel vector of each token scaled to
# unit size and multiplied by a learned gain that starts at 0.1 x the mean embedding size.

def rms_sum(base):
    return ChannelModel(base, 6, sum_norm="rms", projector="levels")


def test_rms_sum_adds_a_vector_of_fixed_size(tiny_model):
    m = rms_sum(tiny_model)
    x, s = ids(), sig()
    added = embeds_seen_by_the_base(m, x, s) - tiny_model.transformer.wte(x)
    table = tiny_model.transformer.wte.weight
    start = 0.1 * table.pow(2).mean(-1).sqrt().mean()
    assert torch.allclose(m.sum_scale * m.sum_gain, start, atol=1e-7)
    assert torch.allclose(added.pow(2).mean(-1).sqrt(), start.expand(2, 12), atol=1e-5)
    assert torch.allclose(added, start * unit(m.projector(s)), atol=1e-7, rtol=1e-3)  # rtol: the epsilon of _unit


def test_the_size_of_the_rms_sum_projection_does_not_matter(tiny_model):
    m = rms_sum(tiny_model)
    before = m(ids(), sig()).logits
    assert not torch.allclose(before, tiny_model(ids()).logits, atol=1e-4)  # it starts at 0.1 x, not at zero
    with torch.no_grad():
        m.projector.net[-1].weight *= 10
        m.projector.net[-1].bias *= 10
    assert torch.allclose(m(ids(), sig()).logits, before, atol=1e-4)


def test_the_rms_sum_gain_trains_with_the_channel(tiny_model):
    m = rms_sum(tiny_model)
    assert m.channel_parameters() == list(m.projector.parameters()) + [m.sum_gain]
    m(ids(), sig()).logits.sum().backward()
    assert m.sum_gain.grad is not None and m.sum_gain.grad.abs() > 0
    assert torch.count_nonzero(m.projector.net[0].weight.grad) > 0


def test_rms_sum_is_an_option_of_the_sum_delivery_only(tiny_model):
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, delivery="lanes", lane_width=8, sum_norm="rms")
    with pytest.raises(ValueError):
        ChannelModel(tiny_model, 6, sum_norm="layer")
    assert ChannelModel(tiny_model, 6).sum_norm is None


# R10 encoder "levels+fourier": the level table plus 16 sin / cos pairs of the value per signal,
# an ordered code in which near values get near features.

def test_levels_fourier_code_is_the_level_code_plus_sixteen_pairs():
    from prio.channel import FOURIER_FREQS

    p = SignalProjector(2, 32, kind="levels+fourier")
    levels = SignalProjector(2, 32, kind="levels")
    x = torch.tensor([[[0.905, 0.31]]])
    code = p.encode(x)[0, 0]
    assert code.shape == (2 * 100 + 2 * 32,) and len(p.net) == 1 and p.net[0].in_features == 264
    assert torch.allclose(code[:200], levels.encode(x)[0, 0])  # the table keeps its layout: signal * 100 + level
    assert FOURIER_FREQS.shape == (16,)
    angles = 0.905 * FOURIER_FREQS
    assert torch.allclose(code[200:232], 0.25 * torch.cat([angles.sin(), angles.cos()]), atol=1e-5)  # signal 0


def test_levels_fourier_code_is_ordered():
    p = SignalProjector(1, 4, kind="levels+fourier")
    smooth = lambda x: p.encode(torch.tensor([[[x]]]))[0, 0, 100:]
    gaps = [(smooth(0.505) - smooth(0.505 + d)).norm().item() for d in (0.01, 0.05, 0.2)]
    assert gaps[0] < gaps[1] < gaps[2]  # unlike the one-hot table, where every other level is equally far


@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_an_untrained_levels_fourier_channel_equals_the_base(tiny_model, delivery):
    base = tiny_model(ids()).logits
    m = ChannelModel(tiny_model, 6, delivery=delivery, lane_width=8, projector="levels+fourier")
    assert torch.allclose(m(ids(), sig()).logits, base, atol=1e-5)
    arm(m)
    assert not torch.allclose(m(ids(), sig()).logits, base, atol=1e-3)


# Every new configuration survives save -> load.

NEW_CONFIGS = {
    "none": dict(delivery="none"),
    "insert": dict(delivery="insert", insert_id=99, projector="levels"),
    "segment": dict(delivery="segment"),
    "steer": dict(delivery="steer", steer_layer=1),
    "sum-rms": dict(sum_norm="rms", projector="levels"),
    "wide-fourier": dict(delivery="wide", lane_width=8, projector="levels+fourier"),
}


@pytest.mark.parametrize("name", sorted(NEW_CONFIGS))
def test_new_configurations_save_load_roundtrip(tiny_model, tmp_path, name):
    config = NEW_CONFIGS[name]
    m = ChannelModel(copy.deepcopy(tiny_model), 6, **config)
    arm(m)
    torch.manual_seed(4)
    if name == "segment":
        torch.nn.init.normal_(m.segment, std=0.5)
    if name == "steer":
        m.set_steering(torch.randn(6, 32), alpha=3.0)
    if name == "sum-rms":
        with torch.no_grad():
            m.sum_gain.fill_(2.5)
    m.save_channel(tmp_path / "c.pt")
    torch.manual_seed(9)  # other random starting weights
    m2 = ChannelModel(copy.deepcopy(tiny_model), 6, **config)
    m2.load_channel(tmp_path / "c.pt")
    x, s = with_inserts(), torch.rand(2, 12, 6, generator=torch.Generator().manual_seed(7))
    s[:, 6:, 0] = 0.95
    harness = torch.zeros(2, 12, dtype=torch.bool)
    harness[:, 2:5] = True
    out = m(x, s, harness_mask=harness).logits
    assert torch.allclose(out, m2(x, s, harness_mask=harness).logits, atol=1e-6)
    if name != "none":
        assert not torch.allclose(out, tiny_model(x).logits, atol=1e-3)
