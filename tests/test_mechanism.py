"""Mechanism analysis (R1): cached-prefix single-position runner, ablation masks, patching,
minimal pairs, probes, and the level-table geometry (B1). Tiny random GPT-2 on CPU."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

import mech_utils as mu  # noqa: E402
from helpers import FAKE_MS, arm, make_episode  # noqa: E402
from prio.channel import ChannelModel  # noqa: E402
from prio.signals import signal_set  # noqa: E402

SPECS = signal_set(6)
NAMES = [s.name for s in SPECS]


@pytest.fixture
def wide(tiny_model):
    torch.manual_seed(1)
    model = ChannelModel(tiny_model, 6, delivery="wide", lane_width=8, projector="levels").eval()
    arm(model, std=0.5)
    return model


@pytest.fixture
def lanes(tiny_model):
    torch.manual_seed(1)
    model = ChannelModel(tiny_model, 6, delivery="lanes", lane_width=8, projector="levels").eval()
    arm(model, std=0.5)
    return model


def _sequence(length=12, seed=0):
    rng = np.random.default_rng(seed)
    ids = [int(i) for i in rng.integers(6, 90, size=length)]
    rows = rng.uniform(0.3, 0.7, size=(length, 6)).astype(np.float32)
    rows[:, 4] = 1.0  # net up
    event, control = rows.copy(), rows.copy()
    event[-1, 0], control[-1, 0] = 0.905, 0.895  # ctx crosses its limit at the last token
    return ids, event, control


def _full_last_logits(model, ids, rows):
    with torch.no_grad():
        return model(torch.tensor([ids]), torch.from_numpy(rows)[None]).logits[0, -1]


# --- sites and layers ---------------------------------------------------------------------------

def test_channel_sites_finds_every_reader_and_writer(wide):
    readers, writers = mu.channel_sites(wide)
    assert len(readers) == len(wide.readers) == 4  # c_attn and c_fc in each of 2 layers
    assert len(writers) == len(wide.writers) == 4  # attn.c_proj and mlp.c_proj in each layer
    assert sorted({s.layer for s in readers}) == [0, 1]
    assert {(s.proj, s.group) for s in readers} == {("c_attn", "attn"), ("c_fc", "ffn")}
    assert {s.group for s in writers} == {"attn", "ffn"}
    assert len(mu.decoder_layers(wide)) == 2


@pytest.fixture
def wide_lora(wide):
    """As a saved run is loaded: channel built first, then the LoRA wrapper, adapter non-zero."""
    from prio.train import add_lora

    add_lora(wide)
    torch.manual_seed(3)
    for name, p in wide.base.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.2)
    return wide.eval()


def test_sites_and_runner_work_through_the_lora_wrapper(wide_lora):
    readers, writers = mu.channel_sites(wide_lora)
    assert len(readers) == 4 and len(writers) == 4
    assert len(mu.decoder_layers(wide_lora)) == 2
    ids, event, _ = _sequence()
    with mu.Interventions(wide_lora) as iv:
        kv = iv.prefix(ids, event)
        logits = iv.last(kv, ids[-1], event[-1:])
        kv_base = iv.prefix(ids, event, enabled=False, adapter=False)
        base = iv.last(kv_base, ids[-1], event[-1:], enabled=False, adapter=False)
    assert torch.allclose(logits[0], _full_last_logits(wide_lora, ids, event), atol=1e-5)
    with wide_lora.base.disable_adapter():
        wide_lora.enabled = False
        ref = _full_last_logits(wide_lora, ids, event)
        wide_lora.enabled = True
    assert torch.allclose(base[0], ref, atol=1e-5)
    assert not torch.allclose(base[0], logits[0], atol=1e-4)


def test_lane_delivery_has_readers_only(lanes):
    readers, writers = mu.channel_sites(lanes)
    assert len(readers) == 4 and writers == []


# --- cached-prefix runner -----------------------------------------------------------------------

@pytest.mark.parametrize("which", ["wide", "lanes"])
def test_last_position_runner_matches_a_full_forward(which, request):
    model = request.getfixturevalue(which)
    ids, event, control = _sequence()
    with mu.Interventions(model) as iv:
        kv = iv.prefix(ids, event)
        logits = iv.last(kv, ids[-1], np.stack([event[-1], control[-1], event[-1]]))
    assert torch.allclose(logits[0], _full_last_logits(model, ids, event), atol=1e-5)
    assert torch.allclose(logits[1], _full_last_logits(model, ids, control), atol=1e-5)
    assert torch.allclose(logits[2], logits[0], atol=1e-6)
    # the prefix is not consumed: a second call gives the same result
    with mu.Interventions(model) as iv:
        again = iv.last(kv, ids[-1], event[-1:])
    assert torch.allclose(again[0], logits[0], atol=1e-6)


def _zeroed_copy_logits(model, kv, ids, row, readers=(), writers=()):
    """Reference: the same single-position pass with the parameters really set to zero."""
    sites_r, sites_w = mu.channel_sites(model)
    saved = {}
    for i in readers:
        saved[("r", i)] = model.readers[sites_r[i].key].data.clone()
        model.readers[sites_r[i].key].data.zero_()
    for i in writers:
        saved[("w", i)] = model.writers[sites_w[i].key].data.clone()
        model.writers[sites_w[i].key].data.zero_()
    try:
        with mu.Interventions(model) as iv:
            return iv.last(kv, ids[-1], row[None])[0]
    finally:
        for (part, i), value in saved.items():
            target = model.readers[sites_r[i].key] if part == "r" else model.writers[sites_w[i].key]
            target.data.copy_(value)


def test_per_element_masks_equal_zeroed_parameters(wide):
    ids, event, _ = _sequence()
    conds = [mu.Cond("intact"), mu.Cond("r0", readers=frozenset({0})), mu.Cond("w3", writers=frozenset({3})),
             mu.Cond("both", readers=frozenset({1, 2}), writers=frozenset({0, 1, 2, 3}))]
    with mu.Interventions(model := wide) as iv:
        kv = iv.prefix(ids, event)
        rows = np.stack([event[-1]] * len(conds))
        logits = iv.last(kv, ids[-1], rows, conds=conds)
    for k, cond in enumerate(conds):
        ref = _zeroed_copy_logits(model, kv, ids, event[-1], cond.readers, cond.writers)
        assert torch.allclose(logits[k], ref, atol=1e-5), cond.name
    assert not torch.allclose(logits[0], logits[1], atol=1e-4)  # the masks do something


def test_masks_touch_only_the_recomputed_position(wide):
    """Zeroing every reader at the last position is not the channel switched off: the cached
    prefix still carries what the earlier positions read."""
    ids, event, _ = _sequence()
    readers, _ = mu.channel_sites(wide)
    with mu.Interventions(wide) as iv:
        kv = iv.prefix(ids, event)
        cut = iv.last(kv, ids[-1], event[-1:], conds=[mu.Cond("all", readers=frozenset(range(len(readers))))])[0]
        kv_off = iv.prefix(ids, event, enabled=False)
        off = iv.last(kv_off, ids[-1], event[-1:], enabled=False)[0]
    wide.enabled = False
    assert torch.allclose(off, _full_last_logits(wide, ids, event), atol=1e-5)
    wide.enabled = True
    assert not torch.allclose(cut, off, atol=1e-4)


def test_hooks_are_removed_on_exit(wide):
    ids, event, _ = _sequence()
    before = _full_last_logits(wide, ids, event)
    with mu.Interventions(wide) as iv:
        kv = iv.prefix(ids, event)
        iv.last(kv, ids[-1], event[-1:], conds=[mu.Cond("x", readers=frozenset({0, 1}))])
    assert torch.allclose(_full_last_logits(wide, ids, event), before, atol=1e-6)
    assert wide._lanes.values is None


# --- recording and patching ---------------------------------------------------------------------

def test_recording_gives_every_layer_input_and_the_last_output(wide):
    ids, event, _ = _sequence()
    with mu.Interventions(wide) as iv:
        kv = iv.prefix(ids, event)
        _, rec = iv.last(kv, ids[-1], event[-1:], record=True)
    assert sorted(rec) == [0, 1, 2]
    assert all(r.shape == (1, 32 + 8) for r in rec.values())  # d + lane width


def test_patching_the_lanes_at_the_input_turns_the_control_into_the_event(wide):
    ids, event, control = _sequence()
    with mu.Interventions(wide) as iv:
        kv = iv.prefix(ids, event)
        both, rec = iv.last(kv, ids[-1], np.stack([event[-1], control[-1]]), record=True)
        source = {layer: r[0] for layer, r in rec.items()}  # the event run's residuals
        conds = [mu.Cond("lanes0", signals="control", patch=(0, "lanes")),
                 mu.Cond("main0", signals="control", patch=(0, "main")),
                 mu.Cond("all1", signals="control", patch=(1, "all"))]
        patched = iv.last(kv, ids[-1], np.stack([control[-1]] * 3), conds=conds, patch_source=source)
    assert not torch.allclose(both[0], both[1], atol=1e-4)
    assert torch.allclose(patched[0], both[0], atol=1e-5)  # same token: only the lanes differ at the input
    assert torch.allclose(patched[1], both[1], atol=1e-5)  # the main stream is the same embedding
    assert torch.allclose(patched[2], both[0], atol=1e-5)  # the whole residual at any layer


def test_adding_a_vector_to_the_main_stream(wide):
    ids, event, _ = _sequence()
    with mu.Interventions(wide) as iv:
        kv = iv.prefix(ids, event)
        base, rec = iv.last(kv, ids[-1], event[-1:], record=True)
        zero = iv.last(kv, ids[-1], event[-1:], conds=[mu.Cond("z", add=(1, "v"))], add_vectors={"v": torch.zeros(32)})
        push = iv.last(kv, ids[-1], event[-1:], conds=[mu.Cond("p", add=(1, "v"))],
                       add_vectors={"v": torch.randn(32, generator=torch.Generator().manual_seed(0)) * 3})
    assert torch.allclose(zero[0], base[0], atol=1e-6)
    assert not torch.allclose(push[0], base[0], atol=1e-3)


def test_collect_records_chosen_positions(wide):
    ids, event, _ = _sequence()
    with mu.Interventions(wide) as iv:
        rec = iv.collect(ids, event, positions=[2, 5, 11])
        off = iv.collect(ids, event, positions=[2, 5, 11], enabled=False)
    assert rec[0].shape == (3, 40) and off[0].shape == (3, 32)
    assert torch.allclose(rec[0][:, :32].float(), off[0].float(), atol=1e-5)  # embedding + position, no lanes yet


# --- minimal pairs ------------------------------------------------------------------------------

def test_nominal_side_value():
    by = dict(zip(NAMES, SPECS))
    assert mu.nominal_side(by["ctx"]) == pytest.approx(0.895)
    assert mu.nominal_side(by["think"]) == pytest.approx(0.105)
    assert mu.nominal_side(by["net"]) == 1.0
    for spec in SPECS:
        assert not spec.past_limit(mu.nominal_side(spec))


def _event_episode(kind="event", signal="ctx", t_cross=3, t_decision=None, target=None):
    prompt = [10, 11, 12, 13]
    target = target or [20, 21, 22, 23, 24, 25, 26]
    rows = np.full((len(prompt) + len(target), 6), 0.5, dtype=np.float32)
    rows[:, 4] = 1.0
    i = NAMES.index(signal)
    start = len(prompt) + (t_cross if t_cross is not None else 0)
    rows[:, i] = mu.nominal_side(SPECS[i])
    rows[start:, i] = 0.905 if SPECS[i].bad_when == "high" else (0.0 if signal == "net" else 0.095)
    return make_episode(kind=kind, signal=signal, phase="thinking", prompt_ids=prompt, target_ids=target,
                        signals=rows, t_cross=t_cross, t_decision=t_decision)


def test_pair_for_an_interrupt_event_differs_only_at_the_crossing_token():
    pair = mu.make_pair(_event_episode(), SPECS, FAKE_MS)
    assert pair.ids == [10, 11, 12, 13, 20, 21, 22, 23]  # forced through target[t_cross]
    diff = np.argwhere(pair.event != pair.control)
    assert diff.tolist() == [[7, 0]]
    assert pair.control[-1, 0] == pytest.approx(0.895) and pair.event[-1, 0] == pytest.approx(0.905)


def test_pair_for_a_gating_decision_ends_before_the_tool_call():
    ep = _event_episode(kind="gating_event", signal="net", t_cross=1, t_decision=5)
    pair = mu.make_pair(ep, SPECS, FAKE_MS)
    assert pair.ids == [10, 11, 12, 13, 20, 21, 22, 23, 24]
    assert np.argwhere(pair.event != pair.control).tolist() == [[8, 4]]
    assert pair.control[-1, 4] == 1.0 and pair.event[-1, 4] == 0.0


def test_no_pair_for_other_kinds_or_a_spent_thinking_budget():
    assert mu.make_pair(make_episode(kind="nominal"), SPECS, FAKE_MS) is None
    close = FAKE_MS.think_close
    ep = _event_episode(signal="think", t_cross=4, target=[20, close, 22, 23, 24, 25, 26])
    assert mu.make_pair(ep, SPECS, FAKE_MS) is None


def test_retained_fraction():
    assert mu.retained(5.0, 5.0, -3.0) == pytest.approx(1.0)
    assert mu.retained(-3.0, 5.0, -3.0) == pytest.approx(0.0)
    assert mu.retained(1.0, 5.0, -3.0) == pytest.approx(0.5)
    assert np.isnan(mu.retained(1.0, 2.0, 2.0))


# --- B1: level-table geometry -------------------------------------------------------------------

import level_geometry as lg  # noqa: E402
from prio.channel import SignalProjector  # noqa: E402


def test_level_vectors_are_the_table_columns_of_each_signal():
    torch.manual_seed(0)
    proj = SignalProjector(6, 16, zero_init=False, kind="levels")
    state = {k: v.clone() for k, v in proj.state_dict().items()}
    vectors = lg.level_vectors(state, n_signals=6)
    assert vectors.shape == (6, 100, 16)
    row = torch.full((1, 1, 6), 0.505)  # every signal at level 50
    row[..., 2] = 0.235  # energy at level 23
    with torch.no_grad():
        out = proj(row)[0, 0].numpy()
    expected = vectors[[0, 1, 3, 4, 5], 50].sum(0) + vectors[2, 23] + state["net.0.bias"].numpy()
    assert np.allclose(out, expected, atol=1e-5)


def test_limit_steps_and_past_levels():
    by = dict(zip(NAMES, SPECS))
    assert lg.limit_step(by["ctx"]) == 89 and lg.limit_step(by["time"]) == 89
    assert lg.limit_step(by["think"]) == 9 and lg.limit_step(by["ram"]) == 9
    assert lg.limit_step(by["net"]) is None
    assert np.flatnonzero(lg.past_levels(by["ctx"])).tolist() == list(range(90, 100))
    assert np.flatnonzero(lg.past_levels(by["think"])).tolist() == list(range(0, 10))


def _table(kind, seed=0, dim=16):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=dim), rng.normal(size=dim)
    noise = rng.normal(scale=0.02, size=(100, dim))
    if kind == "step":
        return np.where(np.arange(100)[:, None] >= 90, b, a) + noise
    return np.arange(100)[:, None] / 100 * (b - a) + a + noise  # ramp


def test_geometry_sees_a_step_at_the_limit():
    ctx = SPECS[0]
    g = lg.geometry(_table("step"), ctx)
    assert g["largest_step"] == 89 and g["largest_step_at_limit"]
    assert g["limit_step_rank"] == 1 and g["step_fraction"] > 0.9
    assert g["cos_within_past"] > 0.9 and g["cos_within_nominal"] > 0.9
    assert g["crossing_level"] == 90 and g["last_nominal_level"] == 89
    big = _table("step")
    big[90] *= 3  # an outsized vector at the crossing level
    norms = np.linalg.norm(big, axis=1)
    assert lg.geometry(big, ctx)["crossing_level_norm_ratio"] == pytest.approx(norms[90] / np.median(norms))
    think = lg.geometry(_table("step")[::-1], SPECS[1])  # low limit: past = levels 0..9
    assert think["crossing_level"] == 9 and think["last_nominal_level"] == 10


def test_geometry_sees_a_ramp_as_an_ordered_code():
    g = lg.geometry(_table("ramp"), SPECS[0])
    assert g["step_fraction"] < 0.2
    assert abs(g["spearman_pc1_nominal"]) > 0.9


def test_visited_levels_restrict_the_statistics():
    table = _table("step")
    table[:50] = np.random.default_rng(5).normal(size=(50, 16))  # never visited: random
    visits = np.zeros(100)
    visits[50:] = 1000
    g = lg.geometry(table, SPECS[0], visits=visits, min_visits=10)
    assert g["n_visited"] == 50 and g["step_fraction"] > 0.9


def test_spearman():
    assert lg.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert lg.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_procrustes_recovers_a_rotation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 8))
    rot, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    r = lg.procrustes(a, a @ rot)
    assert np.allclose(r, rot, atol=1e-8)


def test_init_table_is_the_projector_a_run_started_from(tiny_model):
    torch.manual_seed(7)
    model = ChannelModel(tiny_model, 6, delivery="wide", lane_width=64, projector="levels")
    assert np.allclose(lg.init_table(7, n_signals=6, width=64), model.projector.net[0].weight.detach().numpy())


def test_visit_counts_split_between_neighbors():
    rows = np.array([[0.905] * 6, [0.905] * 6, [0.91] * 6], dtype=np.float32)
    counts = lg.visit_counts([rows], n_signals=6)
    assert counts.shape == (6, 100)
    assert counts[0, 90] == pytest.approx(2.5) and counts[0, 91] == pytest.approx(0.5)


def test_reader_gain_profile():
    d = np.zeros(4)
    d[0] = 1.0
    state = {"readers.model-layers-0-self_attn-q_proj": torch.tensor(np.diag([3.0, 1, 1, 1])).float(),
             "readers.model-layers-1-mlp-up_proj": torch.tensor(np.diag([0.0, 2, 2, 2])).float()}
    prof = lg.reader_profile(state, d)
    by = {(r["layer"], r["proj"]): r for r in prof}
    assert by[(0, "q_proj")]["gain"] == pytest.approx(3.0)
    assert by[(1, "up_proj")]["gain"] == pytest.approx(0.0)
    assert by[(0, "q_proj")]["gain_ratio"] == pytest.approx(3.0 / np.sqrt(12 / 4))


def test_writer_profile_overlap():
    d = np.zeros(4)
    d[1] = 1.0
    rows = np.zeros((4, 6))
    rows[1, :] = 5.0  # the writer only ever writes along lane 1
    prof = lg.writer_profile({"writers.model-layers-2-mlp-down_proj": torch.tensor(rows).float()}, d, top=1)
    assert prof[0]["layer"] == 2 and prof[0]["effective_rank"] == pytest.approx(1.0)
    assert prof[0]["overlap_top"] == pytest.approx(1.0)


# --- B2 / B3 / T3 runner (mechanism.py) ---------------------------------------------------------

import json  # noqa: E402

import mechanism as mech  # noqa: E402


class ModTok:
    """Word -> id inside the tiny vocabulary (ids 6..95), deterministic."""

    def encode(self, text, add_special_tokens=False):
        return [6 + sum(map(ord, w)) % 90 for w in text.split()]


def _episodes(n=4):
    """Interrupt events (ctx crosses at t_cross) and one gating decision (net down), ids < 100."""
    rng = np.random.default_rng(3)
    eps = []
    for k in range(n):
        prompt = [int(x) for x in rng.integers(6, 90, size=6)]
        target = [int(x) for x in rng.integers(6, 90, size=10)]
        if k == n - 1:
            ep = _event_episode(kind="gating_event", signal="net", t_cross=2, t_decision=7, target=target)
        else:
            ep = _event_episode(kind="event", signal=["ctx", "energy", "time"][k % 3], t_cross=4 + k % 3,
                                target=target)
        ep.prompt_ids = prompt + ep.prompt_ids[len(prompt):]
        ep.signals = np.concatenate([np.repeat(ep.signals[:1], len(ep.prompt_ids) - 4, axis=0), ep.signals])
        ep.episode_id, ep.task_id = f"e{k}", f"t{k % 3}"
        eps.append(ep)
    return eps


def test_episode_helper_builds_valid_pairs():
    pairs = [mu.make_pair(ep, SPECS, FAKE_MS) for ep in _episodes()]
    assert all(p is not None for p in pairs)
    assert [p.kind for p in pairs] == ["event", "event", "event", "gating_event"]


def test_condition_lists(wide):
    with mu.Interventions(wide) as iv:
        names = [c.name for c in mech.b2_conditions(iv)]
    assert len(names) == len(set(names))
    for name in ("readers:all", "readers:L0", "readers:L1", "writers:all", "writers:L1", "lanes_from:L1",
                 "lanes_upto:L1", "patch_main:L0", "patch_lanes:L1", "readers:proj=c_fc", "readers:attn"):
        assert name in names


def test_condition_lists_without_writers(lanes):
    with mu.Interventions(lanes) as iv:
        names = [c.name for c in mech.b2_conditions(iv)]
    assert not any(n.startswith(("writers", "patch_lanes")) for n in names)
    assert "patch_main:L1" in names


def test_analyze_pair_is_batch_invariant_and_anchored(wide):
    with mu.Interventions(wide) as iv:
        conds = mech.b2_conditions(iv)
        results = [(p, mech.analyze_pair(iv, p, conds, batch=3))
                   for p in (mu.make_pair(ep, SPECS, FAKE_MS) for ep in _episodes())]
        split = [(p, r) for p, r in results if r["split"]]
        assert split, "the tiny model must split at least one pair for this test to mean anything"
        pair, small = split[0]
        large = mech.analyze_pair(iv, pair, conds, batch=64)
    assert all(not r["conditions"] for _, r in results if not r["split"])  # unsplit pairs are not scored
    assert small["r"] == large["r"] and small["c"] == large["c"]
    assert set(small["conditions"]) == {c.name for c in conds} | {"prefix_off:event", "prefix_off:control", "channel_off"}
    for name, value in small["conditions"].items():
        assert value["ld"] == pytest.approx(large["conditions"][name]["ld"], abs=1e-4), name
    cond = small["conditions"]
    ret = lambda name: mu.retained(cond[name]["ld"], small["ld_intact"], small["ld_control"])  # noqa: E731
    assert ret("patch_lanes:L0") == pytest.approx(1, abs=1e-3)  # same token: the input lanes are the difference
    assert ret("patch_main:L0") == pytest.approx(0, abs=1e-3)
    assert cond["readers:all"]["ld"] == pytest.approx(cond["readers+writers:all"]["ld"], abs=1e-5)  # nobody reads the writes


def test_greedy_order_is_cross_fitted():
    records = []
    for k in range(4):
        # even pairs: layer 1 matters; odd pairs: layer 0 matters
        good, bad = (1, 0) if k % 2 == 0 else (0, 1)
        records.append({"index": k, "split": True, "ld_intact": 1.0, "ld_control": 0.0,
                        "conditions": {f"readers:L{good}": {"ld": 0.0}, f"readers:L{bad}": {"ld": 1.0}}})
    assert mech.greedy_order(records, parity=0, n_layers=2) == [0, 1]  # ranked on the odd pairs
    assert mech.greedy_order(records, parity=1, n_layers=2) == [1, 0]


def test_summary_hypotheses():
    def rec(k, values):
        return {"index": k, "split": True, "kind": "event", "signal": "ctx", "ld_intact": 2.0, "ld_control": 0.0,
                "conditions": {n: {"ld": v, "top_r": v > 1} for n, v in values.items()}}

    n_layers = 8
    values = {"writers:all": 1.9, "readers:all": 0.0, **{f"greedy:k{k}": 2.0 - 0.9 * k for k in range(1, 3)},
              **{f"greedy:k{k}": 0.1 for k in range(3, 9)}}
    s = mech.summarize([rec(0, values), rec(1, values)], n_layers=n_layers)
    assert s["conditions"]["writers:all"]["mean_retained"] == pytest.approx(0.95)
    assert s["H2"]["writers_all_removed"] == pytest.approx(0.05) and s["H2"]["pass"]
    assert s["H1"]["k_for_80pct"] == 2 and s["H1"]["pass"]  # k=2: retained 0.1 <= 0.2, 2 <= 25% of 8


def test_notice_texts_share_a_template():
    for spec in SPECS:
        past, nominal = mech.notice_texts(spec)
        assert past != nominal and past.startswith(" [STATUS:")
        assert len(past.split()) == len(nominal.split())


def test_pipeline_runs_end_to_end_and_resumes(wide, tmp_path):
    eps = _episodes()
    out = tmp_path / "mech"
    mech.run_mechanism(wide, ModTok(), FAKE_MS, SPECS, eps, out, batch=8, t3=True)
    summary = json.loads((out / "summary.json").read_text())
    for key in ("conditions", "H1", "H2", "patch_curves", "prefix_off", "t3", "n_pairs", "n_split"):
        assert key in summary
    assert summary["n_pairs"] == 4 and summary["n_split"] >= 1
    for name in ("greedy:k1", "greedy:k2", "t3_off", "t3_tx:L1", "t3_ch:L1", "channel_off", "prefix_off:event"):
        assert summary["conditions"][name]["n"] == summary["n_split"], name
    c = summary["conditions"]
    assert c["greedy:k2"]["mean_retained"] == pytest.approx(c["readers:all"]["mean_retained"], abs=1e-4)
    assert c["t3_ch:L0"]["mean_retained"] == pytest.approx(c["t3_off"]["mean_retained"], abs=1e-4)  # v_ch(0) = 0
    assert c["t3_off"]["mean_retained"] == pytest.approx(c["channel_off"]["mean_retained"], abs=1e-4)
    assert (out / "summary.md").exists()
    lines = (out / "pairs.jsonl").read_text().splitlines()
    assert len(lines) == 4
    t3 = summary["t3"]["by_signal"]["ctx"]
    assert len(t3["cos_trained"]) == 3 and len(t3["null_sd"]) == 3  # layers 0, 1 and the last output
    mech.run_mechanism(wide, ModTok(), FAKE_MS, SPECS, eps, out, batch=8, t3=True)
    assert len((out / "pairs.jsonl").read_text().splitlines()) == 4  # nothing redone


# --- B3 probes ----------------------------------------------------------------------------------

def test_probe_positions_and_labels():
    ep = _episodes()[0]
    rows = mech.probe_rows(ep, SPECS, FAKE_MS, np.random.default_rng(0), n_random=2)
    decision = len(ep.prompt_ids) + ep.t_cross
    by_pos = {r["position"]: r for r in rows}
    assert by_pos[decision]["past_any"] == 1 and by_pos[decision]["which"] == ep.signal
    assert by_pos[decision - 1]["past_any"] == 0 and by_pos[decision - 1]["which"] == "none"
    assert all(r["position"] < len(ep.prompt_ids) + len(ep.target_ids) for r in rows)


def test_logistic_and_ridge_probes_on_synthetic_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 20)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    fit = mu.fit_logistic(x[:300], y[:300], x[300:], y[300:])
    assert fit["acc_test"] > 0.9
    v = x[:, 1] * 2 + rng.normal(scale=0.1, size=400)
    assert mu.fit_ridge(x[:300], v[:300], x[300:], v[300:])["r2_test"] > 0.9
    assert mu.fit_ridge(x[:300], rng.normal(size=300), x[300:], rng.normal(size=100))["r2_test"] < 0.1


def test_probes_end_to_end(wide, tmp_path):
    out = tmp_path / "probes"
    mech.run_probes(wide, FAKE_MS, SPECS, _episodes(), out, n_random=3)
    result = json.loads((out / "probes.json").read_text())
    assert set(result["conditions"]) == {"on", "off", "base"}
    on = result["conditions"]["on"]
    assert "past_any" in on["main"][0] and "lanes" in on  # wide: the lanes are probed too
    assert "lanes" not in result["conditions"]["off"]
    assert (out / "probes.md").exists()


def test_cli_runs_b2_t3_and_probes(wide, tmp_path, monkeypatch):
    import prio.train as train_module
    from prio.datagen.episodes import write_jsonl

    data, run, out = tmp_path / "data", tmp_path / "run", tmp_path / "out"
    data.mkdir()
    run.mkdir()
    (data / "manifest.json").write_text(json.dumps({"model": "fake", "n_signals": 6}))
    write_jsonl(data / "test.jsonl", _episodes())
    monkeypatch.setattr(train_module, "load_run", lambda model_id, path, n: (wide, ModTok(), FAKE_MS))
    mech.main(["--run", str(run), "--data", str(data), "--out", str(out), "--batch", "8", "--probes",
               "--probe-episodes", "4"])
    assert json.loads((out / "summary.json").read_text())["n_pairs"] == 4
    assert (out / "probes" / "probes.json").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["invocations"][0]["finished"] and meta["invocations"][0]["args"]["batch"] == 8


def test_run_model_id_prefers_the_model_recorded_by_the_run(tmp_path):
    (tmp_path / "train_log.json").write_text('{"model": "Qwen/Qwen3-0.6B"}', encoding="utf-8")
    assert mu.run_model_id(tmp_path, {"model": "Qwen/Qwen3-1.7B"}) == "Qwen/Qwen3-0.6B"


def test_run_model_id_falls_back_to_the_data_manifest(tmp_path):
    assert mu.run_model_id(tmp_path, {"model": "Qwen/Qwen3-1.7B"}) == "Qwen/Qwen3-1.7B"
