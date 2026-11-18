import copy

import numpy as np
import pytest
import torch

import prio.train as train_module
from helpers import arm, make_episode
from prio.channel import ChannelModel
from prio.train import add_lora, collate, fit, load_run, save_run, weighted_loss


def ep(prompt, target, n=6):
    return make_episode(prompt_ids=prompt, target_ids=target, signals=[[0.0] * n] * (len(prompt) + len(target)))


def test_collate_masks_prompt_and_padding():
    b = collate([ep([1, 2, 3], [4, 5]), ep([1], [2])], pad_id=0, n_signals=6)
    assert b["input_ids"].tolist() == [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0]]
    assert b["labels"][0].tolist() == [-100, -100, -100, 4, 5]
    assert b["labels"][1].tolist() == [-100, 2, -100, -100, -100]
    assert b["signals"].shape == (2, 5, 6) and b["attention_mask"][1].tolist() == [1, 1, 0, 0, 0]
    assert b["dropped"] == 0


def test_collate_carries_each_episodes_signals():
    e = make_episode(prompt_ids=[1], target_ids=[2], signals=[[0.5] * 6, [0.75] * 6])
    b = collate([e, ep([1, 2, 3], [4])], pad_id=0, n_signals=6)
    assert b["signals"][0, :2].tolist() == [[0.5] * 6, [0.75] * 6]
    assert torch.count_nonzero(b["signals"][0, 2:]) == 0  # padding rows are zeros


def test_collate_weights_the_reaction_tokens():
    # Found by the phase-0 spike: the decision to react is 1-2 tokens among hundreds and is not
    # learned unless the reaction tokens weigh more in the loss.
    event = make_episode(kind="event", signal="ctx", prompt_ids=[1, 2], target_ids=[3, 4, 5, 6], t_cross=1)
    gating = make_episode(kind="gating_event", signal="net", prompt_ids=[1], target_ids=[3, 4, 5], t_cross=0, t_decision=1)
    twin = make_episode(kind="nominal", prompt_ids=[1, 2], target_ids=[3, 4, 5, 6])
    b = collate([event, gating, twin], pad_id=0, n_signals=6, reaction_weight=20.0)
    assert b["weights"][0].tolist() == [1, 1, 1, 1, 20, 20]  # reaction starts at target index t_cross + 1
    assert b["weights"][1].tolist() == [1, 1, 20, 20, 1, 1]  # gating: from the decision point; padding stays 1
    assert b["weights"][2].tolist() == [1] * 6
    assert collate([event], pad_id=0, n_signals=6)["weights"].tolist() == [[1.0] * 6]  # default: no extra weight


def test_collate_weighs_the_twin_where_it_parts_from_the_reaction():
    # Same text up to here, reaction in one episode and none in the other: both decisions
    # weigh the same, so only the channel can lower the loss.
    event = make_episode(kind="event", signal="ctx", prompt_ids=[1, 2], target_ids=[3, 4, 5, 6], t_cross=0, t_diverge=2)
    twin = make_episode(kind="nominal", prompt_ids=[1, 2], target_ids=[3, 4, 7, 8], t_diverge=2)
    b = collate([event, twin], pad_id=0, n_signals=6, reaction_weight=20.0)
    assert b["weights"][0].tolist() == [1, 1, 1, 20, 20, 20]
    assert b["weights"][1].tolist() == [1, 1, 1, 1, 20, 1]
    assert b["decision"][0].tolist() == [0, 0, 0, 0, 1, 0]  # 1: the token where the reaction shows
    assert b["decision"][1].tolist() == [0, 0, 0, 0, 2, 0]  # 2: the token where the twin carries on


def test_decision_losses_separate_reacting_from_carrying_on():
    # Logged during training: the loss on these tokens falls only if the channel is being used.
    from prio.train import decision_losses

    logits = torch.zeros(1, 4, 5)
    logits[0, 1, 3] = 20.0  # position 1 predicts token 3 almost surely
    labels = torch.tensor([[-100, 2, 3, 4]])
    out = decision_losses(logits, labels, torch.tensor([[0, 0, 1, 2]]))
    assert len(out["react"]) == 1 and out["react"][0] < 1e-3
    assert len(out["hold"]) == 1 and abs(out["hold"][0] - float(torch.log(torch.tensor(5.0)))) < 1e-4
    assert decision_losses(logits, labels, torch.zeros(1, 4, dtype=torch.long)) == {"react": [], "hold": []}


def test_fit_records_the_decision_losses(tiny_model):
    m = ChannelModel(tiny_model, 6)
    event = make_episode(kind="event", signal="ctx", prompt_ids=[5, 6], target_ids=[7, 8, 9], t_cross=0, t_diverge=1)
    twin = make_episode(kind="nominal", prompt_ids=[5, 6], target_ids=[7, 10, 11], t_diverge=1)
    history = {}
    fit(m, [event, twin] * 2, pad_id=0, epochs=1, batch_size=2, grad_accum=1, device="cpu", history=history)
    assert len(history["react"]) == 2 and len(history["hold"]) == 2 and history["skipped"] == 0


def test_weighted_loss_equals_the_plain_loss_when_all_weights_are_one(tiny_model):
    m = ChannelModel(tiny_model, 6)
    b = collate([ep([1, 2, 3], [4, 5]), ep([1], [2, 7, 8])], pad_id=0, n_signals=6)
    out = m(b["input_ids"], b["signals"], attention_mask=b["attention_mask"], labels=b["labels"])
    assert torch.isclose(weighted_loss(out.logits, b["labels"], b["weights"]), out.loss, atol=1e-5)


def test_weighted_loss_leans_toward_the_heavy_tokens():
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 1] = 10.0  # position 0 predicts token 1 almost surely: right for label 1
    labels = torch.tensor([[-100, 1, 2]])  # position 1 predicts token 2 from uniform logits: loss log(4)
    light = weighted_loss(logits, labels, torch.tensor([[1.0, 1.0, 1.0]]))
    heavy = weighted_loss(logits, labels, torch.tensor([[1.0, 1.0, 20.0]]))
    assert light < heavy < torch.log(torch.tensor(4.0))


def test_collate_drops_overlong_instead_of_truncating():  # Review Focus 3
    b = collate([ep([1] * 10, [2] * 10), ep([1], [2])], pad_id=0, n_signals=6, max_len=8)
    assert b["input_ids"].shape[0] == 1 and b["dropped"] == 1


def test_collate_rejects_signal_length_mismatch():  # Review Focus 4
    bad = make_episode(prompt_ids=[1, 2], target_ids=[3], signals=[[0.0] * 6] * 2)
    with pytest.raises(ValueError):
        collate([bad], pad_id=0, n_signals=6)
    with pytest.raises(ValueError):
        collate([ep([1, 2], [3], n=5)], pad_id=0, n_signals=6)


def test_model_learns_from_channel_alone(tiny_model):
    # Same prompt everywhere; the target token depends only on signal 0.
    eps = []
    for i in range(32):
        hi = i % 2 == 0
        eps.append(make_episode(prompt_ids=[5, 6, 7], target_ids=[11 if hi else 22],
                                signals=[[0.95 if hi else 0.2] + [0.0] * 5] * 4))
    m = ChannelModel(tiny_model, 6)
    losses = fit(m, eps, pad_id=0, epochs=60, lr_lora=1e-3, lr_proj=1e-2, batch_size=8, grad_accum=1, device="cpu")
    assert losses[-1] < 0.1 < losses[0]
    b = collate(eps, pad_id=0, n_signals=6)
    pred = m(b["input_ids"], b["signals"]).logits[:, 2].argmax(-1).tolist()
    assert pred == [11, 22] * 16
    m.enabled = False  # without the channel it cannot tell the two cases apart
    blind = m(b["input_ids"], b["signals"]).logits[:, 2].argmax(-1)
    assert len(set(blind.tolist())) == 1


def test_channel_resolves_a_limit_with_a_tiny_margin(tiny_model):
    # Found by the phase-0 spike: values sit just below a limit (0.899) for many tokens and just
    # above it (0.901) after the crossing. The channel must tell the two apart.
    rng = np.random.default_rng(0)

    def at(value, target):
        return make_episode(prompt_ids=[5, 6, 7], target_ids=[target], signals=[[value] + [0.0] * 5] * 4)

    eps = []
    for i in range(256):  # values concentrated near the limit, on both sides
        if i % 2 == 0:
            eps.append(at(0.901 + 0.099 * rng.random() ** 3, 11))
        else:
            eps.append(at(0.899 - 0.899 * rng.random() ** 3, 22))
    m = ChannelModel(tiny_model, 6)
    fit(m, eps, pad_id=0, epochs=30, lr_lora=1e-3, lr_proj=1e-2, batch_size=16, grad_accum=1, device="cpu")
    probe = collate([at(0.899, 22), at(0.901, 11), at(0.85, 22), at(0.95, 11)], pad_id=0, n_signals=6)
    pred = m(probe["input_ids"], probe["signals"]).logits[:, 2].argmax(-1).tolist()
    assert pred == [22, 11, 22, 11]


def test_fit_survives_an_out_of_memory_batch(tiny_model):
    # A long pair of episodes can exhaust the GPU an hour into training: skip that batch, keep going.
    m = ChannelModel(tiny_model, 6)
    real_forward, calls = m.forward, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise torch.OutOfMemoryError("simulated")
        return real_forward(*args, **kwargs)

    m.forward = flaky
    losses = fit(m, [ep([5, 6, 7], [11])] * 8, pad_id=0, epochs=1, batch_size=2, grad_accum=1, device="cpu")
    assert calls["n"] == 4 and len(losses) == 3  # four batches, one skipped


def test_fit_reports_the_end_of_every_epoch(tiny_model):
    m = ChannelModel(tiny_model, 6)
    seen = []
    fit(m, [ep([5, 6, 7], [11])] * 4, pad_id=0, epochs=3, batch_size=2, grad_accum=1, device="cpu",
        on_epoch_end=seen.append)
    assert seen == [0, 1, 2]


def test_lora_freezes_the_base_and_trains_adapter_plus_projector(tiny_model):
    m = ChannelModel(tiny_model, 6)
    add_lora(m)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n or n.startswith("projector.") for n in trainable)
    assert any("lora_" in n for n in trainable) and any(n.startswith("projector.") for n in trainable)


@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_gradient_checkpointing_keeps_the_gradients(tiny_model, delivery):
    # At 1.7B the stored activations of a 1400-token episode do not fit in 16 GB. Recomputing
    # them in the backward pass must change memory use only, never what is learned.
    def gradients(checkpointing):
        torch.manual_seed(0)  # same projector and adapter initialization in both runs
        m = ChannelModel(copy.deepcopy(tiny_model), 6, delivery=delivery, lane_width=8)
        add_lora(m, checkpointing=checkpointing)
        arm(m)
        m.train()
        b = collate([ep([1, 2, 3], [4, 5]), ep([1], [2, 7, 8])], pad_id=0, n_signals=6)
        torch.manual_seed(1)
        logits = m(b["input_ids"], torch.rand(2, 5, 6), attention_mask=b["attention_mask"]).logits
        weighted_loss(logits, b["labels"], b["weights"]).backward()
        on = any(getattr(module, "gradient_checkpointing", False) for module in m.base.modules())
        return on, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    plain_on, plain = gradients(False)
    saved_on, saved = gradients(True)
    assert not plain_on and saved_on
    assert plain.keys() == saved.keys() and any(g.abs().sum() > 0 for g in plain.values())
    assert all(torch.allclose(plain[n], saved[n], atol=1e-6) for n in plain)
    parts = {"sum": ("projector.",), "lanes": ("projector.", "readers."), "wide": ("projector.", "readers.", "writers.")}
    for part in parts[delivery]:
        assert any(n.startswith(part) and g.abs().sum() > 0 for n, g in saved.items())


def test_the_seed_reproduces_a_training_run(tiny_model, tmp_path, monkeypatch):
    # Found in review: adapter and projector were initialized before anything was seeded, so
    # `--seed` did not reproduce a run.
    class Tok:
        pad_token_id = 0

    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(tiny_model), Tok(), None))
    monkeypatch.setattr(train_module, "read_jsonl", lambda path: [ep([5, 6, 7], [11, 12])] * 4)

    def run(name, seed, state):
        torch.manual_seed(state)  # whatever state the process happens to be in
        train_module.main(["--model", "m", "--data", str(tmp_path), "--n-signals", "6", "--out", str(tmp_path / name),
                           "--seed", str(seed), "--epochs", "1"])
        return torch.load(tmp_path / name / "channel.pt")

    a, b, other = run("a", 0, 123), run("b", 0, 456), run("c", 1, 123)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], other[k]) for k in a)


def test_save_load_run_roundtrip(tiny_model, tmp_path, monkeypatch):
    pristine = copy.deepcopy(tiny_model)
    m = ChannelModel(tiny_model, 6)
    add_lora(m)
    arm(m)
    fit(m, [ep([5, 6, 7], [11])] * 8, pad_id=0, epochs=2, batch_size=4, grad_accum=1, device="cpu")
    x, s = torch.tensor([[5, 6, 7]]), torch.rand(1, 3, 6)
    before = m(x, s).logits
    save_run(m, tmp_path)
    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(pristine), None, None))
    m2, _, _ = load_run("unused", tmp_path, 6)
    assert torch.allclose(before, m2(x, s).logits, atol=1e-5)
    assert not torch.allclose(before, pristine(x).logits, atol=1e-3)  # the run really changed the model


# Lane delivery: the signals travel in their own dimensions, read through extra columns.

def lane_model(base, delivery="lanes"):
    return ChannelModel(base, 6, delivery=delivery, lane_width=8)


def test_lora_with_lanes_trains_adapter_projector_and_readers(tiny_model):
    m = lane_model(tiny_model)
    add_lora(m)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all("lora_" in n or n.startswith(("projector.", "readers.")) for n in trainable)
    assert all(any(part in n for n in trainable) for part in ("lora_", "projector.", "readers."))


def test_reader_columns_train_at_the_projectors_rate(tiny_model):
    m = lane_model(tiny_model)
    fit(m, [ep([5, 6, 7], [11])] * 8, pad_id=0, epochs=2, lr_lora=0.0, lr_proj=1e-2, batch_size=4, grad_accum=1,
        device="cpu")
    assert all(torch.count_nonzero(columns) > 0 for columns in m.readers.values())


@pytest.mark.parametrize("delivery", ["lanes", "wide"])
def test_lanes_alone_teach_a_frozen_model(tiny_model, delivery):
    # Nothing of the base model is trained: the lane network and the reader columns (and, in the
    # wide delivery, the writer rows) are enough to make the output follow the signal.
    for p in tiny_model.parameters():
        p.requires_grad_(False)
    eps = []
    for i in range(32):
        hi = i % 2 == 0
        eps.append(make_episode(prompt_ids=[5, 6, 7], target_ids=[11 if hi else 22],
                                signals=[[0.95 if hi else 0.2] + [0.0] * 5] * 4))
    m = lane_model(tiny_model, delivery)
    losses = fit(m, eps, pad_id=0, epochs=60, lr_lora=1e-2, lr_proj=1e-2, batch_size=8, grad_accum=1, device="cpu")
    assert losses[-1] < losses[0] - 0.5  # a frozen random model has a weak output layer: the loss cannot reach 0
    b = collate(eps, pad_id=0, n_signals=6)
    assert m(b["input_ids"], b["signals"]).logits[:, 2].argmax(-1).tolist() == [11, 22] * 16
    m.enabled = False
    assert len(set(m(b["input_ids"], b["signals"]).logits[:, 2].argmax(-1).tolist())) == 1


@pytest.mark.parametrize("delivery", ["lanes", "wide"])
def test_lanes_resolve_a_limit_with_a_tiny_margin(tiny_model, delivery):
    rng = np.random.default_rng(0)

    def at(value, target):
        return make_episode(prompt_ids=[5, 6, 7], target_ids=[target], signals=[[value] + [0.0] * 5] * 4)

    eps = [at(0.901 + 0.099 * rng.random() ** 3, 11) if i % 2 == 0 else at(0.899 - 0.899 * rng.random() ** 3, 22)
           for i in range(256)]
    m = lane_model(tiny_model, delivery)
    fit(m, eps, pad_id=0, epochs=30, lr_lora=1e-3, lr_proj=1e-2, batch_size=16, grad_accum=1, device="cpu")
    probe = collate([at(0.899, 22), at(0.901, 11), at(0.85, 22), at(0.95, 11)], pad_id=0, n_signals=6)
    assert m(probe["input_ids"], probe["signals"]).logits[:, 2].argmax(-1).tolist() == [22, 11, 22, 11]


def test_fit_keeps_the_react_loss_of_each_signal(tiny_model):
    # The mean hides a signal that is learned while the others are not.
    def pair(signal, token):
        event = make_episode(kind="event", signal=signal, prompt_ids=[5, 6], target_ids=[7, token, 9], t_cross=0, t_diverge=1)
        return [event, make_episode(kind="nominal", prompt_ids=[5, 6], target_ids=[7, 10, 11], t_diverge=1)]

    history = {}
    fit(ChannelModel(tiny_model, 6), pair("ctx", 8) + pair("energy", 12) + pair("ctx", 8), pad_id=0, epochs=1,
        batch_size=2, grad_accum=1, device="cpu", history=history)
    assert {k: len(v) for k, v in history["react_by_signal"].items()} == {"ctx": 2, "energy": 1}
    assert sorted(history["react"]) == sorted(v for values in history["react_by_signal"].values() for v in values)


def test_save_load_run_roundtrip_with_lanes(tiny_model, tmp_path, monkeypatch):
    pristine = copy.deepcopy(tiny_model)
    m = lane_model(tiny_model)
    add_lora(m)
    fit(m, [ep([5, 6, 7], [11])] * 8, pad_id=0, epochs=2, lr_proj=1e-2, batch_size=4, grad_accum=1, device="cpu")
    x, s = torch.tensor([[5, 6, 7]]), torch.rand(1, 3, 6)
    before = m(x, s).logits
    save_run(m, tmp_path)
    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(pristine), None, None))
    m2, _, _ = load_run("unused", tmp_path, 6)
    assert m2.delivery == "lanes" and m2.lane_width == 8
    assert torch.allclose(before, m2(x, s).logits, atol=1e-5)
    assert not torch.allclose(before, pristine(x).logits, atol=1e-3)


def test_a_run_saved_before_lanes_existed_loads_as_sum(tiny_model, tmp_path, monkeypatch):
    pristine = copy.deepcopy(tiny_model)
    m = ChannelModel(tiny_model, 6)
    add_lora(m)
    arm(m)
    x, s = torch.tensor([[5, 6, 7]]), torch.rand(1, 3, 6)
    before = m(x, s).logits
    save_run(m, tmp_path)
    (tmp_path / "channel.json").unlink()  # runs of the first phase have no such record
    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(pristine), None, None))
    m2, _, _ = load_run("unused", tmp_path, 6)
    assert m2.delivery == "sum" and torch.allclose(before, m2(x, s).logits, atol=1e-5)


def test_the_command_line_trains_a_lane_run(tiny_model, tmp_path, monkeypatch):
    import json

    class Tok:
        pad_token_id = 0

    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(tiny_model), Tok(), None))
    monkeypatch.setattr(train_module, "read_jsonl", lambda path: [ep([5, 6, 7], [11, 12])] * 4)
    train_module.main(["--model", "m", "--data", str(tmp_path), "--n-signals", "6", "--out", str(tmp_path / "run"),
                       "--epochs", "1", "--delivery", "lanes", "--lane-width", "8"])
    assert json.loads((tmp_path / "run" / "channel.json").read_text(encoding="utf-8")) == {
        "delivery": "lanes", "lane_width": 8, "projector": "mlp"}
    state = torch.load(tmp_path / "run" / "channel.pt")
    assert any(k.startswith("readers.") for k in state) and state["net.4.weight"].shape[0] == 8
    log = json.loads((tmp_path / "run" / "train_log.json").read_text(encoding="utf-8"))
    assert "react_by_signal" in log
    # every option of the command is kept with the run, so a run describes itself
    assert log["args"]["data"] == str(tmp_path) and log["args"]["lane_width"] == 8
    assert log["args"]["batch_size"] == 2 and log["args"]["reaction_weight"] == 20.0


def test_channel_record_fills_the_defaults_of_older_runs(tmp_path):
    import json

    assert train_module.channel_record(tmp_path) == {"delivery": "sum", "lane_width": 64, "projector": "mlp"}
    (tmp_path / "channel.json").write_text(json.dumps({"delivery": "lanes", "lane_width": 8}), encoding="utf-8")
    assert train_module.channel_record(tmp_path) == {"delivery": "lanes", "lane_width": 8, "projector": "mlp"}


# A power cut must not cost more than one epoch: after every epoch `fit` writes all it needs to
# go on (trained parameters, optimizer, schedule, random states, history) to `state_path`.

def resumable(base, delivery="sum"):
    torch.manual_seed(0)  # the same initialization every time, as the command line does
    m = ChannelModel(copy.deepcopy(base), 6, delivery=delivery, lane_width=8)
    add_lora(m)
    return m


def varied_episodes():
    rng = np.random.default_rng(3)
    return [make_episode(prompt_ids=[5, 6, 7], target_ids=[int(rng.integers(10, 90))], signals=rng.random((4, 6)))
            for _ in range(12)]


FIT = dict(pad_id=0, epochs=3, lr_proj=1e-2, batch_size=4, grad_accum=1, device="cpu")


def cut_after(epoch_index):
    def power_cut(epoch):
        if epoch == epoch_index:
            raise RuntimeError("power cut")
    return power_cut


@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_an_interrupted_fit_resumes_to_the_same_result(tiny_model, tmp_path, delivery):
    eps = varied_episodes()
    whole = resumable(tiny_model, delivery)
    expected = fit(whole, eps, **FIT)
    state = tmp_path / "resume.pt"
    with pytest.raises(RuntimeError):
        fit(resumable(tiny_model, delivery), eps, state_path=state, on_epoch_end=cut_after(1), **FIT)
    again, seen = resumable(tiny_model, delivery), []
    losses = fit(again, eps, state_path=state, on_epoch_end=seen.append, **FIT)
    assert seen == [2]  # two epochs were already done
    assert losses == pytest.approx(expected, abs=1e-6)  # all nine batches, the first six from the state
    for (name, a), (_, b) in zip(whole.named_parameters(), again.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), name


def test_a_fit_whose_state_says_finished_trains_no_more(tiny_model, tmp_path):
    eps = varied_episodes()
    state = tmp_path / "resume.pt"
    first = resumable(tiny_model)
    expected = fit(first, eps, state_path=state, **FIT)
    again, seen = resumable(tiny_model), []
    losses = fit(again, eps, state_path=state, on_epoch_end=seen.append, **FIT)
    assert seen == [] and losses == pytest.approx(expected, abs=1e-9)
    for (name, a), (_, b) in zip(first.named_parameters(), again.named_parameters()):
        assert torch.equal(a, b), name


def test_fit_refuses_to_resume_with_other_settings(tiny_model, tmp_path):
    eps = varied_episodes()
    state = tmp_path / "resume.pt"
    with pytest.raises(RuntimeError):
        fit(resumable(tiny_model), eps, state_path=state, on_epoch_end=cut_after(0), **FIT)
    with pytest.raises(ValueError, match="epochs"):
        fit(resumable(tiny_model), eps, state_path=state, **{**FIT, "epochs": 4})
    with pytest.raises(ValueError, match="trainable parameters"):
        fit(resumable(tiny_model, "lanes"), eps, state_path=state, **FIT)


def test_a_state_write_that_is_cut_keeps_the_previous_state(tiny_model, tmp_path, monkeypatch):
    from pathlib import Path

    eps = varied_episodes()
    expected = fit(resumable(tiny_model), eps, **FIT)
    real_save, calls = torch.save, {"n": 0}

    def cut_on_second(obj, path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            Path(path).write_bytes(b"half a file")
            raise OSError("power cut")
        return real_save(obj, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", cut_on_second)
    state = tmp_path / "resume.pt"
    with pytest.raises(OSError):
        fit(resumable(tiny_model), eps, state_path=state, **FIT)
    monkeypatch.setattr(torch, "save", real_save)
    losses = fit(resumable(tiny_model), eps, state_path=state, **FIT)  # goes on after the first epoch
    assert losses == pytest.approx(expected, abs=1e-6)


def test_the_command_line_continues_an_interrupted_run(tiny_model, tmp_path, monkeypatch):
    import json

    class Tok:
        pad_token_id = 0

    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(tiny_model), Tok(), None))
    monkeypatch.setattr(train_module, "read_jsonl", lambda path: [ep([5, 6, 7], [11, 12]), ep([5, 6, 8], [13, 14])] * 2)

    def argv(name):
        return ["--model", "m", "--data", str(tmp_path), "--n-signals", "6", "--out", str(tmp_path / name),
                "--epochs", "3", "--batch-size", "2", "--grad-accum", "1"]

    train_module.main(argv("whole"))
    assert not (tmp_path / "whole" / "resume.pt").exists()  # a finished run keeps adapter and channel only

    real_save_run, calls = train_module.save_run, {"n": 0}

    def cut_on_second(model, out_dir):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("power cut")
        real_save_run(model, out_dir)

    monkeypatch.setattr(train_module, "save_run", cut_on_second)
    with pytest.raises(RuntimeError):
        train_module.main(argv("cut"))
    assert (tmp_path / "cut" / "resume.pt").exists()
    monkeypatch.setattr(train_module, "save_run", real_save_run)
    train_module.main(argv("cut"))  # the same command again
    assert not (tmp_path / "cut" / "resume.pt").exists()
    whole, cut = (torch.load(tmp_path / name / "channel.pt") for name in ("whole", "cut"))
    assert all(torch.allclose(whole[k], cut[k], atol=1e-6) for k in whole)
    logs = [json.loads((tmp_path / name / "train_log.json").read_text(encoding="utf-8")) for name in ("whole", "cut")]
    assert len(logs[1]["losses"]) == 6 and logs[1]["losses"] == pytest.approx(logs[0]["losses"], abs=1e-6)


# Wide delivery: the lanes travel inside the vector; layers read them and write to them.

def test_lora_with_the_wide_delivery_trains_adapter_projector_readers_and_writers(tiny_model):
    m = lane_model(tiny_model, "wide")
    add_lora(m)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all("lora_" in n or n.startswith(("projector.", "readers.", "writers.")) for n in trainable)
    assert all(any(part in n for n in trainable) for part in ("lora_", "projector.", "readers.", "writers."))


def test_columns_and_rows_of_the_wide_delivery_train_at_the_adapters_rate(tiny_model):
    # They are new columns and rows of the model's own matrices; only the signal network has the
    # channel's rate.
    eps = [ep([5, 6, 7], [11])] * 8

    def trained(lr_lora, lr_proj):
        m = lane_model(copy.deepcopy(tiny_model), "wide")
        arm(m, writers=False)  # writers only get a gradient once something reads the lanes
        before = {k: v.clone() for k, v in m.readers.items()}
        fit(m, eps, pad_id=0, epochs=2, lr_lora=lr_lora, lr_proj=lr_proj, batch_size=4, grad_accum=1, device="cpu")
        moved = any(not torch.equal(before[k], v) for k, v in m.readers.items())
        return moved, sorted(k for k, rows in m.writers.items() if torch.count_nonzero(rows) == 0)

    # Nothing reads the lanes after the last block's feed-forward: its writer never gets a gradient.
    assert trained(lr_lora=1e-2, lr_proj=0.0) == (True, ["transformer-h-1-mlp-c_proj"])
    readers_moved, silent = trained(lr_lora=0.0, lr_proj=1e-2)
    assert not readers_moved and len(silent) == 4


def test_save_load_run_roundtrip_with_the_wide_delivery(tiny_model, tmp_path, monkeypatch):
    pristine = copy.deepcopy(tiny_model)
    m = lane_model(tiny_model, "wide")
    add_lora(m)
    arm(m)
    fit(m, [ep([5, 6, 7], [11])] * 8, pad_id=0, epochs=2, lr_proj=1e-2, batch_size=4, grad_accum=1, device="cpu")
    x, s = torch.tensor([[5, 6, 7]]), torch.rand(1, 3, 6)
    before = m(x, s).logits
    save_run(m, tmp_path)
    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(pristine), None, None))
    m2, _, _ = load_run("unused", tmp_path, 6)
    assert m2.delivery == "wide" and m2.lane_width == 8 and len(m2.writers) == 4
    assert torch.allclose(before, m2(x, s).logits, atol=1e-5)
    assert not torch.allclose(before, pristine(x).logits, atol=1e-3)


def test_the_command_line_trains_a_wide_run(tiny_model, tmp_path, monkeypatch):
    import json

    class Tok:
        pad_token_id = 0

    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(tiny_model), Tok(), None))
    monkeypatch.setattr(train_module, "read_jsonl", lambda path: [ep([5, 6, 7], [11, 12])] * 4)
    train_module.main(["--model", "m", "--data", str(tmp_path), "--n-signals", "6", "--out", str(tmp_path / "run"),
                       "--epochs", "1", "--delivery", "wide", "--lane-width", "8"])
    assert json.loads((tmp_path / "run" / "channel.json").read_text(encoding="utf-8")) == {
        "delivery": "wide", "lane_width": 8, "projector": "mlp"}
    state = torch.load(tmp_path / "run" / "channel.pt")
    assert any(k.startswith("writers.") for k in state) and any(k.startswith("readers.") for k in state)
    assert json.loads((tmp_path / "run" / "train_log.json").read_text(encoding="utf-8"))["delivery"] == "wide"


# Levels projector: one learned vector per 1% level of each signal, no hidden layers.

@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_levels_projector_resolves_a_limit_with_a_tiny_margin(tiny_model, delivery):
    rng = np.random.default_rng(0)

    def at(value, target):
        return make_episode(prompt_ids=[5, 6, 7], target_ids=[target], signals=[[value] + [0.0] * 5] * 4)

    eps = [at(0.901 + 0.099 * rng.random() ** 3, 11) if i % 2 == 0 else at(0.899 - 0.899 * rng.random() ** 3, 22)
           for i in range(256)]
    m = ChannelModel(tiny_model, 6, delivery=delivery, lane_width=8, projector="levels")
    fit(m, eps, pad_id=0, epochs=30, lr_lora=1e-3, lr_proj=1e-2, batch_size=16, grad_accum=1, device="cpu")
    probe = collate([at(0.899, 22), at(0.901, 11), at(0.85, 22), at(0.95, 11)], pad_id=0, n_signals=6)
    assert m(probe["input_ids"], probe["signals"]).logits[:, 2].argmax(-1).tolist() == [22, 11, 22, 11]


def test_a_run_remembers_its_projector(tiny_model, tmp_path, monkeypatch):
    import json

    pristine = copy.deepcopy(tiny_model)
    m = ChannelModel(tiny_model, 6, projector="levels")
    add_lora(m)
    arm(m)
    x, s = torch.tensor([[5, 6, 7]]), torch.rand(1, 3, 6)
    before = m(x, s).logits
    save_run(m, tmp_path)
    assert json.loads((tmp_path / "channel.json").read_text(encoding="utf-8"))["projector"] == "levels"
    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(pristine), None, None))
    m2, _, _ = load_run("unused", tmp_path, 6)
    assert m2.projector_kind == "levels" and torch.allclose(before, m2(x, s).logits, atol=1e-5)


def test_the_command_line_trains_with_the_levels_projector(tiny_model, tmp_path, monkeypatch):
    import json

    class Tok:
        pad_token_id = 0

    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(tiny_model), Tok(), None))
    monkeypatch.setattr(train_module, "read_jsonl", lambda path: [ep([5, 6, 7], [11, 12])] * 4)
    train_module.main(["--model", "m", "--data", str(tmp_path), "--n-signals", "6", "--out", str(tmp_path / "run"),
                       "--epochs", "1", "--projector", "levels"])
    assert json.loads((tmp_path / "run" / "channel.json").read_text(encoding="utf-8")) == {
        "delivery": "sum", "lane_width": 64, "projector": "levels"}
    state = torch.load(tmp_path / "run" / "channel.pt")
    assert set(state) == {"net.0.weight", "net.0.bias"} and state["net.0.weight"].shape == (32, 600)
    assert json.loads((tmp_path / "run" / "train_log.json").read_text(encoding="utf-8"))["projector"] == "levels"


# One GPU, several runs to compare: a run can be paused at the end of an epoch and continued later.

def test_a_fit_paused_after_an_epoch_continues_to_the_same_result(tiny_model, tmp_path):
    eps = varied_episodes()
    expected = fit(resumable(tiny_model), eps, **FIT)
    state, history = tmp_path / "resume.pt", {}
    first = fit(resumable(tiny_model), eps, state_path=state, pause_after=1, history=history, **FIT)
    assert len(first) == 3 and history["epochs_done"] == 1  # one epoch of three batches, then it stopped
    history = {}
    losses = fit(resumable(tiny_model), eps, state_path=state, pause_after=1, history=history, **FIT)
    assert history["epochs_done"] == 3  # already past the pause point: it runs to the end
    assert losses == pytest.approx(expected, abs=1e-6)


def test_the_command_line_pauses_and_continues(tiny_model, tmp_path, monkeypatch):
    import json

    class Tok:
        pad_token_id = 0

    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(tiny_model), Tok(), None))
    monkeypatch.setattr(train_module, "read_jsonl", lambda path: [ep([5, 6, 7], [11, 12]), ep([5, 6, 8], [13, 14])] * 2)

    def argv(name, *more):
        return ["--model", "m", "--data", str(tmp_path), "--n-signals", "6", "--out", str(tmp_path / name),
                "--epochs", "3", "--batch-size", "2", "--grad-accum", "1", *more]

    train_module.main(argv("whole"))
    train_module.main(argv("paused", "--pause-after", "1", "--snapshot-minutes", "0"))
    run = tmp_path / "paused"
    assert (run / "resume.pt").exists() and (run / "channel.pt").exists() and not (run / "train_log.json").exists()
    train_module.main(argv("paused"))
    assert not (run / "resume.pt").exists()
    whole, paused = (torch.load(tmp_path / name / "channel.pt") for name in ("whole", "paused"))
    assert all(torch.allclose(whole[k], paused[k], atol=1e-6) for k in whole)
    assert len(json.loads((run / "train_log.json").read_text(encoding="utf-8"))["losses"]) == 6


# A configuration that fails at the default rate gets a second try at another one (sum + levels).

def test_the_command_line_sets_the_channel_learning_rate(tiny_model, tmp_path, monkeypatch):
    import json

    class Tok:
        pad_token_id = 0

    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(tiny_model), Tok(), None))
    monkeypatch.setattr(train_module, "read_jsonl", lambda path: [ep([5, 6, 7], [11, 12]), ep([5, 6, 8], [13, 14])] * 2)

    def argv(name, *more):
        return ["--model", "m", "--data", str(tmp_path), "--n-signals", "6", "--out", str(tmp_path / name),
                "--epochs", "2", "--batch-size", "2", "--grad-accum", "1", "--delivery", "lanes", *more]

    train_module.main(argv("default"))
    train_module.main(argv("frozen", "--lr-proj", "0"))
    torch.manual_seed(0)  # the command line seeds the channel's initialization with --seed (default 0)
    ChannelModel(copy.deepcopy(tiny_model), 6, delivery="lanes").save_channel(tmp_path / "start.pt")
    start, default, frozen = (torch.load(tmp_path / p) for p in ("start.pt", "default/channel.pt", "frozen/channel.pt"))
    assert all(torch.equal(start[k], frozen[k]) for k in start)  # rate 0: the channel stays where it started
    assert any(not torch.equal(start[k], default[k]) for k in start)
    logs = [json.loads((tmp_path / name / "train_log.json").read_text(encoding="utf-8")) for name in ("default", "frozen")]
    assert [log["lr_proj"] for log in logs] == [1e-3, 0.0]


# 2026-11-14: a training was killed one minute before its first per-epoch snapshot (another job
# on the machine had taken the memory). The state is now also written inside an epoch.

def cut_at_forward(model, call):
    real_forward, calls = model.forward, {"n": 0}

    def forward(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == call:
            raise RuntimeError("killed")
        return real_forward(*args, **kwargs)

    model.forward = forward
    return model


@pytest.mark.parametrize("grad_accum, cut_at, epochs_done, batches_done", [(1, 9, 1, 2), (2, 10, 1, 2), (2, 6, 0, 4)])
def test_a_fit_cut_inside_an_epoch_resumes_to_the_same_result(tiny_model, tmp_path, grad_accum, cut_at, epochs_done,
                                                              batches_done):
    eps = varied_episodes()
    setup = {**FIT, "batch_size": 2, "grad_accum": grad_accum}  # 6 batches per epoch, 3 epochs
    whole = resumable(tiny_model)
    expected = fit(whole, eps, **setup)
    state = tmp_path / "resume.pt"
    with pytest.raises(RuntimeError):
        fit(cut_at_forward(resumable(tiny_model), cut_at), eps, state_path=state, snapshot_minutes=0.0, **setup)
    saved = torch.load(state)
    assert (saved["epochs_done"], saved["batches_done"]) == (epochs_done, batches_done)  # the last optimizer step before the cut
    assert len(saved["losses"]) == 6 * epochs_done + batches_done
    again, seen = resumable(tiny_model), []
    losses = fit(again, eps, state_path=state, snapshot_minutes=0.0, on_epoch_end=seen.append, **setup)
    assert seen == list(range(epochs_done, 3))
    assert losses == pytest.approx(expected, abs=1e-6)
    for (name, a), (_, b) in zip(whole.named_parameters(), again.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), name


def test_without_snapshot_minutes_the_state_is_written_at_epoch_ends_only(tiny_model, tmp_path):
    eps = varied_episodes()
    state = tmp_path / "resume.pt"
    with pytest.raises(RuntimeError):
        fit(cut_at_forward(resumable(tiny_model), 3), eps, state_path=state, **FIT)
    assert not state.exists()
    with pytest.raises(RuntimeError):
        fit(cut_at_forward(resumable(tiny_model), 5), eps, state_path=state, **FIT)
    saved = torch.load(state)
    assert (saved["epochs_done"], saved["batches_done"]) == (1, 0)


def test_snapshots_inside_an_epoch_wait_for_their_interval(tiny_model, tmp_path):
    eps = varied_episodes()
    state = tmp_path / "resume.pt"
    with pytest.raises(RuntimeError):  # an hour between snapshots: none before the cut in the first epoch
        fit(cut_at_forward(resumable(tiny_model), 3), eps, state_path=state, snapshot_minutes=60.0, **FIT)
    assert not state.exists()


def test_collate_masks_harness_spans_and_weights_focus_spans():
    ep = make_episode(kind="bg_event", prompt_ids=[1, 2, 3], target_ids=[4, 5, 6, 7, 8, 9],
                      harness_spans=[[2, 4]], focus_spans=[[4, 6]])
    b = train_module.collate([ep], pad_id=0, n_signals=6, reaction_weight=20.0)
    assert b["labels"][0, 3:9].tolist() == [4, 5, -100, -100, 8, 9]
    assert b["weights"][0, 3:9].tolist() == [1.0, 1.0, 1.0, 1.0, 20.0, 20.0]


def test_collate_leaves_phase1_episodes_unchanged():
    ep = make_episode(kind="event", signal="ctx", t_cross=0, target_ids=[4, 5, 6])
    b = train_module.collate([ep], pad_id=0, n_signals=6, reaction_weight=20.0)
    assert b["weights"][0, 3:6].tolist() == [1.0, 20.0, 20.0]  # reaction from t_cross + 1, as before


def test_bg_event_decision_is_marked_as_a_reaction():  # final review C1: the epoch-2 gate needs react vs hold
    ev = make_episode(kind="bg_event", signal="done", t_cross=0, t_diverge=1, target_ids=[4, 5, 6], focus_spans=[[1, 3]])
    tw = make_episode(kind="bg_twin", signal="done", t_cross=0, t_diverge=1, target_ids=[4, 7, 8])
    b = train_module.collate([ev, tw], pad_id=0, n_signals=6, reaction_weight=20.0)
    assert b["decision"][0, 4].item() == 1 and b["decision"][1, 4].item() == 2


# Baseline deliveries and channel options (paper S2, B6a, B6b; program R6, R8, R10).

def test_collate_marks_the_harness_tokens():
    marked = make_episode(kind="bg_event", prompt_ids=[1, 2, 3], target_ids=[4, 5, 6, 7, 8, 9], harness_spans=[[2, 4]])
    b = train_module.collate([marked, ep([1], [2])], pad_id=0, n_signals=6)
    assert b["harness_mask"].dtype == torch.bool
    assert b["harness_mask"].tolist() == [[False] * 5 + [True] * 2 + [False] * 2, [False] * 9]


def test_fit_trains_the_segment_vector_on_harness_tokens_only(tiny_model):
    m = ChannelModel(tiny_model, 6, delivery="segment")
    fit(m, [ep([5, 6], [7, 8, 9])] * 4, pad_id=0, epochs=1, batch_size=2, grad_accum=1, device="cpu")
    assert torch.count_nonzero(m.segment) == 0  # no harness token, no gradient
    marked = make_episode(prompt_ids=[5, 6], target_ids=[7, 8, 9, 10], harness_spans=[[1, 3]])
    fit(m, [marked] * 4, pad_id=0, epochs=1, batch_size=2, grad_accum=1, device="cpu")
    assert torch.count_nonzero(m.segment) > 0


def test_fit_without_a_channel_trains_the_adapter_only(tiny_model):
    m = ChannelModel(tiny_model, 6, delivery="none")
    add_lora(m)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    before = {n: p.clone() for n, p in m.named_parameters() if "lora_B" in n}
    losses = fit(m, [ep([5, 6, 7], [11])] * 8, pad_id=0, epochs=2, batch_size=4, grad_accum=1, device="cpu")
    assert losses[-1] < losses[0]
    assert any(not torch.equal(before[n], p) for n, p in m.named_parameters() if n in before)


def run_inputs():
    x = torch.tensor([[5, 6, 7, 99, 12, 13]])
    harness = torch.zeros(1, 6, dtype=torch.bool)
    harness[0, 3] = True
    s = torch.rand(1, 6, 6, generator=torch.Generator().manual_seed(2))
    s[0, 3:, 0] = 0.95
    return x, s, harness


RUN_CONFIGS = {
    "none": dict(delivery="none"),
    "insert": dict(delivery="insert", insert_id=99, projector="levels"),
    "segment": dict(delivery="segment"),
    "sum-rms": dict(sum_norm="rms", projector="levels"),
    "wide-fourier": dict(delivery="wide", lane_width=8, projector="levels+fourier"),
}


@pytest.mark.parametrize("name", sorted(RUN_CONFIGS))
def test_save_load_run_roundtrip_of_the_new_configurations(tiny_model, tmp_path, monkeypatch, name):
    pristine = copy.deepcopy(tiny_model)
    m = ChannelModel(tiny_model, 6, **RUN_CONFIGS[name])
    add_lora(m)
    rng = np.random.default_rng(1)
    eps = [make_episode(prompt_ids=[5, 6, 7], target_ids=[99, 12, 13], harness_spans=[[0, 1]], signals=rng.random((6, 6)))
           for _ in range(8)]
    fit(m, eps, pad_id=0, epochs=2, lr_proj=1e-2, batch_size=4, grad_accum=1, device="cpu")
    x, s, harness = run_inputs()
    before = m(x, s, harness_mask=harness).logits
    save_run(m, tmp_path)
    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(pristine), None, None))
    m2, _, _ = load_run("unused", tmp_path, 6)
    assert (m2.delivery, m2.sum_norm, m2.insert_id, m2.projector_kind) == (m.delivery, m.sum_norm, m.insert_id,
                                                                            m.projector_kind)
    assert torch.allclose(before, m2(x, s, harness_mask=harness).logits, atol=1e-5)
    assert not torch.allclose(before, pristine(x).logits, atol=1e-3)


def test_a_steering_run_saves_and_loads_without_an_adapter(tiny_model, tmp_path, monkeypatch):
    import json

    from peft import PeftModel

    pristine = copy.deepcopy(tiny_model)
    m = ChannelModel(tiny_model, 6, delivery="steer", steer_layer=1)
    torch.manual_seed(3)
    m.set_steering(torch.randn(6, 32), alpha=2.0)
    x, s, _ = run_inputs()
    before = m(x, s).logits
    save_run(m, tmp_path)
    assert not (tmp_path / "lora").exists()  # training-free: no adapter, and never the whole base model
    assert json.loads((tmp_path / "channel.json").read_text(encoding="utf-8")) == {
        "delivery": "steer", "lane_width": 64, "projector": "mlp", "steer_layer": 1, "adapter": False}
    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(pristine), None, None))
    m2, _, _ = load_run("unused", tmp_path, 6)
    assert not isinstance(m2.base, PeftModel) and m2.steer_layer == 1
    assert torch.allclose(before, m2(x, s).logits, atol=1e-5)
    assert not torch.allclose(before, pristine(x).logits, atol=1e-3)


@pytest.mark.parametrize("projector", ["mlp", "levels"])
@pytest.mark.parametrize("delivery", ["sum", "lanes", "wide"])
def test_runs_of_the_older_kinds_keep_their_records(tiny_model, tmp_path, monkeypatch, delivery, projector):
    import json

    pristine = copy.deepcopy(tiny_model)
    m = ChannelModel(tiny_model, 6, delivery=delivery, lane_width=8, projector=projector)
    add_lora(m)
    arm(m)
    x, s = torch.tensor([[5, 6, 7]]), torch.rand(1, 3, 6)
    before = m(x, s).logits
    save_run(m, tmp_path)
    assert json.loads((tmp_path / "channel.json").read_text(encoding="utf-8")) == {
        "delivery": delivery, "lane_width": 8, "projector": projector}
    assert all(k.startswith(("net.", "readers.", "writers.")) for k in torch.load(tmp_path / "channel.pt"))
    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(pristine), None, None))
    m2, _, _ = load_run("unused", tmp_path, 6)
    assert torch.allclose(before, m2(x, s).logits, atol=1e-5)


def train_cli(tiny_model, tmp_path, monkeypatch, *more):
    import json

    class Tok:
        pad_token_id = 0

        def __len__(self):
            return 90  # the tiny model has 100 embedding rows: 90 is the first free one

    monkeypatch.setattr(train_module, "load_base", lambda *a, **k: (copy.deepcopy(tiny_model), Tok(), None))
    monkeypatch.setattr(train_module, "read_jsonl", lambda path: [ep([5, 6, 7], [11, 12])] * 4)
    train_module.main(["--model", "m", "--data", str(tmp_path), "--n-signals", "6", "--out", str(tmp_path / "run"),
                       "--epochs", "1", *more])
    run = tmp_path / "run"
    return (json.loads((run / "channel.json").read_text(encoding="utf-8")),
            json.loads((run / "train_log.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize("more, record", [
    (["--delivery", "none"], {"delivery": "none", "lane_width": 64, "projector": "mlp"}),
    (["--delivery", "insert", "--projector", "levels"],
     {"delivery": "insert", "lane_width": 64, "projector": "levels", "insert_id": 90}),
    (["--delivery", "segment"], {"delivery": "segment", "lane_width": 64, "projector": "mlp"}),
    (["--sum-norm", "rms", "--projector", "levels"],
     {"delivery": "sum", "lane_width": 64, "projector": "levels", "sum_norm": "rms"}),
    (["--delivery", "wide", "--lane-width", "8", "--projector", "levels+fourier"],
     {"delivery": "wide", "lane_width": 8, "projector": "levels+fourier"}),
])
def test_the_command_line_trains_the_new_configurations(tiny_model, tmp_path, monkeypatch, more, record):
    channel, log = train_cli(tiny_model, tmp_path, monkeypatch, *more)
    assert channel == record
    assert (log["args"]["delivery"], log["args"]["projector"], log["args"]["sum_norm"]) == (
        record["delivery"], record["projector"], record.get("sum_norm"))


def test_the_command_line_refuses_to_train_a_steering_run(tiny_model, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):  # training-free: its vectors come from the base model's activations
        train_cli(tiny_model, tmp_path, monkeypatch, "--delivery", "steer")
