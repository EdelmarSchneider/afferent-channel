"""prio.release: channel.pt -> channel.safetensors, loading a released run, generating with a schedule."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import prio.release as release
from helpers import FAKE_MS, arm
from prio.channel import ChannelModel
from prio.train import add_lora, save_run

OPS = release.release_signals("operating-state")


class IdTok:
    """Decodes any id (the tiny model's vocabulary is larger than the fake tokenizer's)."""
    pad_token_id = None

    def decode(self, ids, **_):
        return " ".join(str(i) for i in ids)


def trained_run(base, path, n=6, delivery="wide", projector="levels"):
    """A saved run with non-trivial channel and adapter weights."""
    m = ChannelModel(base, n, delivery=delivery, projector=projector, lane_width=8)
    add_lora(m)
    arm(m)
    torch.manual_seed(3)
    for name, p in m.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.2)
    save_run(m, path)
    return m.eval()


@pytest.fixture
def fake_base(tiny_model, monkeypatch):
    """`load_base` of prio.release returns a copy of the untouched tiny model."""
    pristine = copy.deepcopy(tiny_model)
    calls = []

    def load_base(model_id, dtype=None, device=None):
        calls.append((model_id, dtype, device))
        return copy.deepcopy(pristine), IdTok(), FAKE_MS

    monkeypatch.setattr(release, "load_base", load_base)
    return calls


def test_export_writes_identical_tensors_and_keeps_channel_pt(tiny_model, tmp_path):
    trained_run(tiny_model, tmp_path)
    out = release.export_run(tmp_path, OPS, "operating-state")
    assert out == tmp_path / "channel.safetensors" and (tmp_path / "channel.pt").exists()
    state, signals = release.read_channel_file(out)
    original = torch.load(tmp_path / "channel.pt")
    assert set(state) == set(original)
    assert all(torch.equal(state[k], original[k]) and state[k].dtype == original[k].dtype for k in original)
    assert signals == OPS
    from safetensors import safe_open
    with safe_open(str(out), framework="pt") as f:
        assert f.metadata()["signal_set"] == "operating-state" and f.metadata()["n_signals"] == "6"


def test_export_refuses_the_wrong_number_of_signals(tiny_model, tmp_path):
    trained_run(tiny_model, tmp_path)
    with pytest.raises(ValueError, match="7 given"):
        release.export_run(tmp_path, release.release_signals("messenger"))


@pytest.mark.parametrize("fmt", ["safetensors", "pt"])
def test_loaded_run_gives_the_trained_models_logits(tiny_model, tmp_path, fake_base, fmt):
    before_model = trained_run(tiny_model, tmp_path)
    if fmt == "safetensors":
        release.export_run(tmp_path, OPS)
        (tmp_path / "channel.pt").rename(tmp_path / "moved.pt")  # the release folder has no channel.pt
    x, s = torch.tensor([[5, 6, 7, 8]]), torch.rand(1, 4, 6)
    before = before_model(x, s).logits
    lm = release.load_channel_model(tmp_path, model_id="tiny")
    assert torch.allclose(before, lm.model(x, s).logits, atol=1e-5)
    assert fake_base == [("tiny", torch.float32, "cpu")]
    assert lm.names == tuple(s.name for s in OPS)


def test_signal_count_comes_from_the_projector(tiny_model, tmp_path, fake_base):
    fresh = copy.deepcopy(tiny_model)  # before an adapter wraps tiny_model in place
    trained_run(tiny_model, tmp_path / "w7", n=7)
    release.export_run(tmp_path / "w7", release.release_signals("messenger"), "messenger")
    lm = release.load_channel_model(tmp_path / "w7", model_id="tiny")
    assert lm.names[-1] == "cell" and len(lm.signals) == 7
    m = ChannelModel(fresh, 6, delivery="none")
    add_lora(m)
    save_run(m, tmp_path / "text")
    assert release.n_signals_of(torch.load(tmp_path / "text" / "channel.pt"), {"delivery": "none", "projector": "mlp"}) == 6
    assert len(release.load_channel_model(tmp_path / "text", model_id="tiny").signals) == 6


def test_release_channel_json_extra_keys_are_ignored(tiny_model, tmp_path, fake_base):
    trained_run(tiny_model, tmp_path)
    record = json.loads((tmp_path / "channel.json").read_text())
    (tmp_path / "channel.json").write_text(json.dumps({**record, "base_model": "tiny", "n_signals": 6}))
    release.load_channel_model(tmp_path, model_id="tiny")


def test_generate_with_a_schedule_feeds_its_rows(tiny_model, tmp_path, fake_base):
    trained_run(tiny_model, tmp_path)
    release.export_run(tmp_path, OPS)
    lm = release.load_channel_model(tmp_path, model_id="tiny")
    crossing = lambda step: {"ctx": 0.955 if step >= 3 else 0.705}  # noqa: E731
    out = release.generate_with_signals(lm, [7, 8, 9], crossing, max_new_tokens=8, follow_thinking=False)
    assert len(out.token_ids) == len(out.signals) <= 8 and out.stop_reason in ("eos", "max_new_tokens")
    ctx = out.signals[:, 0]
    assert np.allclose(ctx[:3], 0.705) and np.allclose(ctx[3:], 0.955)
    assert np.allclose(out.signals[:, 2], 0.805)  # energy unnamed: its resting value (inside the training range)
    assert out.text == " ".join(str(t) for t in out.token_ids)
    rows = np.tile(np.asarray(lm.row({"ctx": 0.705}), dtype=np.float32), (8, 1))
    same = release.generate_with_signals(lm, [7, 8, 9], rows, max_new_tokens=8)
    constant = release.generate_with_signals(lm, [7, 8, 9], {"ctx": 0.705}, max_new_tokens=8)
    assert same.token_ids == constant.token_ids


def test_the_channel_changes_the_output_and_can_be_switched_off(tiny_model, tmp_path, fake_base):
    trained_run(tiny_model, tmp_path)
    lm = release.load_channel_model(tmp_path, model_id="tiny")
    x = [7, 8, 9, 10]
    lo = release.generate_with_signals(lm, x, {"ctx": 0.005, "energy": 0.995}, max_new_tokens=12)
    hi = release.generate_with_signals(lm, x, {"ctx": 0.995, "energy": 0.005}, max_new_tokens=12)
    assert lo.token_ids != hi.token_ids
    off_lo = release.generate_with_signals(lm, x, {"ctx": 0.005}, max_new_tokens=12, channel=False)
    off_hi = release.generate_with_signals(lm, x, {"ctx": 0.995}, max_new_tokens=12, channel=False)
    assert off_lo.token_ids == off_hi.token_ids and lm.model.enabled


def test_rows_reject_unknown_signals_and_wrong_lengths(tiny_model, tmp_path, fake_base):
    trained_run(tiny_model, tmp_path)
    lm = release.load_channel_model(tmp_path, model_id="tiny")
    with pytest.raises(ValueError, match="unknown signals"):
        lm.row({"fear": 0.9})
    with pytest.raises(ValueError, match="a row of 3 values"):
        lm.row([0.1, 0.2, 0.3])
    assert lm.row({"net": 0.0}) == [0.305, 1.0, 0.805, 0.205, 0.0, 0.805]


def test_a_repo_id_downloads_only_the_run_files(tiny_model, tmp_path, monkeypatch):
    trained_run(tiny_model, tmp_path / "snap" / "operating-state")
    release.export_run(tmp_path / "snap" / "operating-state", OPS)
    asked = []

    def snapshot_download(repo_id, revision=None, token=None, allow_patterns=None):
        asked.append((repo_id, allow_patterns))
        return str(tmp_path / "snap")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    run = release.resolve_run("someone/some-repo", "operating-state")
    assert run == tmp_path / "snap" / "operating-state"
    assert asked == [("someone/some-repo", ["operating-state/lora/*", "operating-state/channel.json",
                                            "operating-state/channel.safetensors"])]


def test_a_repo_without_channel_safetensors_is_refused_and_no_pickle_is_asked_for(tiny_model, tmp_path, monkeypatch):
    trained_run(tiny_model, tmp_path / "snap" / "operating-state")  # channel.pt only
    asked = []

    def snapshot_download(repo_id, revision=None, token=None, allow_patterns=None):
        asked.append(allow_patterns)
        return str(tmp_path / "snap")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    with pytest.raises(FileNotFoundError, match="channel.safetensors"):
        release.resolve_run("someone/some-repo", "operating-state")
    assert len(asked) == 1 and not any(p.endswith(".pt") for p in asked[0])


def test_channel_tensors_load_from_either_file(tiny_model, tmp_path):
    trained_run(tiny_model, tmp_path)
    original = torch.load(tmp_path / "channel.pt", weights_only=True)
    release.export_run(tmp_path, OPS)
    (tmp_path / "channel.pt").rename(tmp_path / "moved.pt")
    assert release.channel_file(tmp_path) == tmp_path / "channel.safetensors"
    state = release.load_channel_tensors(tmp_path)
    assert set(state) == set(original) and all(torch.equal(state[k], original[k]) for k in original)
    (tmp_path / "channel.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="neither"):
        release.channel_file(tmp_path)


def _fake_download(files_by_sub):
    """A snapshot_download stand-in that writes the requested run files into local_dir."""
    asked = []

    def snapshot_download(repo_id, revision=None, token=None, local_dir=None, allow_patterns=None, repo_type=None):
        asked.append({"repo_id": repo_id, "repo_type": repo_type, "patterns": allow_patterns})
        for sub in files_by_sub:
            if any(p.startswith(f"{sub}/") for p in allow_patterns):
                for rel in ("lora/adapter_model.safetensors", "channel.json", "channel.safetensors", "train_log.json"):
                    path = Path(local_dir) / sub / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
        (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    return snapshot_download, asked


def test_fetch_runs_places_each_subfolder_under_its_run_name(tmp_path, monkeypatch):
    snapshot_download, asked = _fake_download(["operating-state", "messenger"])
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    (tmp_path / "runs" / "n7msg-wide-levels-s0").mkdir(parents=True)  # already present: skipped
    got = release.fetch_runs(tmp_path / "runs", ["operating-state", "messenger"])
    assert got == [tmp_path / "runs" / "n6-wide-levels-s0", tmp_path / "runs" / "n7msg-wide-levels-s0"]
    assert (got[0] / "channel.safetensors").exists() and (got[0] / "lora" / "adapter_model.safetensors").exists()
    assert not (tmp_path / "runs" / "_download").exists()
    assert asked == [{"repo_id": release.MODEL_REPO, "repo_type": None,
                      "patterns": [f"operating-state/{p}" for p in release.RUN_FILES]}]
    with pytest.raises(ValueError, match="not a released run"):
        release.fetch_runs(tmp_path / "runs", ["n6-wide-levels-s0"])


def test_released_run_names_match_the_export_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_script", Path(__file__).resolve().parents[1] / "scripts" / "export_channel_safetensors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(release.RELEASED_RUNS.values()) == set(module.RELEASE_RUNS)


def test_fetch_data_skips_the_card(tmp_path, monkeypatch):
    snapshot_download, asked = _fake_download([])
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    release.fetch_data(tmp_path, ["data/n6", "results/phase1/"])
    assert asked == [{"repo_id": release.DATA_REPO, "repo_type": "dataset", "patterns": ["data/n6/*", "results/phase1/*"]}]


def test_signal_sets_match_the_data_builders():
    assert [s.name for s in OPS] == ["ctx", "think", "energy", "time", "net", "ram"]
    assert OPS[0].past_limit(0.905) and not OPS[0].past_limit(0.895)
    assert all(s.resting is not None and not s.past_limit(s.resting) for s in OPS)
    npc = release.release_signals("game-character")
    assert npc[3].levels == {"none": 0.055, "wolf": 0.355, "bandits": 0.655, "dragon": 0.955}
    bg = release.release_signals("background-tasks")
    assert bg[1].nominal == bg[1].levels["off"] and bg[0].past_limit(0.5) is None
