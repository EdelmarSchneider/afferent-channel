"""A published run folder (channel.safetensors, no channel.pt) loads through `prio.train.load_run`,
the loader of every evaluation and analysis command, and gives the trained model's logits."""

import copy

import torch

import prio.release as release
import prio.train as train
from helpers import FAKE_MS
from test_release import IdTok, trained_run


def test_load_run_reads_a_published_run_folder(tiny_model, tmp_path, monkeypatch):
    pristine = copy.deepcopy(tiny_model)
    monkeypatch.setattr(train, "load_base", lambda model_id: (copy.deepcopy(pristine), IdTok(), FAKE_MS))
    before_model = trained_run(tiny_model, tmp_path)
    release.export_run(tmp_path, release.release_signals("operating-state"))
    (tmp_path / "channel.pt").rename(tmp_path / "moved.pt")  # a published folder has no channel.pt
    x, s = torch.tensor([[5, 6, 7, 8]]), torch.rand(1, 4, 6)
    model, _, _ = train.load_run("tiny", tmp_path, 6)
    assert torch.allclose(before_model(x, s).logits, model(x, s).logits, atol=1e-5)
