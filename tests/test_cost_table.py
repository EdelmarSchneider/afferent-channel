import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cost_table import overhead_fraction, render_status, text_update_tokens  # noqa: E402

from helpers import FAKE_TOK  # noqa: E402
from prio.signals import CORE_SIGNALS, signal_set  # noqa: E402


def test_overhead_fraction():
    assert overhead_fraction(240, 50) == pytest.approx(240 / 290)
    assert overhead_fraction(0, 50) == 0.0


def test_render_status_format():
    assert render_status(CORE_SIGNALS[:2], [0.5, 0.071]) == "[status ctx=0.50 think=0.07]"


def test_projector_params_match_the_real_module():
    from cost_table import projector_params
    from prio.channel import SignalProjector

    real = sum(p.numel() for p in SignalProjector(6, 64).parameters())
    assert projector_params(6, 64) == real


def test_channel_params_count_every_saved_channel_tensor(tiny_model, tmp_path):
    # Main configuration (wide + levels): projector + lane readers + lane writers, i.e. all of channel.pt.
    import torch

    from cost_table import channel_params
    from prio.channel import ChannelModel

    ChannelModel(tiny_model, 6, delivery="wide", projector="levels").save_channel(tmp_path / "c.pt")
    saved = sum(t.numel() for t in torch.load(tmp_path / "c.pt").values())
    assert channel_params(tiny_model, 6, delivery="wide", projector="levels") == saved


def test_channel_params_need_no_weights():
    # The script counts on a meta-device model: no 3.4 GB download, no memory.
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from cost_table import channel_params

    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    with torch.device("meta"):
        meta = GPT2LMHeadModel(config)
    real = GPT2LMHeadModel(config)
    assert channel_params(meta, 6) == channel_params(real, 6) > 6 * 100 * 64


def test_event_notice_names_only_the_crossing_signal():
    # Event-only text notice (paper plan s7): one signal per notice, whatever the set size.
    from cost_table import notice_tokens

    class CountingTok:
        def encode(self, text, **_):
            return FAKE_TOK.encode(text)

    assert notice_tokens(CountingTok(), signal_set(6), n_samples=3) == 2  # "[status" + "ctx=0.90]"
    assert notice_tokens(CountingTok(), signal_set(16), n_samples=3) == 2


def test_text_update_tokens_grows_with_signal_count():
    class CountingTok:  # one token per whitespace-separated field
        def encode(self, text, **_):
            return FAKE_TOK.encode(text)

    six = text_update_tokens(CountingTok(), signal_set(6), n_samples=5)
    sixteen = text_update_tokens(CountingTok(), signal_set(16), n_samples=5)
    assert six == 7 and sixteen == 17  # "[status" + one field per signal
