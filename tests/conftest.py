import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from helpers import FAKE_MS, FAKE_TOK


@pytest.fixture
def tiny_model():
    """Tiny random GPT-2 on CPU: fast, offline, deterministic."""
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=100, n_positions=512, n_embd=32, n_layer=2, n_head=2,
                        bos_token_id=0, eos_token_id=0)
    return GPT2LMHeadModel(config).eval()


@pytest.fixture
def tok():
    return FAKE_TOK


@pytest.fixture
def ms():
    return FAKE_MS
