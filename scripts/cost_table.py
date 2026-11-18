"""Analytic cost of sending the signals as text versus through the channel (spec G6)."""

from pathlib import Path
from typing import Sequence

import numpy as np

from prio.signals import SignalSpec, signal_set

MODEL_ID = "Qwen/Qwen3-1.7B"
SIGNAL_COUNTS = (6, 16, 64)
INTERVALS = (1, 10, 50, 200)  # generated tokens between two text updates
PROJECTOR_HIDDEN = 256
PROJECTOR_BINS = 100  # piecewise-linear encoding: features per signal


def render_status(specs: Sequence[SignalSpec], values: Sequence[float]) -> str:
    """Compact text form of one status update."""
    return "[status " + " ".join(f"{s.name}={v:.2f}" for s, v in zip(specs, values)) + "]"


def text_update_tokens(tok, specs: Sequence[SignalSpec], n_samples: int = 200, seed: int = 0) -> float:
    """Mean token count of one text update, over random signal values."""
    rng = np.random.default_rng(seed)
    counts = [
        len(tok.encode(render_status(specs, rng.uniform(0, 1, len(specs))), add_special_tokens=False))
        for _ in range(n_samples)
    ]
    return float(np.mean(counts))


def notice_tokens(tok, specs: Sequence[SignalSpec], n_samples: int = 50, seed: int = 0) -> float:
    """Mean token count of an event-only notice: the status format restricted to the one signal
    that crossed (e.g. `[status ctx=0.90]`), over every signal and random values."""
    rng = np.random.default_rng(seed)
    counts = [len(tok.encode(render_status([s], [v]), add_special_tokens=False))
              for s in specs for v in rng.uniform(0, 1, n_samples)]
    return float(np.mean(counts))


def overhead_fraction(update_tokens: float, interval: int) -> float:
    """Share of the context spent on status text when one update is sent every `interval` tokens."""
    return update_tokens / (update_tokens + interval)


def projector_params(n_signals: int, d_model: int, hidden: int = PROJECTOR_HIDDEN, bins: int = PROJECTOR_BINS) -> int:
    """First design (sum + mlp): the projector is the whole channel."""
    return (n_signals * bins * hidden + hidden) + (hidden * hidden + hidden) + (hidden * d_model + d_model)


def channel_params(base, n_signals: int, delivery: str = "wide", projector: str = "levels", lane_width: int = 64) -> int:
    """Every parameter the channel adds to `base` (projector + lane readers + lane writers = all of
    channel.pt). `base` may live on the meta device."""
    from prio.channel import ChannelModel

    model = ChannelModel(base, n_signals, delivery=delivery, projector=projector, lane_width=lane_width)
    return sum(p.numel() for part in (model.projector, model.readers, model.writers) for p in part.parameters())


def main() -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    d_model = config.hidden_size
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(config)
    base_params = sum(p.numel() for p in base.parameters())
    header = ("| Signals | Tokens per text update | " + " | ".join(f"every {k}" for k in INTERVALS)
              + " | Channel tokens | Channel parameters, main (wide + levels) | Channel parameters, first design (sum + mlp) |")
    lines = [
        "# Cost of signals as text vs. through the channel", "",
        f"Tokenizer: `{MODEL_ID}`. Text format: `{render_status(signal_set(6)[:2], [0.5, 0.07])[:-1]} ...]`, two decimals.",
        "Columns `every k`: share of the context taken by status text when one update is sent every k generated tokens.",
        "", header, "|" + "---|" * (len(INTERVALS) + 5),
    ]
    main_params = {}
    for n in SIGNAL_COUNTS:
        update = text_update_tokens(tok, signal_set(n))
        shares = " | ".join(f"{overhead_fraction(update, k):.1%}" for k in INTERVALS)
        main_params[n] = channel_params(base, n)
        lines.append(f"| {n} | {update:.1f} | {shares} | 0 | {main_params[n]:,} | {projector_params(n, d_model):,} |")
    lines += [
        "",
        "The channel updates at every token (interval 1) at zero context cost. Its cost is parameters and compute. "
        f"Main configuration (wide + levels, {SIGNAL_COUNTS[0]} signals): {main_params[SIGNAL_COUNTS[0]]:,} parameters, "
        f"{main_params[SIGNAL_COUNTS[0]] / base_params:.1%} of the base model's {base_params:,}: a lookup table "
        "(signals x 100 levels x 64 lanes) plus 64 extra input columns or output rows on every attention and MLP projection "
        "of every layer. Measured cost: generation 27.9% slower at batch 1 (tokens per second, channel on vs. off; "
        "`results/phase1/throughput.json`). The first design (sum + mlp: one small MLP per token) is listed for reference; "
        "it did not learn the limit crossings (`results/phase1/report.md`).",
    ]
    notice = notice_tokens(tok, signal_set(SIGNAL_COUNTS[0]))
    lines += [
        "", "## Event-only text notices", "",
        "One notice per limit crossing (once more on recovery for the tool-gating signals), naming only the signal "
        f"that crossed, in the format above: `{render_status(signal_set(6)[:1], [0.9])}`. Nothing is sent between events.",
        "", "| Tokens per notice | one crossing per 100 tokens | one crossing per 1,000 tokens | Channel tokens |",
        "|---|---|---|---|",
        f"| {notice:.1f} | {overhead_fraction(notice, 100):.1%} | {overhead_fraction(notice, 1000):.1%} | 0 |",
        "",
        "Our test episodes hold one event each. For rare threshold events a text notice is cheap in tokens; "
        "the `every k` columns above are the cost of keeping a continuous state current in the text.",
    ]
    out = Path("results/cost_table.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
