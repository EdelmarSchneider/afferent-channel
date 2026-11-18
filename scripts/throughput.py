"""Tokens per second of the generation loop with the channel on vs. off (plan Task 12, step 5).

Same trained model, same prompt, same number of new tokens, greedy, never stops at EOS.
Usage: uv run python scripts/throughput.py runs/n6-s0 results/phase1 [n_tokens] [repeats] [n_signals]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from prio.generate import generate
from prio.reactions import TOOLS
from prio.signals import signal_set
from prio.tasks import SYSTEM_PROMPT
from prio.train import load_run
from prio.trajectories import sample_trajectory

PROMPT = "Write a long, detailed history of the bicycle, decade by decade, from 1817 to today."


def timed(model, ids, values, n_tokens: int) -> float:
    """Tokens per second of one generation of exactly `n_tokens` new tokens."""
    n_in = ids.shape[1]
    signals = torch.from_numpy(values[None, :n_in]).to(ids.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = generate(model, ids, signals, lambda step, _: values[n_in + step], n_tokens, eos_ids=())
    torch.cuda.synchronize()
    assert len(out.token_ids) == n_tokens
    return n_tokens / (time.perf_counter() - started)


def main() -> None:
    run_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    model_id = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))["model"]
    n_signals = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    specs = signal_set(n_signals)
    model, tok, _ = load_run(model_id, run_dir, n_signals)
    device = model.projector.net[0].weight.device
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=True, tokenize=False)
    ids = torch.tensor([tok(text, add_special_tokens=False).input_ids], device=device)
    values = sample_trajectory(specs, ids.shape[1] + n_tokens, np.random.default_rng(0)).values

    timed(model, ids, values, 64)  # warm-up: CUDA kernels, allocator
    rates = {"on": [], "off": []}
    for _ in range(repeats):  # interleaved, so drift (thermal, other processes) hits both alike
        for name in ("on", "off"):
            model.enabled = name == "on"
            rates[name].append(timed(model, ids, values, n_tokens))
    model.enabled = True

    on, off = float(np.median(rates["on"])), float(np.median(rates["off"]))
    result = {"model": model_id, "n_signals": n_signals, "new_tokens": n_tokens, "repeats": repeats,
              "tokens_per_s_on": rates["on"], "tokens_per_s_off": rates["off"],
              "median_on": on, "median_off": off, "overhead": 1 - on / off}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "throughput.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"channel on : {on:.1f} tokens/s (runs: {[round(r, 1) for r in rates['on']]})")
    print(f"channel off: {off:.1f} tokens/s (runs: {[round(r, 1) for r in rates['off']]})")
    print(f"overhead   : {result['overhead']:.1%}")


if __name__ == "__main__":
    main()
