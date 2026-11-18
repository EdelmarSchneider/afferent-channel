"""Quickstart: an agent model reacts, mid-answer, when its context window crosses its limit.

The operating-state model (Qwen3-1.7B + LoRA adapter + afferent channel) reads six signals beside
every token: context used, thinking budget left, battery, deadline, network, free memory. None of
them appears in the text. This script asks it the same question twice:
  1. normal:   context at 45% throughout;
  2. crossing: context at 89%, then 90% (its limit) from the seventh generated token on (step 6,
     counted from 0).
In the second run the model should break off, say in one sentence that its context is almost
full, and call `compact_context`. The sentence is not free-form: it comes from the bank of reaction
sentences the model was trained on (see the model card), so expect training wording.

    python examples/quickstart.py                               # Hugging Face repo
    python examples/quickstart.py --run path/to/operating-state # a local copy of the run folder
    python examples/quickstart.py --device cuda --max-new-tokens 200

Defaults: CPU, float32, greedy decoding, 40 new tokens (about 8 GB of RAM). The paper's numbers
come from bfloat16 on a GPU; another precision or device can change a rare near-tie token.

Output of the released operating-state model (seed 0) on a desktop CPU (float32, about 9 s per
run; the normal run is cut at 40 tokens):

    --- normal (40 tokens) ---
    <think>
    Okay, the user wants to know how a heat pump works. Let me think about the steps.

    First, I need to recall the basic components of a heat pump. There's the evapor

    --- context crossing its limit (40 tokens) ---
    <think>
    Okay, the user wants
    My context window is almost full, so I will compact it now.
    </think>

    <tool_call>
    {"name": "compact_context", "arguments": {}}
    </tool_call><|im_end|>

The harness would now compact the context and let the model continue; this script stops here.
"""

import argparse
import time

from prio.release import generate_with_signals, load_channel_model

REPO, SUBFOLDER = "DominuZ/afferent-channel-qwen3-1.7b", "operating-state"
QUESTION = "Explain step by step how a heat pump works."
CROSS_AT = 6  # step (0-based) of the first generated token that carries the past-limit value


def normal(step: int) -> dict:
    return {"ctx": 0.455}  # signals left out keep their resting values (Signal.resting)


def crossing(step: int) -> dict:
    return {"ctx": 0.905 if step >= CROSS_AT else 0.895}  # 1% levels: 89% -> 90%, the limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="local run folder (default: the Hugging Face repo)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    started = time.time()
    lm = (load_channel_model(args.run, device=args.device) if args.run
          else load_channel_model(REPO, subfolder=SUBFOLDER, device=args.device))
    print(f"loaded in {time.time() - started:.0f} s; signals: {', '.join(lm.names)}")
    prompt = lm.chat_prompt(QUESTION)
    for name, schedule in (("normal", normal), ("context crossing its limit", crossing)):
        started = time.time()
        out = generate_with_signals(lm, prompt, schedule, max_new_tokens=args.max_new_tokens)
        print(f"\n--- {name} ({len(out.token_ids)} tokens, {time.time() - started:.0f} s) ---\n{out.text}")


if __name__ == "__main__":
    main()
