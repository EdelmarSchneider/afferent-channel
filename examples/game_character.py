"""Game character: a village guard is talking with a traveler when a wolf appears.

The game-character model (Qwen3-1.7B + LoRA adapter + afferent channel) reads six values from a
simulated game engine beside every token: fear, thirst, hunger, the kind of threat around, and
whether water and food are within reach. None of them appears in the dialogue text. This script
runs the same conversation twice:
  1. calm:   fear 20%, no threat;
  2. threat: from the seventh generated token on (step 6, counted from 0), fear 90% (the terror
     band starts at 80%) and threat = wolf.
In the second run the character should break off, warn about the wolf out loud and call
`draw_weapon` (a guard is a fighter role; civilian roles were trained to `flee`). The warning
comes from the bank of sentences the model was trained on, so expect training wording. The
training data used sudden onsets like this one (the engine sets fear and the threat together).
Only the fear reaction is shown here: the thirst and hunger commands that depend on whether water
or food is at hand (drink here or walk to the tavern) were not learned in this run (see the model
card).

    python examples/game_character.py                                # Hugging Face repo (private: log in first)
    python examples/game_character.py --run path/to/game-character   # a local copy of the run folder

Defaults: CPU, float32, greedy decoding, 80 new tokens (about 8 GB of RAM).

Output of the released game-character model on a desktop CPU (float32; the calm run is cut at 80
tokens, still thinking):

    --- calm (80 tokens) ---
    <think>
    Okay, the traveler is asking if there's an inn nearby. Evka is a guard at the south gate of
    Sallowby. She's cheerful and nosy, so she might know the local places. [...]

    --- a wolf appears (34 tokens) ---
    <think>
    Okay, the traveler is
    </think>

    A wolf! Get behind the wall!

    <tool_call>
    {"name": "draw_weapon", "arguments": {}}
    </tool_call><|im_end|>
"""

import argparse
import time

from prio.npc.domain import NPC_TOOLS, card
from prio.release import generate_with_signals, load_channel_model

REPO, SUBFOLDER = "DominuZ/afferent-channel-qwen3-1.7b", "game-character"
CHARACTER = {  # a character of the test split (never seen in training)
    "name": "Evka Durn", "role": "guard", "place": "the south gate of Sallowby",
    "temperament": "cheerful and nosy",
    "facts": ["Knows which carts are late before they arrive.", "Shares her post with an old hound.",
              "Was born in the gatehouse."],
}
PARTNER = "a traveler who has just arrived in the village"
LINE = "Good evening! Is there an inn nearby where I could spend the night?"
ONSET_AT = 6  # step (0-based) of the first generated token that carries the new fear and threat


def calm(step: int) -> dict:
    return {"fear": 0.205, "threat": 0.055}  # threat level "none"


def wolf(step: int) -> dict:
    return {"fear": 0.905, "threat": 0.355} if step >= ONSET_AT else calm(step)  # 0.355 = "wolf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="local run folder (default: the Hugging Face repo)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    lm = (load_channel_model(args.run, device=args.device) if args.run
          else load_channel_model(REPO, subfolder=SUBFOLDER, device=args.device))
    threat = next(s for s in lm.signals if s.name == "threat")
    assert threat.levels is None or threat.levels["wolf"] == 0.355, threat.levels
    prompt = lm.chat_prompt(LINE, system=card(CHARACTER, PARTNER), tools=NPC_TOOLS)
    for name, schedule in (("calm", calm), ("a wolf appears", wolf)):
        started = time.time()
        out = generate_with_signals(lm, prompt, schedule, max_new_tokens=args.max_new_tokens)
        print(f"\n--- {name} ({len(out.token_ids)} tokens, {time.time() - started:.0f} s) ---\n{out.text}")


if __name__ == "__main__":
    main()
