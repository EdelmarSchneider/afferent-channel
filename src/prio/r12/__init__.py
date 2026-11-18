"""R12: one confirmatory evaluation, run once.

Authority: a frozen analysis plan ("AP" in the comments of this package; not part of this
repository: its lines and amendments are in Appendix A of the paper).
Fresh tasks and one unseen task family (code explanation) under the standard pairings (arm
`paired`), a crossing on the k-th token the trained model wrote itself (arm `free`), counters
computed by the harness from the generation (arm `live`, the arm of line 3), and the same with
`think` / `time` counters that start at zero (arm `Ls`, reported without a line).
"""

import sys


def utf8_console() -> None:
    """Console output as UTF-8, with a replacement for what cannot be written: a command must not
    die on a character outside the console's code page (decoded model text on a cp1252 console).
    Called at the start of every R12 command. A stream without `reconfigure` is left as it is."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # a stream that cannot change its encoding now
                pass


SEED = 12  # master seed: every generator of the plan starts with it
SET_CODE = {"pilot": 0, "confirm": 1}
PREFIX = {"pilot": "r12p", "confirm": "r12c"}
FAMILIES = ("math", "explain", "web", "heavy", "code")  # order of the family permutations: generator [SEED, position]
ARMS = ("paired", "free", "live", "Ls")
STREAM_ARMS = ("free", "live", "Ls")  # no forced prefix: the trained model writes from the prompt
STREAM_FAMILIES = ("math", "explain", "code")

# AP 3.3: consecutive slices of each family's permutation, in this order. The two Ls slices come
# last, so that they move no other slice.
SLICE_ORDER = (("pilot", "paired"), ("pilot", "free"), ("pilot", "live"),
               ("confirm", "paired"), ("confirm", "free"), ("confirm", "live"),
               ("pilot", "Ls"), ("confirm", "Ls"))
SLICES = {
    "math": (16, 16, 16, 160, 120, 120, 8, 40),
    "explain": (16, 16, 16, 160, 120, 120, 8, 40),
    "web": (16, 0, 0, 80, 0, 0, 0, 0),
    "heavy": (16, 0, 0, 80, 0, 0, 0, 0),
    "code": (32, 16, 16, 320, 120, 120, 8, 40),
}

MODEL_ID = "Qwen/Qwen3-1.7B"
RUNS = {"channel": "runs/n6-wide-levels-s0", "notice": "runs/n6-notice-none-s0"}
TAGS = {"channel": "wide-levels-s0", "notice": "notice-none-s0"}
BATCH_SIZE = 16

INTERRUPTS = ("ctx", "think", "energy", "time")  # arm free: signal of task i = INTERRUPTS[i % 4]
LIVE = ("ctx", "think", "time")  # arm live: target counter of task i = LIVE[i % 3]
LIVE_ZERO = ("think", "time")  # arm Ls: target counter of task i = LIVE_ZERO[i % 2]
FREE_K = (16, 96)  # the crossing lands on the k-th generated token, k uniform on this range (inclusive)
LIVE_K = (32, 160)
STREAM_CODE = {"free": 2, "live": 3, "Ls": 4}  # third element of the generators of each arm
TOKEN_SECONDS = 0.068  # the token clock of the live `time` counter (1 / 14.67 tokens per second)

# AP 9: minimum counts of the confirmatory set; under them a line is "not evaluable".
MIN_COUNTS = {"L1": 200, "L1c": 200, "L2a": 240, "L2b": 100, "L3": 240, "L3_counter": 60}
# The commit that froze the analysis plan: classifier and scoring must be unchanged since (AP 9).
FREEZE_COMMIT = "a1aa9533b6d40973f8ddf20594742d371f345370"
