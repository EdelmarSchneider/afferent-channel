"""Live counters (AP 6): `ctx`, `think` and `time` computed by the harness from the generation
itself, through the `signal_fn` that `generate` / `generate_batch` call for every token.

Integer arithmetic: level = min(floor(100 x count / budget), 99); reported value = (level + 0.5) / 100
(`think`: 1 - that, while the thinking is open; nominal 1.0 after `</think>`, which is the last token
that carries the budget).

Count at the token that receives the value:
- `ctx`: tokens in the context so far, this one included (prompt tokens + generated tokens);
- `think`: its starting count + the thinking tokens of this turn so far (generated tokens from
  `<think>` up to and including `</think>`);
- `time`: its starting count + the generated tokens of this turn so far, this one included (a
  token clock: `TOKEN_SECONDS` per token).
Prompt tokens get the same counters: `ctx` rises over the prompt, `think` and `time` are at their start.

Budgets (window W, thinking budget B, deadline D, in tokens) and starting counts depend on the
prompt length, k and the seed only; nothing is tuned to a model's output.
- arm `live` (the arm of line 3): `ctx` counts from zero; `think` and `time` start partly used, with
  budgets of 800 to 4,000 tokens;
- arm `Ls` (no line): every counter starts at zero; target `think` or `time`.
`energy`, `net`, `ram`: simulated and nominal (`base_rows`).
"""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from prio.r12 import LIVE, LIVE_ZERO, SEED, STREAM_CODE
from prio.signals import SignalSpec
from prio.trajectories import sample_trajectory

BUDGET_RANGE = (800, 4000)  # arm `live`: thinking budget and deadline, tokens (log-uniform)
ROOM = (0.05, 0.849)  # a counter that is not the target ends the run at this share of its budget, at the most
TWIN_GAP = (0.005, 0.045)  # the twin's target counter ends the run this far under its limit
LIMIT = 0.90  # share of the budget at which each counter is past its limit
_COLUMN = {"ctx": 0, "think": 1, "time": 3}  # columns of signals.CORE_SIGNALS


def level(count: int, budget: int) -> int:
    """Whole percent of the budget that is used, capped at 99."""
    if budget <= 0:
        raise ValueError(f"budget {budget}: a counter needs a positive budget")
    return min(100 * int(count) // int(budget), 99)


def value(signal: str, count: int, budget: int) -> float:
    """The reported value of a live counter: the center of its 1% step (`think`: what is left)."""
    if signal not in _COLUMN:
        raise ValueError(f"{signal!r} is not a live counter {tuple(_COLUMN)}")
    used = (level(count, budget) + 0.5) / 100
    return 1 - used if signal == "think" else used


@dataclass(frozen=True)
class Budgets:
    ctx: int  # context window W, tokens
    think: int  # thinking budget B, tokens
    time: int  # deadline D, tokens (x TOKEN_SECONDS = seconds)
    think_start: int = 0  # thinking tokens already used when the turn starts
    time_start: int = 0  # tokens already elapsed when the turn starts


def max_count(signal: str, n_prompt: int, k: int, horizon: int, budgets: Budgets) -> int:
    """The largest count the counter can have at generated token k + horizon, known before the run
    (a model that closes its thinking earlier leaves the `think` count lower)."""
    if signal == "ctx":
        return n_prompt + k + horizon
    return getattr(budgets, f"{signal}_start") + k + horizon


def _ceil_div(count: int, share: float) -> int:
    return int(math.ceil(count / share))


def draw_budgets(arm: str, signal: str, n_prompt: int, k: int, horizon: int, set_code: int, index: int) -> dict[str, Budgets]:
    """Budgets and starting counts of task `index` of a live arm: {"event", "twin"}.

    Event: the target counter reaches 90% of its budget exactly on the k-th generated token (for
    `think`: if the model is still thinking); the other two stay under their limits. Twin (the
    false-alarm control): the target's budget enlarged, its starting count unchanged, so that the
    counter ends the run 0.5-4.5 points short of its limit; the other budgets unchanged.
    """
    if arm not in ("live", "Ls"):
        raise ValueError(f"arm {arm!r} has no live counters")
    targets = LIVE if arm == "live" else LIVE_ZERO
    if signal not in targets:
        raise ValueError(f"arm {arm}: the target is one of {targets}, not {signal!r}")
    code = STREAM_CODE[arm]
    rng = np.random.default_rng([SEED, set_code, code, index, 1])
    budget: dict[str, int] = {}
    start = {"think": 0, "time": 0}
    # Order of the draws: for each of ctx, think, time in turn, the budget's u (arm `live`, not for
    # ctx), then e (not for the target).
    for name in ("ctx", "think", "time"):
        partly_used = arm == "live" and name != "ctx"
        if partly_used:
            u = rng.uniform(math.log(BUDGET_RANGE[0]), math.log(BUDGET_RANGE[1]))
            budget[name] = int(math.floor(math.exp(u)))
        if name == signal:
            if name == "ctx":
                budget[name] = 10 * (n_prompt + k) // 9
            elif partly_used:
                start[name] = (9 * budget[name] + 9) // 10 - k  # ceil(0.9 x budget) - k
            else:
                budget[name] = 10 * k // 9
            continue
        e = rng.uniform(*ROOM)
        if partly_used:
            start[name] = max(0, int(math.floor(e * budget[name])) - (k + horizon))
        else:  # counts from zero: its largest count at token k + horizon is known
            budget[name] = _ceil_div(n_prompt + k + horizon if name == "ctx" else k + horizon, e)
    event = Budgets(budget["ctx"], budget["think"], budget["time"], start["think"], start["time"])
    d = np.random.default_rng([SEED, set_code, code, index, 2]).uniform(*TWIN_GAP)
    enlarged = dict(budget)
    enlarged[signal] = _ceil_div(max_count(signal, n_prompt, k, horizon, event), LIMIT - d)
    twin = Budgets(enlarged["ctx"], enlarged["think"], enlarged["time"], start["think"], start["time"])
    return {"event": event, "twin": twin}


def base_rows(arm: str, specs: Sequence[SignalSpec], n_prompt: int, k: int, horizon: int, set_code: int, index: int) -> np.ndarray:
    """The simulated, nominal rows of a live run: [n_prompt + k + horizon, N], no event. The columns
    `energy`, `net`, `ram` are fed as they are; `ctx`, `think`, `time` are overwritten by the counters."""
    rng = np.random.default_rng([SEED, set_code, STREAM_CODE[arm], index, 3])
    return sample_trajectory(specs, n_prompt + k + horizon, rng).values


def thinking_tokens(generated: Sequence[int], ms) -> int | None:
    """Thinking tokens up to and including the last generated token; None once the thinking is over
    (a `</think>` came before the last token)."""
    last = len(generated) - 1
    for j, token in enumerate(generated):
        if token == ms.think_close:
            if j < last:
                return None
            break
    for j, token in enumerate(generated):
        if token == ms.think_open:
            return last - j + 1
    return 0


def _row(stored: np.ndarray, at: int, ctx_count: int, think_count: int | None, time_count: int, budgets: Budgets) -> np.ndarray:
    row = np.array(stored[min(at, len(stored) - 1)], dtype=np.float32)
    row[_COLUMN["ctx"]] = value("ctx", ctx_count, budgets.ctx)
    row[_COLUMN["think"]] = 1.0 if think_count is None else value("think", budgets.think_start + think_count, budgets.think)
    row[_COLUMN["time"]] = value("time", budgets.time_start + time_count, budgets.time)
    return row


def prompt_rows(n_prompt: int, budgets: Budgets, stored: np.ndarray) -> np.ndarray:
    """Rows of the prompt tokens, [n_prompt, N]: `ctx` rises, `think` and `time` are at their start."""
    return np.stack([_row(stored, i, i + 1, 0, 0, budgets) for i in range(n_prompt)])


def counter_rows(n_prompt: int, generated: Sequence[int], budgets: Budgets, stored: np.ndarray, ms) -> np.ndarray:
    """Rows of the generated tokens, [len(generated), N], recomputed from the tokens alone."""
    generated = list(generated)
    rows = [_row(stored, n_prompt + j, n_prompt + j + 1, thinking_tokens(generated[:j + 1], ms), j + 1, budgets)
            for j in range(len(generated))]
    return np.stack(rows) if rows else np.zeros((0, stored.shape[1]), dtype=np.float32)


def live_signal_fn(n_prompt: int, budgets: Budgets, stored: np.ndarray, ms):
    """The `signal_fn` of a live run: (step, generated ids so far, this token included) -> its row.
    Equals `counter_rows` token by token."""
    def signal_fn(step: int, generated: list[int]) -> np.ndarray:
        return _row(stored, n_prompt + step, n_prompt + step + 1, thinking_tokens(generated, ms), step + 1, budgets)

    return signal_fn


def first_past(rows: np.ndarray, signal: str, specs: Sequence[SignalSpec]) -> int | None:
    """Index of the first row whose `signal` value is past its limit, or None."""
    column = next(i for i, s in enumerate(specs) if s.name == signal)
    spec = specs[column]
    return next((j for j in range(len(rows)) if spec.past_limit(float(rows[j, column]))), None)
