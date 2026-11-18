"""Live counters and their budgets (prio.r12.live; analysis plan Section 6, arms `live` and `Ls`)."""

import math

import numpy as np
import pytest
import torch

from helpers import FAKE_MS, arm
from prio.modelspec import ModelSpec
from prio.r12 import live as L
from prio.signals import CORE_SIGNALS
from prio.trajectories import sample_trajectory

OPEN, CLOSE = FAKE_MS.think_open, FAKE_MS.think_close
CTX, THINK, ENERGY, TIME, NET, RAM = range(6)


def base(n: int, seed: int = 0) -> np.ndarray:
    return sample_trajectory(CORE_SIGNALS, n, np.random.default_rng(seed)).values


def horizon_of(signal: str) -> int:
    return 160 if signal == "time" else 96


# --- level and value ----------------------------------------------------------------------------

def test_level_is_the_whole_percent_of_the_budget_capped_at_99():
    assert [L.level(c, 200) for c in (0, 1, 2, 179, 180, 181, 199, 200, 5000)] == [0, 0, 1, 89, 90, 90, 99, 99, 99]
    assert L.value("ctx", 180, 200) == 0.905 and L.value("time", 179, 200) == 0.895
    assert L.value("think", 180, 200) == 1 - 0.905 and L.value("think", 0, 200) == 0.995
    with pytest.raises(ValueError):
        L.value("energy", 1, 2)  # not a live counter
    with pytest.raises(ValueError):
        L.level(1, 0)


def test_levels_use_integer_arithmetic():
    # count / budget = 0.9 exactly must give level 90, whatever a float division would round to
    for budget in range(10, 2001, 10):
        count = 9 * budget // 10
        assert L.level(count, budget) == 90 and L.level(count - 1, budget) < 90
    assert L.level(9 * 10**17, 10**18) == 90  # far past what a float keeps exact


# --- budgets of arm `live` (the arm of line 3) ---------------------------------------------------

def test_ctx_crosses_exactly_on_the_kth_generated_token():
    for n_prompt in range(600, 751, 7):
        for k in range(32, 161):
            window = L.draw_budgets("live", "ctx", n_prompt, k, 96, 0, 0)["event"].ctx
            assert window == 10 * (n_prompt + k) // 9
            assert L.level(n_prompt + k, window) >= 90 and L.level(n_prompt + k - 1, window) <= 89
            assert L.level(n_prompt, window) < 90  # never over the prompt


@pytest.mark.parametrize("signal", ["think", "time"])
def test_think_and_time_of_the_live_arm_start_partly_used_and_cross_on_the_kth_token(signal):
    seen = []
    for index in range(1 if signal == "think" else 2, 300, 3):
        k = 32 + (index * 37) % 129
        b = L.draw_budgets("live", signal, 640, k, horizon_of(signal), 1, index)["event"]
        budget, start = getattr(b, signal), getattr(b, f"{signal}_start")
        assert 800 <= budget <= 4000 and start == math.ceil(0.9 * budget) - k == (9 * budget + 9) // 10 - k > 0
        assert L.level(start + k, budget) == 90 and L.level(start + k - 1, budget) == 89
        seen.append(budget)
    assert min(seen) < 1000 and max(seen) > 3200  # log-uniform over 800 to 4,000


def test_live_arm_draws_follow_the_order_of_the_plan():
    n_prompt, k, index, code = 655, 77, 7, 1
    for signal in ("ctx", "think", "time"):
        h = horizon_of(signal)
        rng = np.random.default_rng([12, code, 3, index, 1])
        lo, hi = math.log(800), math.log(4000)
        expected = {}
        # for each of ctx, think, time in turn: the budget's u (not for ctx), then e (not for the target)
        if signal == "ctx":
            expected["ctx"] = 10 * (n_prompt + k) // 9
        else:
            expected["ctx"] = math.ceil((n_prompt + k + h) / rng.uniform(0.05, 0.849))
        for name in ("think", "time"):
            budget = math.floor(math.exp(rng.uniform(lo, hi)))
            expected[name] = budget
            if name == signal:
                expected[f"{name}_start"] = math.ceil(0.9 * budget) - k
            else:
                expected[f"{name}_start"] = max(0, math.floor(rng.uniform(0.05, 0.849) * budget) - (k + h))
        drawn = L.draw_budgets("live", signal, n_prompt, k, h, code, index)
        assert drawn["event"] == L.Budgets(**expected)
        # twin: the target's budget enlarged, its starting count unchanged, the other budgets unchanged
        d = np.random.default_rng([12, code, 3, index, 2]).uniform(0.005, 0.045)
        end = n_prompt + k + h if signal == "ctx" else expected[f"{signal}_start"] + k + h
        twin = dict(expected)
        twin[signal] = math.ceil(end / (0.90 - d))
        assert drawn["twin"] == L.Budgets(**twin)
        assert L.max_count(signal, n_prompt, k, h, drawn["event"]) == end


def test_live_arm_budgets_are_pinned():
    drawn = L.draw_budgets("live", "think", 655, 77, 96, 1, 7)
    assert drawn == {"event": L.Budgets(ctx=1148, think=2150, time=3509, think_start=1858, time_start=2537),
                     "twin": L.Budgets(ctx=1148, think=2356, time=3509, think_start=1858, time_start=2537)}
    zero = L.draw_budgets("Ls", "time", 655, 77, 160, 1, 7)
    assert zero == {"event": L.Budgets(ctx=1255, think=409, time=85), "twin": L.Budgets(ctx=1255, think=409, time=270)}


# --- budgets of arm `Ls` (counters from zero, no line) --------------------------------------------

@pytest.mark.parametrize("signal", ["think", "time"])
def test_counters_from_zero_cross_exactly_on_the_kth_generated_token(signal):
    for k in range(32, 161):
        b = L.draw_budgets("Ls", signal, 640, k, horizon_of(signal), 0, k)["event"]
        budget = getattr(b, signal)
        assert budget == 10 * k // 9 and (b.think_start, b.time_start) == (0, 0)
        assert L.level(k, budget) >= 90 and L.level(k - 1, budget) <= 89
    assert L.draw_budgets("Ls", signal, 640, 32, 96, 0, 0)["event"].__getattribute__(signal) == 35
    assert L.draw_budgets("Ls", signal, 640, 160, 96, 0, 0)["event"].__getattribute__(signal) == 177


def test_counters_from_zero_draw_the_other_budgets_in_the_order_ctx_think_time():
    n_prompt, k, index, code = 701, 45, 11, 0
    for signal in ("think", "time"):
        h = horizon_of(signal)
        rng = np.random.default_rng([12, code, 4, index, 1])
        expected = {"ctx": math.ceil((n_prompt + k + h) / rng.uniform(0.05, 0.849)), "think_start": 0, "time_start": 0}
        for name in ("think", "time"):
            expected[name] = 10 * k // 9 if name == signal else math.ceil((k + h) / rng.uniform(0.05, 0.849))
        drawn = L.draw_budgets("Ls", signal, n_prompt, k, h, code, index)
        assert drawn["event"] == L.Budgets(**expected)
        d = np.random.default_rng([12, code, 4, index, 2]).uniform(0.005, 0.045)
        twin = dict(expected)
        twin[signal] = math.ceil((k + h) / (0.90 - d))
        assert drawn["twin"] == L.Budgets(**twin)
    with pytest.raises(ValueError):
        L.draw_budgets("Ls", "ctx", n_prompt, k, 96, code, index)  # ctx is never the target of this arm
    with pytest.raises(ValueError):
        L.draw_budgets("free", "think", n_prompt, k, 96, code, index)


# --- what the budgets guarantee -------------------------------------------------------------------

def _cases():
    for arm_name, targets in (("live", ("ctx", "think", "time")), ("Ls", ("think", "time"))):
        for index in range(240):
            signal = targets[index % len(targets)]
            k = 32 + (index * 53) % 129
            n_prompt = 602 + (index * 29) % 143
            yield arm_name, signal, n_prompt, k, horizon_of(signal), index % 2, index


def test_roomy_counters_stay_five_points_from_their_limit_to_the_last_token():
    for arm_name, signal, n_prompt, k, h, code, index in _cases():
        b = L.draw_budgets(arm_name, signal, n_prompt, k, h, code, index)["event"]
        for other in ("ctx", "think", "time"):
            if other != signal:
                top = L.level(L.max_count(other, n_prompt, k, h, b), getattr(b, other))
                assert top <= 84, (arm_name, signal, other, top)


def test_twin_budget_ends_in_the_near_miss_band_and_never_crosses():
    lowest = {"live": 99, "Ls": 99}
    for arm_name, signal, n_prompt, k, h, code, index in _cases():
        drawn = L.draw_budgets(arm_name, signal, n_prompt, k, h, code, index)
        event, twin = drawn["event"], drawn["twin"]
        assert (twin.think_start, twin.time_start) == (event.think_start, event.time_start)
        assert getattr(twin, signal) > getattr(event, signal)
        top = L.level(L.max_count(signal, n_prompt, k, h, twin), getattr(twin, signal))
        assert top <= 89, (arm_name, signal, top)  # 0.5 points short at the closest
        lowest[arm_name] = min(lowest[arm_name], top)
        assert top >= (85 if arm_name == "live" else 84), (arm_name, signal, k, top)
    assert lowest["live"] == 85


# --- the counters over a run ------------------------------------------------------------------------

def test_counters_over_the_prompt():
    b = L.Budgets(ctx=1000, think=200, time=300)
    rows = L.prompt_rows(700, b, base(900))
    assert rows.shape == (700, 6) and rows.dtype == np.float32
    assert np.allclose(rows[:, CTX], [(min(100 * (i + 1) // 1000, 99) + 0.5) / 100 for i in range(700)])
    assert rows[0, CTX] == np.float32(0.005) and rows[-1, CTX] == np.float32(0.705)
    assert (rows[:, THINK] == np.float32(0.995)).all() and (rows[:, TIME] == np.float32(0.005)).all()
    assert np.array_equal(rows[:, [ENERGY, NET, RAM]], base(900)[:700][:, [ENERGY, NET, RAM]])
    started = L.prompt_rows(700, L.Budgets(ctx=1000, think=2000, time=1000, think_start=1000, time_start=400), base(900))
    assert (started[:, THINK] == np.float32(1 - 0.505)).all() and (started[:, TIME] == np.float32(0.405)).all()


def test_generated_tokens_count_into_the_three_counters():
    b = L.Budgets(ctx=1000, think=200, time=300, think_start=100, time_start=30)
    tokens = [OPEN] + [50] * 9
    rows = L.counter_rows(700, tokens, b, base(900), FAKE_MS)
    assert rows.shape == (10, 6) and rows.dtype == np.float32
    for j in range(10):
        assert rows[j, CTX] == np.float32(L.value("ctx", 700 + j + 1, 1000))
        assert rows[j, TIME] == np.float32(L.value("time", 30 + j + 1, 300))
        assert rows[j, THINK] == np.float32(L.value("think", 100 + j + 1, 200))
    assert np.array_equal(rows[:, [ENERGY, NET, RAM]], base(900)[700:710][:, [ENERGY, NET, RAM]])
    longer = L.counter_rows(898, [OPEN] + [50] * 9, b, base(900), FAKE_MS)  # past the stored rows: the last one repeats
    assert np.array_equal(longer[5:, [ENERGY, NET, RAM]], np.repeat(base(900)[-1:, [ENERGY, NET, RAM]], 5, axis=0))


def test_think_counts_only_open_thinking_and_is_nominal_after_the_close():
    b = L.Budgets(ctx=5000, think=40, time=5000)
    tokens = [60, OPEN, 61, 62, 63, CLOSE, 64, OPEN, 65]
    rows = L.counter_rows(10, tokens, b, base(60), FAKE_MS)
    counts = [0, 1, 2, 3, 4, 5]  # nothing before <think>; </think> is the last token that carries the budget
    assert [float(v) for v in rows[:6, THINK]] == [float(np.float32(L.value("think", c, 40))) for c in counts]
    assert (rows[6:, THINK] == 1.0).all()  # nominal afterwards, also when a second <think> is written
    assert L.thinking_tokens(tokens[:1], FAKE_MS) == 0 and L.thinking_tokens(tokens[:6], FAKE_MS) == 5
    assert L.thinking_tokens(tokens[:7], FAKE_MS) is None and L.thinking_tokens(tokens, FAKE_MS) is None


def test_think_crosses_on_token_k_when_the_thinking_is_open_and_never_when_it_closes_first():
    k, n_prompt = 40, 30
    for arm_name, index in (("live", 1), ("Ls", 0)):
        b = L.draw_budgets(arm_name, "think", n_prompt, k, 96, 0, index)["event"]
        rows = base(n_prompt + k + 96)
        open_run = [OPEN] + [50] * (k + 95)
        fed = L.counter_rows(n_prompt, open_run, b, rows, FAKE_MS)
        assert L.first_past(fed, "think", CORE_SIGNALS) == k - 1  # the k-th generated token, 0-based
        assert L.first_past(fed, "ctx", CORE_SIGNALS) is None and L.first_past(fed, "time", CORE_SIGNALS) is None
        closed = [OPEN] + [50] * 20 + [CLOSE] + [50] * (k + 74)
        fed = L.counter_rows(n_prompt, closed, b, rows, FAKE_MS)
        assert L.first_past(fed, "think", CORE_SIGNALS) is None
        late = [50, 50, OPEN] + [50] * (k + 93)  # a model that opens its thinking two tokens late crosses two tokens late
        assert L.first_past(L.counter_rows(n_prompt, late, b, rows, FAKE_MS), "think", CORE_SIGNALS) == k + 1


@pytest.mark.parametrize("arm_name, signal", [("live", "ctx"), ("live", "time"), ("Ls", "time")])
def test_ctx_and_time_cross_on_token_k_whatever_the_model_writes(arm_name, signal):
    k, n_prompt, h = 57, 611, horizon_of(signal)
    index = {"ctx": 0, "time": 2}[signal] if arm_name == "live" else 1
    drawn = L.draw_budgets(arm_name, signal, n_prompt, k, h, 1, index)
    rows = base(n_prompt + k + h)
    tokens = [OPEN] + [50] * 30 + [CLOSE] + [51] * (k + h - 32)
    fed = L.counter_rows(n_prompt, tokens, drawn["event"], rows, FAKE_MS)
    assert L.first_past(fed, signal, CORE_SIGNALS) == k - 1
    assert not any(L.first_past(fed, other, CORE_SIGNALS) is not None for other in ("ctx", "think", "time") if other != signal)
    assert not any(s.past_limit(float(v)) for i, s in enumerate(CORE_SIGNALS) if s.name in ("energy", "net", "ram")
                   for v in fed[:, i])
    quiet = L.counter_rows(n_prompt, tokens, drawn["twin"], rows, FAKE_MS)
    assert all(L.first_past(quiet, name, CORE_SIGNALS) is None for name in ("ctx", "think", "time"))
    column = [s.name for s in CORE_SIGNALS].index(signal)
    assert 0.845 <= float(quiet[-1, column]) <= 0.896  # the twin stops 0.5 to 4.5 points short (5.5 at the most from zero)
    prompt = L.prompt_rows(n_prompt, drawn["event"], rows)
    assert not any(s.past_limit(float(v)) for i, s in enumerate(CORE_SIGNALS) for v in prompt[:, i])


def test_base_rows_are_the_nominal_trajectory_of_the_plans_generator():
    rows = L.base_rows("live", CORE_SIGNALS, 640, 50, 96, 1, 9)
    expected = sample_trajectory(CORE_SIGNALS, 640 + 50 + 96, np.random.default_rng([12, 1, 3, 9, 3])).values
    assert np.array_equal(rows, expected)
    zero = L.base_rows("Ls", CORE_SIGNALS, 640, 50, 96, 1, 9)
    assert np.array_equal(zero, sample_trajectory(CORE_SIGNALS, 786, np.random.default_rng([12, 1, 4, 9, 3])).values)
    assert not any(s.past_limit(float(v)) for i, s in enumerate(CORE_SIGNALS) for v in rows[:, i])


# --- through the generation loop ------------------------------------------------------------------

def test_live_signal_fn_equals_counter_rows_through_generate_batch(tiny_model):
    from prio.channel import ChannelModel
    from prio.generate import generate, generate_batch

    model = ChannelModel(tiny_model, 6, delivery="wide", lane_width=8, projector="levels")
    arm(model)
    g = torch.Generator().manual_seed(0)
    lengths, horizons = [5, 9, 2], [40, 25, 33]
    prompts = [torch.randint(3, 100, (n,), generator=g).tolist() for n in lengths]
    stored = [base(n + h, seed=i) for i, (n, h) in enumerate(zip(lengths, horizons))]
    budgets = [L.Budgets(ctx=40, think=30, time=50), L.Budgets(ctx=45, think=400, time=27, think_start=350),
               L.Budgets(ctx=500, think=20, time=300, time_start=250)]
    prompt_rows = [L.prompt_rows(n, b, s) for n, b, s in zip(lengths, budgets, stored)]

    def run_batch(ms):
        fns = [L.live_signal_fn(n, b, s, ms) for n, b, s in zip(lengths, budgets, stored)]
        return generate_batch(model, prompts, prompt_rows, fns, horizons, set())

    first = run_batch(FAKE_MS)[0].token_ids[0]  # the first token depends on the prompt only: make it <think>
    ms = ModelSpec("fake", think_open=first, think_close=FAKE_MS.think_close, tool_open=FAKE_MS.tool_open,
                   tool_close=FAKE_MS.tool_close, eos_ids=frozenset())
    batch = run_batch(ms)
    assert batch[0].token_ids[0] == first
    for i, gen in enumerate(batch):
        assert len(gen.token_ids) == horizons[i]
        expected = L.counter_rows(lengths[i], gen.token_ids, budgets[i], stored[i], ms)
        assert np.array_equal(gen.signals, expected)
        single = generate(model, torch.tensor([prompts[i]]), torch.from_numpy(prompt_rows[i][None]),
                          L.live_signal_fn(lengths[i], budgets[i], stored[i], ms), horizons[i], set())
        assert single.token_ids == gen.token_ids and np.array_equal(single.signals, gen.signals)
    # the counters moved while the model wrote: each crossed its limit inside one of the runs
    assert L.first_past(batch[0].signals, "ctx", CORE_SIGNALS) is not None
    assert L.first_past(batch[0].signals, "think", CORE_SIGNALS) is not None
    assert L.first_past(batch[1].signals, "time", CORE_SIGNALS) == 25 - 1
