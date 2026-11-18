"""R10: limits given in words + value readout (prio.datagen.limits, prio.eval.limits_eval)."""

import json

import numpy as np
import pytest

from helpers import CALL, FAKE_MS, FAKE_TOK, enc, make_episode
from prio.signals import CORE_SIGNALS, with_limits
from prio.tasks import SYSTEM_PROMPT, limit_rule, parse_limit_rule, system_with_limits
from prio.trajectories import sample_trajectory

NAMES = [s.name for s in CORE_SIGNALS]
BY = {s.name: s for s in CORE_SIGNALS}
LIMITS = {"ctx": 0.75, "think": 0.2, "energy": 0.05, "time": 0.73, "ram": 0.5}


def codec():
    from prio.datagen.limits import Codec

    return Codec(enc, FAKE_TOK.decode, FAKE_TOK.eos_token_id, user_turn="<user> {question} <eos> <assistant> <think> </think>")


# --- limits in words ---------------------------------------------------------------------------

def test_limit_rule_writes_whole_percents_and_parses_back_exactly():
    rule = limit_rule(LIMITS)
    assert "75% full" in rule and "at 5%" in rule and "73% of the deadline" in rule
    assert parse_limit_rule(rule) == LIMITS
    assert parse_limit_rule("some text. " + system_with_limits(LIMITS)) == LIMITS
    assert system_with_limits(LIMITS) == SYSTEM_PROMPT + " " + rule
    with pytest.raises(ValueError):
        limit_rule({**LIMITS, "ctx": 0.735})  # limits are whole percents
    with pytest.raises(ValueError):
        limit_rule({k: v for k, v in LIMITS.items() if k != "ram"})


def test_with_limits_replaces_only_the_named_limits():
    specs = with_limits(CORE_SIGNALS, {"ctx": 0.6, "energy": 0.3})
    assert specs[0].limit == 0.6 and specs[2].limit == 0.3
    assert [s for i, s in enumerate(specs) if i not in (0, 2)] == [s for i, s in enumerate(CORE_SIGNALS) if i not in (0, 2)]
    assert CORE_SIGNALS[0].limit == 0.9  # the shared specs are untouched
    with pytest.raises(ValueError):
        with_limits(CORE_SIGNALS, {"disk": 0.5})


def test_grid_and_held_out_values():
    from prio.datagen.limits import GRID, HELD_OFF, HELD_ON, TRAIN_GRID

    assert TRAIN_GRID["high"] == (0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
    assert TRAIN_GRID["low"] == (0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5)
    assert HELD_ON == {"high": (0.65, 0.85), "low": (0.15, 0.35)} and set(HELD_ON["high"]) <= set(GRID["high"])
    assert HELD_OFF == {"high": (0.73, 0.87), "low": (0.12, 0.28)}
    assert not set(HELD_OFF["high"] + HELD_OFF["low"]) & set(GRID["high"] + GRID["low"])


def test_rewrite_prompt_puts_the_rule_into_the_system_prompt():
    from prio.datagen.limits import rewrite_prompt

    prompt = enc(f"<sys> {SYSTEM_PROMPT} <user> Explain tides. <eos> <assistant>")
    new = rewrite_prompt(prompt, LIMITS, codec())
    assert FAKE_TOK.decode(new) == f"<sys> {system_with_limits(LIMITS)} <user> Explain tides. <eos> <assistant>"
    assert parse_limit_rule(FAKE_TOK.decode(new)) == LIMITS
    with pytest.raises(ValueError):
        rewrite_prompt(enc("<sys> another prompt <user> hi <eos>"), LIMITS, codec())


def test_level_is_the_whole_percent_counted_from_the_limit_side():
    from prio.datagen.limits import level, level_value

    assert (level(0.735, "high"), level(0.115, "low"), level(1.0, "low"), level(0.0, "high")) == (73, 12, 100, 0)
    assert all(level(level_value(p, "high"), "high") == p for p in range(0, 100))
    assert all(level(level_value(p, "low"), "low") == p for p in range(1, 101))
    # The level a limit names is the first level past it, in both orientations.
    high, low = with_limits(CORE_SIGNALS, {"ctx": 0.73, "energy": 0.12})[0], with_limits(CORE_SIGNALS, {"energy": 0.12})[2]
    assert high.past_limit(level_value(73, "high")) and not high.past_limit(level_value(72, "high"))
    assert low.past_limit(level_value(12, "low")) and not low.past_limit(level_value(13, "low"))


# --- a small phase-1-like source dataset in FAKE_TOK words ---------------------------------------

BASE = "<think> " + " ".join(f"w{k}" for k in range(30)) + " </think> " + " ".join(f"a{k}" for k in range(20)) + " <eos>"
REACTION = {
    "ctx": "my context is almost full " + CALL % ("compact_context", "{}") + " <eos>",
    "energy": "my battery is almost empty " + CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}") + " <eos>",
    "think": "my thinking budget is almost gone </think> the answer . <eos>",
    "time": "i am almost out of time so here is the short version . <eos>",
    "ram": "memory is almost exhausted " + CALL % ("run_light", '{"job":"j"}') + " <eos>",
    "net": "i am offline " + CALL % ("local_lookup", '{"query":"q"}') + " <eos>",
}


def _prompt(task):
    return enc(f"<sys> {SYSTEM_PROMPT} <user> please do task {task} <eos> <assistant>")


def _source(episode_id, kind, signal=None, task="t0", t_cross=None, t_decision=None, t_recover=None, split="train",
            reaction=False, t_diverge=None, seed=0):
    prompt, base = _prompt(task), enc(BASE)
    cut = None if t_cross is None else (1 if kind == "starts_past" else t_cross + 1)
    if kind == "gating_event":
        cut = t_decision
    target = base[:cut] + enc(REACTION[signal]) if reaction else base
    n = len(prompt) + len(target)
    rng = np.random.default_rng(seed)
    if signal is None:
        values = sample_trajectory(CORE_SIGNALS, n, rng).values
    else:
        start = 0 if kind == "starts_past" else len(prompt) + t_cross
        recover = None if t_recover is None else len(prompt) + t_recover
        values = sample_trajectory(CORE_SIGNALS, n, rng, event=NAMES.index(signal), t_cross=start, t_recover=recover).values
    return make_episode(episode_id=episode_id, task_id=task, family="explain", split=split, kind=kind, signal=signal,
                        phase="thinking" if signal else None, prompt_ids=prompt, target_ids=target, signals=values,
                        t_cross=t_cross, t_decision=t_decision, t_diverge=t_diverge)


def _pair(i, kind, signal, t_cross, t_decision=None):
    """A reaction episode and its quiet twin (same base run, nominal kind, t_diverge at the cut)."""
    event = _source(f"train-{i:05d}", kind, signal, task=f"t{i}", t_cross=t_cross, t_decision=t_decision, reaction=True, seed=i)
    base = enc(BASE)
    diverge = next(k for k, tok in enumerate(event.target_ids) if k >= len(base) or tok != base[k])
    twin = _source(f"train-9{i:04d}", "nominal", task=f"t{i}", t_diverge=diverge, seed=1000 + i)
    return event, twin


def train_source(n_pairs=24):
    signals = ["ctx", "think", "energy", "time"]
    events, twins = [], []
    for i in range(n_pairs):
        kind = "starts_past" if i % 8 == 7 else "event"
        e, t = _pair(i, kind, signals[i % 4], 0 if kind == "starts_past" else 4 + (i % 5))
        events.append(e)
        twins.append(t)
    e, t = _pair(n_pairs, "gating_event", "ram", 3, t_decision=10)
    g, h = _pair(n_pairs + 1, "gating_event", "net", 2, t_decision=10)
    extra = [
        _source("train-00500", "recovery", "ram", task="r0", t_cross=2, t_decision=12, t_recover=6),
        _source("train-00501", "gating_unused", "net", task="u0", t_cross=5),
        _source("train-00502", "nominal", task="n0"),
        _source("train-00503", "nominal", task="n1"),
    ]
    return events + [e, g] + extra + twins + [t, h]


def test_source_fixture_is_like_phase_1():
    src = train_source()
    for ep in src:  # crossings of the phase-1 limits sit where the build put them
        if ep.kind in ("event", "gating_event"):
            column = ep.signals[:, NAMES.index(ep.signal)]
            past = [BY[ep.signal].past_limit(float(v)) for v in column]
            assert past.index(True) == len(ep.prompt_ids) + ep.t_cross


def _transform(source, split="train", **options):
    from prio.datagen.limits import Options, transform

    return transform(source, split, codec(), FAKE_MS, Options(**options))


def _first_past(ep, name):
    from prio.eval.limits_eval import episode_specs

    spec = episode_specs(ep, CORE_SIGNALS)[NAMES.index(name)]
    past = [spec.past_limit(float(v)) for v in ep.signals[:, NAMES.index(name)]]
    return past.index(True) if True in past else None


@pytest.mark.parametrize("signal", ["ctx", "energy", "think", "time"])
def test_crossing_is_recomputed_against_the_episode_limit(signal):
    episodes, _, _ = _transform(train_source())
    events = [e for e in episodes if e.kind == "event" and e.signal == signal]
    assert events
    for ep in events:
        limits = parse_limit_rule(FAKE_TOK.decode(ep.prompt_ids))
        assert limits == ep.meta["limits"] and set(limits) == {"ctx", "think", "energy", "time", "ram"}
        assert ep.signals.shape == (len(ep.prompt_ids) + len(ep.target_ids), 6)
        assert _first_past(ep, signal) == len(ep.prompt_ids) + ep.t_cross  # the reaction starts at t_cross + 1
        assert all(_first_past(ep, other) is None for other in ("ctx", "think", "energy", "time", "ram") if other != signal)
        assert ep.target_ids == next(s for s in train_source() if s.episode_id == ep.meta["source"]).target_ids


def test_other_kinds_keep_their_timing_against_their_own_limits():
    episodes, _, _ = _transform(train_source())
    by_kind = {}
    for ep in episodes:
        by_kind.setdefault(ep.kind, []).append(ep)
    for ep in by_kind["starts_past"]:
        assert _first_past(ep, ep.signal) == 0
    gating = next(e for e in by_kind["gating_event"] if e.signal == "ram")
    assert _first_past(gating, "ram") == len(gating.prompt_ids) + gating.t_cross
    recovery = by_kind["recovery"][0]
    assert recovery.meta["t_recover"] == 6 and _first_past(recovery, "ram") == len(recovery.prompt_ids) + 2
    column = recovery.signals[len(recovery.prompt_ids) + 6:, NAMES.index("ram")]
    assert not any(v <= recovery.meta["limits"]["ram"] for v in column)
    for ep in by_kind["nominal"]:
        assert all(_first_past(ep, name) is None for name in ("ctx", "think", "energy", "time", "ram"))


def test_held_out_limits_never_appear_in_train():
    from prio.datagen.limits import TRAIN_GRID

    episodes, _, _ = _transform(train_source(40))
    allowed = set(TRAIN_GRID["high"] + TRAIN_GRID["low"])
    for ep in episodes:
        written = parse_limit_rule(FAKE_TOK.decode(ep.prompt_ids))
        assert set(written.values()) <= allowed, (ep.episode_id, written)
        assert ep.meta["limit_split"] in ("seen", None)


def test_a_quiet_twin_shares_the_prompt_of_its_reaction_episode():
    # Same prompt and same text up to the cut: only the channel can tell the pair apart (lesson of the
    # first background-jobs model: whatever separates a pair outside the channel gets memorized).
    episodes, _, _ = _transform(train_source())
    reacting = {e.task_id: e for e in episodes if e.kind in ("event", "starts_past", "gating_event")}
    quiet = [e for e in episodes if e.kind == "nominal" and e.t_diverge is not None]
    assert len(quiet) == len(reacting)
    for twin in quiet:
        assert twin.prompt_ids == reacting[twin.task_id].prompt_ids
        assert twin.meta["limits"] == reacting[twin.task_id].meta["limits"]
    solo = [e for e in episodes if e.kind == "nominal" and e.t_diverge is None]
    assert len({tuple(e.prompt_ids) for e in solo}) == len(solo)


def test_think_follows_the_text_in_train():
    episodes, _, _ = _transform(train_source())
    think = NAMES.index("think")
    closed = [ep for ep in episodes if FAKE_MS.think_close in ep.target_ids]
    assert len(closed) > len(episodes) / 2
    for ep in closed:
        close = len(ep.prompt_ids) + ep.target_ids.index(FAKE_MS.think_close)
        assert (ep.signals[close + 1:, think] == 1.0).all() and (ep.signals[:close + 1, think] < 1.0).all()


def test_limit_twin_has_the_same_text_and_trajectory_and_a_limit_beyond_the_peak():
    from prio.datagen.limits import TRAIN_GRID, level

    source = train_source(40)
    episodes, _, stats = _transform(source)
    by_id = {e.episode_id: e for e in episodes}
    twins = [e for e in episodes if e.kind == "limit_twin"]
    assert twins and stats["limit_twins"] + sum(stats["twin_skipped"].values()) == sum(
        e.kind in ("event", "starts_past", "gating_event") and e.signal != "net" for e in episodes)
    for twin in twins:
        event = by_id[twin.meta["twin_of"]]
        quiet = next(s for s in source if s.kind == "nominal" and s.task_id == twin.task_id)
        assert twin.target_ids == quiet.target_ids and twin.signal == event.signal and twin.t_cross == event.t_cross
        assert twin.meta["twin_kind"] == event.kind and twin.t_decision == event.t_decision
        col, spec = NAMES.index(twin.signal), BY[twin.signal]
        # same trajectory, aligned on the target: every signal identical up to the decision token (after
        # it the texts part, and `think` ends where each text closes its thinking)
        p_e, p_t = len(event.prompt_ids), len(twin.prompt_ids)
        upto = twin.t_diverge + 1
        assert np.array_equal(event.signals[p_e:p_e + upto], twin.signals[p_t:p_t + upto])
        if twin.signal != "think":
            shared = min(len(event.target_ids), len(twin.target_ids))
            assert np.array_equal(event.signals[p_e:p_e + shared, col], twin.signals[p_t:p_t + shared, col])
        # only the event signal's limit moved, to >= 3 points beyond the peak, from the train grid
        moved = {k for k in event.meta["limits"] if event.meta["limits"][k] != twin.meta["limits"][k]}
        assert moved == {twin.signal}
        limit = twin.meta["limits"][twin.signal]
        assert limit in TRAIN_GRID[spec.bad_when]
        column = twin.signals[:, col]
        extreme = level(column.max(), "high") if spec.bad_when == "high" else level(column.min(), "low")
        assert (round(limit * 100) - extreme if spec.bad_when == "high" else extreme - round(limit * 100)) >= 3
        assert _first_past(twin, twin.signal) is None  # the twin never passes its own limit
        assert parse_limit_rule(FAKE_TOK.decode(twin.prompt_ids)) == twin.meta["limits"]
        base = quiet.target_ids
        assert twin.t_diverge == next(k for k, t in enumerate(event.target_ids) if k >= len(base) or t != base[k])


def test_limit_twin_trains_like_a_quiet_twin():
    from prio.train import collate

    episodes, _, _ = _transform(train_source(40))
    twin = next(e for e in episodes if e.kind == "limit_twin")
    batch = collate([twin], pad_id=0, n_signals=6, reaction_weight=20.0)
    position = len(twin.prompt_ids) + twin.t_diverge
    weights = batch["weights"][0]
    assert weights[position] == 20.0 and int((weights == 20.0).sum()) == 1  # no reaction tokens, x20 at the divergence
    assert batch["decision"][0, position] == 2  # "hold": the quiet side of the decision


def test_report_turns_skip_held_out_levels_and_carry_the_value():
    from prio.datagen.limits import REPORT_SIGNALS, level

    episodes, _, stats = _transform(train_source(40), report_share=1.0)
    reports = [e for e in episodes if e.meta.get("report")]
    assert reports and all(e.kind == "nominal" for e in reports) and stats["reports"] == len(reports)
    assert {e.meta["report"]["signal"] for e in reports} <= set(REPORT_SIGNALS)
    for ep in reports:
        info = ep.meta["report"]
        spec = with_limits(CORE_SIGNALS, ep.meta["limits"])[NAMES.index(info["signal"])]
        assert info["level"] % 5 != 3 and not info["held_out"]
        q0, a0 = info["question_start"], info["answer_start"]
        assert ep.target_ids[q0 - 1] == FAKE_TOK.eos_token_id  # after the base run's end of turn
        assert ep.harness_spans == [[q0, a0]] and ep.focus_spans == [[a0, len(ep.target_ids)]]
        answer = FAKE_TOK.decode(ep.target_ids[a0:])
        assert answer.endswith(f"at {info['level']}%. <eos>")
        rows = ep.signals[len(ep.prompt_ids) + q0:, NAMES.index(info["signal"])]
        assert all(level(v, spec.bad_when) == info["level"] for v in rows)
        assert not spec.past_limit(float(rows[0]))  # reported values never ask for a reaction


def test_reports_off_leaves_quiet_episodes_unchanged_in_length():
    episodes, _, stats = _transform(train_source(), reports=False)
    assert stats["reports"] == 0 and not any(e.meta.get("report") for e in episodes)
    source = {s.episode_id: s for s in train_source()}
    assert all(e.target_ids == source[e.meta["source"]].target_ids for e in episodes if e.kind != "limit_twin")


def test_transform_is_deterministic():
    # Same seed, same source: same output (one generator per episode, seeded by its position).
    a, _, _ = _transform(train_source())
    b, _, _ = _transform(train_source())
    assert a == b


# --- test split ----------------------------------------------------------------------------------

def eval_source(n=36):
    eps = []
    signals = ["ctx", "think", "energy", "time"]
    for i in range(n):
        eps.append(_source(f"test-{i:05d}", "event", signals[i % 4], task=f"x{i}", t_cross=4 + i % 5, split="test", seed=i))
    eps.append(_source("test-00900", "gating_event", "ram", task="g0", t_cross=3, t_decision=10, split="test", seed=900))
    eps.append(_source("test-00901", "gating_event", "net", task="g1", t_cross=3, t_decision=10, split="test", seed=901))
    return eps


def test_test_split_cycles_seen_held_on_and_off_grid_limits():
    from prio.datagen.limits import HELD_OFF, HELD_ON, TRAIN_GRID

    episodes, reports, _ = _transform(eval_source(), split="test", test_reports=0)
    events = [e for e in episodes if e.kind == "event"]
    pools = {"seen": TRAIN_GRID, "held_on": HELD_ON, "held_off": HELD_OFF}
    for signal in ("ctx", "think", "energy", "time"):
        mine = [e for e in events if e.signal == signal]
        assert [e.meta["limit_split"] for e in mine][:3] == ["seen", "held_on", "held_off"]
        for e in mine:
            orientation = BY[signal].bad_when
            assert e.meta["limits"][signal] in pools[e.meta["limit_split"]][orientation]
            assert _first_past(e, signal) == len(e.prompt_ids) + e.t_cross
            others = [v for k, v in e.meta["limits"].items() if k != signal]
            assert set(others) <= set(TRAIN_GRID["high"] + TRAIN_GRID["low"])
    net = next(e for e in episodes if e.signal == "net")
    assert net.meta["limit_split"] is None
    assert reports == []
    held_on = [e for e in events if e.meta["limit_split"] == "held_on"]
    assert {e.meta["limits"][e.signal] for e in held_on if e.signal == "ctx"} == {0.65, 0.85}


def test_test_limit_twins_stay_beyond_the_peak_over_the_evaluated_span():
    from prio.datagen.limits import EVAL_SPAN, HELD_OFF, HELD_ON, TRAIN_GRID, level

    episodes, _, stats = _transform(eval_source(120), split="test", test_reports=0)
    by_id = {e.episode_id: e for e in episodes}
    twins = [e for e in episodes if e.kind == "limit_twin"]
    assert twins and stats["limit_twins"] == len(twins)
    # a twin's limit comes preferably from the split after its event's (held-out twin limits are the
    # primary measure), so most twins differ in split from their event and every split is well filled
    splits = [t.meta["limit_split"] for t in twins]
    assert min(splits.count(s) for s in ("seen", "held_on", "held_off")) >= len(twins) // 6
    assert sum(t.meta["limit_split"] != by_id[t.meta["twin_of"]].meta["limit_split"] for t in twins) > 0.6 * len(twins)
    for twin in twins:
        event = by_id[twin.meta["twin_of"]]
        assert twin.target_ids == event.target_ids and twin.t_diverge is None
        spec = BY[twin.signal]
        limit = twin.meta["limits"][twin.signal]
        split = twin.meta["limit_split"]
        assert limit in {"seen": TRAIN_GRID, "held_on": HELD_ON, "held_off": HELD_OFF}[split][spec.bad_when]
        end = twin.t_decision if twin.meta["twin_kind"] == "gating_event" else twin.t_cross + 1
        span = twin.signals[:len(twin.prompt_ids) + end + EVAL_SPAN, NAMES.index(twin.signal)]
        extreme = level(span.max(), "high") if spec.bad_when == "high" else level(span.min(), "low")
        assert (round(limit * 100) - extreme if spec.bad_when == "high" else extreme - round(limit * 100)) >= 3


def test_test_reports_mix_held_out_and_seen_levels():
    episodes, reports, stats = _transform(eval_source(), split="test", test_reports=12)
    assert len(reports) == 12 == stats["reports"] and len({r.task_id for r in reports}) == 12
    assert all(r.kind == "report" and r.split == "test" for r in reports)
    held = [r.meta["report"]["held_out"] for r in reports]
    assert sum(held) == 6
    for r in reports:
        info = r.meta["report"]
        assert (info["level"] % 5 == 3) == info["held_out"] and r.signal == info["signal"]
        assert r.target_ids[:info["question_start"]] == next(e.target_ids for e in eval_source() if e.task_id == r.task_id)
        assert all(_first_past(r, name) is None for name in ("ctx", "think", "energy", "time", "ram"))


# --- build: files and manifest -------------------------------------------------------------------

def test_build_writes_the_files_and_a_manifest(tmp_path):
    from prio.datagen.episodes import read_jsonl, write_jsonl
    from prio.datagen.limits import Options, build

    src = tmp_path / "src"
    src.mkdir()
    write_jsonl(src / "train.jsonl", train_source())
    write_jsonl(src / "test.jsonl", eval_source())
    manifest = build(src, tmp_path / "out", codec(), FAKE_MS, Options(test_reports=8))
    for name in ("train.jsonl", "test.jsonl", "test_reports.jsonl", "manifest.json"):
        assert (tmp_path / "out" / name).exists()
    assert json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["train_grid"]["high"] == [0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
    assert manifest["held_out_on_grid"] == {"high": [0.65, 0.85], "low": [0.15, 0.35]}
    assert manifest["held_out_off_grid"] == {"high": [0.73, 0.87], "low": [0.12, 0.28]}
    assert manifest["report"]["held_out_levels"][:3] == [3, 8, 13]
    assert manifest["episodes"]["test_reports"] == 8 == len(read_jsonl(tmp_path / "out" / "test_reports.jsonl"))
    assert set(manifest["limits_in_train"]) <= set(manifest["train_grid"]["high"] + manifest["train_grid"]["low"])
    assert manifest["episodes"]["train"] == len(read_jsonl(tmp_path / "out" / "train.jsonl"))
    assert manifest["kinds"]["train"]["limit_twin"] == manifest["limit_twins"]["train"]


def test_build_limit_keeps_the_pairs_of_the_episodes_it_takes(tmp_path):
    from prio.datagen.episodes import write_jsonl
    from prio.datagen.limits import Options, build

    src = tmp_path / "src"
    src.mkdir()
    write_jsonl(src / "train.jsonl", train_source())
    write_jsonl(src / "test.jsonl", eval_source())
    manifest = build(src, tmp_path / "out", codec(), FAKE_MS, Options(limit=6, test_reports=2))
    from prio.datagen.episodes import read_jsonl

    train = read_jsonl(tmp_path / "out" / "train.jsonl")
    reacting = {e.task_id for e in train if e.kind in ("event", "starts_past", "gating_event")}
    quiet = {e.task_id for e in train if e.kind == "nominal" and e.t_diverge is not None}
    assert reacting and reacting <= quiet  # every reaction episode taken comes with its quiet twin
    assert manifest["source_episodes"]["test"] == 6


# --- evaluation: per-episode limits, limit twins, reports -----------------------------------------

def gen(text, stop="eos"):
    from prio.generate import GenResult

    ids = enc(text)
    return GenResult(ids, np.zeros((len(ids), 6), dtype=np.float32), stop)


def _eval_episode(episode_id="ev", kind="event", signal="ctx", limits=None, **fields):
    target = enc("a b c d e f g h")
    meta = {"limits": limits or {"ctx": 0.5, "think": 0.3, "energy": 0.45, "time": 0.55, "ram": 0.4}, "limit_split": "held_on"}
    meta.update(fields.pop("meta", {}))
    return make_episode(episode_id=episode_id, signal=signal, kind=kind, phase="answer", target_ids=target, split="test",
                        meta=meta, **fields)


def test_episode_specs_read_the_limits_of_the_episode():
    from prio.eval.limits_eval import episode_specs

    specs = episode_specs(_eval_episode(t_cross=3), CORE_SIGNALS)
    assert [s.limit for s in specs] == [0.5, 0.3, 0.45, 0.55, 0.0, 0.4]
    assert episode_specs(make_episode(), CORE_SIGNALS) == CORE_SIGNALS


def test_any_reaction_in_a_limit_twin_is_a_false_alarm(tok, ms):
    from prio.eval.limits_eval import classify_limits

    twin = _eval_episode("tw", kind="limit_twin", t_cross=3, meta={"twin_kind": "event"})
    specs = with_limits(CORE_SIGNALS, twin.meta["limits"])
    late = "x " * 60 + "my context is almost full " + CALL % ("compact_context", "{}")
    alarm = classify_limits(twin, "event", "on", gen(late), tok, ms, specs)
    assert (alarm.kind, alarm.paired, alarm.event_occurred, alarm.reaction_signal) == ("limit_twin", "event", False, "ctx")
    quiet = classify_limits(twin, "event", "on", gen("so the answer is 42"), tok, ms, specs)
    assert quiet.reaction_signal is None and not quiet.event_occurred
    gating = _eval_episode("tg", kind="limit_twin", signal="ram", t_cross=1, t_decision=5, meta={"twin_kind": "gating_event"})
    called = classify_limits(gating, "event", "on", gen(CALL % ("run_heavy", '{"job":"j"}')), tok, ms, specs)
    assert called.called_gated_tool and called.reaction_signal is None
    reacted = gen("my context is full " + CALL % ("compact_context", "{}"))
    event = classify_limits(_eval_episode(t_cross=3), "event", "on", reacted, tok, ms, specs)
    assert event.event_occurred and event.reaction_signal == "ctx" and event.kind == "event"


def _scripted(seen, text="x y"):
    def scripted(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **_):
        from prio.generate import GenResult

        seen.append((prompt_ids.shape[1], max_new_tokens, prompt_signals[0].cpu().numpy()))
        ids = enc(text)
        return GenResult(ids, np.zeros((len(ids), 6), dtype=np.float32), "eos")

    return scripted


def test_evaluation_draws_every_pairing_against_the_episode_limits(tiny_model, tok, ms, monkeypatch):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel
    from prio.eval.limits_eval import evaluate_limits

    seen = []
    monkeypatch.setattr(run_module, "generate", _scripted(seen))
    eps = [_eval_episode(t_cross=3),
           _eval_episode("tw", kind="limit_twin", t_cross=3, meta={"twin_kind": "event"}),
           _eval_episode("tg", kind="limit_twin", signal="ram", t_cross=1, t_decision=5, meta={"twin_kind": "gating_event"})]
    records = evaluate_limits(ChannelModel(tiny_model, 6), tok, ms, CORE_SIGNALS, eps, ("on", "zeroed"), log_every=0)
    runs = {}
    for r in records:
        runs.setdefault(r.episode_id, []).append((r.condition, r.paired))
    assert runs["ev"] == [("on", "event"), ("on", "nominal"), ("on", "near_miss"), ("on", "swapped"), ("zeroed", "event")]
    assert runs["tw"] == runs["tg"] == [("on", "event")]  # a limit twin runs once, channel on, its own trajectory
    assert [s[0] for s in seen] == [7] * 5 + [7, 8]  # prompt (3) + target[:t_cross + 1], or [:t_decision] at a tool decision
    specs = with_limits(CORE_SIGNALS, eps[0].meta["limits"])
    nominal, near, swapped = seen[1][2], seen[2][2], seen[3][2]
    for fed in (nominal, near):  # quiet pairings: nothing past the EPISODE's limits (phase-1 limits would allow 0.85)
        assert not any(s.past_limit(float(v)) for i, s in enumerate(specs) for v in fed[:, i])
    assert nominal[:, 0].max() < 0.5 and near[:, 0].max() < 0.5
    swap = next(r for r in records if r.paired == "swapped")
    column = NAMES.index(swap.signal)
    assert specs[column].past_limit(float(swapped[-1, column]))
    assert not any(specs[column].past_limit(float(v)) for v in swapped[:-1, column])
    twin = next(r for r in records if r.episode_id == "tw")
    assert twin.kind == "limit_twin" and not twin.event_occurred


def test_parse_report_reads_the_first_percent_of_the_answer():
    from prio.eval.limits_eval import parse_report

    assert parse_report("Battery at 63%.") == 63
    assert parse_report("<think>\nmaybe 20%\n</think>\n\nBattery at 7 %.") == 7
    assert parse_report("Battery is fine.") is None


def test_report_scores_are_mean_absolute_errors_in_points():
    from prio.eval.limits_eval import score_reports

    rows = [{"condition": "on", "signal": "energy", "level": 63, "held_out": True, "predicted": 60},
            {"condition": "on", "signal": "ctx", "level": 13, "held_out": True, "predicted": 13},
            {"condition": "on", "signal": "ctx", "level": 40, "held_out": True, "predicted": None},
            {"condition": "on", "signal": "ctx", "level": 41, "held_out": False, "predicted": 50},
            {"condition": "zeroed", "signal": "ctx", "level": 41, "held_out": False, "predicted": 11}]
    scores = score_reports(rows)
    held = scores["on|all|held_out"]
    assert held["n"] == 3 and held["parsed"] == pytest.approx(2 / 3)
    assert held["mae"] == 1.5 and held["exact"] == 0.5 and held["within_3"] == 1.0
    assert scores["on|ctx|held_out"]["mae"] == 0.0 and scores["on|all|seen"]["mae"] == 9.0
    assert scores["zeroed|all|seen"]["mae"] == 30.0


def test_report_evaluation_forces_the_question_and_reads_the_level(tiny_model, tok, ms, monkeypatch, tmp_path):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel
    from prio.eval.limits_eval import REPORT_HORIZON, evaluate_reports

    _, reports, _ = _transform(eval_source(), split="test", test_reports=4)
    seen = []
    monkeypatch.setattr(run_module, "generate", _scripted(seen, "Battery at 61%."))
    path = tmp_path / "reports.partial.jsonl"
    rows = evaluate_reports(ChannelModel(tiny_model, 6), tok, ms, reports, ("on", "zeroed"), out_path=path)
    assert len(rows) == 8 and {r["predicted"] for r in rows} == {61}
    first = reports[0]
    assert seen[0][0] == len(first.prompt_ids) + first.meta["report"]["answer_start"] and seen[0][1] == REPORT_HORIZON
    assert (rows[0]["level"], rows[0]["held_out"], rows[0]["signal"]) == (
        first.meta["report"]["level"], first.meta["report"]["held_out"], first.meta["report"]["signal"])
    again = evaluate_reports(ChannelModel(tiny_model, 6), tok, ms, reports, ("on", "zeroed"), out_path=path)
    assert again == rows and len(seen) == 8  # resumed from the file, nothing generated twice


def test_limits_summary_has_rows_per_limit_split(tmp_path):
    from prio.eval.limits_eval import write_limits_summary
    from prio.eval.metrics import RunRecord

    def record(episode_id, kind, paired, occurred, reacted):
        return RunRecord(episode_id, "ctx", kind, "answer", "on", paired, occurred, "ctx" if reacted else None, None,
                         reacted, False, False)

    records = [record("a", "event", "event", True, True), record("a", "event", "nominal", False, False),
               record("b", "event", "event", True, False), record("t", "limit_twin", "event", False, True)]
    split = {"a": "held_on", "b": "seen", "t": "held_on"}
    write_limits_summary(records, split, [], tmp_path)
    payload = json.loads((tmp_path / "limits_summary.json").read_text(encoding="utf-8"))
    groups = payload["groups"]
    assert groups["on|ctx [held_on]|all"]["reaction_rate"] == 1.0 and groups["on|ctx [seen]|all"]["reaction_rate"] == 0.0
    assert groups["on|ctx (limit_twin) [held_on]|all"]["false_alarm_rate"] == 1.0
    assert groups["on|all interrupt [held_on]|all"]["n"] == 1
    assert (tmp_path / "limits_summary.md").exists() and (tmp_path / "summary.md").exists()
