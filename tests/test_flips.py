"""Several-flips test set and its summary (preregistration
amendment A2): prio.datagen.flips, prio.eval.flips."""

import json
from dataclasses import replace

import numpy as np
import pytest

from helpers import CALL, FAKE_MS, FAKE_TOK, enc
from prio.datagen import flips as F
from prio.datagen import variants as V
from prio.datagen.episodes import Episode, read_jsonl, write_jsonl
from prio.generate import GenResult
from prio.signals import signal_set
from prio.trajectories import sample_trajectory

SPECS = signal_set(6)
NAMES = [s.name for s in SPECS]
BY_NAME = {s.name: s for s in SPECS}
INSERT_ID = 99_999
PROMPT = enc("<sys> system prompt <user> look this up <eos> <assistant>")
N_PROMPT = len(PROMPT)
WEB = CALL % ("web_search", '{"query":"q"}')
LOCAL = CALL % ("local_lookup", '{"query":"q"}')


def gating(i, signal, n_answer=40, kind="gating_event"):
    """A test gating event as data/n6 stores it: base text up to the tool decision, then the reaction."""
    rng = np.random.default_rng(i)
    base = enc(" ".join(f"a{k}" for k in range(n_answer)) + " " + WEB + " <eos>")
    d = base.index(FAKE_MS.tool_open)
    target = base[:d] + enc(f"I am offline so I use the local store {LOCAL} <eos>")
    t_cross = int(rng.integers(0, d))
    signals = sample_trajectory(SPECS, N_PROMPT + len(target), rng, event=NAMES.index(signal),
                                t_cross=N_PROMPT + t_cross).values
    return Episode(episode_id=f"test-{i:05d}", task_id=f"task-{i}", family="search", split="test", kind=kind,
                   signal=signal, phase="answer", prompt_ids=PROMPT, target_ids=target, signals=signals,
                   t_cross=t_cross, t_decision=d, templated=False, t_diverge=d)


def past(ep):
    """Per target token: is the episode's gating signal past its limit."""
    spec = BY_NAME[ep.signal]
    return [spec.past_limit(float(v)) for v in ep.signals[len(ep.prompt_ids):, NAMES.index(ep.signal)]]


def points(ep):
    return [p for a, b in ep.meta["flips"]["intervals"] for p in (a, b) if p is not None]


# --- change points ----------------------------------------------------------------------------

@pytest.mark.parametrize("m", [2, 3])
@pytest.mark.parametrize("final_bad", [True, False])
@pytest.mark.parametrize("d", [6, 7, 40])
def test_change_points_give_m_intervals_before_the_decision(m, final_bad, d):
    for seed in range(30):
        intervals = F.change_points(d, m, final_bad, np.random.default_rng(seed))
        flat = [p for a, b in intervals for p in (a, b) if p is not None]
        assert len(intervals) == m and len(flat) == (2 * m - 1 if final_bad else 2 * m)
        assert flat == sorted(set(flat)) and flat[0] >= 0 and flat[-1] <= d - 1
        assert (intervals[-1][1] is None) == final_bad and all(b is not None for _, b in intervals[:-1])


def test_change_points_need_one_token_per_change():
    with pytest.raises(ValueError):
        F.change_points(5, 3, False, np.random.default_rng(0))  # six changes, five tokens


# --- channel form -----------------------------------------------------------------------------

@pytest.mark.parametrize("signal", ["net", "ram"])
@pytest.mark.parametrize("m", [2, 3])
@pytest.mark.parametrize("final_bad", [True, False])
def test_flips_episode_redraws_only_the_gating_signal_up_to_the_decision(signal, m, final_bad):
    src = gating(3, signal)
    d = src.t_decision
    ep = F.flips_episode(src, SPECS, m, final_bad, seed=0)
    assert ep.episode_id == f"{src.episode_id}-m{m}-{'bad' if final_bad else 'good'}"
    assert ep.kind == ("gating_event" if final_bad else "recovery") and ep.signal == signal
    assert ep.target_ids == src.target_ids[:d] and ep.t_decision == d and ep.t_diverge is None
    assert ep.signals.shape == (N_PROMPT + d, 6) and ep.signals.dtype == np.float32
    info = ep.meta["flips"]
    assert (info["m"], info["final"], info["source"], info["decision"]) == (m, "bad" if final_bad else "good", src.episode_id, d)
    expected = [False] * d
    for a, b in info["intervals"]:
        expected[a:b] = [True] * ((d if b is None else b) - a)
    assert past(ep) == expected and expected[-1] == final_bad
    assert ep.t_cross == info["intervals"][0][0]
    index = NAMES.index(signal)
    others = [i for i in range(6) if i != index]
    assert np.array_equal(ep.signals[:, others], src.signals[:N_PROMPT + d, others])
    assert not any(BY_NAME[signal].past_limit(float(v)) for v in ep.signals[:N_PROMPT, index])  # prompt: good


def test_flips_episode_is_deterministic_and_follows_the_seed():
    src = gating(4, "ram")
    assert F.flips_episode(src, SPECS, 3, False, seed=0) == F.flips_episode(src, SPECS, 3, False, seed=0)
    assert points(F.flips_episode(src, SPECS, 3, False, seed=1)) != points(F.flips_episode(src, SPECS, 3, False, seed=0))


def test_flips_episode_takes_gating_events_only():
    with pytest.raises(ValueError):
        F.flips_episode(gating(1, "net", kind="recovery"), SPECS, 2, True, seed=0)


# --- notice / insert form ---------------------------------------------------------------------

@pytest.mark.parametrize("signal", ["net", "ram"])
@pytest.mark.parametrize("m, final_bad", [(2, True), (2, False), (3, True), (3, False)])
def test_noticed_puts_one_notice_after_every_state_change(signal, m, final_bad):
    plain = F.flips_episode(gating(5, signal), SPECS, m, final_bad, seed=0)
    ep = F.noticed(plain, SPECS, V.Carrier(FAKE_TOK))
    records = ep.meta["variant"]["insertions"]
    assert ep.meta["variant"]["name"] == "flips" and ep.meta["flips"] == plain.meta["flips"]
    assert (ep.episode_id, ep.kind, ep.signal) == (plain.episode_id, plain.kind, plain.signal)
    assert [r["role"] for r in records] == (["crossing", "recovery"] * m)[:2 * m - 1 if final_bad else 2 * m]
    spec, index, shift = BY_NAME[signal], NAMES.index(signal), 0
    for r, p in zip(records, points(plain)):
        assert r["at"] == p + 1 + shift  # right after the first token that carries the new state
        shift += r["end"] - r["at"]
        assert spec.past_limit(float(r["shown"])) == (r["role"] == "crossing")
        assert FAKE_TOK.decode(ep.target_ids[r["at"]:r["end"]]) == f"[status {signal}={r['shown']}]"
        rows = ep.signals[N_PROMPT + r["at"] - 1:N_PROMPT + r["end"], index]  # that token and the notice
        assert all(spec.past_limit(float(v)) == (r["role"] == "crossing") for v in rows)
    assert ep.harness_spans == [[r["at"], r["end"]] for r in records]
    assert ep.t_decision == len(ep.target_ids) == len(plain.target_ids) + shift
    kept = [t for i, t in enumerate(ep.target_ids) if not any(a <= i < b for a, b in ep.harness_spans)]
    assert kept == plain.target_ids and ep.signals.shape == (N_PROMPT + len(ep.target_ids), 6)


def test_noticed_with_the_insert_carrier_uses_one_position_per_change():
    plain = F.flips_episode(gating(6, "ram"), SPECS, 3, False, seed=0)
    ep = F.noticed(plain, SPECS, V.Carrier(FAKE_TOK, INSERT_ID))
    records = ep.meta["variant"]["insertions"]
    assert len(records) == 6 and ep.target_ids.count(INSERT_ID) == 6
    spec, index = BY_NAME["ram"], NAMES.index("ram")
    for r in records:  # one position, whose row shows the new state
        assert r["end"] - r["at"] == 1 and ep.target_ids[r["at"]] == INSERT_ID
        assert spec.past_limit(float(ep.signals[N_PROMPT + r["at"], index])) == (r["role"] == "crossing")


# --- folder -----------------------------------------------------------------------------------

def _source(tmp_path, n_answer=40):
    src = tmp_path / "n6"
    src.mkdir()
    eps = []
    for i in range(16):
        eps.append(gating(i, ("net", "ram")[i % 2], n_answer=n_answer))
        eps.append(gating(100 + i, "net", kind="recovery"))  # other kinds are not sources
    write_jsonl(src / "test.jsonl", eps)
    (src / "manifest.json").write_text(json.dumps({"model": "fake", "n_signals": 6, "seed": 0}), encoding="utf-8")
    return src


def test_build_takes_the_quick_subset_gating_events_times_m_times_final_state(tmp_path):
    src = _source(tmp_path)
    manifest = F.build(src, tmp_path / "flips", FAKE_TOK)
    eps = read_jsonl(tmp_path / "flips" / "test.jsonl")
    # 8 net + 8 ram gating events, every 4th of each signal: 4 sources x 2 m x 2 final states
    assert len(eps) == 16 and manifest["episodes"]["test"] == 16
    assert {e.meta["flips"]["source"] for e in eps} == {"test-00000", "test-00008", "test-00001", "test-00009"}
    assert manifest["kinds"]["test"] == {"gating_event": 8, "recovery": 8}
    assert manifest["variant"] == "flips" and manifest["options"]["carrier"] is None
    assert manifest["source_manifest"]["model"] == "fake" and manifest["skipped"] == {}
    assert all("variant" not in e.meta and e.harness_spans is None for e in eps)
    assert json.loads((tmp_path / "flips" / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_build_gives_every_carrier_the_same_change_points_and_is_deterministic(tmp_path):
    src = _source(tmp_path)
    F.build(src, tmp_path / "plain", FAKE_TOK)
    F.build(src, tmp_path / "notice", FAKE_TOK, carrier="notice")
    manifest = F.build(src, tmp_path / "insert", FAKE_TOK, carrier="insert", insert_id=INSERT_ID)
    plain, notice, inserted = (read_jsonl(tmp_path / name / "test.jsonl") for name in ("plain", "notice", "insert"))
    assert [e.episode_id for e in plain] == [e.episode_id for e in notice] == [e.episode_id for e in inserted]
    assert [e.meta["flips"] for e in plain] == [e.meta["flips"] for e in notice] == [e.meta["flips"] for e in inserted]
    # per source: 3 + 4 + 5 + 6 changes = 10 crossings and 8 recoveries
    assert manifest["insertions"]["test"] == {"crossing": 40, "recovery": 32}
    assert manifest["options"]["insert_id"] == INSERT_ID and manifest["harness_tokens"]["test"] == 72
    F.build(src, tmp_path / "again", FAKE_TOK, carrier="notice")
    assert (tmp_path / "again" / "test.jsonl").read_bytes() == (tmp_path / "notice" / "test.jsonl").read_bytes()
    with pytest.raises(ValueError):
        F.build(src, tmp_path / "x", FAKE_TOK, carrier="insert")  # needs the delivery's insert id
    with pytest.raises(ValueError):
        F.build(src, src, FAKE_TOK)


def test_build_skips_and_counts_what_does_not_fit_before_the_decision(tmp_path):
    src = _source(tmp_path, n_answer=4)  # four tokens before the decision: m = 2 fits, m = 3 does not
    manifest = F.build(src, tmp_path / "flips", FAKE_TOK)
    eps = read_jsonl(tmp_path / "flips" / "test.jsonl")
    assert {e.meta["flips"]["m"] for e in eps} == {2} and len(eps) == 8
    assert manifest["skipped"] == {"too_short": 8}


# --- training pairs (P-e1b, preregistration amendment A3) --------------------------------------

def twin_of(ep):
    """The no-event twin of a gating event as data/n6 stores it: same prompt and prefix, then the gated call."""
    base = ep.target_ids[:ep.t_decision] + enc(WEB + " <eos>")
    signals = sample_trajectory(SPECS, N_PROMPT + len(base), np.random.default_rng(7)).values
    return replace(ep, episode_id=ep.episode_id + "t", kind="nominal", signal=None, phase=None, target_ids=base,
                   signals=signals, t_cross=None, t_decision=None, t_diverge=ep.t_decision)


def past_of(ep, signal):
    spec = BY_NAME[signal]
    return [spec.past_limit(float(v)) for v in ep.signals[len(ep.prompt_ids):, NAMES.index(signal)]]


@pytest.mark.parametrize("signal", ["net", "ram"])
def test_train_pair_keeps_both_continuations_and_ends_bad_or_good(signal):
    event = gating(3, signal)
    twin, d = twin_of(event), event.t_decision
    bad, good = F.train_pair(event, twin, SPECS, 2, seed=0)
    assert (bad.episode_id, good.episode_id) == (f"{event.episode_id}-m2-bad", f"{twin.episode_id}-m2-good")
    assert (bad.kind, bad.signal, bad.t_decision, bad.t_diverge) == ("gating_event", signal, d, d)
    assert (good.kind, good.signal, good.t_cross, good.t_decision, good.t_diverge) == ("nominal", None, None, None, d)
    assert bad.target_ids == event.target_ids and good.target_ids == twin.target_ids  # nothing cut
    for ep, src, final in ((bad, event, "bad"), (good, twin, "good")):
        info = ep.meta["flips"]
        assert (info["m"], info["final"], info["source"], info["decision"], info["signal"]) == (2, final, src.episode_id, d, signal)
        expected = [False] * len(ep.target_ids)
        for a, b in info["intervals"]:
            assert 0 <= a < d and (b is None or a < b <= d - 1)
            expected[a:b] = [True] * ((len(expected) if b is None else b) - a)
        assert past_of(ep, signal) == expected  # the open interval lasts through the reaction
        assert expected[d - 1] == (final == "bad") and expected[-1] == (final == "bad")
        others = [i for i in range(6) if i != NAMES.index(signal)]
        assert np.array_equal(ep.signals[:, others], src.signals[:, others]) and ep.signals.dtype == np.float32
    assert len(bad.meta["flips"]["intervals"]) == len(good.meta["flips"]["intervals"]) == 2
    assert bad.meta["flips"]["intervals"][-1][1] is None and good.meta["flips"]["intervals"][-1][1] is not None
    assert bad.t_cross == bad.meta["flips"]["intervals"][0][0]


def test_train_pair_is_deterministic_and_needs_the_events_own_twin():
    event = gating(4, "ram")
    twin = twin_of(event)
    assert F.train_pair(event, twin, SPECS, 2, seed=0) == F.train_pair(event, twin, SPECS, 2, seed=0)
    assert points(F.train_pair(event, twin, SPECS, 2, seed=1)[0]) != points(F.train_pair(event, twin, SPECS, 2, seed=0)[0])
    with pytest.raises(ValueError):
        F.train_pair(event, twin_of(gating(5, "ram")), SPECS, 2, seed=0)  # another base run
    with pytest.raises(ValueError):
        F.train_pair(event, replace(twin, t_diverge=twin.t_diverge - 1), SPECS, 2, seed=0)
    with pytest.raises(ValueError):
        F.train_pair(twin, twin, SPECS, 2, seed=0)


@pytest.mark.parametrize("carrier", [V.Carrier(FAKE_TOK), V.Carrier(FAKE_TOK, INSERT_ID)])
def test_noticed_training_pair_keeps_reaction_weights_in_place(carrier):
    from prio.datagen.episodes import reaction_start

    event = gating(6, "net")
    twin, d = twin_of(event), event.t_decision
    bad, good = (F.noticed(ep, SPECS, carrier) for ep in F.train_pair(event, twin, SPECS, 2, seed=0))
    assert [r["role"] for r in bad.meta["variant"]["insertions"]] == ["crossing", "recovery", "crossing"]
    assert [r["role"] for r in good.meta["variant"]["insertions"]] == ["crossing", "recovery", "crossing", "recovery"]
    for ep, src in ((bad, event), (good, twin)):
        kept = [t for i, t in enumerate(ep.target_ids) if not any(a <= i < b for a, b in ep.harness_spans)]
        assert kept == src.target_ids and ep.signals.shape == (N_PROMPT + len(ep.target_ids), 6)
        assert all(r["end"] <= ep.t_diverge for r in ep.meta["variant"]["insertions"])  # every notice before the decision
        assert ep.target_ids[ep.t_diverge:] == src.target_ids[d:]
    assert reaction_start(bad) == bad.t_decision == bad.t_diverge and reaction_start(good) is None
    assert good.t_cross is None and good.kind == "nominal"
    spec, index = BY_NAME["net"], NAMES.index("net")
    for ep in (bad, good):
        for r in ep.meta["variant"]["insertions"]:
            assert spec.past_limit(float(r["shown"])) == (r["role"] == "crossing")
            assert spec.past_limit(float(ep.signals[N_PROMPT + r["at"], index])) == (r["role"] == "crossing")


def _train_source(tmp_path, n_answer=40):
    src, base = tmp_path / "n6", tmp_path / "n6-notice"
    src.mkdir()
    base.mkdir()
    eps = []
    for i in range(6):
        event = replace(gating(i, ("net", "ram")[i % 2], n_answer=n_answer), episode_id=f"train-{i:05d}", split="train")
        eps += [event, replace(twin_of(event), episode_id=f"train-{i:05d}t")]
    eps.append(replace(gating(50, "net"), episode_id="train-00050", split="train", task_id="lonely"))  # no twin: not a source
    eps.append(replace(gating(60, "net", kind="recovery"), episode_id="train-00060", split="train"))
    write_jsonl(src / "train.jsonl", eps)
    (src / "manifest.json").write_text(json.dumps({"model": "fake", "n_signals": 6, "seed": 0}), encoding="utf-8")
    own = [F.noticed(F.flips_episode(gating(70 + i, "net"), SPECS, 2, True), SPECS, V.Carrier(FAKE_TOK)) for i in range(3)]
    write_jsonl(base / "train.jsonl", own)  # stands for the carrier's own training file
    (base / "manifest.json").write_text(json.dumps({"variant": "notice", "options": {"carrier": "notice", "insert_id": None}}),
                                        encoding="utf-8")
    return src, base


def test_build_train_appends_one_pair_per_twinned_gating_event(tmp_path):
    src, base = _train_source(tmp_path)
    before = (base / "train.jsonl").read_bytes()
    manifest = F.build_train(src, base, tmp_path / "m2", FAKE_TOK, carrier="notice")
    eps = read_jsonl(tmp_path / "m2" / "train.jsonl")
    own = read_jsonl(base / "train.jsonl")
    assert eps[:3] == own and (base / "train.jsonl").read_bytes() == before  # the carrier's file first, untouched
    pairs = eps[3:]
    assert [e.episode_id for e in pairs] == [f"train-{i:05d}{s}" for i in range(6) for s in ("-m2-bad", "t-m2-good")]
    assert all(e.meta["variant"]["name"] == "flips" and e.meta["flips"]["m"] == 2 for e in pairs)
    assert [e.kind for e in pairs] == ["gating_event", "nominal"] * 6
    assert manifest["variant"] == "flips-train" and manifest["episodes"]["train"] == 15
    assert manifest["base_episodes"] == 3 and manifest["pairs"] == 6 and manifest["skipped"] == {}
    assert manifest["options"] == {"seed": 0, "m": 2, "carrier": "notice", "insert_id": None}
    assert manifest["insertions"]["train"] == {"crossing": 24, "recovery": 18}  # per pair: 2 + 2 crossings, 1 + 2 recoveries
    assert json.loads((tmp_path / "m2" / "manifest.json").read_text(encoding="utf-8")) == manifest
    F.build_train(src, base, tmp_path / "again", FAKE_TOK, carrier="notice")
    assert (tmp_path / "again" / "train.jsonl").read_bytes() == (tmp_path / "m2" / "train.jsonl").read_bytes()


def test_build_train_checks_its_folders_and_counts_what_does_not_fit(tmp_path):
    src, base = _train_source(tmp_path, n_answer=3)  # three tokens before the decision: four changes do not fit
    manifest = F.build_train(src, base, tmp_path / "m2", FAKE_TOK, carrier="notice")
    assert manifest["pairs"] == 0 and manifest["skipped"] == {"too_short": 6} and manifest["episodes"]["train"] == 3
    for out in (src, base):
        with pytest.raises(ValueError):
            F.build_train(src, base, out, FAKE_TOK, carrier="notice")
    with pytest.raises(ValueError):
        F.build_train(src, base, tmp_path / "x", FAKE_TOK, carrier="insert", insert_id=INSERT_ID)  # base is a notice folder
    with pytest.raises(ValueError):
        F.build_train(src, base, tmp_path / "x", FAKE_TOK, carrier=None)


def test_main_takes_the_numbers_of_intervals(tmp_path, capsys):
    src = _source(tmp_path)
    F.main(["--src", str(src), "--out", str(tmp_path / "m4"), "--ms", "4"])
    eps = read_jsonl(tmp_path / "m4" / "test.jsonl")
    assert {e.meta["flips"]["m"] for e in eps} == {4} and len(eps) == 8  # 4 sources x 2 final states
    assert json.loads(capsys.readouterr().out)["options"]["ms"] == [4]


# --- evaluation and summary -------------------------------------------------------------------

def test_harness_mode_runs_every_flips_episode_once_up_to_its_decision(tiny_model, tok, ms, monkeypatch):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    eps = [F.noticed(F.flips_episode(gating(7, "net"), SPECS, 2, final_bad, seed=0), SPECS, V.Carrier(FAKE_TOK))
           for final_bad in (True, False)]
    seen = []
    refuse = enc(f"I am offline so I use the local store {LOCAL}")
    bad, good = enc("net=0.00]")[0], enc("net=1.00]")[0]

    def generate(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **kw):
        prompt = prompt_ids[0].tolist()
        seen.append((prompt, prompt_signals[0].cpu().numpy(), kw.get("prompt_harness")))
        last = next(t for t in reversed(prompt) if t in (bad, good))  # scripted: obeys the last notice
        ids = refuse if last == bad else enc(WEB)
        return GenResult(list(ids), np.zeros((len(ids), 6), dtype=np.float32), "max_new_tokens")

    monkeypatch.setattr(run_module, "generate", generate)
    records = run_module.evaluate_harness(ChannelModel(tiny_model, 6), tok, ms, SPECS, {"event": eps}, log_every=0)
    assert [(r.episode_id, r.paired, r.kind) for r in records] == [
        (eps[0].episode_id, "event", "gating_event"), (eps[1].episode_id, "event", "recovery")]
    for (prompt, values, harness), ep in zip(seen, eps):
        assert prompt == ep.prompt_ids + ep.target_ids and len(values) == len(prompt)  # the whole prefix, notices in
        assert np.array_equal(values, ep.signals)
        assert [i - N_PROMPT for i, h in enumerate(harness) if h] == [i for a, b in ep.harness_spans for i in range(a, b)]
    from dataclasses import asdict

    from prio.eval.flips import correct

    assert [correct(asdict(r)) for r in records] == [True, True]


def _record(i, kind, signal="net", reaction=None, gated=False):
    return {"episode_id": i, "kind": kind, "signal": signal, "condition": "on", "paired": "event",
            "reaction_signal": reaction, "called_gated_tool": gated}


def test_correct_decision_follows_the_final_state():
    from prio.eval.flips import correct

    assert correct(_record("a", "gating_event", reaction="net"))
    assert not correct(_record("a", "gating_event", reaction=None, gated=True))  # called the gated tool while bad
    assert not correct(_record("a", "gating_event", reaction="ram"))
    assert correct(_record("a", "recovery", gated=True))
    assert not correct(_record("a", "recovery", reaction="net"))  # stale refusal


def test_summary_pools_per_m_and_applies_the_pre_registered_line(tmp_path):
    from prio.eval import flips as S

    src = _source(tmp_path)
    F.build(src, tmp_path / "flips", FAKE_TOK)
    eps = read_jsonl(tmp_path / "flips" / "test.jsonl")
    records = []
    for e in eps:  # everything right, except every final-good run with m = 3
        bad, wrong = e.kind == "gating_event", e.meta["flips"]["m"] == 3 and e.kind == "recovery"
        records.append(_record(e.episode_id, e.kind, e.signal, reaction=e.signal if bad or wrong else None,
                               gated=not bad and not wrong))
    base = [_record(f"b{i}", "gating_event", reaction="net") for i in range(9)] + [_record("b9", "recovery", gated=False)]
    base += [dict(_record("z", "gating_event"), paired="quiet"), dict(_record("y", "event"), signal="ctx")]  # not counted
    table = S.summarize(records, eps, base)
    assert table["base"] == {"k": 9, "n": 10, "rate": 0.9}
    by_m = {row["m"]: row for row in table["by_m"]}
    assert (by_m[2]["k"], by_m[2]["n"], by_m[3]["k"], by_m[3]["n"]) == (8, 8, 4, 8)
    assert by_m[2]["passes"] and not by_m[3]["passes"] and not table["passes"]
    cells = {(row["m"], row["final"]): (row["k"], row["n"]) for row in table["by_final"]}
    assert cells == {(2, "bad"): (4, 4), (2, "good"): (4, 4), (3, "bad"): (4, 4), (3, "good"): (0, 4)}
    assert {(row["m"], row["signal"]): row["n"] for row in table["by_signal"]} == {
        (2, "net"): 4, (2, "ram"): 4, (3, "net"): 4, (3, "ram"): 4}
    assert sum(row["n"] for row in table["by_distance"]) == 16 and len(table["by_distance"]) == 3
    assert [row["lo"] for row in table["by_distance"]] == sorted(row["lo"] for row in table["by_distance"])
    text = S.write_summary(table, tmp_path / "out", arm="scripted")
    assert "| 3 | 4 / 8 | 50.0%" in text and (tmp_path / "out" / "flips_summary.md").read_text(encoding="utf-8") == text
    assert json.loads((tmp_path / "out" / "flips_summary.json").read_text(encoding="utf-8"))["passes"] is False


def test_summary_line_needs_95_percent_and_no_more_than_three_points_below_m_1():
    from prio.eval.flips import passes

    assert passes(0.96, 0.98) and passes(0.95, 0.977)
    assert not passes(0.949, 0.90)  # under 95%
    assert not passes(0.955, 0.99)  # more than 3 points under the m <= 1 rate
    assert passes(0.96, None)  # no m <= 1 reference given: the 95% half only
