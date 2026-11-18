"""Context eviction on the several-flips test files (R21, preregistration amendment A7)."""
import json

import numpy as np
import pytest

from helpers import FAKE_TOK, enc
from prio.datagen import evict as E
from prio.datagen import flips as F
from prio.datagen import variants as V
from prio.datagen.episodes import read_jsonl
from prio.generate import GenResult
from test_flips import INSERT_ID, LOCAL, N_PROMPT, SPECS, WEB, _source, gating, points


def plain(i=5, signal="net", m=2, final_bad=True):
    return F.flips_episode(gating(i, signal), SPECS, m, final_bad, seed=0)


def base_tokens(ep):
    """The turn without what the harness inserted."""
    return [t for i, t in enumerate(ep.target_ids) if not any(a <= i < b for a, b in ep.harness_spans or ())]


# --- one episode ------------------------------------------------------------------------------

@pytest.mark.parametrize("m, final_bad", [(1, True), (1, False), (2, True), (3, False)])
@pytest.mark.parametrize("window", [4, 16])
def test_evict_keeps_the_prompt_and_the_last_tokens_before_the_decision(m, final_bad, window):
    ep = plain(m=m, final_bad=final_bad)
    d, cut = ep.t_decision, ep.t_decision - window
    ev = E.evict(ep, window)
    assert ev.prompt_ids == ep.prompt_ids and ev.target_ids == ep.target_ids[cut:] and ev.t_decision == window
    assert np.array_equal(ev.signals, np.concatenate([ep.signals[:N_PROMPT], ep.signals[N_PROMPT + cut:]]))
    assert (ev.episode_id, ev.kind, ev.signal) == (ep.episode_id, ep.kind, ep.signal)
    assert ev.meta["flips"] == ep.meta["flips"]  # the summary reads m, final state and distances from here
    inside = [p + 1 >= cut for p in points(ep)]  # a change is inside when its first full token of the new state is
    assert ev.meta["evict"] == {"window": window, "cut": cut, "decision": d, "changes": len(inside),
                                "changes_inside": sum(inside), "last_inside": inside[-1]}
    assert inside == sorted(inside)  # whatever is inside is a suffix of the changes
    assert ev.t_cross is not None and 0 <= ev.t_cross <= window


def test_a_window_longer_than_the_turn_evicts_nothing():
    ep = plain()
    ev = E.evict(ep, 1000)
    assert ev.target_ids == ep.target_ids and np.array_equal(ev.signals, ep.signals) and ev.t_decision == ep.t_decision
    assert ev.meta["evict"]["cut"] == 0 and ev.meta["evict"]["changes_inside"] == ev.meta["evict"]["changes"] == 3
    assert ev.t_cross == ep.t_cross


@pytest.mark.parametrize("m, final_bad", [(1, True), (2, True), (2, False), (3, True)])
@pytest.mark.parametrize("window", [4, 12, 30])
def test_notices_before_the_cut_go_whole_and_the_others_stay_whole(m, final_bad, window):
    channel_form = plain(m=m, final_bad=final_bad)
    ep = F.noticed(channel_form, SPECS, V.Carrier(FAKE_TOK))
    ev, ref = E.evict(ep, window), E.evict(channel_form, window)
    cut = channel_form.t_decision - window
    assert ev.meta["evict"] == ref.meta["evict"]  # the cut is at the same place for every carrier
    assert base_tokens(ev) == channel_form.target_ids[cut:]  # W counts tokens of the original turn
    records, kept = ep.meta["variant"]["insertions"], ev.meta["variant"]["insertions"]
    inside = [p + 1 >= cut for p in points(channel_form)]
    assert len(kept) == sum(inside) == ev.meta["evict"]["changes_inside"]
    for old, new in zip([r for r, k in zip(records, inside) if k], kept):
        assert {k: old[k] for k in ("role", "signal", "shown")} == {k: new[k] for k in ("role", "signal", "shown")}
        assert FAKE_TOK.decode(ev.target_ids[new["at"]:new["end"]]) == f"[status net={new['shown']}]"  # never split
        assert np.array_equal(ev.signals[N_PROMPT + new["at"]:N_PROMPT + new["end"]],
                              ep.signals[N_PROMPT + old["at"]:N_PROMPT + old["end"]])
    assert (ev.harness_spans or []) == [[r["at"], r["end"]] for r in kept]
    assert ev.t_decision == len(ev.target_ids) and ev.signals.shape == (N_PROMPT + len(ev.target_ids), 6)
    assert ev.meta["variant"]["name"] == "flips"


def test_a_notice_right_at_the_cut_is_kept_and_one_token_earlier_is_dropped():
    channel_form = plain(m=2, final_bad=True)
    ep = F.noticed(channel_form, SPECS, V.Carrier(FAKE_TOK))
    d, last = channel_form.t_decision, points(channel_form)[-1]
    at_cut = E.evict(ep, d - (last + 1))  # the cut falls on the token the last notice follows
    assert at_cut.meta["evict"]["changes_inside"] == 1 and at_cut.meta["evict"]["last_inside"] is True
    assert at_cut.harness_spans[0][0] == 0  # the window opens with that notice
    one_later = E.evict(ep, d - (last + 2))
    assert one_later.meta["evict"]["changes_inside"] == 0 and one_later.harness_spans is None
    assert "[status" not in FAKE_TOK.decode(one_later.target_ids)


def test_inserted_positions_follow_the_same_rule():
    channel_form = plain(i=6, signal="ram", m=3, final_bad=False)
    ep = F.noticed(channel_form, SPECS, V.Carrier(FAKE_TOK, INSERT_ID))
    for window in (4, 12, 30):
        ev = E.evict(ep, window)
        assert ev.target_ids.count(INSERT_ID) == ev.meta["evict"]["changes_inside"]
        assert ev.meta["evict"] == E.evict(channel_form, window).meta["evict"]
        assert base_tokens(ev) == channel_form.target_ids[channel_form.t_decision - window:]


def test_evict_takes_flips_episodes_only():
    with pytest.raises(ValueError):
        E.evict(gating(3, "net"), 16)  # no change points to count
    with pytest.raises(ValueError):
        E.evict(plain(), 0)


# --- folder -----------------------------------------------------------------------------------

@pytest.mark.parametrize("carrier, insert_id", [(None, None), ("notice", None), ("insert", INSERT_ID)])
def test_build_writes_the_same_runs_with_the_cut_and_counts_what_stays_inside(tmp_path, carrier, insert_id):
    src = _source(tmp_path)
    F.build(src, tmp_path / "flips", FAKE_TOK, carrier=carrier, insert_id=insert_id)
    manifest = E.build(tmp_path / "flips", tmp_path / "w8", 8)
    before, after = read_jsonl(tmp_path / "flips" / "test.jsonl"), read_jsonl(tmp_path / "w8" / "test.jsonl")
    assert [e.episode_id for e in after] == [e.episode_id for e in before] and len(after) == 16
    assert all(e == E.evict(b, 8) for e, b in zip(after, before))
    counts = manifest["changes_inside"]["test"]
    assert sum(counts.values()) == 16 and set(counts) <= {"0", "1", "2+"}
    assert counts.get("0", 0) == sum(e.meta["evict"]["changes_inside"] == 0 for e in after)
    assert manifest["variant"] == "evict" and manifest["options"]["window"] == 8
    assert manifest["evicted_from"] == str(tmp_path / "flips") and manifest["source"] == str(src)
    assert manifest["source_manifest"]["model"] == "fake" and manifest["options"]["carrier"] == carrier
    assert json.loads((tmp_path / "w8" / "manifest.json").read_text(encoding="utf-8")) == manifest
    first = (tmp_path / "w8" / "test.jsonl").read_bytes()
    E.build(tmp_path / "flips", tmp_path / "w8", 8)
    assert (tmp_path / "w8" / "test.jsonl").read_bytes() == first  # deterministic


def test_build_checks_its_folders(tmp_path):
    src = _source(tmp_path)
    F.build(src, tmp_path / "flips", FAKE_TOK)
    with pytest.raises(ValueError):
        E.build(src, tmp_path / "out", 8)  # not a flips folder
    with pytest.raises(ValueError):
        E.build(tmp_path / "flips", tmp_path / "flips", 8)  # would overwrite its source


def test_main_builds_one_folder_per_window(tmp_path, capsys):
    src = _source(tmp_path)
    F.build(src, tmp_path / "flips", FAKE_TOK)
    E.main(["--src", str(tmp_path / "flips"), "--out", str(tmp_path / "w16"), "--window", "16"])
    assert json.loads(capsys.readouterr().out)["options"]["window"] == 16
    assert len(read_jsonl(tmp_path / "w16" / "test.jsonl")) == 16


# --- evaluation -------------------------------------------------------------------------------

def test_harness_mode_gives_the_model_the_prompt_and_the_window_only(tiny_model, tok, ms, monkeypatch):
    import prio.eval.run as run_module
    from prio.channel import ChannelModel

    eps = [E.evict(F.noticed(plain(i=7, m=2, final_bad=final_bad), SPECS, V.Carrier(FAKE_TOK)), 6)
           for final_bad in (True, False)]
    seen = []

    def generate(model, prompt_ids, prompt_signals, signal_fn, max_new_tokens, eos_ids, **kw):
        seen.append((prompt_ids[0].tolist(), prompt_signals[0].cpu().numpy(), kw.get("prompt_harness")))
        ids = enc(WEB)
        return GenResult(list(ids), np.zeros((len(ids), 6), dtype=np.float32), "max_new_tokens")

    monkeypatch.setattr(run_module, "generate", generate)
    records = run_module.evaluate_harness(ChannelModel(tiny_model, 6), tok, ms, SPECS, {"event": eps}, log_every=0)
    assert [(r.episode_id, r.kind) for r in records] == [(e.episode_id, e.kind) for e in eps]
    for (prompt, values, harness), ep in zip(seen, eps):
        assert prompt == ep.prompt_ids + ep.target_ids and np.array_equal(values, ep.signals)
        marked = [i - N_PROMPT for i, h in enumerate(harness or []) if h]
        assert marked == [i for a, b in ep.harness_spans or () for i in range(a, b)]
    assert LOCAL  # (the scripted model never refuses here; decisions are judged by prio.eval.flips)
