"""The three pre-registered lines of R12 and the report (prio.r12.lines; analysis plan Sections 7-9)."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

import prio.eval.run as run_module
from helpers import CALL, make_episode
from prio import r12
from prio.channel import ChannelModel
from prio.datagen.episodes import write_jsonl
from prio.eval.metrics import RunRecord
from prio.r12 import ledger
from prio.r12 import lines as N
from prio.r12 import stream as S
from prio.signals import CORE_SIGNALS
from r12_helpers import MS, R12Tok
from test_r12_stream import N_PROMPT, Scripted, plan, prompt_of

ZERO = {name: 0 for name in r12.MIN_COUNTS}


# --- records by hand ------------------------------------------------------------------------------------

def rr(episode: str, hit: bool = True, *, signal: str = "ctx", kind: str = "event", paired: str = "event",
       condition: str = "on", occurred: bool = True, reaction=...) -> RunRecord:
    if reaction is ...:
        reaction = signal if hit else None
    return RunRecord(episode_id=episode, signal=signal, kind=kind, phase="thinking", condition=condition, paired=paired,
                     event_occurred=occurred, reaction_signal=reaction, delay=1 if reaction == signal else None,
                     named=reaction is not None, called_gated_tool=False, used_alternative=False)


def paired_records(code=(300, 270), seen=(300, 290), twins=(300, 6), quiet="nominal"):
    """`code` / `seen`: (event runs, hits) of the new family / of math + explain; `twins`: (quiet twins
    of the new family, false alarms). Returns the records and the family of each episode."""
    records, family = [], {}
    for name, (n, k) in (("code", code), ("math", (seen[0] // 2, seen[1] // 2)), ("explain", (seen[0] - seen[0] // 2, seen[1] - seen[1] // 2))):
        for i in range(n):
            episode = f"{name}-{i}"
            family[episode] = name
            records.append(rr(episode, i < k))
    for i in range(twins[0]):
        records.append(rr(f"code-{i}", paired=quiet, occurred=False, reaction="energy" if i < twins[1] else None))
    return records, family


def sr(i: int, *, arm="free", signal="ctx", run_type="event", occurred=True, why_not=None, hit=True, onset=0, k=40,
       false_alarm=None, model="channel", family="math", budget=None, premature=False, onset_last=None, at_start=None,
       deadline=False) -> S.StreamRecord:
    hit = bool(hit and occurred and run_type == "event")
    quiet = run_type != "event"
    return S.StreamRecord(
        task_id=f"r12c-{family}-{i:05d}", family=family, arm=arm, model=model, signal=signal, run_type=run_type, k=k,
        occurred=occurred and run_type == "event", why_not=why_not, phase="thinking", hit=hit,
        reaction_signal=signal if hit else None, named=hit, delay=2 if hit else None, onset=onset if hit else None,
        onset_word=onset if hit else None, onset_first_text=0 if hit else None, premature=premature,
        false_alarm=false_alarm if run_type != "event" else None, unnamed_call=False,
        crossing_at=k - 1 if occurred and run_type == "event" else None, same_prefix=True if run_type == "twin" else None,
        n_pre=k, n_post=96, stop_reason="max_new_tokens", budget=budget, start=0 if budget else None,
        onset_last=(onset if onset_last is None else onset_last) if hit else None,
        alarm_start=(bool(false_alarm) if at_start is None else at_start) if quiet else None,
        alarm_at_k=bool(false_alarm) if quiet else None, alarm_at_close=False if quiet else None, deadline_near_end=deadline)


# --- verdicts and intervals -----------------------------------------------------------------------------

def test_thresholds_are_exact_fractions():
    line = Fraction(85, 100)
    assert N.verdict(254, 299, at_least=line) == "not met"  # 84.95%: rounds to 85.0%, does not meet 85%
    assert N.verdict(255, 300, at_least=line) == "met" and N.verdict(17, 20, at_least=line) == "met"
    assert N.verdict(15, 300, at_most=Fraction(5, 100)) == "met" and N.verdict(16, 300, at_most=Fraction(5, 100)) == "not met"
    assert N.verdict(0, 250, at_most=Fraction(5, 100)) == "met"
    assert N.THRESHOLD == {"L1a": Fraction(17, 20), "L1b": Fraction(1, 20), "L1c": Fraction(1, 10), "L2a": Fraction(9, 10),
                           "L2b": Fraction(19, 20), "L3": Fraction(17, 20)}


def test_too_few_runs_is_not_evaluable_never_met():
    assert N.verdict(199, 199, at_least=Fraction(85, 100), minimum=200) == "not evaluable"
    assert N.verdict(0, 199, at_least=Fraction(85, 100), minimum=200) == "not evaluable"  # nor "not met"
    assert N.verdict(200, 200, at_least=Fraction(85, 100), minimum=200) == "met"
    assert N.verdict(0, 0, at_least=Fraction(85, 100)) == "not evaluable" and N.verdict(0, 0, at_most=Fraction(5, 100)) == "not evaluable"


def test_newcombe_interval_against_a_worked_example():
    # Newcombe (1998), Statistics in Medicine 17: 56 / 70 against 48 / 80, method 10: 0.0524 to 0.3339
    low, high = N.newcombe(56, 70, 48, 80)
    assert low == pytest.approx(0.0524, abs=5e-4) and high == pytest.approx(0.3339, abs=5e-4)
    back = N.newcombe(48, 80, 56, 70)
    assert back == (pytest.approx(-high), pytest.approx(-low))
    assert N.newcombe(10, 10, 10, 10)[0] < 0 < N.newcombe(10, 10, 10, 10)[1]


# --- line 1 ----------------------------------------------------------------------------------------------

def test_line_1_needs_reaction_false_alarms_and_the_gap():
    records, family = paired_records()
    line = N.line1(records, family)
    assert (line["L1a"]["k"], line["L1a"]["n"], line["L1a"]["verdict"]) == (270, 300, "met")
    assert (line["L1b"]["k"], line["L1b"]["n"], line["L1b"]["verdict"]) == (6, 300, "met")
    assert (line["L1c"]["seen"]["k"], line["L1c"]["seen"]["n"], line["L1c"]["verdict"]) == (290, 300, "met")
    assert line["L1c"]["gap_points"] == pytest.approx(100 * (290 / 300 - 270 / 300)) and line["verdict"] == "met"
    assert line["L1a"]["rate"] == pytest.approx(0.9) and line["L1a"]["ci"][0] < 0.9 < line["L1a"]["ci"][1]
    assert line["L1a"]["bound_clears"] is True and line["L1b"]["bound_clears"] is True  # [86.1%, ...] and [..., 4.3%]
    assert line["L1c"]["gap_ci_points"][0] < line["L1c"]["gap_points"] < line["L1c"]["gap_ci_points"][1]

    for change, part in ((dict(code=(300, 254), seen=(300, 280)), "L1a"), (dict(twins=(300, 16)), "L1b"),
                         (dict(code=(300, 255), seen=(300, 286)), "L1c")):
        line = N.line1(*paired_records(**change))
        assert line[part]["verdict"] == "not met" and line["verdict"] == "not met", part
        assert [line[p]["verdict"] for p in ("L1a", "L1b", "L1c") if p != part] == ["met", "met"]
    edge = N.line1(*paired_records(code=(300, 255), seen=(300, 285)))  # 95% against 85%: exactly 10 points
    assert edge["L1c"]["verdict"] == "met" and edge["L1c"]["bound_clears"] is False and edge["verdict"] == "met"
    close = N.line1(*paired_records(code=(300, 256)))  # 85.3%: met, but the interval reaches under 85%
    assert close["L1a"]["verdict"] == "met" and close["L1a"]["bound_clears"] is False


def test_line_1_counts_only_what_the_plan_names():
    records, family = paired_records()
    family.update({"web-0": "web", "code-x": "code"})
    extra = [rr("code-x", False, kind="starts_past"), rr("code-x", False, condition="zeroed"), rr("code-x", False, condition="base"),
             rr("code-x", paired="near_miss", occurred=False, reaction="ctx"),  # a near-miss alarm is not a false alarm of L1b
             rr("code-x", False, kind="swap", paired="swapped"), rr("web-0", False, kind="gating_event", signal="net"),
             rr("code-x", False, occurred=False),  # `think` after the thinking closed: no event occurred
             rr("code-x", paired="nominal", occurred=False, kind="starts_past", reaction="ctx")]
    line = N.line1(records + extra, family)
    assert (line["L1a"]["k"], line["L1a"]["n"], line["L1b"]["k"], line["L1b"]["n"], line["L1c"]["seen"]["n"]) == (270, 300, 6, 300, 300)
    # a hit is the reaction to the event's own signal: a reaction to another signal is a miss
    wrong = [rr(f"code-{i}", reaction="energy") if r.episode_id == f"code-{i}" and r.paired == "event" and i == 0 else r
             for i, r in enumerate(records)]
    assert N.line1(wrong, family)["L1a"]["k"] == 269
    # the text-notice model's quiet twins are its `quiet` runs
    quiet = N.line1(*paired_records(quiet="quiet", twins=(300, 16)))
    assert (quiet["L1b"]["k"], quiet["L1b"]["n"], quiet["L1b"]["verdict"], quiet["quiet_pairing"]) == (16, 300, "not met", "quiet")


def test_seen_family_gap_is_one_sided():
    better = N.line1(*paired_records(code=(300, 297), seen=(300, 258)))  # the new family 13 points above the seen ones
    assert better["L1c"]["gap_points"] == pytest.approx(-13.0) and better["L1c"]["verdict"] == "met"
    assert better["verdict"] == "met"


def test_line_1_minimum_counts():
    few = N.line1(*paired_records(code=(199, 199), twins=(199, 0)))
    assert few["L1a"]["verdict"] == few["L1b"]["verdict"] == "not evaluable" and few["verdict"] == "not evaluable"
    few_twins = N.line1(*paired_records(twins=(150, 0)))
    assert few_twins["L1a"]["verdict"] == "met" and few_twins["L1b"]["verdict"] == "not evaluable"
    assert few_twins["verdict"] == "not evaluable"
    few_seen = N.line1(*paired_records(seen=(198, 190)))  # L1c not evaluable; L1a and L1b still reported
    assert (few_seen["L1a"]["verdict"], few_seen["L1b"]["verdict"], few_seen["L1c"]["verdict"]) == ("met", "met", "not evaluable")
    assert few_seen["verdict"] == "not evaluable" and few_seen["L1c"]["seen"]["n"] == 198
    failed = N.line1(*paired_records(code=(300, 200), seen=(198, 190)))
    assert failed["verdict"] == "not met"  # a part that is not met decides, whatever cannot be evaluated
    pilot = N.line1(*paired_records(code=(20, 19), seen=(20, 19), twins=(20, 0)), minimums=ZERO)
    assert pilot["verdict"] == "met"


# --- lines 2 and 3 ---------------------------------------------------------------------------------------

def free_records(occurred=300, hits=280, ended=40, closed=20, onset_ok=None, last_ok=None):
    onset_ok = hits if onset_ok is None else onset_ok
    last_ok = onset_ok if last_ok is None else last_ok
    rows = [sr(i, signal=("ctx", "think", "energy", "time")[i % 4], hit=i < hits, onset=0 if i < onset_ok else 2,
               onset_last=0 if i < last_ok else 7) for i in range(occurred)]
    rows += [sr(occurred + i, occurred=False, why_not="ended") for i in range(ended)]
    rows += [sr(occurred + ended + i, signal="think", occurred=False, why_not="thinking_over") for i in range(closed)]
    rows += [sr(i, run_type="twin", false_alarm=i < 3) for i in range(occurred)] + [sr(i, run_type="off", false_alarm=False) for i in range(occurred)]
    return rows


def test_line_2_excludes_and_counts_crossings_that_did_not_occur():
    line = N.line2(free_records())
    assert (line["L2a"]["k"], line["L2a"]["n"], line["L2a"]["verdict"]) == (280, 300, "met")
    assert line["planned"] == 360 and line["not_occurred"] == {"ended": 40, "thinking_over": 20, "no_crossing": 0,
                                                               "crossing_off_k": 0}
    assert line["verdict"] == "met"
    assert N.line2(free_records(hits=269))["L2a"]["verdict"] == "not met"  # 89.7%
    assert N.line2(free_records(hits=270))["L2a"]["verdict"] == "met"  # exactly 90%
    few = N.line2(free_records(occurred=239, hits=239, ended=121, closed=0))
    assert few["L2a"]["verdict"] == "not evaluable" and few["verdict"] == "not evaluable" and few["L2a"]["n"] == 239
    live = [sr(i, arm="live") for i in range(300)]  # another arm's records never enter
    assert N.line2(live)["L2a"]["n"] == 0


def test_onset_line_is_over_hits_only():
    line = N.line2(free_records(hits=280, onset_ok=266))
    assert (line["L2b"]["k"], line["L2b"]["n"], line["L2b"]["verdict"]) == (266, 280, "met")  # exactly 95% of the hits
    late = N.line2(free_records(hits=280, onset_ok=265))
    assert late["L2b"]["verdict"] == "not met" and late["L2a"]["verdict"] == "met" and late["verdict"] == "not met"
    rows = free_records(hits=280)
    rows[0].onset = 1  # index 1 still counts
    rows[1].onset = None  # a hit whose naming sentence cannot be placed does not
    assert N.line2(rows)["L2b"]["k"] == 279
    few = N.line2(free_records(occurred=240, hits=99, ended=120, closed=0))
    assert (few["L2a"]["verdict"], few["L2b"]["verdict"], few["verdict"]) == ("not met", "not evaluable", "not met")


def test_the_second_onset_reading_is_printed_beside_the_line_and_never_decides_it():
    line = N.line2(free_records(hits=280, onset_ok=280, last_ok=250))  # a keyword in ordinary text flattered 30 hits
    assert (line["L2b"]["k"], line["L2b"]["n"], line["L2b"]["verdict"], line["verdict"]) == (280, 280, "met", "met")
    beside = line["onset_last"]
    assert (beside["k"], beside["n"]) == (250, 280) and beside["rate"] == pytest.approx(250 / 280) and "verdict" not in beside
    assert beside["ci"][0] < 250 / 280 < beside["ci"][1]
    rows = free_records(hits=280)
    rows[0].onset_last = None  # no keyword before the marker: not at most 1
    assert N.line2(rows)["onset_last"]["k"] == 279


def test_a_live_crossing_off_token_k_is_out_of_line_3_and_counted_by_reason():
    rows = live_records()
    for r in rows[:7]:  # seven `ctx` runs whose first past-limit row was on another token
        r.occurred, r.why_not, r.hit, r.crossing_at = False, "crossing_off_k", False, r.k + 1
    line = N.line3(rows)
    assert (line["L3"]["k"], line["L3"]["n"]) == (248, 293) and line["by_counter"]["ctx"]["n"] == 93
    assert line["not_occurred"] == {"ended": 0, "thinking_over": 0, "no_crossing": 0, "crossing_off_k": 7}
    assert N.no_line(rows, "live")["not_occurred"]["crossing_off_k"] == 7


def live_records(per_counter=(100, 100, 100), hits=(85, 85, 85), arm_name="live"):
    rows, i = [], 0
    for signal, n, k in zip(("ctx", "think", "time"), per_counter, hits):
        for j in range(n):
            rows.append(sr(i, arm=arm_name, signal=signal, hit=j < k, budget=1000))
            i += 1
    return rows


def test_line_3_pools_the_three_counters_and_needs_60_per_counter():
    line = N.line3(live_records())
    assert (line["L3"]["k"], line["L3"]["n"], line["L3"]["verdict"], line["verdict"]) == (255, 300, "met", "met")
    assert {name: (c["k"], c["n"]) for name, c in line["by_counter"].items()} == {"ctx": (85, 100), "think": (85, 100), "time": (85, 100)}
    uneven = N.line3(live_records(hits=(100, 100, 55)))  # one counter far under 85%: the pooled rate decides
    assert uneven["L3"]["verdict"] == "met" and uneven["by_counter"]["time"]["rate"] == pytest.approx(0.55)
    assert N.line3(live_records(hits=(85, 85, 84)))["L3"]["verdict"] == "not met"
    thin = N.line3(live_records(per_counter=(120, 121, 59), hits=(120, 121, 59)))
    assert thin["L3"]["n"] == 300 and thin["L3"]["verdict"] == "not evaluable" and thin["verdict"] == "not evaluable"
    few = N.line3(live_records(per_counter=(80, 80, 79), hits=(80, 80, 79)))
    assert few["L3"]["verdict"] == "not evaluable"
    mixed = live_records() + live_records(hits=(0, 0, 0), arm_name="Ls")  # arm Ls is never in line 3
    assert N.line3(mixed)["L3"]["k"] == 255 and N.line3(mixed)["L3"]["n"] == 300
    beside = N.no_line(live_records(per_counter=(0, 60, 60), hits=(0, 30, 45), arm_name="Ls"), "Ls")
    assert beside["verdict"] == "no line" and (beside["pooled"]["k"], beside["pooled"]["n"]) == (75, 120)
    assert beside["by_counter"]["think"]["rate"] == pytest.approx(0.5) and "ctx" not in beside["by_counter"]


# --- tables that are not lines ---------------------------------------------------------------------------

def test_k_terciles_are_fixed_by_the_range_of_k():
    assert N.k_terciles("free") == [(16, 42), (43, 69), (70, 96)]
    assert N.k_terciles("live") == N.k_terciles("Ls") == [(32, 74), (75, 117), (118, 160)]
    rows = [sr(i, k=16 + i, hit=i < 40) for i in range(81)]
    table = N.stream_tables(rows, "free")
    assert table["by_k"] == {"16-42": {"planned": 27, "occurred": 27, "hits": 27}, "43-69": {"planned": 27, "occurred": 27, "hits": 13},
                             "70-96": {"planned": 27, "occurred": 27, "hits": 0}}
    live = [sr(i, arm="live", signal="think", budget=b, k=40) for i, b in enumerate((800, 1000, 1999, 2000, 2001, 4000))]
    live += [sr(9, arm="live", signal="ctx", budget=700, k=40)]
    table = N.stream_tables(live, "live")
    # points per token = 100 / budget; a budget of 1,000 is exactly 0.10, one of 2,000 exactly 0.05 (each in the bin it opens)
    assert {name: v["planned"] for name, v in table["by_steepness"].items()} == {"under 0.05": 2, "0.05 to 0.10": 2, "0.10 and over": 3}
    assert table["budgets_drawn"]["think"] == {"n": 6, "budget": [800, 1999.5, 4000], "start": [0, 0.0, 0]}


def test_models_are_compared_paired_in_the_paired_arm_and_unpaired_elsewhere():
    channel, family = paired_records(code=(300, 290), seen=(300, 290))
    notice, _ = paired_records(code=(300, 270), seen=(300, 290), quiet="quiet")
    both = N.between_paired(channel, notice, family)
    assert both["all"]["pairs"] == 600 and both[r12.FAMILIES[-1]]["pairs"] == 300
    assert (both["code"]["channel_hits"], both["code"]["notice_hits"], both["code"]["channel_only"], both["code"]["notice_only"]) == (290, 270, 20, 0)
    assert both["code"]["p"] < 1e-5 and both["code"]["claim"] is True and both["code"]["test"] == "exact McNemar"
    same = N.between_paired(channel, channel, family)
    assert same["all"]["p"] == 1.0 and same["all"]["claim"] is False
    free = N.between_stream(free_records(hits=280), free_records(hits=270), "free")
    assert free["channel"] == [280, 300] and free["notice"] == [270, 300] and free["test"] == "Fisher exact"
    assert 0.05 < free["p"] < 0.5 and free["claim"] is False
    assert N.between_stream([], free_records(), "free")["p"] is None


def test_expected_runs_of_a_paired_evaluation():
    episodes = ([{"kind": "event"}] * 10 + [{"kind": "gating_event"}] * 4 + [{"kind": "starts_past"}] * 3 + [{"kind": "recovery"}] * 2
                + [{"kind": "gating_unused"}])
    # on: 20 + 2 x 14 main + 10 swaps = 58; channel model: + 20 with the channel off + 14 on the untrained model
    assert N.expected_paired_runs(episodes, "notice") == 58 and N.expected_paired_runs(episodes, "channel") == 92
    many = [{"kind": "event"}] * 400
    assert N.expected_paired_runs(many, "channel") == 400 * 4 + 400 + 150


def test_reference_rates_are_read_from_the_full_evaluations(tmp_path):
    assert N.REFERENCE == {"channel": "results/phase1", "notice": "results/b3-notice-none-full-b16"}
    assert N.reference_rates("channel", tmp_path) is None  # no such folder: nothing printed
    folder = tmp_path / "results" / "phase1"
    folder.mkdir(parents=True)
    group = {"n": 343, "reaction_rate": 0.95, "false_alarm_rate": 0.003, "n_nominal": 343, "delay_median": 1.0}
    (folder / "summary.json").write_text(json.dumps({"groups": {"on|all interrupt|all": group, "on|ctx|all": {}}}), encoding="utf-8")
    assert N.reference_rates("channel", tmp_path) == {"folder": "results/phase1", "n": 343, "reaction_rate": 0.95, "n_quiet": 343,
                                                      "false_alarm_rate": 0.003}


def test_scoring_is_unchanged_since_the_freeze_commit():
    # The real repository, read only: reactions.py, eval/metrics.py and the source of classify as frozen.
    state = N.scoring_unchanged(Path("."))
    if state["passed"] is None:  # a copy without the project's history: nothing to compare with
        assert state == {"freeze_commit": r12.FREEZE_COMMIT, "files_unchanged": None, "classify_unchanged": None, "passed": None}
        pytest.skip("the freeze commit is not in this repository's history")
    assert state == {"freeze_commit": r12.FREEZE_COMMIT, "files_unchanged": True, "classify_unchanged": True, "passed": True}
    assert N._function_source("def a():\n    return 1\n\n\ndef b():\n    return 2\n", "b") == "def b():\n    return 2"


def test_stream_reading_lists_every_miss_and_false_alarm_of_the_new_family(tmp_path):
    tok = R12Tok()
    w = tok.encode("w")
    pre = [MS.think_open] + w * 19
    ctx = tok.encode("my context is almost full " + CALL % ("compact_context", "{}") + " <eos>")
    energy = tok.encode("my battery is almost empty " + CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}") + " <eos>")
    budget = tok.encode("my thinking budget is almost gone </think> so the answer is short")
    early = [MS.think_open] + w * 3 + budget + w * (20 - 4 - len(budget))
    runs = [
        (plan(0, "ctx", 20, family="code"), "event", pre, tok.encode("the function adds two numbers") + w * 91, "max_new_tokens"),  # a miss
        (plan(1, "ctx", 20, family="code"), "event", early, ctx, "eos"),  # a reaction before the crossing: never a hit
        (plan(2, "ctx", 20, family="code"), "event", pre, ctx, "eos"),  # a hit: not listed
        (plan(3, "ctx", 20, family="code"), "event", pre[:10] + tok.encode("<eos>"), [], "eos"),  # no crossing: not listed
        (plan(4, "ctx", 20, family="math"), "event", pre, w * 96, "max_new_tokens"),  # a miss of a seen family: not listed
        (plan(0, "ctx", 20, family="code"), "twin", pre, w * 130 + energy, "eos"),  # a false alarm far into the run
        (plan(2, "ctx", 20, family="code"), "off", pre, energy, "eos"),  # a false alarm right after token k
        (plan(2, "ctx", 20, family="code"), "twin", pre, w * 96, "max_new_tokens"),  # quiet: not listed
        # a thinking-budget reaction right after token 100: seen at the crossing position, not from the first token
        (plan(5, "think", 100, family="code"), "twin", [MS.think_open] + w * 99, budget + tok.encode("<eos>"), "eos"),
    ]
    records, rows = [], []
    for p, run_type, first, rest, stop in runs:
        records.append(S.judge(p, run_type, "channel", first, rest, stop, tok, MS, CORE_SIGNALS, crossing_at=19))
        rows.append({"task_id": p.task_id, "run_type": run_type, "n_prompt": 12, "pre": first, "post": rest, "stop_reason": stop,
                     "notice": None, "notice_text": None, "rows_milli": None, "budgets": None})
    assert [(r.occurred, r.hit, r.premature) for r in records[:5]] == [(True, False, False), (True, False, True), (True, True, False),
                                                                        (False, False, False), (True, False, False)]
    assert [r.false_alarm for r in records[5:]] == [True, True, False, True]
    assert (records[8].alarm_start, records[8].alarm_at_k, records[8].alarm_at_close) == (False, True, True)
    (tmp_path / "generations.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    listed = N.stream_reading(tmp_path, records, tok, MS)["new_family"]
    assert [(x["what"], x["task_id"], x["run_type"], x["premature"]) for x in listed] == [
        ("miss", "r12p-code-00000", "event", False), ("miss", "r12p-code-00001", "event", True),
        ("false alarm", "r12p-code-00000", "twin", False), ("false alarm", "r12p-code-00002", "off", False),
        ("false alarm", "r12p-code-00005", "twin", False)]
    miss, premature, far, near, at_k = listed
    assert (at_k["found"], at_k["from_token"], at_k["reaction"]) == ("at the crossing position", 100 + 6 - 60, "think")
    assert "thinking budget is almost gone" in at_k["text"] and (far["found"], near["found"]) == ("from the first token",) * 2
    assert miss["text"].startswith(" the function adds two numbers w w") and len(miss["text"]) <= 300 and miss["from_token"] == 0
    assert (miss["signal"], miss["k"], miss["reaction"], miss["tokens"], miss["stop_reason"]) == ("ctx", 20, None, 116, "max_new_tokens")
    assert "context is almost full" in premature["text"] and premature["reaction"] == "ctx"  # the judged part, as it was written
    # a false alarm is shown around its reaction, wherever in the run it is; the same cap on the printed text
    assert far["from_token"] == 20 + 130 + 5 - 60 and "battery is almost empty" in far["text"] and far["reaction"] == "energy"
    assert near["from_token"] == 0 and "battery" in near["text"] and len(far["text"]) <= 300
    assert N.stream_reading(tmp_path, [r for r in records if r.family != "code"], tok, MS) == {"new_family": []}


# --- the report -------------------------------------------------------------------------------------------

EPISODES =([("code", "event", "ctx")] * 4 + [("math", "event", "time")] * 2 + [("explain", "event", "think")] * 2
            + [("web", "gating_event", "net")] * 2 + [("code", "starts_past", "energy")])


@pytest.fixture
def project(tmp_path, monkeypatch, tiny_model):
    state = {"root": tmp_path, "tok": R12Tok(), "model": ChannelModel(tiny_model, 6), "monkeypatch": monkeypatch}
    monkeypatch.setattr(ledger, "code_tree", lambda: "t" * 40)
    monkeypatch.setattr(ledger, "commit", lambda: "c" * 40)
    monkeypatch.setattr(ledger, "tree_clean", lambda: True)
    monkeypatch.setattr(N, "scoring_unchanged", lambda root=Path("."): {"freeze_commit": "f" * 40, "files_unchanged": True,
                                                                       "classify_unchanged": True, "passed": True})
    monkeypatch.setattr(N, "_load_tokenizer", lambda model_id: (state["tok"], MS))
    (tmp_path / "results" / "r12p").mkdir(parents=True)
    (tmp_path / "results" / "r12p" / "smoke.json").write_text(json.dumps({"passed": True, "tree": "t" * 40}), encoding="utf-8")
    return state


def stream_plans_of(arm_name: str) -> list[S.Plan]:
    signals = {"free": ("ctx", "think", "energy", "time"), "live": ("ctx", "think", "time"), "Ls": ("think", "time")}[arm_name]
    offset = {"free": 0, "live": 10, "Ls": 20}[arm_name]
    return [plan(offset + i, signals[i % len(signals)], 34 + 3 * i, arm_name, ("math", "explain", "code")[i % 3]) for i in range(6)]


def write_data(state, set_name: str) -> Path:
    data = state["root"] / "data" / r12.PREFIX[set_name]
    for folder in ("paired", "paired-notice"):
        (data / folder).mkdir(parents=True, exist_ok=True)
        episodes = [make_episode(episode_id=f"ep-{i:03d}", family=family, kind=kind, signal=signal, phase="thinking", t_cross=1,
                                 t_decision=1 if kind == "gating_event" else None) for i, (family, kind, signal) in enumerate(EPISODES)]
        write_jsonl(data / folder / "test.jsonl", episodes)
    for name in ("test_quiet.jsonl", "test_near_miss.jsonl", "test_swap.jsonl"):
        (data / "paired-notice" / name).write_text("", encoding="utf-8")
    manifest = {"tasks_requested": {"math": 3, "explain": 2, "web": 3, "code": 6},
                "dropped": {"math": {"not_ended": 1, "reaction_tool": 0, "no_position": 0, "too_long": 0},
                            "explain": {"not_ended": 0, "reaction_tool": 0, "no_position": 0, "too_long": 0},
                            "web": {"not_ended": 0, "reaction_tool": 0, "no_position": 1, "too_long": 0},
                            "code": {"not_ended": 0, "reaction_tool": 1, "no_position": 0, "too_long": 1}},
                "main_episodes": {"math": 2, "explain": 2, "web": 2, "code": 4},
                "kept_share": {"math": 2 / 3, "explain": 1.0, "web": 2 / 3, "code": 2 / 3}, "under_70_percent": ["math", "web", "code"]}
    (data / "paired" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    plans = [vars(p) for arm_name in ("free", "live", "Ls") for p in stream_plans_of(arm_name)]
    (data / "stream.json").write_text(json.dumps(plans), encoding="utf-8")
    (data / "tasks.json").write_text("[]", encoding="utf-8")
    return data


def paired_folder(state, set_name: str, model: str, misses=(), alarms=()) -> Path:
    """A paired results folder with exactly the runs an evaluation makes; `misses` / `alarms`:
    episode numbers whose event run misses / whose quiet twin reacts."""
    tok, out = state["tok"], state["root"] / N.out_folder(set_name, "paired", model)
    out.mkdir(parents=True, exist_ok=True)
    quiet = "nominal" if model == "channel" else "quiet"
    records, generations = [], []

    def add(i, signal, kind, paired, condition, occurred, reaction, text):
        records.append(vars(rr(f"ep-{i:03d}", signal=signal, kind=kind, paired=paired, condition=condition, occurred=occurred,
                               reaction=reaction)))
        generations.append({"episode_id": f"ep-{i:03d}", "condition": condition, "paired": paired, "signal": signal,
                            "stop_reason": "eos", "token_ids": tok.encode(text)})

    for i, (family, kind, signal) in enumerate(EPISODES):
        hit = i not in misses
        add(i, signal, kind, "event", "on", True, signal if hit else None,
            "I am almost out of time so here it is <eos>" if signal == "time" else "the model went on with plain words <eos>")
        if kind in ("event", "gating_event"):
            add(i, signal, kind, quiet, "on", False, "energy" if i in alarms else None, "my battery is almost empty <eos>")
            add(i, signal, kind, "near_miss", "on", False, None, "plain words <eos>")
        if kind == "event":
            add(i, "energy" if signal != "energy" else "ctx", "swap", "swapped", "on", True, None, "plain words <eos>")
        if model == "channel":
            add(i, signal, kind, "event", "zeroed", True, None, "plain words <eos>")
            if kind in ("event", "gating_event"):
                add(i, signal, kind, "event", "base", True, None, "plain words <eos>")
    (out / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (out / "generations.jsonl").write_text("\n".join(json.dumps(g) for g in generations) + "\n", encoding="utf-8")
    (out / "eval_meta.json").write_text(json.dumps({"invocations": [{"args": {"batch_size": 16}}]}), encoding="utf-8")
    return out


def stream_folder(state, set_name: str, arm_name: str, model: str, options=None, out: Path | None = None) -> Path:
    tok = state["tok"]
    script = Scripted(tok, options)
    state["monkeypatch"].setattr(run_module, "generate", script)
    plans = stream_plans_of(arm_name)
    prompts = {p.task_id: prompt_of(tok, p.index) for p in plans}
    out = out or state["root"] / N.out_folder(set_name, arm_name, model)
    runner = S.run_channel if model == "channel" else S.run_notice
    runner(state["model"], tok, MS, CORE_SIGNALS, plans, prompts, r12.SET_CODE[set_name], out, batch_size=1)
    (out / "eval_meta.json").write_text(json.dumps({"invocations": [{"args": {"batch_size": 16}}]}), encoding="utf-8")
    return out


def test_lines_are_judged_per_model(project):
    root = project["root"]
    write_data(project, "pilot")
    paired_folder(project, "pilot", "channel")
    paired_folder(project, "pilot", "notice", misses=(0, 1), alarms=(2,))
    options = {("free", "notice"): {4: {"end_at": 9}},  # a turn that ends before its crossing
               ("Ls", "channel"): {22: {"mute": True}}}  # a code task whose crossing the model ignores
    for arm_name in ("free", "live", "Ls"):
        for model in ("channel", "notice"):
            stream_folder(project, "pilot", arm_name, model, options.get((arm_name, model)))
    rep = N.report("pilot", root, project["tok"], MS)
    assert rep["pilot"] is True and rep["set"] == "pilot"
    assert rep["statement"] == {"channel": {"line 1": "met", "line 2": "met", "line 3": "met"},
                                "notice": {"line 1": "not met", "line 2": "met", "line 3": "met"}}
    channel, notice = rep["lines"]["channel"], rep["lines"]["notice"]
    assert (channel["line 1"]["L1a"]["k"], channel["line 1"]["L1a"]["n"]) == (4, 4)
    assert (notice["line 1"]["L1a"]["k"], notice["line 1"]["L1a"]["n"], notice["line 1"]["L1a"]["verdict"]) == (2, 4, "not met")
    assert (notice["line 1"]["L1b"]["k"], notice["line 1"]["L1b"]["n"], notice["line 1"]["L1b"]["verdict"]) == (1, 4, "not met")
    assert notice["line 1"]["L1c"]["seen"]["n"] == 4 and notice["line 1"]["quiet_pairing"] == "quiet"
    assert (channel["line 2"]["L2a"]["k"], channel["line 2"]["L2a"]["n"]) == (6, 6)
    assert (notice["line 2"]["L2a"]["k"], notice["line 2"]["L2a"]["n"]) == (5, 5)  # one turn ended before its crossing: excluded
    assert notice["line 2"]["not_occurred"] == {"ended": 1, "thinking_over": 0, "no_crossing": 0, "crossing_off_k": 0}
    assert (notice["line 2"]["onset_last"]["k"], notice["line 2"]["onset_last"]["n"]) == (5, 5)
    assert channel["line 3"]["L3"]["n"] == 6 and set(channel["line 3"]["by_counter"]) == {"ctx", "think", "time"}
    assert rep["models"]["channel"]["Ls"]["beside"]["verdict"] == "no line"
    assert set(rep["models"]["channel"]["Ls"]["beside"]["by_counter"]) == {"think", "time"}

    # accounting: every planned run produced; integrity: every check passed
    runs = rep["accounting"]["runs"]
    assert runs["paired / channel"] == {"planned": N.expected_paired_runs(N.episode_rows(root / "data/r12p/paired/test.jsonl"), "channel"),
                                        "produced": 60, "judged": 60}
    assert runs["paired / notice"]["planned"] == runs["paired / notice"]["produced"] == 39
    assert runs["free / notice"] == {"planned": 18, "produced": 18, "event_judged": 5,
                                     "event_not_occurred": {"ended": 1, "thinking_over": 0, "no_crossing": 0, "crossing_off_k": 0},
                                     "twin_judged": 6, "off_judged": 6}
    assert rep["accounting"]["paired_tasks"]["code"] == {"requested": 6, "not_ended": 0, "reaction_tool": 1, "no_position": 0,
                                                         "too_long": 1, "episode": 4}
    assert rep["integrity_passed"] is True and all(c["passed"] is not False for c in rep["integrity"])
    names = [c["check"] for c in rep["integrity"]]
    assert "one batch size" in names and "classifier and scoring unchanged since the freeze commit" in names
    assert sum(name.startswith("planned and produced runs") for name in names) == 8
    # saved rows against recomputed rows: a check for the channel model; for the text-notice model the row says what it is
    rows_checks = {c["check"]: c for c in rep["integrity"] if c["check"].startswith("saved rows equal the rows recomputed")}
    assert len(rows_checks) == 6 and all(c["passed"] is True for name, c in rows_checks.items() if name.endswith("channel"))
    for arm_name in ("live", "Ls"):
        row = rows_checks[f"saved rows equal the rows recomputed from the tokens, {arm_name} / notice"]
        assert row["passed"] is None and "true by construction, not a check" in row["detail"]
    assert rows_checks["saved rows equal the rows recomputed from the tokens, free / notice"]["passed"] is None
    # a crossing off token k: an integrity row in arm free only (planned rows); in the live arms a count by reason
    assert [name for name in names if name.startswith("crossing on token k")] == ["crossing on token k, free / channel",
                                                                                 "crossing on token k, free / notice"]
    assert all(c["passed"] is True for c in rep["integrity"] if not c["check"].startswith("saved rows"))

    # the comparison of the two models and the readings of the saved generations
    assert set(rep["between_models"]) == {"paired", "free", "live", "Ls"}
    assert rep["between_models"]["paired"]["code"]["channel_only"] == 2 and rep["between_models"]["paired"]["all"]["pairs"] == 8
    reading = rep["models"]["notice"]["paired"]["reading"]
    assert reading["time"] == {"n": 2, "strict": 2, "practical": 2}
    assert [(x["what"], x["episode_id"]) for x in reading["new_family"]] == [("miss", "ep-000"), ("miss", "ep-001"), ("false alarm", "ep-002")]
    assert "battery" in reading["new_family"][2]["text"]
    assert rep["models"]["channel"]["paired"]["tables"]["by_family"]["code"]["reaction_rate"] == 1.0
    assert rep["models"]["notice"]["paired"]["tables"]["by_family"]["code"]["false_alarm_rate"] == 0.25
    assert rep["base_runs"]["under_70_percent"] == ["math", "web", "code"]
    # the arms without a forced prefix: the new family's misses and false alarms decoded, the twins' end levels, unplaced onsets
    ls = rep["models"]["channel"]["Ls"]
    assert [(x["what"], x["task_id"], x["run_type"]) for x in ls["reading"]["new_family"]] == [("miss", "r12p-code-00022", "event")]
    assert ls["reading"]["new_family"][0]["text"].strip().startswith("w w w")
    assert rep["models"]["notice"]["free"]["reading"] == {"new_family": []}
    for model in ("channel", "notice"):
        for arm_name in ("live", "Ls"):
            ends = rep["models"][model][arm_name]["twin_end_levels"]
            assert ends["n"] == 6 and 84 <= ends["min"] <= ends["max"] <= 89 and ends["above_89"] == 0
            assert (ends["fed_max"] is None) == (model == "notice") and (ends["fed_max"] or 0) <= 89
        assert rep["models"][model]["free"]["twin_end_levels"] is None  # no budget in arm free
    assert rep["models"]["channel"]["free"]["summary"]["event"]["onset"]["b"].get("none", 0) == 0

    json.dumps(rep)  # the whole report is plain data
    text = N.render(rep)
    assert text.startswith("# R12 lines report, PILOT set") and "never reported as results" in text
    assert text.count("could not be placed") == 6 and "seen by neither side" in text  # readings I and J, in words
    # the three anchors of the classifier, the count from the first token alone, the keyword-only deadline count
    assert "at the crossing position" in text and "at the close of the thinking" in text and "first generated token" in text
    assert text.count("from the first generated token alone") >= 6
    assert "keyword match only" in text and "is not seen by the frozen classifier" in text and "not a counted reaction" in text
    assert "### Turns that ended and name the deadline in their last 128 tokens (keyword match only, not a counted reaction)" in text
    assert "| live | channel | 2 | 0 | 0 | 0 |" in text and "| free | text-notice | 1 | 0 | 0 | 0 |" in text  # the `time` hits name it
    assert "At the last keyword before the reaction's marker" in text and "| text-notice | 5 / 5 | 100.0% |" in text
    assert "INTEGRITY" not in text and "crossing_off_k 0" in text
    live = rep["models"]["channel"]["live"]["summary"]
    assert live["twin"]["at_start"] == live["twin"]["false_alarms"] == 0 and live["deadline_near_end"]["twin"] == 0
    assert text.count("Twins' target counter") == 4 and "miss `r12p-code-00022` (event" in text
    assert "INTEGRITY FAILURE" not in text and "## Arm Ls beside arm live (no line)" in text
    assert "| text-notice | L1a | new family reaction >= 85% | 2 / 4 | 50.0% |" in text and "**not met**" in text
    assert text.count("(not a line)") >= 12  # every table that is not a line says so
    without = N.report("pilot", root)  # no tokenizer: the readings are left out, the row check is reported as not made
    assert without["models"]["notice"]["paired"]["reading"] is None and without["integrity_passed"] is True
    assert without["models"]["channel"]["Ls"]["reading"] is None and without["models"]["channel"]["Ls"]["twin_end_levels"]["n"] == 6
    assert sum(c["passed"] is None for c in without["integrity"]) == 6 and "not checked" in N.render(without)

    N.main(["--set", "pilot", "--root", str(root)])
    assert (root / "results" / "r12p" / "r12_lines.md").read_text(encoding="utf-8").startswith("# R12 lines report, PILOT set")
    saved = json.loads((root / "results" / "r12p" / "r12_lines.json").read_text(encoding="utf-8"))
    assert saved["statement"] == rep["statement"]
    assert b"\r" not in (root / "results" / "r12p" / "r12_lines.md").read_bytes()


def test_lines_main_prints_a_report_that_holds_characters_outside_the_console_code_page(project, monkeypatch):
    import io
    import sys

    root = project["root"]
    write_data(project, "pilot")
    monkeypatch.setattr(N, "render", lambda rep: "- miss `r12p-code-00022`: ' so 3 → 4 and then …'\n")  # decoded model text
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")  # the Windows console of this machine
    monkeypatch.setattr(sys, "stdout", console)
    N.main(["--set", "pilot", "--root", str(root)])  # used to end in UnicodeEncodeError after writing its files
    console.flush()
    assert "3 → 4 and then …".encode("utf-8") in console.buffer.getvalue()
    assert "3 → 4" in (root / "results" / "r12p" / "r12_lines.md").read_text(encoding="utf-8")


def test_report_prints_every_marker_entry_and_labels_reruns(project, capsys):
    root = project["root"]
    data = write_data(project, "confirm")
    files = [data / "tasks.json", data / "stream.json"]

    def start(model="channel", arm_name="free", reason=None, options=None, finish=True):
        entry = ledger.begin("confirm", arm_name, model, root / N.out_folder("confirm", arm_name, model), files,
                             ["--set", "confirm", "--arm", arm_name, "--model", model], rerun_reason=reason, root=root)
        out = stream_folder(project, "confirm", arm_name, model, options, out=root / entry["out"])
        if finish:
            ledger.finish(entry, out / "records.jsonl", root)
        return entry

    start()  # the first run: the confirmatory result
    start(reason="defect D-999 in the runner", options={i: {"end_at": 5} for i in range(6)})  # a rerun in which nothing occurs
    start(model="notice", finish=False)  # a run that never recorded its end
    rep = N.report("confirm", root, project["tok"], MS)
    assert rep["pilot"] is False
    line = rep["lines"]["channel"]["line 2"]
    assert (line["L2a"]["k"], line["L2a"]["n"]) == (6, 6)  # the first run, not the rerun
    assert line["L2a"]["verdict"] == "not evaluable" and line["verdict"] == "not evaluable"  # 6 crossings: far under 240
    assert rep["statement"]["channel"] == {"line 1": "not run", "line 2": "not evaluable", "line 3": "not run"}
    again = rep["models"]["channel"]["free"]["reruns"]
    assert [(r["rerun"], r["reason"], r["folder"], r["summary"]["event"]["occurred"]) for r in again] == [
        (1, "defect D-999 in the runner", "results/r12-free-wide-levels-s0-b16-rerun1", 0)]
    assert [(e["arm"], e["model"], e["kind"]) for e in rep["marker"]] == [("free", "channel", "run"), ("free", "channel", "rerun"),
                                                                         ("free", "notice", "run")]
    failed = [c for c in rep["integrity"] if c["passed"] is False]
    assert [c["check"] for c in failed] == ["marker entry free / notice"] and "unfinished" in failed[0]["detail"]
    assert rep["integrity_passed"] is False
    text = N.render(rep)
    assert text.startswith("# R12 lines report, confirmatory set") and "INTEGRITY FAILURE" in text
    assert "RERUN 1 of arm free, channel model (reason: defect D-999 in the runner" in text
    assert text.count("- free / channel, first run:") == 1 and "- free / channel, RERUN 1 (reason: defect D-999 in the runner)" in text
    assert "- free / notice, first run:" in text and "finished None" in text

    # a records file that is not the one the run recorded
    ledger.finish(ledger.entries("confirm", root)[2], root / "results" / "r12-free-notice-none-s0-b16" / "records.jsonl", root)
    whole = N.report("confirm", root, project["tok"], MS)
    assert whole["integrity_passed"] is True
    # the data files named in each entry are hashed again: are they still the files the run read?
    hashed = [c for c in whole["integrity"] if c["check"].startswith("data files of the marker entry")]
    assert [c["check"] for c in hashed] == ["data files of the marker entry free / channel",
                                            "data files of the marker entry free / channel (rerun 1)",
                                            "data files of the marker entry free / notice"]
    assert all(c["passed"] is True and c["detail"].startswith("2 of 2 files") for c in hashed)
    original = (data / "stream.json").read_bytes()
    (data / "stream.json").write_bytes(original + b"\n")  # the same plans in another file
    changed = N.report("confirm", root, project["tok"], MS)
    hashed = [c for c in changed["integrity"] if c["check"].startswith("data files of the marker entry")]
    assert all(c["passed"] is False and "stream.json" in c["detail"] and "tasks.json" not in c["detail"] for c in hashed)
    assert changed["integrity_passed"] is False
    (data / "stream.json").write_bytes(original)
    assert N.report("confirm", root, project["tok"], MS)["integrity_passed"] is True
    with open(root / "results" / "r12-free-wide-levels-s0-b16" / "records.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(SystemExit) as stop:
        N.main(["--set", "confirm", "--root", str(root)])
    assert stop.value.code == 2 and "INTEGRITY FAILURE" in capsys.readouterr().err
    written = (root / "results" / "r12" / "r12_lines.md").read_text(encoding="utf-8")
    assert "records.jsonl is not the file the run recorded" in written
