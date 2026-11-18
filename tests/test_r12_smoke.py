"""Smoke checks on the pilot set (prio.r12.smoke; implementation plan Section 6, analysis plan Section 9).

One complete pilot is built once (scripted base runs, scripted generation, the real builder and
runners; no GPU); every test works on its own copy and breaks one thing."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import prio.eval.run as run_module
from prio.channel import ChannelModel
from prio.eval.run import spread
from prio.r12 import STREAM_ARMS, ledger
from prio.r12 import build as B
from prio.r12 import lines as N
from prio.r12 import smoke as K
from prio.r12 import stream as S
from prio.r12 import tasks as T
from prio.signals import CORE_SIGNALS
from prio.tasks import build_prompt
from r12_helpers import MS, R12Tok, ScriptedRuns
from test_r12_lines import rr
from test_r12_stream import Scripted

TREE, COMMIT = "t" * 40, "c" * 40
NOTE = "read 10 event runs of every arm and model: each reaction is a reaction, each notice sits after token k"


def paired_results(root: Path, model: str) -> None:
    """A paired results folder with exactly the runs an evaluation makes for the built episodes (all hits)."""
    data = root / "data" / "r12p" / ("paired" if model == "channel" else "paired-notice")
    episodes = N.episode_rows(data / "test.jsonl")
    quiet = "nominal" if model == "channel" else "quiet"
    main = [e for e in episodes if e["kind"] in ("event", "gating_event")]
    records = []
    for e in episodes:
        records.append(rr(e["episode_id"], signal=e["signal"], kind=e["kind"]))
        if e["kind"] in ("event", "gating_event"):
            records.append(rr(e["episode_id"], signal=e["signal"], kind=e["kind"], paired=quiet, occurred=False, reaction=None))
            records.append(rr(e["episode_id"], signal=e["signal"], kind=e["kind"], paired="near_miss", occurred=False, reaction=None))
        if e["kind"] == "event":
            records.append(rr(e["episode_id"], False, signal="energy" if e["signal"] != "energy" else "ctx", kind="swap", paired="swapped"))
        if model == "channel":
            records.append(rr(e["episode_id"], False, signal=e["signal"], kind=e["kind"], condition="zeroed"))
    if model == "channel":
        records += [rr(e["episode_id"], False, signal=e["signal"], kind=e["kind"], condition="base") for e in spread(main, 150)]
    out = root / N.out_folder("pilot", "paired", model)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("\n".join(json.dumps(vars(r)) for r in records) + "\n", encoding="utf-8")


def onset_folder(root: Path, name: str, tok, texts: list[str]) -> None:
    """A saved paired evaluation with one hit per text (the reference folders of the onset check)."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    records = [vars(rr(f"e{i}")) for i in range(len(texts))]
    rows = [{"episode_id": f"e{i}", "condition": "on", "paired": "event", "signal": "ctx", "stop_reason": "eos",
             "token_ids": tok.encode(text)} for i, text in enumerate(texts)]
    (folder / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (folder / "generations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot")
    tok = R12Tok()
    torch.manual_seed(0)
    tiny = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=64, n_embd=16, n_layer=1, n_head=1)).eval()
    model = ChannelModel(tiny, 6)  # the scripted generation never calls it
    sampler = ScriptedRuns(tok, every=10**9)  # every base run is kept
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(ledger, "code_tree", lambda: TREE)
        patch.setattr(ledger, "commit", lambda: COMMIT)
        patch.setattr(ledger, "tree_clean", lambda: True)
        patch.setattr(B, "_load_tokenizer", lambda model_id: (tok, MS))
        patch.setattr(B, "load_base", lambda model_id: ("the model", tok, MS))
        patch.setattr(B, "sample_batch", lambda m, t, ms, inputs, n, batch, label="": sampler(inputs))
        (root / "data" / "n6").mkdir(parents=True)
        (root / "data" / "n6" / "test.jsonl").write_text("", encoding="utf-8")
        B.build("pilot", "fake/model", root=root)
        for name in ("channel", "notice"):
            paired_results(root, name)
        tasks = T.read_tasks(root / "data" / "r12p")
        for arm in STREAM_ARMS:
            plans = S.load_plans(root / "data" / "r12p" / "stream.json", arm)
            by_id = {t.task_id: t for t in tasks[arm]}
            prompts = {p.task_id: build_prompt(tok, by_id[p.task_id]) for p in plans}
            for name, runner in (("channel", S.run_channel), ("notice", S.run_notice)):
                patch.setattr(run_module, "generate", Scripted(tok, prompts=prompts.values()))
                runner(model, tok, MS, CORE_SIGNALS, plans, prompts, 0, root / N.out_folder("pilot", arm, name), batch_size=1)
            if arm in ("free", "live"):  # the batch-1 slices
                S.run_channel(model, tok, MS, CORE_SIGNALS, spread(plans, 16), prompts, 0,
                              root / N.out_folder("pilot", arm, "channel", 1), batch_size=1, run_types=("event",))
        onset_folder(root, "results/phase1", tok, ["my context is almost full", "so . my context is almost full"])
        onset_folder(root, "results/b3-notice-none-full-b16", tok, ["my context is almost full"])
    finally:
        patch.undo()
    return {"root": root, "tok": tok}


@pytest.fixture
def project(pilot, tmp_path, monkeypatch):
    root = tmp_path / "p"
    shutil.copytree(pilot["root"], root)
    state = {"root": root, "tok": pilot["tok"], "tree": TREE, "clean": True}
    monkeypatch.setattr(ledger, "code_tree", lambda: state["tree"])
    monkeypatch.setattr(ledger, "commit", lambda: COMMIT)
    monkeypatch.setattr(ledger, "tree_clean", lambda: state["clean"])
    monkeypatch.setattr(K, "_load_tokenizer", lambda model_id: (pilot["tok"], MS))
    monkeypatch.setattr(K, "NOTICE_TOKENS", 2)  # ` [status`, `ctx=0.90]` in whitespace words
    monkeypatch.setattr(K, "ONSET_EXPECTED", {
        "results/phase1": {"hits": 2, "a_at_most_1": 2, "b_at_most_1": 1, "c_at_most_1": 1, "b_by_kind": {"event": {"0": 1, "2": 1}}},
        "results/b3-notice-none-full-b16": {"hits": 1, "b": {"0": 1}}})
    return state


def verdict(state, note=NOTE):
    return K.verdict(state["root"], state["tok"], MS, hand_read=note)


def folder(state, arm, model, batch=16) -> Path:
    return state["root"] / N.out_folder("pilot", arm, model, batch)


def edit_jsonl(path: Path, change) -> None:
    """Rewrite a .jsonl file with `change(rows)` applied (rows changed in place, or a new list returned)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = change(rows)
    rows = changed if isinstance(changed, list) else rows
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def edit_json(path: Path, change) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def failed(result) -> list[str]:
    return [name for name, check in result["checks"].items() if not check["passed"]]


# --- the whole verdict ------------------------------------------------------------------------------------

def test_smoke_passes_on_a_complete_pilot_and_opens_the_confirmatory_gate(project):
    result = verdict(project)
    assert failed(result) == [] and list(result["checks"]) == ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
    assert result["passed"] is True and (result["tree"], result["commit"], result["clean"], result["hand_read"]) == (TREE, COMMIT, True, NOTE)
    assert all(set(check) == {"what", "passed", "numbers", "failures"} and check["failures"] == [] for check in result["checks"].values())
    numbers = {name: check["numbers"] for name, check in result["checks"].items()}
    assert numbers["S1"]["pilot"] == {"paired": 96, "free": 48, "live": 48, "Ls": 24}
    assert numbers["S1"]["confirm"] == {"paired": 800, "free": 360, "live": 360, "Ls": 120} and numbers["S1"]["shared_prompts"] == 0
    assert numbers["S1"]["plans"] == {"free": 48, "live": 48, "Ls": 24}
    assert min(numbers["S2"]["kept_share"].values()) == 1.0 and set(numbers["S2"]["kinds"]) == set(K.KINDS)
    assert numbers["S3"]["channel"]["planned"] == numbers["S3"]["channel"]["produced"]
    assert set(numbers["S3"]["channel"]["planned"]) == {"on / event", "on / nominal", "on / near_miss", "on / swapped",
                                                        "zeroed / event", "base / event"}
    assert set(numbers["S3"]["notice"]["planned"]) == {"on / event", "on / quiet", "on / near_miss", "on / swapped"}
    assert (numbers["S4"]["runs"], numbers["S4"]["planned"], numbers["S4"]["rows_differ"], numbers["S4"]["crossings_off_k"]) == (144, 144, 0, 0)
    assert numbers["S4"]["twins_sharing_prefix"] == numbers["S4"]["twins"] == 48
    assert set(numbers["S5"]) == set(numbers["S7"]["slices"]) | {"Ls"} == {"free", "live", "Ls"}  # arm Ls is checked like arm live
    assert numbers["S5"]["Ls"]["runs"] == 72 and numbers["S5"]["live"]["events_with_notice"] == 48
    assert set(numbers["S6"]) == {"live", "Ls"} and numbers["S6"]["Ls"]["rows_differ"] == 0
    for arm in ("live", "Ls"):
        ends = numbers["S6"][arm]["twin_end_levels"]
        assert ends["above_89"] == 0 and ends["fed_max"] <= 89 and numbers["S6"][arm]["other_counters_highest_level"] <= 84
        assert numbers["S6"][arm]["twins_sharing_prefix"] <= numbers["S6"][arm]["twins"]  # reported, no gate
    assert (numbers["S7"]["compared"], numbers["S7"]["differing"]) == (32, 0)
    assert numbers["S9"]["words"] == len(NOTE.split())
    assert all(t["fired"] is False for t in result["triggers"]) and len(result["triggers"]) >= 8
    json.dumps(result)  # plain data

    K.main(["--hand-read", NOTE, "--root", str(project["root"])])
    saved = json.loads(ledger.smoke_file(project["root"]).read_text(encoding="utf-8"))
    assert saved["passed"] is True and saved["tree"] == TREE and set(saved) >= {"passed", "tree", "commit", "checks", "hand_read", "triggers"}
    assert b"\r" not in ledger.smoke_file(project["root"]).read_bytes()
    assert ledger.pilot_passed(project["root"])["passed"] is True  # what the confirmatory builder and runs ask


def test_smoke_needs_the_hand_read_note(project, capsys):
    for note in (None, "", "   "):
        result = verdict(project, note)
        assert failed(result) == ["S9"] and result["passed"] is False
    with pytest.raises(SystemExit) as stop:
        K.main(["--root", str(project["root"])])
    assert stop.value.code == 1 and "S9" in capsys.readouterr().out
    assert json.loads(ledger.smoke_file(project["root"]).read_text(encoding="utf-8"))["passed"] is False
    with pytest.raises(ledger.RunOnceError, match="not passed"):
        ledger.pilot_passed(project["root"])


def test_smoke_records_the_code_tree_and_needs_a_clean_one(project):
    project["tree"] = "u" * 40
    assert verdict(project)["tree"] == "u" * 40
    project["clean"] = False
    result = verdict(project)
    assert failed(result) == [] and result["clean"] is False and result["passed"] is False  # every check passes, the tree is dirty
    states = result["code_states"]  # where the pilot's runs were made: reported, never a gate
    assert states["current"] == "u" * 40 and states["data_built_on"] == TREE


def test_a_missing_folder_fails_its_check_and_nothing_else(project):
    shutil.move(str(folder(project, "Ls", "notice")), str(project["root"] / "elsewhere"))
    result = verdict(project)
    assert failed(result) == ["S5"] and any("missing" in line and "r12p-Ls-notice-none-s0-b16" in line for line in result["checks"]["S5"]["failures"])
    assert result["passed"] is False


# --- S1, S2, S3: task sets and arm paired -------------------------------------------------------------------

def test_smoke_fails_when_the_task_list_or_the_freshness_audit_is_not_the_planned_one(project):
    data = project["root"] / "data" / "r12p"
    edit_json(data / "freshness.json", lambda audit: audit.update(passed=False))
    assert failed(verdict(project)) == ["S1"]
    edit_json(data / "freshness.json", lambda audit: (audit.update(passed=True), audit["by_seed"]["tasks"].update(confirm=1520)))
    assert "S1" in failed(verdict(project))  # an audit that did not cover the confirmatory set
    edit_json(data / "freshness.json", lambda audit: audit["by_seed"]["tasks"].update(confirm=1640))
    assert failed(verdict(project)) == []
    edit_json(data / "tasks.json", lambda rows: rows.pop())
    result = verdict(project)
    assert failed(result) == ["S1"] and any("tasks.json" in line for line in result["checks"]["S1"]["failures"])


def test_smoke_fails_when_a_family_keeps_under_70_percent(project):
    manifest = project["root"] / "data" / "r12p" / "paired" / "manifest.json"
    edit_json(manifest, lambda m: m["kept_share"].update(code=0.6875, web=0.7))
    result = verdict(project)
    assert failed(result) == ["S2"] and result["checks"]["S2"]["failures"] == ["code keeps 68.8% of its base runs (under 70%)"]
    edit_json(manifest, lambda m: m["kept_share"].update(code=1.0))
    assert failed(verdict(project)) == []
    for name in ("paired", "paired-notice"):  # the kinds are read from the episode files themselves
        edit_jsonl(project["root"] / "data" / "r12p" / name / "test.jsonl", lambda rows: [r for r in rows if r["kind"] != "recovery"])
    result = verdict(project)
    assert result["checks"]["S2"]["failures"] == ["no episode of kind recovery"] and failed(result) == ["S2", "S3"]


def test_smoke_fails_when_the_notice_files_do_not_pair_or_a_notice_is_not_the_planned_text(project, monkeypatch):
    notice = project["root"] / "data" / "r12p" / "paired-notice"
    edit_jsonl(notice / "test_quiet.jsonl", lambda rows: rows.pop())
    result = verdict(project)
    assert failed(result) == ["S2"] and any("test_quiet.jsonl" in line for line in result["checks"]["S2"]["failures"])
    monkeypatch.setattr(K, "NOTICE_TOKENS", 9)  # the real tokenizer's count: these two-word notices are not it
    assert any("tokens" in line for line in verdict(project)["checks"]["S2"]["failures"])


def test_smoke_fails_when_the_paired_runs_are_not_the_planned_ones(project):
    records = folder(project, "paired", "notice") / "records.jsonl"
    edit_jsonl(records, lambda rows: rows.pop())
    result = verdict(project)
    assert failed(result) == ["S3"] and result["checks"]["S3"]["numbers"]["channel"]["planned"] == result["checks"]["S3"]["numbers"]["channel"]["produced"]
    assert any(line.startswith("notice:") for line in result["checks"]["S3"]["failures"])
    edit_jsonl(records, lambda rows: rows.append(dict(rows[0])))  # the count is right again, one run is there twice
    assert failed(verdict(project)) == ["S3"]


# --- S4, S5, S6: the arms without a forced prefix ------------------------------------------------------------

def test_smoke_fails_when_fed_rows_differ_from_recomputed_counters(project):
    def shift(arm_name, run_type):
        def change(rows):
            row = next(r for r in rows if r["run_type"] == run_type)
            name = next(iter(row["rows_milli"]))
            row["rows_milli"][name][3] += 10
        edit_jsonl(folder(project, arm_name, "channel") / "generations.jsonl", change)

    shift("live", "event")
    result = verdict(project)
    assert failed(result) == ["S6"] and result["checks"]["S6"]["numbers"]["live"]["rows_differ"] == 1
    shift("free", "twin")  # arm free: the fed column against the planned trajectory
    result = verdict(project)
    assert failed(result) == ["S4", "S6"] and result["checks"]["S4"]["numbers"]["rows_differ"] == 1


def test_smoke_fails_when_a_crossing_is_not_on_token_k(project):
    def late(rows):
        row = next(r for r in rows if r["run_type"] == "event" and r["occurred"])
        row["crossing_at"] += 1

    edit_jsonl(folder(project, "Ls", "channel") / "records.jsonl", late)
    result = verdict(project)
    assert failed(result) == ["S6"] and result["checks"]["S6"]["numbers"]["Ls"]["crossings_off_k"] == 1
    edit_jsonl(folder(project, "free", "channel") / "records.jsonl", late)
    assert failed(verdict(project)) == ["S4", "S6"]


def test_any_live_run_excluded_as_crossing_off_k_stops_the_smoke_checks_for_a_reading(project):
    def excluded(rows):  # what the judge records when the first past-limit row is on another token than the k-th
        row = next(r for r in rows if r["run_type"] == "event" and r["occurred"])
        row.update(occurred=False, why_not="crossing_off_k", hit=False, crossing_at=row["k"] + 1)

    edit_jsonl(folder(project, "live", "channel") / "records.jsonl", excluded)
    result = verdict(project)
    assert "S6" in failed(result) and result["checks"]["S6"]["numbers"]["live"]["crossings_off_k"] == 1
    assert any("crossing_off_k" in line or "not on token k" in line for line in result["checks"]["S6"]["failures"])
    edit_jsonl(folder(project, "Ls", "notice") / "records.jsonl", excluded)
    result = verdict(project)
    assert "S5" in failed(result) and any(line.startswith("Ls") and "crossing_off_k" in line for line in result["checks"]["S5"]["failures"])


def test_the_text_notice_models_saved_counters_are_not_a_check(project):
    def shift(rows):  # these rows are what the harness computed from the same tokens: comparing them again proves nothing
        row = next(r for r in rows if r["run_type"] == "event" and r["rows_milli"])
        row["rows_milli"]["ctx"][0] += 10

    edit_jsonl(folder(project, "live", "notice") / "generations.jsonl", shift)
    assert failed(verdict(project)) == []  # the notice itself is still compared with the value recomputed from the tokens


def test_twins_must_share_their_prefix_in_arm_free_only(project):
    def part(n):
        def change(rows):
            for r in [r for r in rows if r["run_type"] == "twin"][:n]:
                r["same_prefix"] = False
        return change

    edit_jsonl(folder(project, "live", "channel") / "records.jsonl", part(24))  # live twins differ from the first token: no gate
    edit_jsonl(folder(project, "free", "channel") / "records.jsonl", part(4))
    result = verdict(project)
    assert failed(result) == [] and result["checks"]["S4"]["numbers"]["twins_sharing_prefix"] == 44  # 44 / 48 = 91.7%
    edit_jsonl(folder(project, "free", "channel") / "records.jsonl", part(5))
    result = verdict(project)
    assert failed(result) == ["S4"] and result["checks"]["S4"]["numbers"]["twins_sharing_prefix"] == 43  # 89.6%: under 90%
    assert (result["checks"]["S6"]["numbers"]["live"]["twins_sharing_prefix"], result["checks"]["S6"]["numbers"]["live"]["twins"]) == (24, 48)


def test_smoke_fails_when_a_notice_is_not_what_the_harness_computed(project):
    def wrong_value(rows):
        row = next(r for r in rows if r["run_type"] == "event" and r["notice_text"])
        row["notice_text"] = row["notice_text"].replace("=0.", "=1.")

    edit_jsonl(folder(project, "live", "notice") / "generations.jsonl", wrong_value)
    result = verdict(project)
    assert failed(result) == ["S5"] and len(result["checks"]["S5"]["failures"]) >= 1

    def other_prefix(rows):
        row = next(r for r in rows if r["run_type"] == "off")
        row["pre"] = row["pre"][:-1] + [row["pre"][-1] + 1]

    edit_jsonl(folder(project, "free", "notice") / "generations.jsonl", other_prefix)
    lines = verdict(project)["checks"]["S5"]["failures"]
    assert any(line.startswith("free") and "pass 1" in line for line in lines) and any(line.startswith("live") for line in lines)


def test_smoke_fails_when_a_notice_is_given_without_a_crossing_or_withheld_with_one(project):
    def withheld(rows):
        row = next(r for r in rows if r["run_type"] == "event" and r["notice"])
        row["notice"], row["notice_text"] = None, None

    edit_jsonl(folder(project, "Ls", "notice") / "generations.jsonl", withheld)
    result = verdict(project)
    assert failed(result) == ["S5"] and any(line.startswith("Ls") and "no notice" in line for line in result["checks"]["S5"]["failures"])


def test_a_twin_above_level_89_fails_and_one_below_85_is_only_reported(project, monkeypatch):
    real = N.twin_end_levels

    def low(folder_, plans):
        ends = real(folder_, plans)
        return {**ends, "min": 84, "below_85": 1}

    monkeypatch.setattr(N, "twin_end_levels", low)
    result = verdict(project)
    assert failed(result) == [] and result["checks"]["S6"]["numbers"]["Ls"]["twin_end_levels"]["below_85"] == 1
    monkeypatch.setattr(N, "twin_end_levels", lambda folder_, plans: {**real(folder_, plans), "max": 90, "above_89": 1})
    result = verdict(project)
    assert failed(result) == ["S6"] and any("above 89" in line for line in result["checks"]["S6"]["failures"])
    # the same when the plan was right and a level above 89 was nevertheless fed to a twin
    monkeypatch.setattr(N, "twin_end_levels", lambda folder_, plans: {**real(folder_, plans), "fed_max": 90})
    result = verdict(project)
    assert failed(result) == ["S6"] and any("above 89" in line for line in result["checks"]["S6"]["failures"])


def test_another_counter_near_its_limit_fails_the_live_check(project):
    def near(rows):
        row = next(r for r in rows if r["run_type"] == "off")
        row["rows_milli"]["ctx"] = [855] * len(row["rows_milli"]["ctx"])  # the Ls arm never targets ctx

    edit_jsonl(folder(project, "Ls", "channel") / "generations.jsonl", near)
    result = verdict(project)
    assert failed(result) == ["S6"] and result["checks"]["S6"]["numbers"]["Ls"]["other_counters_highest_level"] == 85


# --- S7, S8 ---------------------------------------------------------------------------------------------------

def test_batch_agreement_allows_one_differing_run_of_32(project):
    def flip(n):
        def change(rows):
            for row in rows[:n]:
                row["hit"] = not row["hit"]
        return change

    edit_jsonl(folder(project, "free", "channel", 1) / "records.jsonl", flip(1))
    result = verdict(project)
    assert failed(result) == [] and (result["checks"]["S7"]["numbers"]["compared"], result["checks"]["S7"]["numbers"]["differing"]) == (32, 1)
    edit_jsonl(folder(project, "live", "channel", 1) / "records.jsonl", flip(1))
    result = verdict(project)
    assert failed(result) == ["S7"] and result["checks"]["S7"]["numbers"]["differing"] == 2
    def back_and_one_less(rows):
        rows[0]["hit"] = not rows[0]["hit"]
        rows.pop()

    edit_jsonl(folder(project, "live", "channel", 1) / "records.jsonl", back_and_one_less)
    result = verdict(project)  # one differing run again, but a slice of 15 is not the planned one
    assert failed(result) == ["S7"] and (result["checks"]["S7"]["numbers"]["compared"], result["checks"]["S7"]["numbers"]["differing"]) == (31, 1)


def test_onset_check_compares_the_saved_evaluations_with_the_plans_numbers(project, monkeypatch):
    assert verdict(project)["checks"]["S8"]["numbers"]["results/phase1"]["b_at_most_1"] == 1
    expected = json.loads(json.dumps(K.ONSET_EXPECTED))
    expected["results/phase1"]["b_at_most_1"] = 2
    monkeypatch.setattr(K, "ONSET_EXPECTED", expected)
    result = verdict(project)
    assert failed(result) == ["S8"] and "b_at_most_1" in result["checks"]["S8"]["failures"][0]


def test_onset_numbers_of_the_frozen_plan():
    import prio.r12.smoke as fresh  # the constants as shipped (reading 13 of the analysis plan)

    assert fresh.ONSET_EXPECTED == {
        "results/phase1": {"hits": 550, "a_at_most_1": 550, "b_at_most_1": 549, "c_at_most_1": 544,
                           "b_by_kind": {"event": {"0": 320, "1": 5, "3": 1}, "gating_event": {"0": 224}}},
        "results/b3-notice-none-full-b16": {"hits": 557, "b": {"0": 557}}}
    assert fresh.NOTICE_TOKENS == 9


# --- triggers: computed and stored, never a gate ----------------------------------------------------------------

def test_triggers_start_a_bug_hunt_and_never_gate(project):
    def miss_most(rows):
        for row in [r for r in rows if r["condition"] == "on" and r["paired"] == "event" and r["kind"] == "event"][5:]:
            row["reaction_signal"] = None

    def alarms(rows):
        for row in [r for r in rows if r["paired"] == "nominal" and r["kind"] == "event"][:20]:
            row["reaction_signal"] = "energy"

    def off_reacts(rows):
        for row in [r for r in rows if r["condition"] == "zeroed" and r["kind"] == "event"]:
            row["reaction_signal"] = row["signal"]

    for change in (miss_most, alarms, off_reacts):
        edit_jsonl(folder(project, "paired", "channel") / "records.jsonl", change)

    def few_crossings(rows):
        for row in [r for r in rows if r["run_type"] == "event"][:20]:
            row.update(occurred=False, why_not="ended", hit=False, crossing_at=None)

    edit_jsonl(folder(project, "free", "notice") / "records.jsonl", few_crossings)

    def no_think(rows):
        for row in [r for r in rows if r["run_type"] == "event" and r["signal"] == "think"][:8]:
            row.update(occurred=False, why_not="thinking_over", hit=False, crossing_at=None)

    edit_jsonl(folder(project, "live", "channel") / "records.jsonl", no_think)
    result = K.verdict(project["root"], project["tok"], MS, hand_read=NOTE)
    fired = {t["trigger"] for t in result["triggers"] if t["fired"]}
    assert fired == {"arm paired, channel model: event reaction under 70%", "arm paired, channel model: quiet-twin alarms over 10%",
                     "arm paired, channel model: reaction with the channel off over 10%",
                     "arm free, text-notice model: crossings that occur under 75% of the planned",
                     "arm live, channel model: think crossings that occur under 60%"}
    # The triggers gate nothing. (The records edited here no longer agree with the saved notices and with
    # the batch-1 slice, which S5 and S7 notice; the paired arm, where only outcomes changed, still passes.)
    assert set(failed(result)) <= {"S5", "S7"} and result["checks"]["S3"]["passed"] and result["checks"]["S6"]["passed"]
    assert all(set(t) == {"trigger", "fired", "detail"} for t in result["triggers"])
    assert K.triggers(project["root"]) == result["triggers"]
    detail = {t["trigger"]: t["detail"] for t in result["triggers"]}
    assert detail["arm free, text-notice model: crossings that occur under 75% of the planned"] == "28 / 48 = 58.3%"
    assert detail["arm live, channel model: think crossings that occur under 60%"] == "8 / 16 = 50.0%"
    shutil.move(str(folder(project, "paired", "notice")), str(project["root"] / "elsewhere"))  # no records: not computed, not fired
    missing = [t for t in K.triggers(project["root"]) if t["trigger"].startswith("arm paired, text-notice")]
    assert len(missing) == 2 and all(t["fired"] is None and "not computed" in t["detail"] for t in missing)
