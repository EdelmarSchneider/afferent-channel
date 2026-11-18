"""The R12 builder (prio.r12.build): arm P data from scripted base runs (no GPU), the notice files of
the text-notice model, the plans of the arms without a forced prefix, the gates of the confirmatory set."""

import json
import re
from collections import Counter

import numpy as np
import pytest

from prio import r12
from prio.datagen.episodes import read_jsonl
from prio.r12 import build as B
from prio.r12 import ledger
from prio.r12 import tasks as R
from prio.signals import CORE_SIGNALS
from prio.tasks import build_prompt
from r12_helpers import MS, R12Tok, ScriptedRuns

MAIN = ("event", "gating_event")


@pytest.fixture(scope="module")
def pilot():
    return R.task_sets()["pilot"]


@pytest.fixture
def tok():
    return R12Tok()


def paired(tmp_path, tok, tasks, sampler=None, name="paired"):
    sampler = sampler or ScriptedRuns(tok)
    return B.build_paired(tok, MS, sampler, tasks, tmp_path / name, "pilot", "fake/model"), sampler


def test_paired_build_gives_one_main_episode_per_kept_run_and_counts_the_dropped(tmp_path, tok, pilot):
    tasks = pilot["paired"]
    manifest, sampler = paired(tmp_path, tok, tasks)
    episodes = read_jsonl(tmp_path / "paired" / "test.jsonl")
    style = {t.task_id: sampler.styles[tuple(build_prompt(tok, t))] for t in tasks}
    family = {t.task_id: t.family for t in tasks}

    main = [ep for ep in episodes if ep.kind in MAIN]
    assert sorted(ep.task_id for ep in main) == sorted(t for t, s in style.items() if s == "ok")  # one each, no more
    assert len({ep.episode_id for ep in episodes}) == len(episodes)
    assert all(re.fullmatch(r"r12p-P-\d{5}", ep.episode_id) for ep in episodes)
    assert all(ep.split == "test" and ep.family == family[ep.task_id] for ep in episodes)
    assert all(ep.signals.shape == (len(ep.prompt_ids) + len(ep.target_ids), 6) for ep in episodes)

    requested = Counter(family.values())
    assert manifest["tasks_requested"] == dict(requested) == {"math": 16, "explain": 16, "web": 16, "heavy": 16, "code": 32}
    reasons = {"unfinished": "not_ended", "reaction_tool": "reaction_tool", "no_position": "no_position", "too_long": "too_long"}
    for name in r12.FAMILIES:
        expected = Counter(reasons[s] for t, s in style.items() if family[t] == name and s != "ok")
        assert manifest["dropped"][name] == {r: expected.get(r, 0) for r in ("not_ended", "reaction_tool", "no_position", "too_long")}
        kept = sum(1 for t, s in style.items() if family[t] == name and s == "ok")
        assert manifest["main_episodes"][name] == kept == requested[name] - sum(manifest["dropped"][name].values())
        assert manifest["kept_share"][name] == kept / requested[name]
    assert sum(sum(d.values()) for d in manifest["dropped"].values()) > 0  # the script did drop some
    assert manifest["under_70_percent"] == [f for f in r12.FAMILIES if manifest["kept_share"][f] < 0.7]
    assert manifest["episodes"] == len(episodes) and manifest["kinds"] == dict(Counter(ep.kind for ep in episodes))
    assert (manifest["set"], manifest["model"], manifest["n_signals"], manifest["seed"]) == ("pilot", "fake/model", 6, 12)
    assert manifest["base_run_seed"] == 1200 and manifest["episode_generator"] == [12, 0, 1]
    assert manifest["sha256"]["test.jsonl"] == ledger.sha256(tmp_path / "paired" / "test.jsonl")
    assert manifest["sha256"]["base_runs.json"] == ledger.sha256(tmp_path / "paired" / "base_runs.json")
    assert json.loads((tmp_path / "paired" / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_code_family_gets_interrupt_events_only_and_tool_families_get_gating_events(tmp_path, tok, pilot):
    paired(tmp_path, tok, pilot["paired"], ScriptedRuns(tok, every=10**9))  # every base run is kept
    episodes = read_jsonl(tmp_path / "paired" / "test.jsonl")
    by_family = {}
    for ep in episodes:
        by_family.setdefault(ep.family, Counter())[ep.kind] += 1
    for name in ("math", "explain", "code"):
        assert set(by_family[name]) <= {"event", "starts_past", "gating_unused"} and by_family[name]["event"] > 0
    for name in ("web", "heavy"):
        assert set(by_family[name]) <= {"gating_event", "recovery"} and by_family[name]["gating_event"] > 0
    assert {ep.signal for ep in episodes if ep.family == "web"} == {"net"}
    assert {ep.signal for ep in episodes if ep.family == "heavy"} == {"ram"}
    events = [ep for ep in episodes if ep.kind == "event"]
    assert {ep.signal for ep in events} == {"ctx", "think", "energy", "time"}
    assert [ep.signal for ep in events[:8]] == ["ctx", "think", "energy", "time"] * 2  # in rotation over the runs
    assert all(ep.phase == "thinking" for ep in events if ep.signal == "think")
    assert {"starts_past", "gating_unused", "recovery"} <= {ep.kind for ep in episodes}
    names = [s.name for s in CORE_SIGNALS]
    for ep in events:  # the crossing token is the first that carries the past-limit value
        column, spec = names.index(ep.signal), CORE_SIGNALS[names.index(ep.signal)]
        row = len(ep.prompt_ids) + ep.t_cross
        assert spec.past_limit(float(ep.signals[row, column])) and not spec.past_limit(float(ep.signals[row - 1, column]))


def test_build_resumes_from_the_base_run_cache_and_writes_the_same_bytes(tmp_path, tok, pilot):
    manifest, sampler = paired(tmp_path, tok, pilot["paired"])
    assert sampler.calls == 1
    before = {name: (tmp_path / "paired" / name).read_bytes() for name in ("test.jsonl", "manifest.json", "base_runs.json")}

    def refuse(prompts):
        raise AssertionError("the cache holds every base run")

    again, _ = paired(tmp_path, tok, pilot["paired"], refuse)
    assert again == manifest
    assert {name: (tmp_path / "paired" / name).read_bytes() for name in before} == before
    # the generator of the episodes is [12, set code, 1]: another set code gives other positions
    B.build_paired(tok, MS, ScriptedRuns(tok), pilot["paired"], tmp_path / "other", "confirm", "fake/model")
    assert (tmp_path / "other" / "test.jsonl").read_bytes() != before["test.jsonl"]
    assert read_jsonl(tmp_path / "other" / "test.jsonl")[0].episode_id == "r12c-P-00000"


def test_notice_folder_holds_four_test_files_with_the_same_main_episode_ids(tmp_path, tok, pilot):
    paired(tmp_path, tok, pilot["paired"])
    manifest = B.build_notice(tmp_path / "paired", tmp_path / "paired-notice", tok, MS)
    folder = tmp_path / "paired-notice"
    source = read_jsonl(tmp_path / "paired" / "test.jsonl")
    files = {name: read_jsonl(folder / name) for name in ("test.jsonl", "test_quiet.jsonl", "test_near_miss.jsonl", "test_swap.jsonl")}
    main_ids = [ep.episode_id for ep in source if ep.kind in MAIN]
    assert [ep.episode_id for ep in files["test.jsonl"]] == [ep.episode_id for ep in source]  # every kind, same order
    assert [ep.episode_id for ep in files["test_quiet.jsonl"]] == main_ids
    assert [ep.episode_id for ep in files["test_near_miss.jsonl"]] == main_ids
    assert [ep.episode_id for ep in files["test_swap.jsonl"]] == [ep.episode_id for ep in source if ep.kind == "event"]
    for name in ("test.jsonl", "test_near_miss.jsonl", "test_swap.jsonl"):
        for ep in files[name]:
            spans = ep.harness_spans
            assert spans and len(ep.signals) == len(ep.prompt_ids) + len(ep.target_ids)
            for a, b in spans:
                assert re.fullmatch(r" \[status (ctx|think|energy|time|net|ram)=\d\.\d\d\]", tok.decode(ep.target_ids[a:b]))
            assert ep.t_cross == spans[0][1] - 1  # the forced prefix ends with the notice
    assert all(not ep.harness_spans for ep in files["test_quiet.jsonl"])
    by_id = {ep.episode_id: ep for ep in source}
    names = [s.name for s in CORE_SIGNALS]
    for ep in files["test.jsonl"]:  # the event notice shows a value past the limit, the near-miss one a value under it
        if ep.kind in MAIN:
            shown = float(ep.meta["variant"]["insertions"][0]["shown"])
            assert CORE_SIGNALS[names.index(ep.signal)].past_limit(shown)
            assert ep.target_ids[:by_id[ep.episode_id].t_cross + 1] == by_id[ep.episode_id].target_ids[:by_id[ep.episode_id].t_cross + 1]
    for ep in files["test_near_miss.jsonl"]:
        shown = float(ep.meta["variant"]["insertions"][0]["shown"])
        assert not CORE_SIGNALS[names.index(ep.signal)].past_limit(shown)
    assert manifest["variant"] == "notice" and set(manifest["test_only"]) == {"swap", "near_miss", "quiet"}
    assert set(manifest["sha256"]) == set(files) and manifest["sha256"]["test.jsonl"] == ledger.sha256(folder / "test.jsonl")
    assert json.loads((folder / "manifest.json").read_text(encoding="utf-8")) == manifest
    again = B.build_notice(tmp_path / "paired", tmp_path / "paired-notice", tok, MS)  # the same bytes again
    assert again == manifest


def test_stream_plans_balance_signals_and_keep_k_in_range():
    sets = R.task_sets()
    confirm = sets["confirm"]
    plans = B.stream_plans(confirm["free"], confirm["live"], confirm["Ls"], "confirm")
    assert plans == B.stream_plans(confirm["free"], confirm["live"], confirm["Ls"], "confirm")
    assert Counter(p["arm"] for p in plans) == {"free": 360, "live": 360, "Ls": 120}
    assert list(plans[0]) == ["arm", "index", "task_id", "family", "signal", "k", "horizon"]
    for arm_name, tasks in (("free", confirm["free"]), ("live", confirm["live"]), ("Ls", confirm["Ls"])):
        mine = [p for p in plans if p["arm"] == arm_name]
        assert [p["index"] for p in mine] == list(range(len(tasks)))
        assert [(p["task_id"], p["family"]) for p in mine] == [(t.task_id, t.family) for t in tasks]
    per = Counter((p["arm"], p["family"], p["signal"]) for p in plans)
    for family in ("math", "explain", "code"):
        assert [per[("free", family, s)] for s in ("ctx", "think", "energy", "time")] == [30] * 4
        assert [per[("live", family, s)] for s in ("ctx", "think", "time")] == [40] * 3
        assert [per[("Ls", family, s)] for s in ("think", "time")] == [20] * 2
    assert len(per) == 3 * (4 + 3 + 2)
    for p in plans:
        lo, hi = (16, 96) if p["arm"] == "free" else (32, 160)
        assert lo <= p["k"] <= hi and isinstance(p["k"], int)
        assert p["horizon"] == (160 if p["signal"] == "time" else 96)
        code = {"free": 2, "live": 3, "Ls": 4}[p["arm"]]
        if p["index"] % 40 == 0:  # the generators of the plan
            draw = np.random.default_rng([12, 1, code, p["index"]]).integers(16 if p["arm"] == "free" else 32,
                                                                           97 if p["arm"] == "free" else 161)
            assert p["k"] == int(draw)
    free_k = [p["k"] for p in plans if p["arm"] == "free"]
    assert min(free_k) == 16 and max(free_k) == 96  # both ends are drawn
    assert [p["k"] for p in plans if p["arm"] == "free"][:6] == [45, 78, 54, 96, 58, 35]  # pinned
    assert [p["k"] for p in plans if p["arm"] == "live"][:6] == [144, 135, 111, 154, 42, 72]  # pinned
    assert [p["k"] for p in plans if p["arm"] == "Ls"][:6] == [57, 151, 125, 35, 105, 33]  # pinned

    pilot = sets["pilot"]
    small = B.stream_plans(pilot["free"], pilot["live"], pilot["Ls"], "pilot")
    assert Counter(p["arm"] for p in small) == {"free": 48, "live": 48, "Ls": 24}
    assert [p["k"] for p in small][:3] != [p["k"] for p in plans][:3]  # the set code is in the generator
    assert all(p["task_id"].startswith("r12p-") for p in small)


# --- the whole build and its gates ---------------------------------------------------------------

@pytest.fixture
def project(tmp_path, monkeypatch, tok):
    """A project root with an empty data folder; git and the model loaders are stand-ins."""
    state = {"tree": "t" * 40, "clean": True, "loads": [], "sampler": ScriptedRuns(tok), "root": tmp_path}
    (tmp_path / "data" / "n6").mkdir(parents=True)
    (tmp_path / "data" / "n6" / "test.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(ledger, "code_tree", lambda: state["tree"])
    monkeypatch.setattr(ledger, "commit", lambda: "c" * 40)
    monkeypatch.setattr(ledger, "tree_clean", lambda: state["clean"])
    monkeypatch.setattr(B, "_load_tokenizer", lambda model: (tok, MS))
    monkeypatch.setattr(B, "load_base", lambda model: state["loads"].append(model) or ("the model", tok, MS))

    def sample_batch(model, tok_, ms, inputs, max_new_tokens, batch_size, label=""):
        state["seen"] = (model, max_new_tokens, batch_size)
        return state["sampler"](inputs)

    monkeypatch.setattr(B, "sample_batch", sample_batch)
    return state


def smoke(state, passed=True, tree=None):
    path = state["root"] / "results" / "r12p" / "smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"passed": passed, "tree": tree or state["tree"]}), encoding="utf-8")


def test_main_builds_the_pilot_set_and_loads_the_model_only_when_a_base_run_is_missing(project, capsys):
    root = project["root"]
    B.main(["--set", "pilot", "--model", "fake/model", "--batch-size", "8", "--root", str(root)])
    out = root / "data" / "r12p"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == manifest
    assert project["loads"] == ["fake/model"] and project["seen"] == ("the model", 1024, 8)
    for name in ("tasks.json", "freshness.json", "stream.json", "manifest.json", "paired/test.jsonl", "paired/base_runs.json",
                 "paired/manifest.json", "paired-notice/test.jsonl", "paired-notice/test_quiet.jsonl",
                 "paired-notice/test_near_miss.jsonl", "paired-notice/test_swap.jsonl", "paired-notice/manifest.json"):
        assert (out / name).exists(), name
    assert (manifest["set"], manifest["model"], manifest["tree"], manifest["clean"]) == ("pilot", "fake/model", "t" * 40, True)
    assert manifest["tasks"] == {"paired": 96, "free": 48, "live": 48, "Ls": 24} and manifest["freshness_passed"] is True
    assert manifest["plans"] == {"free": 48, "live": 48, "Ls": 24}
    assert manifest["paired"]["episodes"] == len(read_jsonl(out / "paired" / "test.jsonl"))
    for name in ("tasks.json", "stream.json", "freshness.json"):
        assert manifest["sha256"][name] == ledger.sha256(out / name)
        assert b"\r" not in (out / name).read_bytes()
    plans = json.loads((out / "stream.json").read_text(encoding="utf-8"))
    assert len(plans) == 120 and plans[0]["task_id"] == R.task_sets()["pilot"]["free"][0].task_id
    assert R.read_tasks(out) == R.task_sets()["pilot"]

    before = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    B.main(["--set", "pilot", "--model", "fake/model", "--root", str(root)])  # the pilot may be built again: same bytes, no model
    assert project["loads"] == ["fake/model"]
    assert {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"} == before


def test_confirm_build_refuses_without_a_passed_pilot_on_the_same_tree(project):
    root = project["root"]
    argv = ["--set", "confirm", "--model", "fake/model", "--root", str(root)]
    for prepare, word in ((lambda: None, "smoke"), (lambda: smoke(project, passed=False), "not passed"),
                          (lambda: smoke(project, tree="x" * 40), "another")):
        prepare()
        with pytest.raises(ledger.RunOnceError, match=word):
            B.build("confirm", "fake/model", root=root)
        with pytest.raises(SystemExit) as stop:
            B.main(argv)
        assert stop.value.code == 3
    smoke(project)
    project["clean"] = False
    with pytest.raises(ledger.RunOnceError, match="uncommitted"):
        B.build("confirm", "fake/model", root=root)
    assert not (root / "data" / "r12c").exists() and project["loads"] == []  # nothing written, nothing loaded
    project["clean"] = True
    manifest = B.build("confirm", "fake/model", root=root)
    assert manifest["tasks"] == {"paired": 800, "free": 360, "live": 360, "Ls": 120}
    assert manifest["paired"]["base_run_seed"] == 1201 and (root / "data" / "r12c" / "manifest.json").exists()


def test_confirm_build_refuses_to_rebuild(project):
    root = project["root"]
    smoke(project)
    B.build("confirm", "fake/model", root=root)
    before = (root / "data" / "r12c" / "paired" / "test.jsonl").read_bytes()
    with pytest.raises(ledger.RunOnceError, match="never rebuilt"):
        B.build("confirm", "fake/model", root=root)
    assert (root / "data" / "r12c" / "paired" / "test.jsonl").read_bytes() == before
    # an interrupted build has no manifest yet: it goes on from the base-run cache
    (root / "data" / "r12c" / "manifest.json").rename(root / "data" / "r12c" / "manifest.kept")
    loads = len(project["loads"])
    B.build("confirm", "fake/model", root=root)
    assert len(project["loads"]) == loads and (root / "data" / "r12c" / "paired" / "test.jsonl").read_bytes() == before


def test_build_stops_when_the_freshness_audit_finds_a_match(project, tok):
    from helpers import make_episode
    from prio.datagen.episodes import write_jsonl

    root = project["root"]
    planted = build_prompt(tok, R.task_sets()["confirm"]["Ls"][3])  # a confirmatory task, found while the pilot is built
    write_jsonl(root / "data" / "n6" / "test.jsonl", [make_episode(prompt_ids=planted)])
    with pytest.raises(B.FreshnessError, match="r12c-"):
        B.build("pilot", "fake/model", root=root)
    out = root / "data" / "r12p"
    assert json.loads((out / "freshness.json").read_text(encoding="utf-8"))["passed"] is False
    assert not (out / "paired").exists() and not (out / "manifest.json").exists() and project["loads"] == []
    with pytest.raises(SystemExit) as stop:
        B.main(["--set", "pilot", "--model", "fake/model", "--root", str(root)])
    assert stop.value.code == 4
