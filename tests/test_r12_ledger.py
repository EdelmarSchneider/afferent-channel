"""The run-once marker file of R12 (prio.r12.ledger; analysis plan Section 9, "Run once, checkable")."""

import hashlib
import json
import re

import pytest

from prio.r12 import ledger as G

TREE, COMMIT = "t" * 40, "c" * 40


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A project folder with a clean source tree, a passed pilot on that tree and two data files."""
    state = {"tree": TREE, "commit": COMMIT, "clean": True}
    monkeypatch.setattr(G, "code_tree", lambda: state["tree"])
    monkeypatch.setattr(G, "commit", lambda: state["commit"])
    monkeypatch.setattr(G, "tree_clean", lambda: state["clean"])
    (tmp_path / "results" / "r12p").mkdir(parents=True)
    (tmp_path / "data" / "r12c").mkdir(parents=True)
    (tmp_path / "data" / "r12c" / "tasks.json").write_bytes(b"[1, 2]\n")
    (tmp_path / "data" / "r12c" / "stream.json").write_bytes(b"[3]\n")
    state["root"] = tmp_path
    state["data"] = [tmp_path / "data" / "r12c" / "tasks.json", tmp_path / "data" / "r12c" / "stream.json"]
    state["smoke"] = tmp_path / "results" / "r12p" / "smoke.json"
    state["ledger"] = tmp_path / "results" / "r12" / "RUN_ONCE.json"
    state["out"] = tmp_path / "results" / "r12-free-wide-levels-s0-b16"
    write_smoke(state)
    return state


def write_smoke(state, passed=True, tree=None):
    state["smoke"].write_text(json.dumps({"passed": passed, "tree": tree or state["tree"]}), encoding="utf-8")


def begin(state, arm="free", model="channel", set_name="confirm", **kw):
    return G.begin(set_name, arm, model, kw.pop("out", state["out"]), kw.pop("data", state["data"]),
                   kw.pop("argv", ["--set", set_name, "--arm", arm, "--model", model]), root=state["root"], **kw)


def stored(state, set_name="confirm"):
    return json.loads(G.marker_file(set_name, state["root"]).read_text(encoding="utf-8"))["entries"]


def test_marker_files_of_the_two_sets():
    from pathlib import Path

    assert G.marker_file("confirm") == Path("results/r12/RUN_ONCE.json")
    assert G.marker_file("pilot") == Path("results/r12p/RUN_ONCE.json")
    assert G.smoke_file() == Path("results/r12p/smoke.json")


def test_first_confirmatory_run_writes_commit_tree_and_data_hashes(repo):
    entry = begin(repo)
    assert entry["resumed"] is False
    saved = stored(repo)
    assert len(saved) == 1 and saved[0] == {k: v for k, v in entry.items() if k != "resumed"}
    e = saved[0]
    assert (e["set"], e["arm"], e["model"], e["kind"], e["rerun"]) == ("confirm", "free", "channel", "run", 0)
    assert e["commit"] == COMMIT and e["tree"] == TREE and e["batch_size"] == 16
    assert e["argv"] == ["--set", "confirm", "--arm", "free", "--model", "channel"]
    assert e["out"] == "results/r12-free-wide-levels-s0-b16"  # relative to the project root, forward slashes
    assert e["data"] == {"data/r12c/tasks.json": hashlib.sha256(b"[1, 2]\n").hexdigest(),
                         "data/r12c/stream.json": hashlib.sha256(b"[3]\n").hexdigest()}
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", e["started"])
    assert e["finished"] is None and e["records_sha256"] is None and e["resumes"] == [] and e["rerun_reason"] is None
    raw = repo["ledger"].read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")


def test_second_run_of_the_same_arm_and_model_is_refused(repo):
    first = begin(repo)
    G.finish(first, _records(repo), root=repo["root"])
    before = repo["ledger"].read_bytes()
    with pytest.raises(G.RunOnceError, match="already"):
        begin(repo)
    assert repo["ledger"].read_bytes() == before  # nothing written
    # another arm or another model is another entry
    begin(repo, arm="live", out=repo["root"] / "results" / "r12-live-wide-levels-s0-b16")
    begin(repo, model="notice", out=repo["root"] / "results" / "r12-free-notice-none-s0-b16")
    assert [(e["arm"], e["model"]) for e in stored(repo)] == [("free", "channel"), ("live", "channel"), ("free", "notice")]


def _records(state, text=b'{"a": 1}\n', out=None):
    folder = out or state["out"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "records.jsonl").write_bytes(text)
    return folder / "records.jsonl"


def test_finish_records_the_hash_of_the_records_file(repo):
    entry = begin(repo)
    G.finish(entry, _records(repo), root=repo["root"])
    e = stored(repo)[0]
    assert e["records_sha256"] == hashlib.sha256(b'{"a": 1}\n').hexdigest() == G.sha256(repo["out"] / "records.jsonl")
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", e["finished"])
    with pytest.raises(G.RunOnceError):
        G.finish({**entry, "started": "1999-01-01 00:00:00"}, _records(repo), root=repo["root"])  # no such entry


def test_repairs_of_torn_files_are_written_into_the_entry(repo):
    entry = begin(repo)
    assert stored(repo)[0]["repairs"] == []
    first = {"file": "results/r12-free-wide-levels-s0-b16/generations.jsonl", "bytes": 271, "time": "2026-11-17 11:00:00",
             "what": "torn last line"}
    G.add_repairs(entry, [first], root=repo["root"])
    again = begin(repo)  # the resume that found the torn line
    second = {**first, "file": "results/r12-free-wide-levels-s0-b16/records.partial.jsonl", "bytes": 12}
    G.add_repairs(again, [second], root=repo["root"])
    G.add_repairs(again, [], root=repo["root"])  # nothing to add: the file is left as it is
    e = stored(repo)
    assert len(e) == 1 and e[0]["repairs"] == [first, second] and len(e[0]["resumes"]) == 1
    G.finish(again, _records(repo), root=repo["root"])
    assert stored(repo)[0]["repairs"] == [first, second]
    with pytest.raises(G.RunOnceError):
        G.add_repairs({**entry, "started": "1999-01-01 00:00:00"}, [first], root=repo["root"])


def test_unfinished_run_resumes_only_with_the_same_tree_and_data(repo):
    first = begin(repo)
    again = begin(repo)  # crash or power cut: the entry was never finished
    assert again["resumed"] is True and again["started"] == first["started"] and again["out"] == first["out"]
    e = stored(repo)
    assert len(e) == 1 and len(e[0]["resumes"]) == 1 and re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", e[0]["resumes"][0])
    begin(repo)
    assert len(stored(repo)[0]["resumes"]) == 2

    repo["data"][1].write_bytes(b"[4]\n")  # a data file changed
    before = repo["ledger"].read_bytes()
    with pytest.raises(G.RunOnceError, match="data"):
        begin(repo)
    repo["data"][1].write_bytes(b"[3]\n")
    repo["tree"] = "u" * 40  # the code changed (and the pilot passed again on the new tree)
    write_smoke(repo)
    with pytest.raises(G.RunOnceError, match="tree"):
        begin(repo)
    assert repo["ledger"].read_bytes() == before
    repo["tree"] = TREE
    write_smoke(repo)
    G.finish(begin(repo), _records(repo), root=repo["root"])
    with pytest.raises(G.RunOnceError, match="already"):
        begin(repo)  # finished: no resume


def test_rerun_needs_a_reason_and_gets_a_new_folder(repo):
    with pytest.raises(G.RunOnceError, match="no first run"):
        begin(repo, rerun_reason="nothing to run again")
    first = begin(repo)
    G.finish(first, _records(repo), root=repo["root"])
    with pytest.raises(G.RunOnceError):
        begin(repo)
    with pytest.raises(G.RunOnceError, match="reason"):
        begin(repo, rerun_reason="  ")
    rerun = begin(repo, rerun_reason="defect D-999: the notice was spliced one token late")
    assert rerun["kind"] == "rerun" and rerun["rerun"] == 1 and rerun["resumed"] is False
    assert rerun["out"] == "results/r12-free-wide-levels-s0-b16-rerun1"
    assert rerun["rerun_reason"] == "defect D-999: the notice was spliced one token late"
    saved = stored(repo)
    assert [e["kind"] for e in saved] == ["run", "rerun"] and saved[0]["records_sha256"] is not None  # the first run untouched
    # an unfinished rerun resumes like any unfinished entry; a new reason starts another one
    assert begin(repo)["resumed"] is True and len(stored(repo)) == 2
    G.finish(rerun, _records(repo, out=repo["root"] / rerun["out"]), root=repo["root"])
    second = begin(repo, rerun_reason="second look")
    assert second["out"].endswith("-rerun2") and second["rerun"] == 2 and len(stored(repo)) == 3
    assert G.first_run(stored(repo), "free", "channel")["kind"] == "run"
    assert [e["rerun"] for e in G.reruns(stored(repo), "free", "channel")] == [1, 2]


def test_a_rerun_never_goes_into_a_folder_that_exists(repo):
    G.finish(begin(repo), _records(repo), root=repo["root"])
    (repo["root"] / "results" / "r12-free-wide-levels-s0-b16-rerun1").mkdir(parents=True)
    with pytest.raises(G.RunOnceError, match="exists"):
        begin(repo, rerun_reason="a reason")
    assert len(stored(repo)) == 1


def test_a_first_run_never_adopts_records_that_no_entry_explains(repo):
    _records(repo)  # records in the folder, no entry in the marker file
    with pytest.raises(G.RunOnceError, match="holds records"):
        begin(repo)
    assert not repo["ledger"].exists()


def test_dirty_source_tree_is_refused(repo):
    repo["clean"] = False
    with pytest.raises(G.RunOnceError, match="uncommitted"):
        begin(repo)
    assert not repo["ledger"].exists()


def test_confirm_needs_a_passed_pilot_on_the_same_tree(repo):
    repo["smoke"].unlink()
    with pytest.raises(G.RunOnceError, match="smoke"):
        begin(repo)
    write_smoke(repo, passed=False)
    with pytest.raises(G.RunOnceError, match="not passed"):
        begin(repo)
    write_smoke(repo, tree="x" * 40)
    with pytest.raises(G.RunOnceError, match="another"):
        begin(repo)
    assert not repo["ledger"].exists()
    write_smoke(repo)
    G.pilot_passed(repo["root"])  # the builder's gate: the same check
    begin(repo)


def test_pilot_runs_are_recorded_and_never_refused(repo):
    repo["clean"] = False
    repo["smoke"].unlink()
    out = repo["root"] / "results" / "r12p-free-wide-levels-s0-b16"
    first = begin(repo, set_name="pilot", out=out)
    G.finish(first, _records(repo, out=out), root=repo["root"])
    second = begin(repo, set_name="pilot", out=out, batch_size=1)
    saved = stored(repo, "pilot")
    assert len(saved) == 2 and [e["set"] for e in saved] == ["pilot", "pilot"] and saved[1]["batch_size"] == 1
    assert saved[0]["finished"] and saved[1]["finished"] is None and saved[0]["clean"] is False
    assert second["out"] == first["out"] and not repo["ledger"].exists()  # the confirmatory file is not touched
    with pytest.raises(G.RunOnceError, match="rerun"):
        begin(repo, set_name="pilot", out=out, rerun_reason="pilot runs are simply run again")


def test_a_missing_data_file_is_an_error_before_anything_is_written(repo):
    with pytest.raises(FileNotFoundError):
        begin(repo, data=[repo["root"] / "data" / "r12c" / "missing.jsonl"])
    assert not repo["ledger"].exists()


def test_sha256_reads_the_file_in_pieces(tmp_path):
    blob = bytes(range(256)) * 9000  # 2.3 MB: several pieces
    (tmp_path / "blob").write_bytes(blob)
    assert G.sha256(tmp_path / "blob") == hashlib.sha256(blob).hexdigest()


def test_git_is_read_only_and_tells_the_tree_of_the_source():
    # The real repository, read only: the three commands the ledger is allowed to run.
    assert re.fullmatch(r"[0-9a-f]{40}", G.code_tree()) and re.fullmatch(r"[0-9a-f]{40}", G.commit())
    assert G.tree_clean() in (True, False)
    assert G.GIT_COMMANDS == (("rev-parse", "HEAD:src/prio"), ("rev-parse", "HEAD"),
                              ("--no-optional-locks", "status", "--porcelain", "--", "src/prio"))
