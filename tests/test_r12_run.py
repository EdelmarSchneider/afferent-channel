"""The run command of R12 (prio.r12.run): one evaluation per (set, arm, model), started through the
run-once marker file."""

import json
from pathlib import Path

import pytest

from prio import r12
from prio.r12 import build as B
from prio.r12 import ledger
from prio.r12 import run as R
from prio.r12 import stream as S
from prio.r12 import tasks as T
from prio.tasks import build_prompt
from r12_helpers import MS, R12Tok

NOTICE_FILES = ("test.jsonl", "test_quiet.jsonl", "test_near_miss.jsonl", "test_swap.jsonl")


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project root with both sets' task lists and plans, stand-in data files, the two runs'
    logs, a passed pilot; git, the model loader and the evaluations are stand-ins that record
    what they were given."""
    state = {"root": tmp_path, "tree": "t" * 40, "clean": True, "eval": [], "loads": [], "stream": [], "tok": R12Tok()}
    monkeypatch.setattr(ledger, "code_tree", lambda: state["tree"])
    monkeypatch.setattr(ledger, "commit", lambda: "c" * 40)
    monkeypatch.setattr(ledger, "tree_clean", lambda: state["clean"])
    sets = T.task_sets()
    for name, folder in (("pilot", "r12p"), ("confirm", "r12c")):
        out = tmp_path / "data" / folder
        T.write_tasks(name, out)
        plans = B.stream_plans(sets[name]["free"], sets[name]["live"], sets[name]["Ls"], name)
        (out / "stream.json").write_text(json.dumps(plans), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps({"set": name}), encoding="utf-8")  # the builder writes it last
        (out / "paired").mkdir()
        (out / "paired" / "test.jsonl").write_text(f"{name} paired\n", encoding="utf-8")
        (out / "paired-notice").mkdir()
        for file in NOTICE_FILES:
            (out / "paired-notice" / file).write_text(f"{name} {file}\n", encoding="utf-8")
    for run in r12.RUNS.values():
        (tmp_path / run).mkdir(parents=True)
        (tmp_path / run / "train_log.json").write_text('{"model": "fake/model"}', encoding="utf-8")
    (tmp_path / "results" / "r12p").mkdir(parents=True)
    (tmp_path / "results" / "r12p" / "smoke.json").write_text(json.dumps({"passed": True, "tree": state["tree"]}), encoding="utf-8")

    def eval_main(argv):
        state["eval"].append(list(argv))
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "records.jsonl").write_text('{"paired": 1}\n', encoding="utf-8")
        (out / "summary.md").write_text("# Evaluation summary\n", encoding="utf-8")

    def load_run(model_id, run, n_signals):
        state["loads"].append((model_id, Path(run).as_posix(), n_signals))
        return "the model", state["tok"], MS

    def runner(name):
        def run(model, tok, ms, specs, plans, prompts, set_code, out_dir, batch_size=16, run_types=S.RUN_TYPES):
            state["stream"].append({"runner": name, "model": model, "plans": list(plans), "prompts": prompts, "set_code": set_code,
                                    "out": Path(out_dir), "batch_size": batch_size, "run_types": tuple(run_types),
                                    "specs": [s.name for s in specs]})
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "records.jsonl").write_text('{"stream": 1}\n', encoding="utf-8")
            (Path(out_dir) / "stream_summary.md").write_text("# counts\n", encoding="utf-8")
            return []
        return run

    monkeypatch.setattr(R, "eval_main", eval_main)
    monkeypatch.setattr(R, "load_run", load_run)
    monkeypatch.setattr(R, "run_channel", runner("channel"))
    monkeypatch.setattr(R, "run_notice", runner("notice"))
    return state


def go(state, *args):
    R.main([*args, "--root", str(state["root"])])


def entries(state, set_name="confirm"):
    return ledger.entries(set_name, state["root"])


def test_out_folders_name_the_set_the_arm_the_model_and_the_batch_size():
    assert R.out_folder("confirm", "paired", "channel") == Path("results/r12-paired-wide-levels-s0-b16")
    assert R.out_folder("confirm", "Ls", "notice") == Path("results/r12-Ls-notice-none-s0-b16")
    assert R.out_folder("pilot", "free", "channel") == Path("results/r12p-free-wide-levels-s0-b16")
    assert R.out_folder("pilot", "live", "channel", batch_size=1) == Path("results/r12p-live-wide-levels-s0-b1")
    folders = {R.out_folder(s, a, m) for s in ("pilot", "confirm") for a in r12.ARMS for m in ("channel", "notice")}
    assert len(folders) == 16 and len({str(f).lower() for f in folders}) == 16  # distinct also where case is ignored


def test_paired_arm_calls_eval_run_with_the_planned_arguments(project, capsys):
    root = project["root"]
    go(project, "--set", "confirm", "--arm", "paired", "--model", "channel")
    go(project, "--set", "confirm", "--arm", "paired", "--model", "notice")
    channel, notice = project["eval"]
    out = root / "results" / "r12-paired-wide-levels-s0-b16"
    assert channel == ["--run", str(root / "runs" / "n6-wide-levels-s0"), "--data", str(root / "data" / "r12c" / "paired"),
                       "--out", str(out), "--batch-size", "16", "--preservation-n", "0"]
    assert "--mode" not in channel and "--base-n" not in channel  # standard mode; the untrained floor stays at its default 150
    assert notice == ["--run", str(root / "runs" / "n6-notice-none-s0"), "--data", str(root / "data" / "r12c" / "paired-notice"),
                      "--out", str(root / "results" / "r12-paired-notice-none-s0-b16"), "--batch-size", "16", "--mode", "harness"]
    assert "--preservation-n" not in notice
    first, second = entries(project)
    assert (first["arm"], first["model"], first["out"], first["batch_size"]) == ("paired", "channel", "results/r12-paired-wide-levels-s0-b16", 16)
    assert first["argv"] == ["--set", "confirm", "--arm", "paired", "--model", "channel", "--root", str(root)]
    assert list(first["data"]) == ["data/r12c/paired/test.jsonl"]
    assert list(second["data"]) == [f"data/r12c/paired-notice/{name}" for name in NOTICE_FILES]
    assert first["finished"] and first["records_sha256"] == ledger.sha256(out / "records.jsonl")
    assert project["loads"] == []  # the evaluation loads its own model (and prints its own summary)


def test_confirm_refuses_limit_run_types_and_other_batch_sizes(project, capsys):
    for extra in (["--batch-size", "1"], ["--limit", "16"], ["--run-types", "event"], ["--batch-size", "32"]):
        with pytest.raises(SystemExit) as stop:
            go(project, "--set", "confirm", "--arm", "free", "--model", "channel", *extra)
        assert stop.value.code == 3 and "refused" in capsys.readouterr().err
    assert entries(project) == [] and project["loads"] == [] and project["stream"] == []
    assert not (project["root"] / "results" / "r12-free-wide-levels-s0-b16").exists()


def test_second_confirmatory_run_exits_with_code_3_before_anything_is_loaded(project, capsys):
    go(project, "--set", "confirm", "--arm", "live", "--model", "channel")
    assert len(project["loads"]) == 1 and len(entries(project)) == 1
    with pytest.raises(SystemExit) as stop:
        go(project, "--set", "confirm", "--arm", "live", "--model", "channel")
    assert stop.value.code == 3 and "already" in capsys.readouterr().err
    assert len(project["loads"]) == 1 and len(project["stream"]) == 1 and len(entries(project)) == 1
    project["clean"] = False  # a dirty tree stops every confirmatory start
    with pytest.raises(SystemExit) as stop:
        go(project, "--set", "confirm", "--arm", "free", "--model", "channel")
    assert stop.value.code == 3 and len(project["loads"]) == 1


def test_stream_arm_runs_the_plans_of_its_arm_through_the_right_runner(project, capsys):
    root, tok = project["root"], project["tok"]
    go(project, "--set", "confirm", "--arm", "Ls", "--model", "channel")
    go(project, "--set", "confirm", "--arm", "free", "--model", "notice")
    first, second = project["stream"]
    tasks = T.task_sets()["confirm"]
    assert (first["runner"], first["model"], first["set_code"], first["batch_size"], first["run_types"]) == (
        "channel", "the model", 1, 16, ("event", "twin", "off"))
    assert first["out"] == root / "results" / "r12-Ls-wide-levels-s0-b16"
    assert [p.task_id for p in first["plans"]] == [t.task_id for t in tasks["Ls"]] and {p.arm for p in first["plans"]} == {"Ls"}
    assert first["plans"] == S.load_plans(root / "data" / "r12c" / "stream.json", "Ls") and len(first["plans"]) == 120
    assert set(first["prompts"]) == {t.task_id for t in tasks["Ls"]}
    sample = tasks["Ls"][7]
    assert first["prompts"][sample.task_id] == build_prompt(tok, sample)
    assert first["specs"] == ["ctx", "think", "energy", "time", "net", "ram"]
    assert (second["runner"], len(second["plans"]), second["out"]) == ("notice", 360, root / "results" / "r12-free-notice-none-s0-b16")
    assert project["loads"] == [("fake/model", (root / "runs" / "n6-wide-levels-s0").as_posix(), 6),
                                ("fake/model", (root / "runs" / "n6-notice-none-s0").as_posix(), 6)]
    assert tok.padding_side == "left"
    saved = entries(project)
    assert [list(e["data"]) for e in saved] == [["data/r12c/tasks.json", "data/r12c/stream.json"]] * 2
    assert all(e["finished"] and e["records_sha256"] == ledger.sha256(root / e["out"] / "records.jsonl") for e in saved)
    meta = json.loads((first["out"] / "eval_meta.json").read_text(encoding="utf-8"))["invocations"][-1]
    assert (meta["run"], meta["model"], meta["arm"], meta["set"]) == ("runs/n6-wide-levels-s0", "fake/model", "Ls", "confirm")
    assert "finished" in meta and meta["args"]["batch_size"] == 16
    assert "# counts" in capsys.readouterr().out


def test_pilot_slice_at_batch_1_goes_to_its_own_folder(project):
    root = project["root"]
    project["clean"] = False  # the pilot runs while the code is still being fixed
    go(project, "--set", "pilot", "--arm", "live", "--model", "channel", "--batch-size", "1", "--limit", "16", "--run-types", "event")
    go(project, "--set", "pilot", "--arm", "live", "--model", "channel")
    slice_run, whole = project["stream"]
    assert (slice_run["out"], slice_run["batch_size"], slice_run["run_types"], slice_run["set_code"]) == (
        root / "results" / "r12p-live-wide-levels-s0-b1", 1, ("event",), 0)
    every = S.load_plans(root / "data" / "r12p" / "stream.json", "live")
    # --limit takes plans at even spacing over the arm, so that a slice of 16 covers its families and counters
    assert slice_run["plans"] == [every[i] for i in (0, 3, 6, 9, 13, 16, 19, 22, 25, 28, 31, 34, 38, 41, 44, 47)]
    assert {p.family for p in slice_run["plans"]} == {"math", "explain", "code"}
    assert {p.signal for p in slice_run["plans"]} == {"ctx", "think", "time"}
    assert set(slice_run["prompts"]) == {p.task_id for p in slice_run["plans"]}
    assert whole["out"] == root / "results" / "r12p-live-wide-levels-s0-b16" and len(whole["plans"]) == 48
    assert [e["batch_size"] for e in entries(project, "pilot")] == [1, 16] and entries(project) == []
    go(project, "--set", "pilot", "--arm", "paired", "--model", "channel", "--limit", "8")
    assert project["eval"][-1][-2:] == ["--limit", "8"]
    with pytest.raises(SystemExit):
        go(project, "--set", "pilot", "--arm", "paired", "--model", "channel", "--run-types", "event")  # no run types in the paired arm
    with pytest.raises(SystemExit):
        go(project, "--set", "pilot", "--arm", "free", "--model", "channel", "--run-types", "event,quiet")


def test_rerun_goes_to_a_new_folder_with_its_reason(project, capsys):
    root = project["root"]
    go(project, "--set", "confirm", "--arm", "free", "--model", "channel")
    go(project, "--set", "confirm", "--arm", "free", "--model", "channel", "--rerun-reason", "defect D-999, fixed in the runner")
    assert [s["out"].name for s in project["stream"]] == ["r12-free-wide-levels-s0-b16", "r12-free-wide-levels-s0-b16-rerun1"]
    first, rerun = entries(project)
    assert (first["kind"], rerun["kind"], rerun["rerun_reason"]) == ("run", "rerun", "defect D-999, fixed in the runner")
    assert first["records_sha256"] and rerun["records_sha256"] and (root / first["out"] / "records.jsonl").exists()


def test_an_interrupted_confirmatory_run_resumes_into_its_folder(project, monkeypatch):
    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(R, "run_channel", crash)
    with pytest.raises(KeyboardInterrupt):
        go(project, "--set", "confirm", "--arm", "free", "--model", "channel")
    assert entries(project)[0]["finished"] is None
    seen = []
    monkeypatch.setattr(R, "run_channel", lambda *a, **kw: seen.append(a[7]) or (Path(a[7]) / "records.jsonl").write_text("x\n") or [])
    monkeypatch.setattr(R, "_show", lambda path: None)
    go(project, "--set", "confirm", "--arm", "free", "--model", "channel")
    saved = entries(project)
    assert len(saved) == 1 and len(saved[0]["resumes"]) == 1 and saved[0]["finished"]
    assert seen == [project["root"] / "results" / "r12-free-wide-levels-s0-b16"]


def test_run_refuses_a_set_whose_build_is_not_finished(project, capsys):
    root = project["root"]
    (root / "data" / "r12c" / "manifest.json").rename(root / "data" / "r12c" / "manifest.kept")
    for arm in ("paired", "Ls"):  # the builder writes the manifest last: without it the data may be half built
        with pytest.raises(SystemExit) as stop:
            go(project, "--set", "confirm", "--arm", arm, "--model", "channel")
        assert stop.value.code == 3 and "manifest.json" in capsys.readouterr().err
    assert entries(project) == [] and project["eval"] == [] and project["loads"] == []
    (root / "data" / "r12p" / "manifest.json").rename(root / "data" / "r12p" / "manifest.kept")
    with pytest.raises(SystemExit) as stop:  # the pilot: a plain error, nothing recorded
        go(project, "--set", "pilot", "--arm", "free", "--model", "notice")
    assert stop.value.code not in (0, 3) and "manifest.json" in str(stop.value.code)
    assert entries(project, "pilot") == [] and project["stream"] == []
    (root / "data" / "r12p" / "manifest.kept").rename(root / "data" / "r12p" / "manifest.json")
    go(project, "--set", "pilot", "--arm", "free", "--model", "notice")
    assert len(project["stream"]) == 1


def test_torn_partial_files_are_repaired_before_the_run_goes_on_and_written_into_the_entry(project, monkeypatch):
    root = project["root"]

    def killed(*args, **kwargs):
        raise KeyboardInterrupt

    # arm paired: the evaluator (prio.eval.run) reads its partial file line by line; it is repaired before it is called
    paired = root / "results" / "r12-paired-wide-levels-s0-b16"
    monkeypatch.setattr(R, "eval_main", killed)
    with pytest.raises(KeyboardInterrupt):
        go(project, "--set", "confirm", "--arm", "paired", "--model", "channel")
    row = {"episode_id": "r12c-P-00000", "condition": "on", "paired": "event"}
    paired.mkdir(parents=True, exist_ok=True)
    (paired / "records.partial.jsonl").write_text(json.dumps(row) + "\n" + '{"episode_id": "r12c-P-00001", "condi', encoding="utf-8")
    (paired / "generations.jsonl").write_text(json.dumps({**row, "token_ids": [1]}) + "\n", encoding="utf-8")
    seen = []

    def evaluator(argv):
        seen.append((paired / "records.partial.jsonl").read_text(encoding="utf-8"))  # what the evaluator finds
        (paired / "records.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(R, "eval_main", evaluator)
    go(project, "--set", "confirm", "--arm", "paired", "--model", "channel")
    assert seen == [json.dumps(row) + "\n"]  # whole lines only
    entry = entries(project)[0]
    assert len(entry["resumes"]) == 1 and entry["finished"]
    assert [(r["file"], r["bytes"], r["what"]) for r in entry["repairs"]] == [
        ("results/r12-paired-wide-levels-s0-b16/records.partial.jsonl", 37, "torn last line")]
    assert (paired / "records.partial.jsonl.torn").read_text(encoding="utf-8").endswith('{"episode_id": "r12c-P-00001", "condi\n')

    # the other arms: the same before the runner opens the folder (pilot entries carry the repairs too)
    free = root / "results" / "r12p-free-wide-levels-s0-b16"
    free.mkdir(parents=True)
    (free / "generations.jsonl").write_text('{"task_id": "r12p-math-00016", "run_type": "eve', encoding="utf-8")
    (free / "pass1.jsonl").write_text('{"task_id": "r12p-math-00016", "pre": [1, 2', encoding="utf-8")
    go(project, "--set", "pilot", "--arm", "free", "--model", "channel")
    pilot = entries(project, "pilot")[-1]
    assert sorted(r["file"] for r in pilot["repairs"]) == ["results/r12p-free-wide-levels-s0-b16/generations.jsonl",
                                                          "results/r12p-free-wide-levels-s0-b16/pass1.jsonl"]
    assert (free / "generations.jsonl").read_bytes() == b"" and (free / "generations.jsonl.torn").exists()
    go(project, "--set", "pilot", "--arm", "live", "--model", "channel")
    assert entries(project, "pilot")[-1]["repairs"] == []  # nothing torn: nothing written


@pytest.mark.parametrize("module", ["run", "build", "lines", "smoke", "freshness"])
def test_every_r12_command_writes_utf8_to_a_console_of_another_code_page(module, monkeypatch):
    import importlib
    import io
    import sys

    for name in ("stdout", "stderr"):
        monkeypatch.setattr(sys, name, io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    with pytest.raises(UnicodeEncodeError):
        sys.stdout.write("crossing → reaction")  # what a cp1252 console does with a character it does not hold
    main = importlib.import_module(f"prio.r12.{module}").main
    with pytest.raises(SystemExit) as stop:
        main(["--help"])  # the first thing a command does, before it parses or prints anything
    assert stop.value.code == 0
    for stream in (sys.stdout, sys.stderr):
        assert (stream.encoding.lower(), stream.errors) == ("utf-8", "replace")
    sys.stdout.write("crossing → reaction")
    sys.stdout.flush()
    assert "crossing → reaction".encode("utf-8") in sys.stdout.buffer.getvalue()


def test_utf8_console_leaves_a_stream_that_cannot_be_reconfigured_alone(monkeypatch):
    import io
    import sys

    plain = io.StringIO()  # no `reconfigure`: a captured or replaced stream
    monkeypatch.setattr(sys, "stdout", plain)
    r12.utf8_console()
    assert sys.stdout is plain


def test_main_runs_a_pilot_arm_end_to_end_on_the_tiny_model(project, monkeypatch, tiny_model):
    from helpers import arm
    from prio.channel import ChannelModel
    from test_r12_stream import SmallTok

    tok = SmallTok()
    model = ChannelModel(tiny_model, 6, delivery="wide", lane_width=8, projector="levels")
    arm(model)
    monkeypatch.undo()  # the real runners
    monkeypatch.setattr(ledger, "code_tree", lambda: "t" * 40)
    monkeypatch.setattr(ledger, "commit", lambda: "c" * 40)
    monkeypatch.setattr(ledger, "tree_clean", lambda: True)
    monkeypatch.setattr(R, "load_run", lambda model_id, run, n: (model, tok, MS))
    root = project["root"]
    R.main(["--set", "pilot", "--arm", "Ls", "--model", "channel", "--limit", "2", "--batch-size", "2", "--root", str(root)])
    out = root / "results" / "r12p-Ls-wide-levels-s0-b2"
    records = [S.StreamRecord(**json.loads(line)) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r.arm, r.run_type) for r in records] == [("Ls", t) for _ in range(2) for t in S.RUN_TYPES]
    arm_tasks = T.task_sets()["pilot"]["Ls"]
    assert {r.task_id for r in records} == {arm_tasks[0].task_id, arm_tasks[-1].task_id}  # two plans spread over the arm
    entry = ledger.entries("pilot", root)[-1]
    assert entry["finished"] and entry["records_sha256"] == ledger.sha256(out / "records.jsonl")
    assert (out / "stream_summary.md").exists() and (out / "eval_meta.json").exists()
