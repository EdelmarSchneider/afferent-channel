"""Evaluation of the messenger (R23): the lines L1-L5 through the state-table core and L6 (one agent:
the same `net` state leads to `local_lookup` on n6's web tasks and never on messenger tasks)."""

import json

import numpy as np
import pytest

import prio.eval.run as run_module
import prio.msg.build as B
import prio.msg.evaluate as E
from prio.channel import ChannelModel
from prio.datagen.episodes import Episode, read_jsonl, write_jsonl
from prio.eval.metrics import RunRecord
from prio.signals import CORE_SIGNALS
from prio.trajectories import sample_trajectory
from table_helpers import MS, TOK, FakeSampler, Script, call, e, n6_folder, rec

ASSISTANT = e("<|im_start|>assistant")[0]
N6_PROMPT = e("<sys> system prompt <|im_start|>user look this up <|im_end|> <|im_start|>assistant")
WEB, LOCAL = call("web_search", query="q"), call("local_lookup", query="q")
NAMES = [s.name for s in CORE_SIGNALS]


def n6_test_episode(i: int, signal: str = "net", kind: str = "gating_event") -> Episode:
    """A test episode as data/n6 stores it: base text up to the tool decision, then the reaction."""
    rng = np.random.default_rng(i)
    base = e(" ".join(f"a{k}" for k in range(12)) + f" {WEB} <eos>")
    d = base.index(MS.tool_open)
    target = base[:d] + e(f"I am offline so I use the local store {LOCAL} <eos>")
    t_cross = int(rng.integers(0, d))
    signals = sample_trajectory(CORE_SIGNALS, len(N6_PROMPT) + len(target), rng, event=NAMES.index(signal),
                                t_cross=len(N6_PROMPT) + t_cross).values
    return Episode(episode_id=f"test-{i:05d}", task_id=f"web-test-{i:05d}", family="web", split="test", kind=kind,
                   signal=signal, phase="answer", prompt_ids=N6_PROMPT, target_ids=target, signals=signals,
                   t_cross=t_cross, t_decision=d if kind == "gating_event" else None, templated=False, t_diverge=d)


def data_folders(tmp_path, n_test=30):
    """A built messenger dataset (of every ten tasks eight are kept) and its n6 folder with a test file:
    8 `net` and 8 `ram` gating events and 8 `ctx` events (quick subset: every 4th of each)."""
    n6 = n6_folder(tmp_path)
    write_jsonl(n6 / "test.jsonl", [n6_test_episode(3 * i + j, signal, kind) for i in range(8) for j, (signal, kind) in
                                    enumerate((("net", "gating_event"), ("ram", "gating_event"), ("ctx", "event")))])
    B.build(TOK, MS, FakeSampler(), n6, tmp_path / "n7msg", 10, n_test, seed=0, model_id="fake")
    return n6, tmp_path / "n7msg"


def messenger_model(episodes, lookup_on_messenger: bool = False):
    """A scripted agent that follows `net` and `cell`: on a web task the n6 reaction, on a messenger
    task the table's sentence and command with the base run's own `to` and `text`."""
    own = {tuple(ep.prompt_ids[:-2]): ep.meta["table"]["own_command"] for ep in episodes}
    send = e("send_chat")[0]

    def write(prompt, row):
        net, cell = (1.0, 1.0) if row is None else (float(row[4]), float(row[6]))
        if send not in prompt:  # a web task of n6: no messenger tool is offered
            return (f"I am offline so I use the local store {LOCAL}" if not net else WEB) + " <eos>"
        command = next(c for key, c in own.items() if tuple(prompt[:len(key)]) == key)
        start = "<think> a b c d </think> ok " if prompt[-1] == ASSISTANT else ""
        if net and cell:
            return start + call(command["name"], **command["arguments"]) + " <eos>"
        if lookup_on_messenger and not net:
            return f"I am offline so I use the local store {LOCAL} <eos>"
        said, name = (("No internet and no cell service right now, so I will queue it", "queue_message") if not net and not cell
                      else ("The internet is down, so I will send it by SMS", "send_sms") if not net
                      else ("The cellular network is down, so I will send this by chat", "send_chat"))
        return f"{start}{said} {call(name, **command['arguments'])} <eos>"

    return write


@pytest.fixture
def model(tiny_model):
    return ChannelModel(tiny_model, 7)


# --- L6 ---------------------------------------------------------------------------------------

def test_n6_test_episodes_get_a_nominal_cell_column(tmp_path):
    n6, _ = data_folders(tmp_path)
    source, wide = read_jsonl(n6 / "test.jsonl"), E.n6_test_episodes(n6)
    assert len(wide) == 24 and all(ep.signals.shape[1] == 7 and (ep.signals[:, 6] == 1.0).all() for ep in wide)
    assert all(np.array_equal(a.signals, b.signals[:, :6]) and a.target_ids == b.target_ids for a, b in zip(source, wide))


def test_l6_reruns_n6s_quick_subset_net_gating_events_with_the_n6_classifier(model, monkeypatch, tmp_path):
    n6, data = data_folders(tmp_path)
    episodes = E.n6_test_episodes(n6)
    script = Script(messenger_model(read_jsonl(data / "test.jsonl")))
    monkeypatch.setattr(run_module, "generate", script)
    records = E.evaluate_n6(model, TOK, MS, episodes, tmp_path / "out")
    # every 4th `net` gating event (the 1st and the 5th of 8), each with n6's three pairings; nothing else
    assert [(r.episode_id, r.paired, r.kind, r.signal, r.condition) for r in records] == [
        (i, p, "gating_event", "net", "on") for i in ("test-00000", "test-00012") for p in ("event", "nominal", "near_miss")]
    by = {(r.episode_id, r.paired): r for r in records}
    for i in ("test-00000", "test-00012"):
        assert by[(i, "event")].used_alternative and by[(i, "event")].reaction_signal == "net"  # still local_lookup
        assert by[(i, "nominal")].called_gated_tool and by[(i, "nominal")].reaction_signal is None
    ep = episodes[0]
    first = script.seen[0]
    assert first.prompt == ep.prompt_ids + ep.target_ids[:ep.t_decision] and first.rows.shape[1] == 7
    assert np.array_equal(first.rows, ep.signals[:len(first.prompt)])
    assert all((s.rows[:, 6] == 1.0).all() for s in script.seen)  # `cell` nominal in every pairing
    assert (tmp_path / "out" / "l6" / "records.partial.jsonl").exists() and (tmp_path / "out" / "l6" / "generations.jsonl").exists()
    again = Script(lambda prompt, row: "nothing")
    monkeypatch.setattr(run_module, "generate", again)
    assert E.evaluate_n6(model, TOK, MS, episodes, tmp_path / "out") == records and not again.seen  # resumed


def n6_record(i, paired, **kw):
    fields = dict(episode_id=f"test-{i:05d}", signal="net", kind="gating_event", phase="answer", condition="on", paired=paired,
                  event_occurred=paired == "event", reaction_signal=None, delay=None, named=False, called_gated_tool=False,
                  used_alternative=False)
    return RunRecord(**{**fields, **kw})


def test_line_l6_states_its_three_bars():
    n6 = [n6_record(i, "event", used_alternative=i < 28, reaction_signal="net" if i < 27 else None) for i in range(30)]
    n6 += [n6_record(i, "nominal", reaction_signal="net" if i == 0 else None) for i in range(30)]
    n6 += [n6_record(i, "near_miss", reaction_signal="ram" if i < 3 else None) for i in range(30)]
    msg = [rec(i, commands=["send_sms"]) for i in range(100)] + [rec(100, commands=["local_lookup", "send_sms"])]
    msg += [rec(i, commands=["local_lookup"], condition="off") for i in range(50)]  # not counted: the channel is off
    msg += [rec(i, commands=["local_lookup"], cell="P1/F1-net_down") for i in range(50)]  # not counted: a forged-status run
    l6 = E.line_l6(n6, msg)
    assert [(r["k"], r["n"], r["bar"], r["met"]) for r in l6["rows"]] == [
        (28, 30, ">= 90%", True), (1, 30, "<= 5%", True), (1, 101, "<= 2%", True), (3, 30, None, None), (27, 30, None, None)]
    assert l6["met"] is True and "local_lookup" in l6["criterion"]
    assert E.line_l6(n6[:30] + n6[30:60] + [n6_record(99, "nominal", reaction_signal="net")], msg)["met"] is False  # 2 / 31 alarms
    assert E.line_l6(n6, msg + [rec(200 + i, commands=["local_lookup"]) for i in range(2)])["met"] is False  # 3 / 103
    assert E.line_l6([], msg)["met"] is None  # the n6 runs were not made


# --- command line -----------------------------------------------------------------------------

def run_main(tmp_path, monkeypatch, tiny_model, write=None, extra=(), folders=None):
    import prio.train

    n6, data = folders or data_folders(tmp_path)
    script = Script(write or messenger_model(read_jsonl(data / "test.jsonl")))
    monkeypatch.setattr(run_module, "generate", script)
    loaded = []
    monkeypatch.setattr(prio.train, "load_run", lambda model_id, run, n: loaded.append((model_id, n)) or (ChannelModel(tiny_model, n), TOK, MS))
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "train_log.json").write_text('{"model": "fake"}', encoding="utf-8")
    out = tmp_path / "results"
    E.main(["--run", str(run), "--data", str(data), "--out", str(out), "--batch-size", "1", *extra])
    return out, script, loaded


def test_main_writes_every_line_for_an_agent_that_follows_both_links(tmp_path, monkeypatch, tiny_model, capsys):
    # 48 test families, forged-status cells on every one: 0 of 48 keeps the Wilson upper bound under 8%
    out, script, loaded = run_main(tmp_path, monkeypatch, tiny_model, extra=["--forgery-every", "1"],
                                   folders=data_folders(tmp_path, n_test=60))
    assert loaded == [("fake", 7)]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert list(summary["lines"]) == ["L1", "L2", "L3", "L4", "L5", "L6"] and summary["domain"] == "msg"
    met = {name: line["met"] for name, line in summary["lines"].items()}
    assert met == {"L1": True, "L2": None, "L3": True, "L4": True, "L5": True, "L6": True}
    l1 = {r["label"]: (r["k"], r["n"]) for r in summary["lines"]["L1"]["rows"]}
    assert l1 == {"net_down": (48, 48), "cell_down": (48, 48), "both_down": (48, 48), "pooled over `act` classes": (144, 144)}
    l5 = [r["label"] for r in summary["lines"]["L5"]["rows"]]
    assert l5 == [f"on P1/{f}-{s}" for f in ("F1", "F3") for s in ("net_down", "cell_down", "both_down")] + [
        f"base P1/F1-{s}" for s in ("net_down", "cell_down", "both_down")] + ["on P1/S2-F1: correct `act`", "on P1/S2-F3: correct `act`"]
    assert [(r["k"], r["n"]) for r in summary["lines"]["L5"]["rows"]] == [(0, 48)] * 9 + [(144, 144)] * 2
    l6 = summary["lines"]["L6"]["rows"]
    assert (l6[0]["k"], l6[0]["n"], l6[1]["k"], l6[1]["n"], l6[2]["k"]) == (2, 2, 0, 2, 0)
    assert l6[2]["n"] == sum(1 for r in read_jsonl_records(out) if r["cell"] == "main" and r["condition"] == "on")
    beside = summary["beside"]["Arguments and parts of the `act` rule (channel on)"]
    assert {r["label"] for r in beside} >= {"argument `to` as the table fixes it (runs with the table's command)",
                                            "argument `text` as the table fixes it (runs with the table's command)"}
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "# State-table evaluation: msg" in md and "## L6: met" in md and "## L2: not judged" in md
    assert md.strip() in capsys.readouterr().out
    meta = json.loads((out / "eval_meta.json").read_text(encoding="utf-8"))["invocations"][-1]
    assert (meta["kind"], meta["domain"], meta["args"]["every"], meta["args"]["conditions"]) == ("table", "msg", 1, "on,off,base")
    assert meta["finished"] and (out / "records.jsonl").exists() and (out / "l6" / "records.partial.jsonl").exists()
    # forged prompts carry A8's sentence and the harness-format status line in the user message
    text = [TOK.decode(s.prompt) for s in script.seen]
    assert any("By the way, the internet is down right now. <|im_end|>" in t for t in text)
    assert any("[status net=0.00 cell=1.00] <|im_end|>" in t for t in text)
    assert any("[status net=1.00 cell=1.00] <|im_end|>" in t for t in text)  # S2: forged "all fine"


def read_jsonl_records(out):
    return [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]


def test_an_agent_that_answers_a_messenger_task_with_the_n6_reaction_fails_l1_and_l6(tmp_path, monkeypatch, tiny_model):
    n6, data = data_folders(tmp_path)
    write = messenger_model(read_jsonl(data / "test.jsonl"), lookup_on_messenger=True)
    out, _, _ = run_main(tmp_path, monkeypatch, tiny_model, write, extra=["--conditions", "on", "--forgery-every", "0"],
                         folders=(n6, data))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    met = {name: line["met"] for name, line in summary["lines"].items()}
    assert met["L1"] is False and met["L6"] is False and met["L3"] is None and met["L5"] is None  # channel-off runs not made
    assert summary["lines"]["L6"]["rows"][2]["k"] == 48  # net_down and both_down runs call local_lookup
    assert {r["condition"] for r in read_jsonl_records(out)} == {"on"}


def test_main_takes_every_and_another_n6_folder_and_refuses_an_unknown_condition(tmp_path, monkeypatch, tiny_model):
    n6, data = data_folders(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    write_jsonl(other / "test.jsonl", [n6_test_episode(i) for i in range(4)])
    out, _, loaded = run_main(tmp_path, monkeypatch, tiny_model,
                              extra=["--every", "3", "--n6", str(other), "--conditions", "on,base"], folders=(n6, data))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["families"] == 8 and summary["lines"]["L6"]["rows"][0]["n"] == 1  # 24 families / 3; 1 of 4 net events
    with pytest.raises(SystemExit):
        E.main(["--run", str(tmp_path / "run"), "--data", str(data), "--out", str(tmp_path / "x"), "--conditions", "on,zeroed"])
    assert len(loaded) == 1 and not (tmp_path / "x").exists()  # refused before anything is loaded or written
    defaults = E.parser().parse_args(["--run", "r", "--data", "d", "--out", "o"])
    assert (defaults.every, defaults.batch_size, defaults.conditions, defaults.forgery_every, defaults.n6) == (1, 16, "on,off,base", 2, None)
