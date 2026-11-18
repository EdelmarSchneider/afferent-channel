"""Build of the messenger dataset (R23): base runs through a fake sampler (no GPU), cleaning,
sibling families, the merge with n6's training episodes, the manifest, the CLI, the reader script."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import prio.msg.build as B
from helpers import make_episode
from prio.datagen.episodes import read_jsonl
from prio.msg import domain as D
from table_helpers import MS, TOK, FakeSampler, call, e, n6_folder


def run_build(tmp_path, sampler=None, n_train=40, n_test=20, out="n7msg"):
    return B.build(TOK, MS, sampler or FakeSampler(), tmp_path / "n6", tmp_path / out, n_train, n_test, seed=0, model_id="fake")


def table(ep) -> dict:
    return ep.meta["table"]


def test_train_file_carries_n6_with_a_nominal_cell_column_and_everything_else_as_stored(tmp_path):
    n6 = n6_folder(tmp_path)
    manifest = run_build(tmp_path)
    source, train = read_jsonl(n6 / "train.jsonl"), read_jsonl(tmp_path / "n7msg" / "train.jsonl")
    assert manifest["n6_episodes_carried"] == 3 and manifest["n6"] == str(n6)
    for old, new in zip(source, train[:3]):
        assert new.signals.shape == (len(old.prompt_ids) + len(old.target_ids), 7) and new.signals.dtype == np.float32
        assert np.array_equal(new.signals[:, :6], old.signals) and (new.signals[:, 6] == 1.0).all()
        new.signals = old.signals
        assert new == old  # ids, kind, positions, meta: untouched
    assert all(ep.family == "msg" and ep.signals.shape[1] == 7 for ep in train[3:])
    assert len({ep.episode_id for ep in train}) == len(train)


def test_a_wrong_number_of_columns_is_refused():
    with pytest.raises(ValueError):
        B.widen(make_episode(signals=np.zeros((5, 7), dtype=np.float32)), 1.0)


def test_base_runs_are_cleaned_and_every_kept_run_gives_all_four_state_classes(tmp_path):
    n6_folder(tmp_path)
    manifest = run_build(tmp_path)
    assert manifest["tasks_requested"] == {"train": 40, "test": 20}
    assert manifest["base_runs_finished"] == {"train": 36, "test": 18}
    assert manifest["base_runs_kept"] == {"train": 32, "test": 16}  # of every ten: one unfinished, one without a send
    assert manifest["dropped"] == {"train": {"not_finished": 4, "first_command": 4}, "test": {"not_finished": 2, "first_command": 2}}
    assert manifest["own_command"] == {"train": {"send_chat": 28, "send_sms": 4}, "test": {"send_chat": 14, "send_sms": 2}}
    assert manifest["base_runs_kept_share"] == {"train": 0.8, "test": 0.8} and manifest["stop_rule"] == {
        "base_runs_kept_min": 0.7, "passed": True}
    assert manifest["clean_rule"] == D.CLEAN_RULE and manifest["families"] == {"train": 32, "test": 16}
    assert (manifest["domain"], manifest["model"], manifest["seed"], manifest["n_signals"]) == ("msg", "fake", 0, 7)
    assert manifest["signals"] == ["ctx", "think", "energy", "time", "net", "ram", "cell"]

    for split, n_n6 in (("train", 3), ("test", 0)):
        eps = read_jsonl(tmp_path / "n7msg" / f"{split}.jsonl")[n_n6:]
        assert manifest["msg_episodes"][split] == len(eps) and manifest["episodes"][split] == len(eps) + n_n6
        groups: dict[str, list] = {}
        for ep in eps:
            assert ep.split == split and ep.episode_id.startswith(f"msg-{split}-")
            groups.setdefault(table(ep)["family"], []).append(ep)
        assert len(groups) == manifest["families"][split]
        for siblings in groups.values():
            variants = [table(ep)["variant"] for ep in siblings]
            assert variants[0] == "base" and {"net_down", "cell_down", "both_down"} <= set(variants)
            cut = table(siblings[0])["cut"]
            assert cut == siblings[0].target_ids.index(MS.tool_open)  # the decision: the first command
            for ep in siblings:
                assert ep.prompt_ids == siblings[0].prompt_ids and ep.target_ids[:cut] == siblings[0].target_ids[:cut]
                own = table(ep)["own_command"]
                if table(ep)["behavior"] == "act":
                    assert table(ep)["command"]["arguments"] == own["arguments"]  # `to` and `text` as in the base run
        counts = manifest["by_class_timing"][split]
        assert counts["base/none"] == len(groups) and sum(counts.values()) == len(eps)
        for name in ("net_down", "cell_down", "both_down"):
            assert counts.get(f"{name}/from_start", 0) + counts.get(f"{name}/onset", 0) == len(groups)
        assert manifest["variants"][split]["base"] == len(groups)
        assert sum(n for v, n in manifest["variants"][split].items() if v.endswith("-back")) == counts.get("base/back", 0)
    assert manifest["kinds"]["train"]["event"] == 1 and manifest["kinds"]["train"]["table_act"] == 96
    assert manifest["kinds"]["test"] == {"table_base": manifest["by_class_timing"]["test"]["base/none"]
                                         + manifest["by_class_timing"]["test"].get("base/back", 0), "table_act": 48}
    assert json.loads((tmp_path / "n7msg" / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_too_few_kept_base_runs_trip_the_stop_rule(tmp_path):
    n6_folder(tmp_path)

    def mostly_text(prompts):
        return [(e("<think> hm hm hm hm </think> " + (call("send_chat", to="A", text="b") if i % 2 else "done") + " <eos>"), "eos")
                for i in range(len(prompts))]

    manifest = run_build(tmp_path, mostly_text)
    assert manifest["base_runs_kept_share"]["train"] == 0.5 and manifest["stop_rule"]["passed"] is False


def test_build_resumes_from_the_base_run_cache_and_is_deterministic(tmp_path):
    n6_folder(tmp_path)
    sampler = FakeSampler()
    run_build(tmp_path, sampler)
    assert sampler.calls == 1 and (tmp_path / "n7msg" / "base_runs.json").exists()
    before = {name: (tmp_path / "n7msg" / name).read_bytes() for name in ("train.jsonl", "test.jsonl")}

    def refuse(prompts):
        raise AssertionError("the cache holds every base run")

    run_build(tmp_path, refuse)
    assert {name: (tmp_path / "n7msg" / name).read_bytes() for name in before} == before


def test_the_test_split_does_not_depend_on_the_train_count(tmp_path):
    n6_folder(tmp_path)

    def by_prompt(prompts):  # the same run for the same prompt, whatever its place in the list
        return [(e(f"<think> a b c d </think> ok {call('send_chat', to='R', text=f'm {len(p)} {sum(p) % 97}')} <eos>"), "eos")
                for p in prompts]

    run_build(tmp_path, by_prompt, n_train=40, out="a")
    run_build(tmp_path, by_prompt, n_train=8, out="b")
    assert (tmp_path / "a" / "test.jsonl").read_bytes() == (tmp_path / "b" / "test.jsonl").read_bytes()


def test_the_output_folder_must_not_be_the_n6_folder(tmp_path):
    n6 = n6_folder(tmp_path)
    with pytest.raises(ValueError):
        B.build(TOK, MS, FakeSampler(), n6, n6, 10, 5, seed=0, model_id="fake")


def test_main_parses_the_command_line_and_loads_the_model_only_when_a_base_run_is_missing(tmp_path, monkeypatch, capsys):
    n6 = n6_folder(tmp_path)
    loads, seen = [], {}
    fake = FakeSampler()

    def sample_batch(model, tok, ms, inputs, max_new_tokens, batch_size, label=""):
        seen.update(model=model, max_new_tokens=max_new_tokens, batch_size=batch_size)
        return fake(inputs)

    monkeypatch.setattr(B, "_load_tokenizer", lambda model: (TOK, MS))
    monkeypatch.setattr(B, "load_base", lambda model: loads.append(model) or ("the model", TOK, MS))
    monkeypatch.setattr(B, "sample_batch", sample_batch)
    argv = ["--n6", str(n6), "--model", "fake/model", "--out", str(tmp_path / "mini"), "--seed", "3", "--train", "20",
            "--test", "10", "--batch-size", "8"]
    B.main(argv)
    printed = json.loads(capsys.readouterr().out)
    assert printed["tasks_requested"] == {"train": 20, "test": 10} and (printed["seed"], printed["model"]) == (3, "fake/model")
    assert loads == ["fake/model"] and seen == {"model": "the model", "max_new_tokens": B.MAX_NEW_TOKENS, "batch_size": 8}
    B.main(argv)  # every base run is in the cache: no model
    assert loads == ["fake/model"]
    with pytest.raises(SystemExit):
        B.main(["--model", "fake/model", "--out", str(tmp_path / "x")])  # --n6 is required
    defaults = B.parser().parse_args(["--n6", "a", "--model", "m", "--out", "o"])
    assert (defaults.train, defaults.test, defaults.seed, defaults.batch_size) == (1500, 300, 0, 32)


def test_reader_script_prints_sibling_families_decoded(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import inspect_table

    n6_folder(tmp_path)
    run_build(tmp_path)
    inspect_table.show(tmp_path / "n7msg", 2, "test", TOK)
    out = capsys.readouterr().out
    assert out.count("family msg-test-") == 2 and "user: " in out
    for word in ("base", "net_down", "cell_down", "both_down", "from_start", "send_sms", "queue_message", "net=0", "cell=0"):
        assert word in out
    assert "the user wants a message sent" in out  # the tail of the shared prefix
