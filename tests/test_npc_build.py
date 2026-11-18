"""Build of the game-character dataset (R22): base runs through a fake sampler (no GPU), cleaning,
one sibling family per kept run, the manifest (episodes per task, false-naming floor), the CLI and
the reader script."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

import prio.npc.build as B
from npc_helpers import MS, TOK, WTOK, NpcSampler, call, e, turn, words
from prio.datagen.episodes import read_jsonl
from prio.npc import domain as D
from prio.table.evaluate import families

ACTS = ("terror", "thirst_water", "thirst_dry", "hunger_food", "hunger_none")


def run_build(tmp_path, sampler=None, n_train=60, n_test=30, out="npc6"):
    return B.build(WTOK, MS, sampler or NpcSampler(), tmp_path / out, n_train, n_test, seed=0, model_id="fake")


def table(ep) -> dict:
    return ep.meta["table"]


def test_base_runs_are_cleaned_and_the_manifest_says_what_was_kept_and_why(tmp_path):
    manifest = run_build(tmp_path)
    assert manifest["tasks_requested"] == {"train": 60, "test": 30} and manifest["max_new_tokens"] == D.MAX_NEW_TOKENS
    assert manifest["base_runs_finished"] == {"train": 54, "test": 27}
    assert manifest["base_runs_kept"] == {"train": 42, "test": 21}  # of every ten: unfinished, flees unasked, one short sentence
    reasons = {"not_finished": 6, "reaction_command": 6, "short_speech": 6}
    assert manifest["dropped"] == {"train": reasons, "test": {k: 3 for k in reasons}}
    assert manifest["base_runs_kept_share"] == {"train": 0.7, "test": 0.7}
    assert manifest["stop_rule"] == {"base_runs_kept_min": 0.7, "passed": True} and manifest["clean_rule"] == D.CLEAN_RULE
    assert manifest["own_command"] == {"train": {"none": 36, "point_direction": 6}, "test": {"none": 18, "point_direction": 3}}
    assert (manifest["domain"], manifest["model"], manifest["seed"], manifest["n_signals"]) == ("npc", "fake", 0, 6)
    assert manifest["signals"] == ["fear", "thirst", "hunger", "threat", "water_near", "food_near"]
    assert json.loads((tmp_path / "npc6" / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_every_kept_run_gives_one_family_and_the_manifest_counts_episodes_per_task(tmp_path):
    manifest = run_build(tmp_path)
    assert manifest["families"] == {"train": 42, "test": 21} and manifest["families_dropped"] == {"train": {}, "test": {}}
    assert manifest["families_without_mention"] == {"train": 6, "test": 3}  # the runs with a command of their own
    assert manifest["episodes"]["train"] == 36 * 5 + 6 * 4  # five siblings; four without the mention sibling
    assert manifest["episodes_per_task"]["train"] == pytest.approx(204 / 60)
    assert manifest["episodes_per_family"]["train"] == pytest.approx(204 / 42)
    for split in ("train", "test"):
        eps = read_jsonl(tmp_path / "npc6" / f"{split}.jsonl")
        groups = families(eps)
        assert len(eps) == manifest["episodes"][split] and len(groups) == manifest["families"][split]
        assert len({ep.episode_id for ep in eps}) == len(eps) and all(ep.split == split and ep.family == "npc" for ep in eps)
        assert manifest["episodes_per_task"][split] == pytest.approx(len(eps) / manifest["tasks_requested"][split])
        counts = Counter(f"{table(ep)['cls']}/{table(ep)['timing'] or 'none'}" for ep in eps)
        assert manifest["by_class_timing"][split] == dict(counts) and counts["calm/none"] == len(groups)
        assert manifest["kinds"][split] == dict(Counter(ep.kind for ep in eps))
        assert manifest["variants"][split] == dict(Counter(table(ep)["variant"] for ep in eps))
        assert manifest["by_role"][split] == dict(Counter(table(ep)["facts"]["role"] for ep in eps))
        assert manifest["by_phase"][split] == dict(Counter(ep.phase for ep in eps))
        assert not any("back" in key for key in counts)  # R22 has no such timing
        for name, siblings in groups.items():
            first = siblings[0]
            cut = table(first)["cut"]
            assert name == first.task_id and table(first)["timing"] is None and first.kind == "table_base"
            for ep in siblings:
                assert ep.prompt_ids == first.prompt_ids and ep.target_ids[:cut] == first.target_ids[:cut]
                assert table(ep)["own_command"] == table(first)["own_command"]
                assert (table(ep)["variant"].startswith(D.PAIR)) <= (split == "test")  # pairs: test split only
            assert len({table(ep)["timing"] for ep in siblings[1:]}) == 1  # one timing kind per family
            has_mention = any(ep.kind == "table_mention" for ep in siblings)
            assert has_mention == (table(first)["own_command"] is None)
    train = manifest["by_class_timing"]["train"]
    assert train["terror/onset"] == 2 * 14 and "terror/from_start" not in train  # fear: an onset inside the turn
    for cls in ACTS[1:]:  # thirst and hunger: half from the start, half with an onset
        assert train[f"{cls}/from_start"] == train[f"{cls}/onset"] == 7
    assert set(manifest["by_role"]["test"]) == set(D.KIND) and set(manifest["by_role"]["train"]) == set(D.KIND) - D.HELD_OUT
    assert set(manifest["by_phase"]["train"]) == {"thinking", "answer"}
    assert manifest["length_max"] <= 2048 and manifest["length_p50"] > 0


def test_the_six_kinds_of_family_come_in_a_drawn_order_so_that_a_stride_does_not_pick_one_kind(tmp_path):
    run_build(tmp_path, n_train=120)
    groups = families(read_jsonl(tmp_path / "npc6" / "train.jsonl"))
    kinds = [(table(s[-1])["variant"].split("-")[0].replace("terror", "fear"), table(s[-1])["timing"]) for s in groups.values()]
    blocks = [kinds[i:i + 6] for i in range(0, len(kinds) - 5, 6)]
    assert all(sorted(block) == sorted(D.SLOTS) for block in blocks) and len({tuple(block) for block in blocks}) > 3
    for every in (2, 3, 4, 6):
        assert {felt for felt, _ in kinds[::every]} == {"fear", "thirst", "hunger"}


def test_the_manifest_states_how_often_the_untrained_models_own_text_names_a_state(tmp_path):
    def chatty(prompts):
        said = ["", "you look hungry , traveler .", "the bandits are long gone .", "I am parched , and so hungry ."]
        return [(e(f"<think> {words('t', 30)} </think> {words('a', 42)} {said[i % 4]} <eos>"), "eos") for i in range(len(prompts))]

    manifest = run_build(tmp_path, chatty, n_train=40, n_test=20)
    # since 2026-11-16 19:55 such runs are not used (cleaning rule): the floor of the kept runs is zero by construction
    assert manifest["dropped"]["train"] == {"names_state": 30} and manifest["dropped"]["test"] == {"names_state": 15}
    assert manifest["false_naming_floor"]["train"] == {
        "runs": 10, "any": 0, "by_list": {"wolf": 0, "bandits": 0, "dragon": 0, "thirst": 0, "hunger": 0}}
    assert manifest["false_naming_floor"]["test"]["runs"] == 5 and manifest["false_naming_floor"]["test"]["any"] == 0
    quiet = run_build(tmp_path, out="quiet")["false_naming_floor"]
    assert quiet["train"] == {"runs": 42, "any": 0, "by_list": {"wolf": 0, "bandits": 0, "dragon": 0, "thirst": 0, "hunger": 0}}


def test_too_few_kept_base_runs_trip_the_stop_rule(tmp_path):
    def mostly_fleeing(prompts):
        return [(turn(tail=call("flee") if i % 2 else ""), "eos") for i in range(len(prompts))]

    manifest = run_build(tmp_path, mostly_fleeing, n_train=20, n_test=10)
    assert manifest["base_runs_kept_share"]["train"] == 0.5 and manifest["stop_rule"]["passed"] is False


def test_a_run_too_short_for_a_cut_is_counted_not_built(tmp_path):
    def short(prompts):  # speech enough to be kept, no word start twenty tokens from both ends
        return [(e(f"<think> t0 </think> {words('a', 28)} <eos>"), "eos") for _ in prompts]

    manifest = run_build(tmp_path, short, n_train=12, n_test=6)
    assert manifest["base_runs_kept"] == {"train": 12, "test": 6} and manifest["families"]["train"] == 4  # from the start only
    assert manifest["families_dropped"]["train"] == {"no_cut": 8}


def test_build_resumes_from_the_base_run_cache_and_is_deterministic(tmp_path):
    sampler = NpcSampler()
    run_build(tmp_path, sampler)
    assert sampler.calls == 1 and (tmp_path / "npc6" / "base_runs.json").exists()
    before = {name: (tmp_path / "npc6" / name).read_bytes() for name in ("train.jsonl", "test.jsonl")}

    def refuse(prompts):
        raise AssertionError("the cache holds every base run")

    run_build(tmp_path, refuse)
    assert {name: (tmp_path / "npc6" / name).read_bytes() for name in before} == before


def test_the_test_split_does_not_depend_on_the_train_count(tmp_path):
    def by_prompt(prompts):  # the same run for the same prompt, whatever its place in the list
        return [(turn(n_think=24 + sum(p) % 9, n_speech=36 + len(p) % 7), "eos") for p in prompts]

    run_build(tmp_path, by_prompt, n_train=30, out="a")
    run_build(tmp_path, by_prompt, n_train=6, out="b")
    assert (tmp_path / "a" / "test.jsonl").read_bytes() == (tmp_path / "b" / "test.jsonl").read_bytes()


def test_main_parses_the_command_line_and_loads_the_model_only_when_a_base_run_is_missing(tmp_path, monkeypatch, capsys):
    loads, seen = [], {}
    fake = NpcSampler()

    def sample_batch(model, tok, ms, inputs, max_new_tokens, batch_size, label=""):
        seen.update(model=model, max_new_tokens=max_new_tokens, batch_size=batch_size)
        return fake(inputs)

    monkeypatch.setattr(B, "_load_tokenizer", lambda model: (WTOK, MS))
    monkeypatch.setattr(B, "load_base", lambda model: loads.append(model) or ("the model", WTOK, MS))
    monkeypatch.setattr(B, "sample_batch", sample_batch)
    argv = ["--model", "fake/model", "--out", str(tmp_path / "mini"), "--seed", "3", "--train", "20", "--test", "10",
            "--batch-size", "8"]
    B.main(argv)
    printed = json.loads(capsys.readouterr().out)
    assert printed["tasks_requested"] == {"train": 20, "test": 10} and (printed["seed"], printed["model"]) == (3, "fake/model")
    assert printed["episodes_per_task"]["train"] > 2  # printed, so the task count can be chosen
    assert loads == ["fake/model"] and seen == {"model": "the model", "max_new_tokens": D.MAX_NEW_TOKENS, "batch_size": 8}
    B.main(argv)  # every base run is in the cache: no model
    assert loads == ["fake/model"]
    with pytest.raises(SystemExit):
        B.main(["--out", str(tmp_path / "x")])  # --model is required
    defaults = B.parser().parse_args(["--model", "m", "--out", "o"])
    assert (defaults.train, defaults.test, defaults.seed, defaults.batch_size) == (1250, 300, 0, 32)


def test_reader_script_prints_the_families_of_this_domain(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import inspect_table

    run_build(tmp_path)
    inspect_table.show(tmp_path / "npc6", 21, "test", TOK)
    out = capsys.readouterr().out
    assert out.count("family npc-test-") == 21 and "user: " in out
    for word in ("Role: knight", "Role: baker", "You are talking with", "calm", "terror-swap", "thirst-water", "hunger-none",
                 "pair-terror-thirst", "onset t_on=", "from_start", "draw_weapon", "flee", "drink_water", "go_to", "tavern",
                 "fear=0.", "threat=0.", "water_near=1.00", "food_near=0.00", "</think>", " -- "):
        assert word in out, word
