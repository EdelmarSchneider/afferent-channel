"""Freshness audit of the R12 task sets (prio.r12.freshness; analysis plan Section 3.2)."""

import json

import pytest

from helpers import make_episode
from prio.datagen.episodes import write_jsonl
from prio.r12 import freshness as F
from prio.r12 import tasks as R
from prio.tasks import Task, build_prompt
from table_helpers import TOK


def task(i: int, name: str = "pilot", family: str = "explain") -> Task:
    prefix = {"pilot": "r12p", "confirm": "r12c"}[name]
    return Task(f"{prefix}-{family}-{i:05d}", family, f"Explain thing number {i} of the {name} set.", None, "test")


def fake_sets() -> dict:
    return {"pilot": {"paired": [task(0), task(1)], "free": [task(2)], "live": [], "Ls": []},
            "confirm": {"paired": [task(3, "confirm"), task(4, "confirm")], "free": [], "live": [task(5, "confirm")],
                        "Ls": [task(6, "confirm")]}}


def data_folder(root, name: str, prompts: list[list[int]], file: str = "test.jsonl"):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / file, [make_episode(episode_id=f"e{i}", prompt_ids=p) for i, p in enumerate(prompts)])
    return folder


def other_prompt(i: int) -> list[int]:
    return build_prompt(TOK, Task(f"old-{i}", "explain", f"Explain an old thing {i}.", None, "test"))


def test_audit_passes_on_disjoint_sets(tmp_path):
    data = tmp_path / "data"
    data_folder(data, "n6", [other_prompt(0), other_prompt(1), other_prompt(0)])
    data_folder(data, "n6-notice", [other_prompt(2)], "train.jsonl")
    result = F.audit(data, TOK, tmp_path / "out", sets=fake_sets(), used=set())
    assert result["passed"] is True and result["by_seed"]["passed"] and result["by_file"]["passed"]
    by_file = result["by_file"]
    assert (by_file["folders"], by_file["files"], by_file["episodes"], by_file["distinct_prompts"]) == (2, 2, 4, 3)
    assert by_file["r12_prompts"] == 7 and by_file["matches"] == [] and by_file["lines_without_prompt"] == 0
    assert result["by_seed"]["tasks"] == {"pilot": 3, "confirm": 4}
    raw = (tmp_path / "out" / "freshness.json").read_bytes()
    assert b"\r" not in raw and json.loads(raw.decode("utf-8")) == result


def test_audit_flags_a_planted_prompt_in_a_data_file(tmp_path):
    data, sets = tmp_path / "data", fake_sets()
    planted = build_prompt(TOK, sets["confirm"]["live"][0])
    data_folder(data, "n6", [other_prompt(0)])
    data_folder(data / "n6-notice", "deeper", [other_prompt(1), planted, other_prompt(2)], "test_swap.jsonl")
    result = F.audit(data, TOK, tmp_path / "out", sets=sets, used=set())
    assert result["passed"] is False and result["by_seed"]["passed"] and not result["by_file"]["passed"]
    assert result["by_file"]["matches"] == [{"task_id": "r12c-explain-00005", "set": "confirm", "arm": "live",
                                             "file": "n6-notice/deeper/test_swap.jsonl", "line": 2}]


def test_audit_flags_a_prompt_of_an_earlier_draw_and_an_overlap_of_the_two_sets():
    sets = fake_sets()
    old = R.earlier_draws()["load_tasks(0, 20, seed 3)"][7]
    sets["confirm"]["paired"].append(Task("r12c-explain-00009", "explain", old.prompt, None, "test"))
    result = F.by_seed(sets)
    assert result["passed"] is False
    assert result["used"] == [{"task_id": "r12c-explain-00009", "set": "confirm", "arm": "paired",
                               "draws": ["load_tasks(0, 20, seed 3)"]}]
    assert result["in_both_sets"] == []

    sets = fake_sets()
    sets["confirm"]["Ls"].append(Task("r12c-explain-00010", "explain", sets["pilot"]["free"][0].prompt, None, "test"))
    result = F.by_seed(sets, used=set())
    assert result["passed"] is False and result["used"] == []
    assert result["in_both_sets"] == [{"pilot": "r12p-explain-00002", "confirm": "r12c-explain-00010"}]


def test_prompt_ids_are_read_without_parsing_the_signals(tmp_path):
    folder = tmp_path / "data" / "x"
    folder.mkdir(parents=True)
    lines = [b'{"episode_id": "a", "prompt_ids": [1, 2, 3], "target_ids": [4], "signals": {"b64": "broken',
             b'{"episode_id": "b", "note": "no prompt here"}',
             b"",
             b'{"episode_id": "c", "prompt_ids": [], "target_ids": []}',
             b'{"episode_id": "d", "prompt_ids": [7,8 , 9]}']
    (folder / "odd.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    assert list(F.prompts_of(folder / "odd.jsonl")) == [(1, (1, 2, 3)), (4, ()), (5, (7, 8, 9))]
    result = F.by_file(fake_sets(), tmp_path / "data", TOK)
    assert result["episodes"] == 3 and result["lines_without_prompt"] == 1 and result["passed"]


def test_r12_folders_and_other_file_types_are_not_scanned(tmp_path):
    data, sets = tmp_path / "data", fake_sets()
    own = [build_prompt(TOK, t) for t in sets["pilot"]["paired"]]
    data_folder(data / "r12p", "paired", own)
    data_folder(data, "r12c", own)
    data_folder(data, "n6", [other_prompt(0)])
    (data / "n6" / "base_runs.json").write_text(json.dumps({"prompt_ids": own[0]}), encoding="utf-8")
    (data / "loose.jsonl").write_text(json.dumps({"prompt_ids": own[0]}) + "\n", encoding="utf-8")  # not in a folder
    result = F.by_file(sets, data, TOK)
    assert result["passed"] and result["folders"] == 1 and result["skipped_folders"] == ["r12c", "r12p"]


def test_a_missing_data_root_is_an_error_not_a_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        F.by_file(fake_sets(), tmp_path / "nothing", TOK)


def test_the_real_sets_pass_the_seed_audit():
    result = F.by_seed(R.task_sets())
    assert result["passed"] is True and result["tasks"] == {"pilot": 216, "confirm": 1640}
    assert result["used"] == [] and result["in_both_sets"] == [] and result["new_items_in_old_lists"] == []
    assert result["earlier_prompts"] == len(R.used_prompts())


def test_main_writes_the_report_of_a_set_and_fails_on_a_match(tmp_path, monkeypatch, capsys):
    data, sets = tmp_path / "data", fake_sets()
    data_folder(data, "n6", [other_prompt(0)])
    monkeypatch.setattr(F, "_load_tokenizer", lambda model: TOK)
    monkeypatch.setattr(F, "task_sets", lambda: sets)
    monkeypatch.setattr(F, "used_prompts", lambda: set())
    F.main(["--set", "pilot", "--data", str(data)])
    assert json.loads((data / "r12p" / "freshness.json").read_text(encoding="utf-8"))["passed"] is True
    assert '"passed": true' in capsys.readouterr().out
    data_folder(data, "n6-notice", [build_prompt(TOK, sets["pilot"]["free"][0])])
    with pytest.raises(SystemExit) as stop:
        F.main(["--set", "confirm", "--data", str(data)])
    assert stop.value.code == 1
    assert json.loads((data / "r12c" / "freshness.json").read_text(encoding="utf-8"))["passed"] is False
