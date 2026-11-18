"""A last line torn by a kill or a power cut in the middle of a write (prio.r12.torn): it is
moved to `<file>.torn`, the file continues from its last whole line, records and generations stay
paired, and the resume finishes with the records of an uninterrupted run."""

import json
import re

import pytest

from prio.r12 import stream as S
from prio.r12 import torn as T
from r12_helpers import R12Tok
from test_r12_stream import plan, run

STAMP = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d"


def lines_of(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def rows_of(path) -> list[dict]:
    return [json.loads(line) for line in lines_of(path)]


# --- the tail of one file ----------------------------------------------------------------------------------

def test_a_torn_last_line_is_moved_aside_and_the_file_continues_from_its_last_whole_line(tmp_path):
    path = tmp_path / "generations.jsonl"
    whole = b'{"a": 1}\n{"a": 2}\n'
    path.write_bytes(whole + b'{"a": 3, "tokens": [1, 2')  # the write never finished: no line break
    repairs = T.repair_tail(path)
    assert path.read_bytes() == whole
    assert len(repairs) == 1 and set(repairs[0]) == {"file", "bytes", "time", "what"}
    assert (repairs[0]["file"], repairs[0]["bytes"], repairs[0]["what"]) == ("generations.jsonl", 24, "torn last line")
    assert re.fullmatch(STAMP, repairs[0]["time"])
    kept = (tmp_path / "generations.jsonl.torn").read_bytes()
    assert kept.endswith(b'{"a": 3, "tokens": [1, 2\n') and re.match(rb"# " + STAMP.encode() + rb" torn last line of generations.jsonl, 24 bytes\n", kept)
    assert T.repair_tail(path) == [] and path.read_bytes() == whole  # nothing more to repair

    path.write_bytes(whole + b'{"a": 3, "tokens": [1, 2\n')  # a line break after a piece that does not parse
    assert [r["bytes"] for r in T.repair_tail(path)] == [25] and path.read_bytes() == whole
    assert (tmp_path / "generations.jsonl.torn").read_bytes().count(b"# ") == 2  # appended: nothing is deleted

    path.write_bytes(whole + b'{"a": 3}{"a": 4}\n')  # two rows glued together by an append after a torn write
    assert len(T.repair_tail(path)) == 1 and path.read_bytes() == whole
    path.write_bytes(b'{"a": ')  # nothing whole at all
    assert len(T.repair_tail(path)) == 1 and path.read_bytes() == b""


def test_a_whole_file_and_a_missing_file_are_left_alone(tmp_path):
    path = tmp_path / "records.partial.jsonl"
    assert T.repair_tail(path) == [] and not path.exists()
    for content in (b"", b'{"a": 1}\n', b'{"a": 1}\n{"b": [1, 2]}\n', b'{"a": 1}\n\n'):
        path.write_bytes(content)
        assert T.repair_tail(path) == [] and path.read_bytes() == content
    assert not (tmp_path / "records.partial.jsonl.torn").exists()


# --- records and generations stay paired ---------------------------------------------------------------------

def _pair(tmp_path, records: list[dict], generations: list[dict]):
    (tmp_path / "records.partial.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (tmp_path / "generations.jsonl").write_text("".join(json.dumps(g) + "\n" for g in generations), encoding="utf-8")


def test_an_unpaired_last_row_is_dropped_back_so_that_the_run_is_made_again(tmp_path):
    key = lambda i, t="event": {"task_id": f"t{i}", "run_type": t}  # noqa: E731
    _pair(tmp_path, [key(0), key(1)], [key(0), key(1), key(2)])  # killed between the two writes of the third run
    repairs = T.repair_stream_folder(tmp_path)
    assert [(r["file"], r["what"]) for r in repairs] == [("generations.jsonl", "generation row without its record")]
    assert rows_of(tmp_path / "generations.jsonl") == [key(0), key(1)] and rows_of(tmp_path / "records.partial.jsonl") == [key(0), key(1)]
    assert json.loads((tmp_path / "generations.jsonl.torn").read_text(encoding="utf-8").splitlines()[-1]) == key(2)
    assert T.repair_stream_folder(tmp_path) == []

    _pair(tmp_path, [key(0), key(1)], [key(0), key(1)])
    with open(tmp_path / "records.partial.jsonl", "a", encoding="utf-8") as f:
        f.write('{"task_id": "t2", "run_ty')  # the record's line torn; its generation row was written whole
    with open(tmp_path / "generations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(key(2)) + "\n")
    repairs = T.repair_stream_folder(tmp_path)
    assert [(r["file"], r["what"]) for r in repairs] == [("records.partial.jsonl", "torn last line"),
                                                        ("generations.jsonl", "generation row without its record")]
    assert rows_of(tmp_path / "generations.jsonl") == rows_of(tmp_path / "records.partial.jsonl") == [key(0), key(1)]


def test_the_paired_arms_files_are_repaired_by_their_own_key(tmp_path):
    key = lambda i, paired="event": {"episode_id": f"e{i}", "condition": "on", "paired": paired}  # noqa: E731
    _pair(tmp_path, [key(0), key(0, "nominal"), key(1)], [key(0), key(0, "nominal")])
    with open(tmp_path / "generations.jsonl", "a", encoding="utf-8") as f:
        f.write('{"episode_id": "e1", "condition": "on", "pai')  # prio.eval.run writes the record first: its generation is torn
    repairs = T.repair_paired_folder(tmp_path)
    assert [(r["file"], r["what"]) for r in repairs] == [("generations.jsonl", "torn last line"),
                                                        ("records.partial.jsonl", "record without its generation row")]
    assert rows_of(tmp_path / "records.partial.jsonl") == rows_of(tmp_path / "generations.jsonl") == [key(0), key(0, "nominal")]
    assert T.repair_paired_folder(tmp_path) == [] and T.repair_paired_folder(tmp_path / "not there") == []


def test_files_that_do_not_pair_beyond_one_cut_are_left_for_a_reading(tmp_path):
    key = lambda i: {"task_id": f"t{i}", "run_type": "event"}  # noqa: E731
    _pair(tmp_path, [key(0)], [key(0), key(1), key(2), key(3), key(4)])  # not what one kill can leave behind
    before = (tmp_path / "generations.jsonl").read_bytes()
    with pytest.raises(T.TornError, match="do not pair"):
        T.repair_stream_folder(tmp_path)
    assert (tmp_path / "generations.jsonl").read_bytes() == before and not (tmp_path / "generations.jsonl.torn").exists()


# --- the resume of the stream runner ---------------------------------------------------------------------------

PLANS = [plan(0, "ctx", 20), plan(1, "time", 24), plan(2, "energy", 30, family="code")]


def reference(tmp_path, tok, monkeypatch, tiny_model, kind="channel"):
    records, _ = run(kind, tmp_path / "whole", tok, PLANS, monkeypatch, tiny_model)
    return records, tmp_path / "whole" / "out"


@pytest.mark.parametrize("torn_file", ["generations.jsonl", "records.partial.jsonl", "between the two writes"])
def test_a_resume_after_a_torn_write_finishes_with_the_records_of_an_uninterrupted_run(tmp_path, monkeypatch, tiny_model, torn_file):
    tok = R12Tok()
    whole, source = reference(tmp_path, tok, monkeypatch, tiny_model)
    run("channel", tmp_path / "cut", tok, PLANS[:2], monkeypatch, tiny_model)  # two tasks done, then the third is killed mid-write
    out = tmp_path / "cut" / "out"
    generation = next(line for line in lines_of(source / "generations.jsonl") if '"r12p-code-00002", "run_type": "event"' in line)
    record = next(line for line in lines_of(source / "records.partial.jsonl") if '"r12p-code-00002"' in line and '"run_type": "event"' in line)
    with open(out / "generations.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(generation[:len(generation) // 2] if torn_file == "generations.jsonl" else generation + "\n")
    if torn_file == "records.partial.jsonl":
        with open(out / "records.partial.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(record[:len(record) // 2])

    store = S._Store(out)  # what the reviewer's probe did: the resume opens the folder
    assert len(store.records) == 6 and len(store.repairs) == {"generations.jsonl": 1, "records.partial.jsonl": 2, "between the two writes": 1}[torn_file]
    resumed, script = run("channel", tmp_path / "cut", tok, PLANS, monkeypatch, tiny_model)
    assert len(script.seen) == 3 and resumed == whole  # only the third task is run, and the records are those of the whole run
    keys = [(g["task_id"], g["run_type"]) for g in rows_of(out / "generations.jsonl")]
    assert len(keys) == len(set(keys)) == 9 and rows_of(out / "generations.jsonl") == rows_of(source / "generations.jsonl")
    assert rows_of(out / "records.partial.jsonl") == rows_of(source / "records.partial.jsonl")
    assert (out / "records.jsonl").read_bytes() == (source / "records.jsonl").read_bytes()
    torn = sorted(p.name for p in out.iterdir() if p.name.endswith(".torn"))
    assert torn == {"generations.jsonl": ["generations.jsonl.torn"], "records.partial.jsonl": ["generations.jsonl.torn", "records.partial.jsonl.torn"],
                    "between the two writes": ["generations.jsonl.torn"]}[torn_file]


def test_a_torn_first_pass_of_the_notice_model_is_repaired_and_the_run_is_the_same(tmp_path, monkeypatch, tiny_model):
    tok = R12Tok()
    whole, source = reference(tmp_path, tok, monkeypatch, tiny_model, "notice")
    run("notice", tmp_path / "cut", tok, PLANS[:2], monkeypatch, tiny_model)
    out = tmp_path / "cut" / "out"
    third = lines_of(source / "pass1.jsonl")[2]
    with open(out / "pass1.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(third[:40])
    resumed, _ = run("notice", tmp_path / "cut", tok, PLANS, monkeypatch, tiny_model)
    assert resumed == whole and rows_of(out / "pass1.jsonl") == rows_of(source / "pass1.jsonl")
    assert (out / "pass1.jsonl.torn").exists() and (out / "records.jsonl").read_bytes() == (source / "records.jsonl").read_bytes()
