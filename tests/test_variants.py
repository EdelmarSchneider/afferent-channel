"""Variant datasets for the text and inserted-embedding baselines (prio.datagen.variants, S3)."""

import json

import numpy as np
import pytest

from helpers import CALL, FAKE_MS, FAKE_TOK, enc
from prio.datagen import variants as V
from prio.datagen.episodes import Episode, read_jsonl, reaction_start, write_jsonl
from prio.signals import signal_set
from prio.tasks import Task
from prio.train import IGNORE, collate
from prio.trajectories import sample_trajectory

SPECS = signal_set(6)
NAMES = [s.name for s in SPECS]
BY_NAME = {s.name: s for s in SPECS}
INSERT_ID = 99_999


class WordTok:
    """FAKE_TOK whose single-token decodes start with a space, like the real tokenizer's word
    pieces, so teacher.valid_cut_points accepts every word start."""

    eos_token_id = FAKE_TOK.eos_token_id

    def encode(self, text):
        return FAKE_TOK.encode(text)

    def decode(self, ids, **_):
        return " " + FAKE_TOK.decode(ids)


class ChatTok(WordTok):
    """WordTok with a whitespace chat template, for tasks.build_prompt."""

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Out(FAKE_TOK.encode(text))

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, enable_thinking=True,
                            tokenize=False):
        return (f"<|im_start|>system {messages[0]['content']} <|im_end|> "
                f"<|im_start|>user {messages[1]['content']} <|im_end|> <|im_start|>assistant")


TOK = WordTok()
PROMPT = enc("<sys> system prompt <user> explain tides <eos> <assistant>")
N_PROMPT = len(PROMPT)
CLOSE = FAKE_MS.think_close


def run_ids(call=None, n_think=40, n_answer=40):
    words = ["<think>"] + [f"t{i}" for i in range(n_think)] + ["</think>"] + [f"a{i}" for i in range(n_answer)]
    return enc(" ".join(words) + (" " + call if call else "") + " <eos>")


def traj(length, rng, signal=None, t_cross=None, t_recover=None):
    event = NAMES.index(signal) if signal else None
    return sample_trajectory(SPECS, length, rng, event=event, t_cross=t_cross, t_recover=t_recover).values


def ep(eid, task, kind, target, signals, split="train", **fields):
    base = dict(episode_id=eid, task_id=task, family="explain", split=split, kind=kind, signal=None, phase=None,
                prompt_ids=PROMPT, target_ids=target, signals=signals, t_cross=None, t_decision=None, templated=False)
    base.update(fields)
    return Episode(**base)


WEB = CALL % ("web_search", '{"query":"q"}')
LOCAL = CALL % ("local_lookup", '{"query":"q"}')


def reaction_pair(i, signal, kind, rng, cut=30):
    """A train reaction episode and its no-event twin over the same base run."""
    base = run_ids(WEB if kind == "gating_event" else None)
    if kind == "gating_event":
        d = base.index(FAKE_MS.tool_open)
        reaction = enc(f"I am offline so I use the local store {LOCAL} <eos>")
        target, t_cross, t_decision, diverge, phase = base[:d] + reaction, d - 30, d, d, "answer"
        signals = traj(N_PROMPT + len(target), rng, signal, N_PROMPT + t_cross)
    else:
        cut = 1 if kind == "starts_past" else cut
        reaction = enc(f"my {signal} is past its limit so I act now <eos>")
        target, t_cross, t_decision, diverge = base[:cut] + reaction, cut - 1, None, cut
        phase = "thinking" if cut < base.index(CLOSE) else "answer"
        signals = traj(N_PROMPT + len(target), rng, signal, 0 if kind == "starts_past" else N_PROMPT + t_cross)
    event = ep(f"train-{i:03d}", f"task-{i}", kind, target, signals, signal=signal, phase=phase, t_cross=t_cross,
               t_decision=t_decision, t_diverge=diverge)
    twin = ep(f"train-{i + 500:03d}", f"task-{i}", "nominal", base, traj(N_PROMPT + len(base), rng), t_diverge=diverge)
    return event, twin


def make_source(root, n_nominal=6):
    rng = np.random.default_rng(0)
    pairs = [reaction_pair(0, "ctx", "event", rng, cut=30), reaction_pair(1, "energy", "event", rng, cut=45),
             reaction_pair(2, "think", "event", rng, cut=25), reaction_pair(3, "time", "event", rng, cut=55),
             reaction_pair(4, "ctx", "starts_past", rng), reaction_pair(5, "net", "gating_event", rng),
             reaction_pair(6, "ram", "gating_event", rng)]
    gated = run_ids(WEB)
    d = gated.index(FAKE_MS.tool_open)
    plain = run_ids()
    train = [e for e, _ in pairs]
    train.append(ep("train-100", "task-100", "recovery", gated, traj(N_PROMPT + len(gated), rng, "net", N_PROMPT + 20,
                                                                      N_PROMPT + 35),
                    signal="net", phase="answer", t_cross=20, t_decision=d))
    train.append(ep("train-101", "task-101", "gating_unused", plain, traj(N_PROMPT + len(plain), rng, "ram", N_PROMPT + 40),
                    signal="ram", t_cross=40))
    train += [ep(f"train-2{j:02d}", f"task-2{j:02d}", "nominal", plain, traj(N_PROMPT + len(plain), rng))
              for j in range(n_nominal)]
    train += [t for _, t in pairs]

    test = []
    for j, (signal, cut) in enumerate((("ctx", 30), ("energy", 60), ("think", 25), ("time", 55))):
        test.append(ep(f"test-{j:03d}", f"tt-{j}", "event", plain, traj(N_PROMPT + len(plain), rng, signal, N_PROMPT + cut - 1),
                       split="test", signal=signal, phase="thinking" if cut < plain.index(CLOSE) else "answer",
                       t_cross=cut - 1))
    test.append(ep("test-010", "tt-10", "gating_event", gated, traj(N_PROMPT + len(gated), rng, "net", N_PROMPT + d - 30),
                   split="test", signal="net", phase="answer", t_cross=d - 30, t_decision=d))
    test.append(ep("test-011", "tt-0", "starts_past", plain, traj(N_PROMPT + len(plain), rng, "time", 0), split="test",
                   signal="time", phase="thinking", t_cross=0))  # same run as test-000, as plan_test does
    test.append(ep("test-012", "tt-12", "recovery", gated, traj(N_PROMPT + len(gated), rng, "ram", N_PROMPT + 20, N_PROMPT + 35),
                   split="test", signal="ram", phase="answer", t_cross=20, t_decision=d))
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "test.jsonl", test)
    (root / "manifest.json").write_text(json.dumps({"model": "fake", "n_signals": 6, "seed": 0}), encoding="utf-8")
    return {e.episode_id: e for e in train + test}


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src"
    return src, make_source(src)


def build(src, out, variant, **options):
    manifest = V.build_variant(src, out, variant, TOK, FAKE_MS, **options)
    files = {split: {e.episode_id: e for e in read_jsonl(out / f"{split}.jsonl")}
             for split in ("train", "test") if (out / f"{split}.jsonl").exists()}
    return manifest, files


def spans_of(e, role):
    return [(r["at"], r["end"]) for r in e.meta["variant"]["insertions"] if r["role"] == role]


def words(ids):
    return FAKE_TOK.decode(ids).split()


# --- shown values and texts -------------------------------------------------------------------

@pytest.mark.parametrize("name,value,text,past", [
    ("ctx", 0.905, "0.90", True), ("ctx", 0.895, "0.89", False), ("think", 0.095, "0.10", True),
    ("think", 0.105, "0.11", False), ("energy", 0.035, "0.04", True), ("net", 0.0, "0.00", True),
    ("net", 1.0, "1.00", False), ("time", 0.955, "0.95", True), ("ram", 0.125, "0.13", False),
])
def test_shown_value_stays_on_its_side_of_the_limit(name, value, text, past):
    spec = BY_NAME[name]
    assert V.shown(spec, np.float32(value)) == text
    assert spec.past_limit(float(text)) == past == spec.past_limit(float(np.float32(value)))


@pytest.mark.parametrize("name", NAMES)
def test_below_limit_values_sit_in_the_near_miss_band(name):
    spec, rng = BY_NAME[name], np.random.default_rng(0)
    for _ in range(50):
        value = V.below_limit_value(spec, rng)
        assert not spec.past_limit(float(V.shown(spec, value)))
        if spec.dynamics != "binary":
            gap = abs(float(V.shown(spec, value)) - spec.limit)
            assert 0.01 - 1e-9 <= gap <= 0.05 + 1e-9


def test_notice_and_status_line_texts():
    assert V.notice_text(BY_NAME["ctx"], 0.905) == " [status ctx=0.90]"
    row = np.array([0.125, 1.0, 0.645, 0.055, 1.0, 0.715], dtype=np.float32)
    assert V.status_line(SPECS, row) == " [status ctx=0.12 think=1.00 energy=0.65 time=0.05 net=1.00 ram=0.72]"


# --- insertion mechanics ----------------------------------------------------------------------

def test_insertion_shifts_later_indices_and_marks_a_harness_span():
    rng = np.random.default_rng(1)
    e, _ = reaction_pair(0, "ctx", "event", rng, cut=30)
    e.harness_spans, e.focus_spans = [[40, 42]], [[30, 33]]
    row = e.signals[N_PROMPT + 29]
    out = V.insert(e, [V.Insertion(at=30, ids=[7, 8, 9], row=row, role="crossing", signal="ctx", shown="0.90",
                                   anchor=True)], "notice")
    assert out.target_ids == e.target_ids[:30] + [7, 8, 9] + e.target_ids[30:]
    assert out.t_cross == 32 and reaction_start(out) == 33 and out.t_diverge == 33
    assert out.target_ids[reaction_start(out):] == e.target_ids[30:]
    assert out.harness_spans == [[30, 33], [43, 45]] and out.focus_spans == [[33, 36]]
    assert out.signals.shape == (N_PROMPT + len(out.target_ids), 6)
    assert np.array_equal(out.signals[N_PROMPT + 30:N_PROMPT + 33], np.repeat(row[None], 3, axis=0))
    assert np.array_equal(out.signals[N_PROMPT + 33:], e.signals[N_PROMPT + 30:])
    assert out.meta["variant"]["insertions"] == [
        {"at": 30, "end": 33, "role": "crossing", "signal": "ctx", "shown": "0.90", "anchor": True}]


def test_insertion_refuses_an_episode_that_is_already_a_variant():
    rng = np.random.default_rng(1)
    e, _ = reaction_pair(0, "ctx", "event", rng)
    once = V.insert(e, [V.Insertion(at=30, ids=[7], row=e.signals[0], role="crossing")], "notice")
    with pytest.raises(ValueError):
        V.insert(once, [V.Insertion(at=30, ids=[7], row=e.signals[0], role="crossing")], "notice")


# --- notice (B3) ------------------------------------------------------------------------------

def test_notice_goes_right_after_the_crossing_and_the_reaction_moves_with_it(source, tmp_path):
    src, eps = source
    _, files = build(src, tmp_path / "out", "notice")
    for eid, e in eps.items():
        if e.kind not in ("event", "starts_past", "gating_event"):
            continue
        out = files[e.split][eid]
        (a, b), = spans_of(out, "crossing")
        assert a == e.t_cross + 1 and out.t_cross == b - 1 and out.harness_spans == [[a, b]]
        value = e.signals[N_PROMPT + e.t_cross, NAMES.index(e.signal)]
        assert words(out.target_ids[a:b]) == words(enc(V.notice_text(BY_NAME[e.signal], value)))
        assert BY_NAME[e.signal].past_limit(float(out.meta["variant"]["insertions"][0]["shown"]))
        start = reaction_start(e)
        assert out.target_ids[reaction_start(out):] == e.target_ids[start:]  # same reaction, moved by the notice
        if e.t_decision is not None:
            assert out.t_decision == e.t_decision + (b - a)
        if e.t_diverge is not None:
            assert out.t_diverge == e.t_diverge + (b - a)


def test_harness_tokens_get_no_loss_and_the_reaction_keeps_its_weight(source, tmp_path):
    src, eps = source
    _, files = build(src, tmp_path / "out", "notice")
    train = list(files["train"].values())
    batch = collate(train, pad_id=0, n_signals=6, reaction_weight=20.0)
    for row, out in enumerate(train):
        for a, b in out.harness_spans or ():
            assert (batch["labels"][row, N_PROMPT + a:N_PROMPT + b] == IGNORE).all()
        start = reaction_start(out)
        if start is not None:
            n = len(out.target_ids)
            assert (batch["weights"][row, N_PROMPT + start:N_PROMPT + n] == 20.0).all()
            assert batch["labels"][row, N_PROMPT + start:N_PROMPT + n].tolist() == out.target_ids[start:]
            assert all(b <= start for _, b in out.harness_spans)  # the notice is context, never weighted


def test_value_twins_differ_from_their_event_only_in_the_number(source, tmp_path):
    src, eps = source
    manifest, files = build(src, tmp_path / "out", "notice")
    train = files["train"]
    value_twins = [e for e in train.values() if e.meta and spans_of(e, "value_twin")]
    assert len(value_twins) == round(7 * 0.5) == manifest["insertions"]["train"]["value_twin"]
    for twin in value_twins:
        event = next(e for e in train.values() if e.task_id == twin.task_id and e.kind != "nominal")
        source_twin = eps[twin.episode_id]
        assert twin.t_diverge == event.t_diverge
        assert spans_of(twin, "value_twin") == spans_of(event, "crossing")
        cut = twin.t_diverge
        differ = [i for i in range(cut) if twin.target_ids[i] != event.target_ids[i]]
        (a, b), = spans_of(twin, "value_twin")
        assert differ and all(a <= i < b for i in differ)  # only inside the notice: its number
        record = twin.meta["variant"]["insertions"][0]
        assert record["signal"] == event.signal and not BY_NAME[event.signal].past_limit(float(record["shown"]))
        assert twin.target_ids[cut] != event.target_ids[cut]
        assert twin.target_ids[cut] == source_twin.target_ids[source_twin.t_diverge]  # the token the event departs at
        assert twin.target_ids[b:] == source_twin.target_ids[a:]
        index = NAMES.index(event.signal)
        assert np.isclose(twin.signals[N_PROMPT + a, index], float(record["shown"]), atol=0.006)


def test_quiet_twins_and_untouched_nominals_stay_as_they_were(source, tmp_path):
    src, eps = source
    manifest, files = build(src, tmp_path / "out", "notice")
    train = files["train"]
    nominal = [e for e in train.values() if e.kind == "nominal"]
    untouched = [e for e in nominal if not e.harness_spans]
    assert len(untouched) == 13 - round(7 * 0.5) - round(6 / 3)
    for e in untouched:
        assert e.target_ids == eps[e.episode_id].target_ids and e.t_diverge == eps[e.episode_id].t_diverge
    near = [e for e in nominal if e.harness_spans and spans_of(e, "near_miss")]
    assert len(near) == round(6 / 3) == manifest["insertions"]["train"]["near_miss"]
    for e in near:
        record = e.meta["variant"]["insertions"][0]
        assert e.t_diverge is None and e.t_cross is None
        assert not BY_NAME[record["signal"]].past_limit(float(record["shown"]))


def test_recovery_gets_a_second_notice_and_gating_unused_one(source, tmp_path):
    src, eps = source
    _, files = build(src, tmp_path / "out", "notice")
    rec, unused = files["train"]["train-100"], files["train"]["train-101"]
    (a1, b1), = spans_of(rec, "crossing")
    (a2, b2), = spans_of(rec, "recovery")
    assert a1 == 21 and rec.t_cross == b1 - 1 and a2 == 36 + (b1 - a1)
    assert rec.t_decision == eps["train-100"].t_decision + (b1 - a1) + (b2 - a2)
    assert rec.meta["variant"]["insertions"][1]["shown"] == "1.00"
    assert spans_of(unused, "crossing") == [(41, 41 + (b1 - a1))] and reaction_start(unused) is None


def test_notice_counts_are_written_to_the_manifest(source, tmp_path):
    src, _ = source
    manifest, files = build(src, tmp_path / "out", "notice")
    assert manifest["variant"] == "notice" and manifest["source"] == str(src)
    assert manifest["episodes"] == {"train": 22, "test": 7}
    ins = manifest["insertions"]
    assert ins["train"] == {"crossing": 9, "recovery": 1, "value_twin": 4, "near_miss": 2}
    assert ins["test"] == {"crossing": 7, "recovery": 1}
    assert manifest["kinds"]["train"]["nominal"] == 13


def test_rerun_is_identical_and_a_variant_is_never_a_source(source, tmp_path):
    src, _ = source
    build(src, tmp_path / "a", "notice")
    first = {n: (tmp_path / "a" / n).read_bytes() for n in ("train.jsonl", "test.jsonl", "manifest.json")}
    build(src, tmp_path / "a", "notice")
    assert {n: (tmp_path / "a" / n).read_bytes() for n in first} == first
    with pytest.raises(ValueError):
        V.build_variant(tmp_path / "a", tmp_path / "b", "notice", TOK, FAKE_MS)
    with pytest.raises(ValueError):
        V.build_variant(src, src, "notice", TOK, FAKE_MS)


# --- insert (B6b) -----------------------------------------------------------------------------

def test_insert_puts_one_position_at_the_crossing_with_the_crossing_row(source, tmp_path):
    src, eps = source
    manifest, files = build(src, tmp_path / "out", "insert", insert_id=INSERT_ID)
    for eid, out in files["train"].items():
        e = eps[eid]
        for (a, b) in out.harness_spans or ():
            assert b - a == 1 and out.target_ids[a] == INSERT_ID
        if e.kind == "event":
            assert out.t_cross == e.t_cross + 1 and out.target_ids[out.t_cross] == INSERT_ID
            assert np.array_equal(out.signals[N_PROMPT + out.t_cross], e.signals[N_PROMPT + e.t_cross])
    twin = next(e for e in files["train"].values() if e.meta and spans_of(e, "value_twin"))
    (a, _), = spans_of(twin, "value_twin")
    record = twin.meta["variant"]["insertions"][0]
    row = twin.signals[N_PROMPT + a]
    assert not BY_NAME[record["signal"]].past_limit(float(row[NAMES.index(record["signal"])]))
    assert manifest["insertions"]["train"] == {"crossing": 9, "recovery": 1, "value_twin": 4, "near_miss": 2}
    with pytest.raises(ValueError):
        V.build_variant(src, tmp_path / "x", "insert", TOK, FAKE_MS)  # INSERT_ID is required


# --- periodic (B4) ----------------------------------------------------------------------------

def test_periodic_lines_follow_the_schedule_and_carry_the_values(source, tmp_path):
    src, eps = source
    manifest, files = build(src, tmp_path / "out", "periodic", periods=(16,))
    from prio.datagen.teacher import valid_cut_points

    nominal = files["train"]["train-200"]
    base = eps["train-200"]
    points = [1] + V.every_k(valid_cut_points(base.target_ids, FAKE_MS, TOK), 16, after=1)  # 1: after <think>
    assert points[:4] == [1, 20, 32, 48]
    lines = spans_of(nominal, "status")
    assert len(lines) == len(points)
    shift = 0
    for p, (a, b) in zip(points, lines):
        assert a == p + shift
        row = base.signals[N_PROMPT + p - 1]
        assert words(nominal.target_ids[a:b]) == words(enc(V.status_line(SPECS, row)))
        shift += b - a
    assert manifest["periods"]["train"] == {"16": manifest["episodes"]["train"]}


def test_periodic_reaction_follows_the_first_line_after_the_crossing(source, tmp_path):
    src, eps = source
    _, files = build(src, tmp_path / "out", "periodic", periods=(16,))
    train = files["train"]
    for eid in ("train-000", "train-001", "train-005"):
        e, out = eps[eid], train[eid]
        twin = next(t for t in train.values() if t.task_id == e.task_id and t.kind == "nominal")
        anchor = next(r for r in out.meta["variant"]["insertions"] if r.get("anchor"))
        assert out.t_cross == anchor["end"] - 1
        shown = dict(item.split("=") for item in words(out.target_ids[anchor["at"]:anchor["end"]])[1:])
        assert BY_NAME[e.signal].past_limit(float(shown[e.signal].rstrip("]")))
        start = reaction_start(out)
        reaction = e.target_ids[reaction_start(e):]
        assert out.target_ids[start:start + 3] == reaction[:3]
        # Same layout as the twin up to the decision; only the numbers in the lines differ.
        assert out.t_diverge == twin.t_diverge
        assert [s for s in out.harness_spans if s[1] <= start] == [s for s in twin.harness_spans if s[1] <= start]
        differ = [i for i in range(out.t_diverge) if out.target_ids[i] != twin.target_ids[i]]
        assert all(any(a <= i < b for a, b in out.harness_spans) for i in differ)
        assert out.target_ids[out.t_diverge] != twin.target_ids[twin.t_diverge]
    batch = collate(list(train.values()), pad_id=0, n_signals=6, reaction_weight=20.0)
    for row, out in enumerate(train.values()):
        for a, b in out.harness_spans or ():
            assert (batch["labels"][row, N_PROMPT + a:N_PROMPT + b] == IGNORE).all()


def test_periodic_drops_events_without_a_line_in_their_phase_and_counts_them(source, tmp_path):
    src, eps = source
    manifest, files = build(src, tmp_path / "out", "periodic", periods=(256,))
    train = files["train"]
    assert "train-002" not in train and "train-502" not in train  # think event at 25: no line before </think>
    assert "train-004" in train  # starts past: the turn-start line shows it
    assert manifest["dropped"]["train"]["no_line_in_phase"] == 2 * 4  # four mid-run events, each with its twin
    assert manifest["dropped"]["train"]["no_line_before_decision"] == 2 * 2


# --- prompt-state (B2) ------------------------------------------------------------------------

def test_prompt_state_writes_the_start_state_and_the_policy_into_the_system_prompt(tmp_path):
    from prio.tasks import SYSTEM_PROMPT, build_prompt

    tok, rng = ChatTok(), np.random.default_rng(3)
    task = Task("tt-0", "explain", "explain tides", None, "test")
    prompt = build_prompt(tok, task)
    plain = run_ids()
    e = Episode("test-000", "tt-0", "explain", "test", "starts_past", "time", "thinking", prompt, plain,
                traj(len(prompt) + len(plain), rng, "time", 0), 0, None, False)
    src = tmp_path / "src"
    src.mkdir()
    write_jsonl(src / "test.jsonl", [e])
    (src / "manifest.json").write_text(json.dumps({"model": "fake"}), encoding="utf-8")
    V.build_variant(src, tmp_path / "out", "prompt-state", tok, FAKE_MS)
    out, = read_jsonl(tmp_path / "out" / "test.jsonl")
    status = V.status_line(SPECS, e.signals[len(prompt) - 1])
    expected = build_prompt(tok, task, system=SYSTEM_PROMPT + V.prompt_state_text(SPECS, e.signals[len(prompt) - 1]))
    assert out.prompt_ids == expected and status.strip() in FAKE_TOK.decode(out.prompt_ids)
    assert "time >= 0.90" in V.policy_text(SPECS)
    assert out.target_ids == e.target_ids and out.t_cross == e.t_cross and not out.harness_spans
    assert out.signals.shape == (len(out.prompt_ids) + len(out.target_ids), 6)
    assert np.array_equal(out.signals[len(out.prompt_ids):], e.signals[len(prompt):])


# --- policy prompt (B3z zero-shot arm) --------------------------------------------------------

def test_policy_copies_every_test_file_of_a_notice_folder_with_the_policy_in_the_system_prompt(tmp_path):
    from dataclasses import replace

    from prio.tasks import SYSTEM_PROMPT, build_prompt

    tok, rng = ChatTok(), np.random.default_rng(3)
    task = Task("tt-0", "explain", "explain tides", None, "test")
    prompt, plain = build_prompt(tok, task), run_ids()
    e = Episode("test-000", "tt-0", "explain", "test", "event", "time", "thinking", prompt, plain,
                traj(len(prompt) + len(plain), rng, "time", 3), 3, None, False)
    e = replace(e, harness_spans=[[1, 3]])
    names = ("test.jsonl", "test_swap.jsonl", "test_near_miss.jsonl", "test_quiet.jsonl")
    src = tmp_path / "n6-notice"
    src.mkdir()
    for name in names:
        write_jsonl(src / name, [e])
    write_jsonl(src / "train.jsonl", [e])
    (src / "manifest.json").write_text(json.dumps({"variant": "notice", "source": "data/n6"}), encoding="utf-8")
    V.build_variant(src, tmp_path / "out", "policy", tok, FAKE_MS)
    expected = build_prompt(tok, task, system=SYSTEM_PROMPT + " " + V.policy_text(SPECS))
    for name in names:
        out, = read_jsonl(tmp_path / "out" / name)
        assert out.prompt_ids == expected
        assert out.target_ids == e.target_ids and out.t_cross == e.t_cross and out.harness_spans == e.harness_spans
        assert out.signals.shape == (len(expected) + len(plain), 6)
        assert np.array_equal(out.signals[len(expected):], e.signals[len(prompt):])
    assert not (tmp_path / "out" / "train.jsonl").exists()  # zero-shot: nothing to train on
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "notice" and manifest["source"] == "data/n6"  # dataset_tasks still finds the tasks
    assert manifest["policy_prompt"] == V.policy_text(SPECS) and manifest["policy_source"] == str(src)
    (src / "manifest.json").write_text(json.dumps({"variant": "insert", "source": "data/n6"}), encoding="utf-8")
    with pytest.raises(ValueError, match="notice"):
        V.build_variant(src, tmp_path / "out2", "policy", tok, FAKE_MS)


def test_tokenizer_model_is_read_from_a_source_or_a_variant_manifest():
    assert V.manifest_model({"model": "Qwen/Qwen3-1.7B"}) == "Qwen/Qwen3-1.7B"
    assert V.manifest_model({"variant": "notice", "source_manifest": {"model": "Qwen/Qwen3-1.7B"}}) == "Qwen/Qwen3-1.7B"


# --- test-only files: swap, near_miss, quiet --------------------------------------------------

def test_swap_injects_another_signal_past_its_limit_at_the_cut(source, tmp_path):
    from prio.eval.run import _swap_signal

    src, eps = source
    manifest, files = build(src, tmp_path / "out", "swap")
    assert "train" not in files
    order = [e.episode_id for e in read_jsonl(src / "test.jsonl")]
    test = files["test"]
    assert set(test) == {i for i in order if eps[i].kind == "event"}
    for eid, out in test.items():
        e = eps[eid]
        thinking_open = CLOSE not in e.target_ids[:e.t_cross + 1]
        swap = _swap_signal(e, SPECS, order.index(eid), thinking_open)
        assert out.kind == "swap" and out.signal == swap != e.signal
        assert out.meta["variant"]["original_signal"] == e.signal
        (a, b), = spans_of(out, "swap")
        assert a == e.t_cross + 1 and out.t_cross == b - 1
        record = out.meta["variant"]["insertions"][0]
        assert record["signal"] == swap and BY_NAME[swap].past_limit(float(record["shown"]))
        assert out.target_ids[:a] == e.target_ids[:a]
    assert manifest["episodes"] == {"test": 4}


def test_near_miss_and_quiet_test_files(source, tmp_path):
    src, eps = source
    _, near = build(src, tmp_path / "near", "near_miss")
    _, quiet = build(src, tmp_path / "quiet", "quiet")
    paired = {i for i, e in eps.items() if e.split == "test" and e.kind in ("event", "gating_event")}
    assert set(near["test"]) == set(quiet["test"]) == paired
    for eid in paired:
        e, n, q = eps[eid], near["test"][eid], quiet["test"][eid]
        (a, b), = spans_of(n, "near_miss")
        record = n.meta["variant"]["insertions"][0]
        assert a == e.t_cross + 1 and n.t_cross == b - 1 and n.kind == e.kind and n.signal == e.signal
        assert record["signal"] == e.signal and not BY_NAME[e.signal].past_limit(float(record["shown"]))
        assert q.target_ids == e.target_ids and q.t_cross == e.t_cross and not q.harness_spans
        if e.t_decision is not None:
            assert n.t_decision == e.t_decision + (b - a)


def test_insert_carrier_for_test_only_files(source, tmp_path):
    src, eps = source
    _, files = build(src, tmp_path / "out", "near_miss", carrier="insert", insert_id=INSERT_ID)
    for eid, out in files["test"].items():
        assert out.target_ids[out.t_cross] == INSERT_ID and out.harness_spans == [[out.t_cross, out.t_cross + 1]]
        signal = eps[eid].signal
        assert not BY_NAME[signal].past_limit(float(out.signals[N_PROMPT + out.t_cross, NAMES.index(signal)]))
