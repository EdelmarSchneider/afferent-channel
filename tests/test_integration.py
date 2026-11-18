"""Integration of the program-A code (WP7): the pieces of WP1-WP6 working together on tiny models.

Harness marks at evaluation (segment delivery), variant files meeting harness mode, the limits
evaluation through `run.evaluate`, and end-to-end smoke runs (build -> train -> evaluate)."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from helpers import FAKE_TOK, enc, make_episode
from prio.channel import ChannelModel
from prio.signals import CORE_SIGNALS

TARGET = "a b c d e f g h i j"


@pytest.fixture(autouse=True)
def gsm8k_stub(monkeypatch):
    """`load_tasks` without the GSM8K download (as tests/test_tasks.py)."""
    import prio.tasks as tasks_module

    monkeypatch.setattr(tasks_module, "_gsm8k", lambda split: [(f"{split} question {i}: what is {i} + 1?", f"work\n#### {i + 1}")
                                                               for i in range(100)])


def vocab_size(margin: int = 16, toks=()) -> int:
    return max(max(len(t.words) for t in (FAKE_TOK, *toks)) + margin, 100)


def tiny_gpt2(vocab: int | None = None, toks=()):
    """Tiny random GPT-2 whose vocabulary covers every word so far of FAKE_TOK and of the other
    whitespace tokenizers `toks` (encode all texts a test uses BEFORE building it); every id
    decodes (fillers up to the vocabulary size)."""
    torch.manual_seed(0)
    n = vocab or vocab_size(toks=toks)
    for t in (FAKE_TOK, *toks):
        while len(t.words) < n:
            t.encode(f"filler{len(t.words)}")
    config = GPT2Config(vocab_size=n, n_positions=1024, n_embd=32, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=0)
    return GPT2LMHeadModel(config).eval()


def segment_model(base):
    """Segment delivery with a segment vector (100 everywhere) that stands out of any embedding,
    and the list that receives, for every prompt pass of the base (more than one position), the
    positions that carry it, per batch row (left padding included)."""
    model = ChannelModel(base, 6, delivery="segment")
    with torch.no_grad():
        model.segment.fill_(100.0)
    marks = []

    def hook(module, args, kwargs):
        embeds = kwargs["inputs_embeds"]
        if embeds.shape[1] > 1:
            marks.append([[i for i, m in enumerate(row) if m] for row in (embeds.abs().amax(-1) > 50).tolist()])

    base.transformer.register_forward_pre_hook(hook, with_kwargs=True)
    return model, marks


def notice_variant(ep, notice: str, **kw):
    """`ep` with `notice` inserted right after its crossing token, as WP3 writes it: harness span
    over the notice, t_cross on its last token, rows copied, later indices shifted."""
    words = enc(notice)
    at = ep.t_cross + 1
    row = len(ep.prompt_ids) + at
    signals = np.concatenate([ep.signals[:row], np.repeat(ep.signals[row - 1:row], len(words), axis=0), ep.signals[row:]])
    fields = dict(target_ids=ep.target_ids[:at] + words + ep.target_ids[at:], signals=signals,
                  t_cross=at + len(words) - 1, harness_spans=[[at, at + len(words)]],
                  t_decision=None if ep.t_decision is None else ep.t_decision + len(words))
    fields.update(kw)
    return replace(ep, **fields)


def harness_files():
    ev = make_episode(episode_id="ev", signal="ctx", kind="event", phase="answer", t_cross=3, target_ids=enc(TARGET))
    ga = make_episode(episode_id="ga", signal="net", kind="gating_event", phase="answer", t_cross=1, t_decision=6,
                      target_ids=enc(TARGET))
    return {"event": [notice_variant(ev, "[status ctx=0.95]"), notice_variant(ga, "[status net=0.00]")],
            "quiet": [ev, ga],
            "near_miss": [notice_variant(ev, "[status ctx=0.87]"), notice_variant(ga, "[status net=1.00]")],
            "swapped": [notice_variant(ev, "[status energy=0.04]", signal="energy")]}


def span_positions(ep, n_forced: int) -> list[int]:
    """Prompt positions of the episode's harness tokens inside its forced prefix."""
    n = len(ep.prompt_ids)
    return [n + i for a, b in ep.harness_spans or () for i in range(a, b) if i < n_forced]


# --- harness marks at evaluation (segment delivery) -------------------------------------------

def test_harness_mode_marks_exactly_the_inserted_notices_of_each_pairing():
    import prio.eval.run as run_module

    files = harness_files()
    model, marks = segment_model(tiny_gpt2())
    records = run_module.evaluate_harness(model, FAKE_TOK, _ms(), CORE_SIGNALS, files, log_every=0, horizon=3)
    by_name = {name: {ep.episode_id: ep for ep in eps} for name, eps in files.items()}
    expected = []
    for r in records:
        src = by_name["event" if r.paired == "event" else r.paired][r.episode_id]
        cut = src.t_decision if src.kind == "gating_event" else src.t_cross + 1
        expected.append(span_positions(src, cut))
    assert [m[0] for m in marks] == expected
    assert [(r.episode_id, r.paired) for r in records] == [("ev", "event"), ("ev", "quiet"), ("ev", "near_miss"),
                                                           ("ev", "swapped"), ("ga", "event"), ("ga", "quiet"),
                                                           ("ga", "near_miss")]
    assert expected[1] == [] and expected[0] == [3 + 4 + k for k in range(len(enc("[status ctx=0.95]")))]

    # Batched: the same records, and every prompt row still marks its own notice (left padding shifts it).
    marks.clear()
    batched = run_module.evaluate_harness(model, FAKE_TOK, _ms(), CORE_SIGNALS, files, log_every=0, horizon=3,
                                          batch_size=2)
    assert batched == records
    assert sum(len(m) for m in marks) == len(records) and all(len(m) <= 2 for m in marks)
    shifted = sorted(len(p) for p in expected)
    assert sorted(len(row) for m in marks for row in m) == shifted


def test_standard_and_rules_evaluations_mark_harness_tokens_and_nothing_else():
    import prio.eval.run as run_module

    ev = harness_files()["event"][0]
    plain = harness_files()["quiet"][0]
    thinking = make_episode(episode_id="th", task_id="t0", signal="think", kind="event", phase="thinking", t_cross=2,
                            target_ids=enc("<think> a b c d e f"))
    variant = notice_variant(thinking, "[status think=0.05]")
    model, marks = segment_model(tiny_gpt2())
    with run_module.fixed_horizon(3):
        records = run_module.evaluate(model, FAKE_TOK, _ms(), CORE_SIGNALS, [ev, plain], ("on",), log_every=0)
    notice = span_positions(ev, ev.t_cross + 1)
    assert len(marks) == len(records) == 8
    assert [m[0] for m in marks[:4]] == [notice] * 4  # every pairing of the variant episode runs its prefix
    assert all(m == [[]] for m in marks[4:])  # the original: no harness token, nothing marked

    # Rules mode: the harness's forced </think> is marked too (condition "on" so the vector is applied).
    ms = _ms()
    marks.clear()
    run_module.evaluate_rules(model, FAKE_TOK, ms, CORE_SIGNALS, [variant], {}, conditions=("on",), horizon=3, log_every=0)
    close = run_module.forced_close_ids(FAKE_TOK, ms)
    n = len(variant.prompt_ids) + variant.t_cross + 1
    assert marks == [[span_positions(variant, variant.t_cross + 1) + list(range(n, n + len(close)))]]


def test_spoofed_status_text_is_never_marked_but_the_true_notice_is():
    from test_spoof import TASK, TOK  # TOK: a chat template over FAKE_TOK words

    import prio.eval.run as run_module
    from prio.eval.spoof import cell_prompt, run_spoof
    from prio.tasks import build_prompt

    prompt = build_prompt(TOK, TASK)
    target = TOK.encode("<think> a b c d e f g h")
    ev = make_episode(episode_id="ev", task_id=TASK.task_id, signal="energy", kind="event", phase="thinking", t_cross=3,
                      prompt_ids=prompt, target_ids=target,
                      signals=np.full((len(prompt) + len(target), 6), 0.5, dtype=np.float32))
    ev = notice_variant(ev, "[status energy=0.04]")
    cells = ["P1/F3"]
    for content in ("none", "F3", "S2-F3"):  # every word the runs will need, before the model exists
        cell_prompt(TOK, TASK, "P1", content, False, "energy")
    model, marks = segment_model(tiny_gpt2(toks=[TOK]))
    records = run_spoof(model, TOK, _ms(), CORE_SIGNALS, [ev], {TASK.task_id: TASK}, trigger_cells=cells,
                        suppress_cells=["P1/S2-F3"], conditions=("on",), horizon=3, every=1, log_every=0)
    keys = [(r.arm, r.cell) for r in records]
    assert keys == [("trigger", "P1/none"), ("trigger", "P1/F3"), ("suppress", "P1/none"), ("suppress", "P1/S2-F3")]
    assert marks[0] == marks[1] == [[]]  # trigger runs: only spoofed text, no harness token
    for m, content in zip(marks[2:], ("none", "S2-F3")):
        spoofed, _ = cell_prompt(TOK, TASK, "P1", content, False, "energy")
        n = len(spoofed)
        assert m == [[n + i for a, b in ev.harness_spans for i in range(a, b)]]  # the true notice, moved with the prompt


def _ms():
    from helpers import FAKE_MS

    return FAKE_MS


# --- R10 evaluation through run.evaluate ------------------------------------------------------

def _limits_reference(model, tok, ms, base_specs, test_eps, conditions, seed=0, out_path=None, every=1):
    """`prio.eval.limits_eval.evaluate_limits` as WP6 wrote it (a copy of `run.evaluate`'s loop),
    kept here as the reference of the refactor that made it call `run.evaluate`."""
    import json
    from dataclasses import asdict

    from prio.eval import run as base_eval
    from prio.eval.limits_eval import _kind, classify_limits, episode_specs
    from prio.trajectories import sample_trajectory

    chosen = set(base_eval.quick_subset(test_eps, every))
    records = []
    stops = base_eval.stop_ids(tok, ms)
    for index, ep in enumerate(test_eps):
        if index not in chosen:
            continue
        specs = episode_specs(ep, base_specs)
        names = [s.name for s in specs]
        think = base_eval._think_index(specs)
        nominal_think = specs[think].nominal if think is not None else 1.0
        at_decision = _kind(ep) in base_eval._AT_DECISION
        prefix = ep.target_ids[:ep.t_decision if at_decision else ep.t_cross + 1]
        prompt = ep.prompt_ids + prefix
        closed_at = prefix.index(ms.think_close) if ms.think_close in prefix else None
        think_end = None if closed_at is None else len(ep.prompt_ids) + closed_at + 1
        swap = base_eval._swap_signal(ep, specs, index, closed_at is None) if ep.kind == "event" else None

        def trajectory(paired, total):
            if paired == "event":
                return base_eval._stored_trajectory(ep, total, think, nominal_think, closed_at, ms)
            if paired == "nominal":
                return sample_trajectory(specs, total, np.random.default_rng([seed, index]), think_end=think_end).values
            if paired == "near_miss":
                rng = np.random.default_rng([seed, index, 1])
                return sample_trajectory(specs, total, rng, near_miss=True, think_end=think_end).values
            rng = np.random.default_rng([seed, index, 2])
            return sample_trajectory(specs, total, rng, event=names.index(swap), t_cross=len(prompt) - 1,
                                     think_end=think_end).values

        for condition in conditions:
            if ep.kind == "limit_twin" and condition != "on":
                continue
            pairings = ["event"]
            if condition == "on" and ep.kind in base_eval._PAIRED_KINDS:
                pairings += ["nominal", "near_miss"] + (["swapped"] if swap else [])
            for paired in pairings:
                judged = replace(ep, signal=swap, kind="swap") if paired == "swapped" else ep
                horizon = base_eval._horizon(at_decision, judged.signal)
                values = trajectory(paired, len(prompt) + horizon)
                signal_fn = base_eval.follow_thinking(values, len(prompt), think if closed_at is None else None,
                                                      nominal_think, ms.think_close)
                with base_eval._condition(model, condition):
                    gen = base_eval._run(model, prompt, values, horizon, stops, signal_fn)
                records.append(classify_limits(judged, paired, condition, gen, tok, ms, specs))
                if out_path is not None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(records[-1])) + "\n")
                    with open(out_path.with_name("generations.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"episode_id": ep.episode_id, "condition": condition, "paired": paired,
                                            "signal": judged.signal, "stop_reason": gen.stop_reason,
                                            "token_ids": [int(t) for t in gen.token_ids]}) + "\n")
    return records


def test_the_limits_evaluation_through_run_evaluate_keeps_its_records(tmp_path):
    from test_limits import _transform, eval_source

    from helpers import arm
    from prio.eval import limits_eval
    from prio.eval import run as run_module

    episodes, _, _ = _transform(eval_source(12), split="test", test_reports=0)
    assert {e.kind for e in episodes} >= {"event", "gating_event", "limit_twin"}
    model = ChannelModel(tiny_gpt2(), 6, delivery="wide", projector="levels")
    arm(model)  # the channel moves the tokens: a different trajectory shows in the generations
    read = lambda folder, name: (tmp_path / folder / name).read_text(encoding="utf-8").splitlines()  # noqa: E731
    with run_module.fixed_horizon(6):
        expected = _limits_reference(model, FAKE_TOK, _ms(), CORE_SIGNALS, episodes, ("on", "zeroed"),
                                     out_path=tmp_path / "ref" / "r.jsonl")
        one = limits_eval.evaluate_limits(model, FAKE_TOK, _ms(), CORE_SIGNALS, episodes, ("on", "zeroed"),
                                          out_path=tmp_path / "one" / "r.jsonl", log_every=0)
        batched = limits_eval.evaluate_limits(model, FAKE_TOK, _ms(), CORE_SIGNALS, episodes, ("on", "zeroed"),
                                              out_path=tmp_path / "two" / "r.jsonl", log_every=0, batch_size=3)
    assert one == expected and batched == expected
    assert {r.kind for r in expected} >= {"limit_twin", "swap"}
    assert read("one", "r.jsonl") == read("ref", "r.jsonl")
    assert read("one", "generations.jsonl") == read("ref", "generations.jsonl")
    assert sorted(read("two", "generations.jsonl")) == sorted(read("ref", "generations.jsonl"))


def test_the_standard_evaluation_is_unchanged_by_the_episode_hooks(tmp_path):
    from helpers import arm
    from prio.eval import run as run_module
    from prio.eval.limits_eval import classify_limits

    ev = make_episode(episode_id="ev", signal="ctx", kind="event", phase="answer", t_cross=3, target_ids=enc(TARGET))
    model = ChannelModel(tiny_gpt2(), 6, delivery="wide", projector="levels")
    arm(model)
    with run_module.fixed_horizon(5):
        plain = run_module.evaluate(model, FAKE_TOK, _ms(), CORE_SIGNALS, [ev], ("on", "zeroed"), log_every=0)
        hooked = run_module.evaluate(model, FAKE_TOK, _ms(), CORE_SIGNALS, [ev], ("on", "zeroed"), log_every=0,
                                     setup=lambda ep, conditions: run_module.EpisodeSetup(CORE_SIGNALS, ep.kind, conditions),
                                     judge=classify_limits)
    assert hooked == plain and len(plain) == 5


# --- test-only variant files meet harness mode ------------------------------------------------

def _notice_with_test_files(tmp_path, carrier="notice"):
    """Source dataset (test_variants.make_source), its notice / insert variant, and the three
    test-only files written into the variant folder."""
    from test_variants import TOK as WORD_TOK
    from test_variants import make_source

    from prio.datagen.variants import build_variant

    src, variant = tmp_path / "n6", tmp_path / f"n6-{carrier}"
    make_source(src)
    insert_id = 99_999 if carrier == "insert" else None
    build_variant(src, variant, carrier, WORD_TOK, _ms(), insert_id=insert_id)
    for name in ("swap", "near_miss", "quiet"):
        build_variant(src, None, name, WORD_TOK, _ms(), into=variant)
    return src, variant


def test_test_only_files_land_beside_the_event_file_where_harness_mode_finds_them(tmp_path):
    import json

    from prio.datagen.episodes import read_jsonl
    from prio.eval.run import harness_files

    src, notice = _notice_with_test_files(tmp_path)
    assert sorted(p.name for p in notice.glob("*.jsonl")) == ["test.jsonl", "test_near_miss.jsonl", "test_quiet.jsonl",
                                                             "test_swap.jsonl", "train.jsonl"]
    files = harness_files(notice)  # no --quiet-data, no --pairing-file
    assert files == {"event": notice / "test.jsonl", "quiet": notice / "test_quiet.jsonl",
                     "near_miss": notice / "test_near_miss.jsonl", "swapped": notice / "test_swap.jsonl"}
    manifest = json.loads((notice / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "notice" and set(manifest["test_only"]) == {"swap", "near_miss", "quiet"}
    assert manifest["test_only"]["swap"]["episodes"]["test"] == len(read_jsonl(files["swapped"]))
    assert manifest["test_only"]["swap"]["file"] == "test_swap.jsonl"
    swap = read_jsonl(files["swapped"])[0]
    assert swap.kind == "swap" and "[status" in " ".join(FAKE_TOK.words[i] for i in swap.target_ids)

    # The insert folder gets the insert carrier and its own INSERT_ID without being told.
    _, insert = _notice_with_test_files(tmp_path / "ins", carrier="insert")
    near = read_jsonl(insert / "test_near_miss.jsonl")[0]
    assert near.target_ids[near.harness_spans[0][0]:near.harness_spans[0][1]] == [99_999]


def test_test_only_files_refuse_a_folder_of_another_carrier_or_source(tmp_path):
    from test_variants import TOK as WORD_TOK
    from test_variants import make_source

    from prio.datagen.variants import build_variant

    src, notice = _notice_with_test_files(tmp_path)
    with pytest.raises(ValueError, match="carrier"):
        build_variant(src, None, "swap", WORD_TOK, _ms(), into=notice, carrier="insert", insert_id=7)
    with pytest.raises(ValueError, match="test-only"):
        build_variant(src, None, "notice", WORD_TOK, _ms(), into=notice)
    other = tmp_path / "other"
    make_source(other)
    with pytest.raises(ValueError, match="source"):
        build_variant(other, None, "swap", WORD_TOK, _ms(), into=notice)


def test_variants_command_line_writes_into_the_variant_folder(tmp_path, monkeypatch):
    import transformers

    import prio.modelspec
    from test_variants import TOK as WORD_TOK

    from prio.datagen import variants

    src, notice = _notice_with_test_files(tmp_path)
    (notice / "test_swap.jsonl").unlink()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda model: WORD_TOK)
    monkeypatch.setattr(prio.modelspec, "resolve_spec", lambda model, tok: _ms())
    variants.main(["--src", str(src), "--variant", "swap", "--test-only-into", str(notice)])
    assert (notice / "test_swap.jsonl").exists()


def test_dataset_tasks_of_a_variant_folder_come_from_its_source(tmp_path):
    import json

    from prio.eval.run import dataset_tasks
    from prio.tasks import load_tasks

    src, notice = _notice_with_test_files(tmp_path)
    (src / "manifest.json").write_text(json.dumps({"model": "fake", "n_signals": 6, "seed": 3,
                                                   "tasks_requested": {"train": 10, "test": 20}}), encoding="utf-8")
    from test_variants import TOK as WORD_TOK

    from prio.datagen.variants import build_variant

    build_variant(src, notice, "notice", WORD_TOK, _ms())  # the variant copies what the tasks need
    expected = {t.task_id: t for t in load_tasks(0, 20, seed=3)}
    assert dataset_tasks(notice) == expected
    meta = json.loads((notice / "manifest.json").read_text(encoding="utf-8"))
    del meta["source_manifest"]["tasks_requested"]  # a variant built before: follow its source path
    (notice / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
    assert dataset_tasks(notice) == expected


# --- end-to-end smoke: build -> train -> evaluate on tiny models ------------------------------

class E2ETok:
    """FAKE_TOK words with what the real pipeline asks of a tokenizer: word-piece decodes that start
    with a space (teacher.valid_cut_points), a chat template with special-token parsing (spoof, from
    test_spoof.ChatTok), a pad id, and a length (`channel.insert_id` = len(tokenizer))."""

    pad_token_id = 0
    eos_token_id = FAKE_TOK.eos_token_id
    unk_token_id = FAKE_TOK.unk_token_id
    SPECIAL_TEXT = ("<|im_start|>", "<|im_end|>")

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def encode(self, text):
        return FAKE_TOK.encode(text)

    def decode(self, ids, **_):
        return " " + FAKE_TOK.decode(ids)

    def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
        from test_spoof import ChatTok

        return ChatTok.__call__(self, text, add_special_tokens, split_special_tokens)

    def apply_chat_template(self, messages, **kw):
        from test_spoof import ChatTok

        return ChatTok.apply_chat_template(self, messages, **kw)


def chat_source(root, tok):
    """test_variants.make_source whose test episodes belong to real test tasks: task ids and prompts
    (`build_prompt` under `tok`'s chat template) of the tasks the manifest names, as the build writes
    them; signal rows aligned to the new prompts."""
    import json

    from test_variants import make_source

    from prio.datagen.episodes import read_jsonl, write_jsonl
    from prio.eval.spoof import _align
    from prio.tasks import build_prompt, load_tasks

    make_source(root)
    test = read_jsonl(root / "test.jsonl")
    ids = list(dict.fromkeys(e.task_id for e in test))
    tasks = dict(zip(ids, load_tasks(0, len(ids), seed=0)))
    moved = []
    for e in test:
        prompt = build_prompt(tok, tasks[e.task_id])
        moved.append(replace(e, task_id=tasks[e.task_id].task_id, prompt_ids=prompt,
                             signals=_align(e.signals, len(e.prompt_ids), len(prompt))))
    write_jsonl(root / "test.jsonl", moved)
    (root / "manifest.json").write_text(json.dumps({"model": "fake", "n_signals": 6, "seed": 0,
                                                    "tasks_requested": {"train": 0, "test": len(ids)}}), encoding="utf-8")


def build_and_train(tmp_path, monkeypatch, carrier: str, delivery: str):
    """Source -> variant (+ the test-only files into it) -> one optimizer step of training through
    the train command line, the base model loading patched to a tiny GPT-2. Returns (tok, variant
    folder, run folder)."""
    import copy

    import prio.train
    from prio.datagen.variants import build_variant
    from prio.eval.spoof import cell_text

    tok = E2ETok(0)
    src, variant, run = tmp_path / "n6", tmp_path / f"n6-{carrier}", tmp_path / f"run-{delivery}"
    chat_source(src, tok)
    for content in ("F3", "S2-F3"):  # words of the spoof cells (spoof test), before the vocabulary is fixed
        for signal in ("ctx", "energy", "think", "time", "net", "ram"):
            FAKE_TOK.encode(cell_text("P1", content, signal))
    if carrier == "notice":
        build_variant(src, variant, "notice", tok, _ms())
    n = vocab_size()
    tok.size = n - 1  # INSERT_ID = len(tokenizer) = the last embedding row
    if carrier == "insert":
        build_variant(src, variant, "insert", tok, _ms(), insert_id=n - 1)
    for name in ("swap", "near_miss", "quiet"):
        build_variant(src, None, name, tok, _ms(), into=variant)
    pristine = tiny_gpt2(n)
    monkeypatch.setattr(prio.train, "load_base", lambda *a, **k: (copy.deepcopy(pristine), tok, _ms()))
    prio.train.main(["--model", "fake", "--data", str(variant), "--n-signals", "6", "--out", str(run), "--epochs", "1",
                     "--batch-size", "2", "--grad-accum", "64", "--delivery", delivery])
    return tok, variant, run


@pytest.mark.parametrize("carrier, delivery", [("notice", "none"), ("insert", "insert"), ("notice", "segment")])
def test_smoke_variant_build_train_and_batched_harness_evaluation(tmp_path, monkeypatch, carrier, delivery):
    import json

    from prio.datagen.episodes import read_jsonl
    from prio.eval import run as run_module
    from prio.train import channel_record

    tok, variant, run = build_and_train(tmp_path, monkeypatch, carrier, delivery)
    log = json.loads((run / "train_log.json").read_text(encoding="utf-8"))
    assert log["delivery"] == delivery and len(log["losses"]) > 0 and np.isfinite(log["final_loss"])
    assert channel_record(run)["delivery"] == delivery
    assert (channel_record(run).get("insert_id") == tok.size) == (delivery == "insert")

    out = tmp_path / f"eval-{delivery}"
    run_module.main(["--run", str(run), "--data", str(variant), "--out", str(out), "--mode", "harness",
                     "--batch-size", "2", "--horizon", "8"])
    records = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    events = read_jsonl(variant / "test.jsonl")
    paired = [e for e in events if e.kind in ("event", "gating_event")]
    assert {r["paired"] for r in records} == {"event", "quiet", "near_miss", "swapped"}
    assert len([r for r in records if r["paired"] == "event"]) == len(events)
    assert len([r for r in records if r["paired"] == "quiet"]) == len(paired)
    assert (out / "summary.md").exists()
    meta = json.loads((out / "eval_meta.json").read_text(encoding="utf-8"))["invocations"][-1]
    assert meta["args"]["batch_size"] == 2 and meta["channel"]["delivery"] == delivery and "finished" in meta
    gens = [json.loads(line) for line in (out / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(gens) == len(records) and all(len(g["token_ids"]) <= 8 for g in gens)
    if delivery == "insert":  # the inserted position is in the prompts, never in what the model writes
        assert any(tok.size in e.target_ids for e in events)
        assert all(tok.size not in g["token_ids"] for g in gens)


def test_smoke_forged_status_cell_on_a_trained_text_model(tmp_path, monkeypatch):
    import json

    import prio.eval.run as run_module
    import prio.eval.spoof as spoof_module

    # spoof binds dataset_tasks at import: a stub patched into run by another test may be the one it holds.
    monkeypatch.setattr(spoof_module, "dataset_tasks", run_module.dataset_tasks)
    tok, variant, run = build_and_train(tmp_path, monkeypatch, "notice", "none")
    out = tmp_path / "spoof"
    spoof_module.main(["--run", str(run), "--data", str(variant), "--out", str(out), "--every", "1",
                       "--cells", "P1/F3", "--suppress-cells", "P1/S2-F3", "--conditions", "on", "--horizon", "8"])
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {"on|P1/F3", "on|P1/none"} <= set(s["trigger"]) and "on|P1/S2-F3" in s["suppress"]
    assert s["trigger"]["on|P1/F3"]["n"] == 5  # event + gating_event episodes of the test file (every = 1)
    assert "on" in s["verdict"] and (out / "summary.md").exists()


# --- batch-size equivalence check --------------------------------------

def test_batch_equivalence_compares_records_and_generations_of_two_folders(tmp_path):
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
    import batch_equivalence

    def write(folder, records, gens):
        folder.mkdir()
        (folder / "records.partial.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        (folder / "generations.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gens), encoding="utf-8")

    rec = lambda e, c, p, react: {"episode_id": e, "condition": c, "paired": p, "kind": "event",  # noqa: E731
                                  "reaction_signal": "ctx" if react else None, "named": react}
    gen = lambda e, c, p, ids: {"episode_id": e, "condition": c, "paired": p, "token_ids": ids}  # noqa: E731
    write(tmp_path / "ref", [rec("a", "on", "event", True), rec("a", "on", "nominal", False), rec("b", "on", "event", True)],
          [gen("a", "on", "event", [1, 2, 3]), gen("a", "on", "nominal", [4, 5]), gen("b", "on", "event", [6, 7, 8])])
    write(tmp_path / "new", [rec("b", "on", "event", False), rec("a", "on", "event", True), rec("c", "on", "event", True)],
          [gen("b", "on", "event", [6, 9, 9]), gen("a", "on", "event", [1, 2, 3]), gen("c", "on", "event", [1])])
    out = batch_equivalence.compare(tmp_path / "ref", tmp_path / "new")
    assert out["key"] == ["episode_id", "condition", "paired", "kind"]
    assert (out["matched"], out["only_reference"], out["only_candidate"]) == (2, 1, 1)
    assert out["records_identical"] == 0.5 and out["fields"]["reaction_signal"] == 0.5 and out["fields"]["named"] == 0.5
    assert out["generations_identical"] == 0.5 and out["first_difference_median"] == 1
    assert out["differing"][0]["key"] == ["b", "on", "event", "event"]
    batch_equivalence.main([str(tmp_path / "ref"), str(tmp_path / "new"), "--out", str(tmp_path / "eq.json")])
    assert json.loads((tmp_path / "eq.json").read_text(encoding="utf-8"))["matched"] == 2


def test_batched_spoof_runs_make_the_records_of_batch_1(tmp_path):
    from test_spoof import MATH, TASK, TOK

    from helpers import arm
    from prio.eval.spoof import cell_prompt, run_spoof
    from prio.tasks import build_prompt

    episodes = []
    for eid, task, kind, signal, t_cross, t_decision in (("ev", TASK, "event", "energy", 3, None),
                                                         ("ga", MATH, "gating_event", "net", 1, 5),
                                                         ("e2", MATH, "event", "ctx", 2, None)):
        prompt, target = build_prompt(TOK, task), TOK.encode("<think> a b c d e f g h")
        episodes.append(make_episode(episode_id=eid, task_id=task.task_id, signal=signal, kind=kind, phase="thinking",
                                     t_cross=t_cross, t_decision=t_decision, prompt_ids=prompt, target_ids=target,
                                     signals=np.full((len(prompt) + len(target), 6), 0.5, dtype=np.float32)))
    cells, suppress = ["P1/F3", "P2/F3", "P4/F1"], ["P1/S2-F3", "P1/S1"]
    for ep in episodes:  # every word the runs need, before the vocabulary is fixed
        for cell in cells + suppress + ["P1/none", "P2/none", "P4/none"]:
            placement, content = cell.split("/")
            cell_prompt(TOK, TASK if ep.task_id == TASK.task_id else MATH, placement, content, False, ep.signal)
    model = ChannelModel(tiny_gpt2(toks=[TOK]), 6, delivery="wide", projector="levels")
    arm(model)
    kw = dict(trigger_cells=cells, suppress_cells=suppress, conditions=("on", "base"), horizon=6, every=1, log_every=0)
    tasks = {TASK.task_id: TASK, MATH.task_id: MATH}
    one = run_spoof(model, TOK, _ms(), CORE_SIGNALS, episodes, tasks, out_dir=tmp_path / "one", **kw)
    three = run_spoof(model, TOK, _ms(), CORE_SIGNALS, episodes, tasks, out_dir=tmp_path / "three", batch_size=3, **kw)
    assert three == one and len(one) == 3 * 2 * 6 + 2 * 3
    read = lambda folder: sorted((tmp_path / folder / "generations.jsonl").read_text(encoding="utf-8").splitlines())  # noqa: E731
    assert read("one") == read("three")
    again = run_spoof(model, TOK, _ms(), CORE_SIGNALS, episodes, tasks, out_dir=tmp_path / "three", batch_size=3, **kw)
    assert again == one  # resumed: nothing new
