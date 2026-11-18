import json

from prio.registry import build_index, write_index


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _group(rate, **extra):
    return {"n": 10, "reaction_rate": rate, "reaction_ci": [rate - 0.1, rate], "false_alarm_rate": 0.0,
            "false_alarm_ci": [0.0, 0.1], "near_miss_alarm_rate": 0.0, "identification_rate": 1.0,
            "delay_median": 2.0, **extra}


def _tree(tmp_path):
    react = {"ctx": [6.0, 4.0, 1.0, 0.0], "net": [3.0, 3.0, 2.0, 1.0]}  # 2 epochs x 2 reactions
    _write(tmp_path / "runs" / "a" / "train_log.json",
           {"model": "m", "epochs": 2, "seed": 1, "minutes": 10.0, "final_loss": 0.05, "losses": [0.4, 0.2, 0.1, 0.1],
            "react_by_signal": react, "delivery": "wide", "projector": "levels", "lr_proj": 1e-3})
    _write(tmp_path / "runs" / "a" / "channel.json", {"delivery": "wide", "lane_width": 64, "projector": "levels"})
    (tmp_path / "runs" / "b").mkdir(parents=True)  # paused: snapshot, no log
    (tmp_path / "runs" / "b" / "resume.pt").write_bytes(b"x")
    groups = {"on|all interrupt|all": _group(0.9), "zeroed|all interrupt|all": _group(0.0), "on|ctx|all": _group(1.0)}
    _write(tmp_path / "results" / "q" / "summary.json", {"groups": groups, "preservation": None})
    _write(tmp_path / "results" / "q" / "eval_meta.json",
           {"invocations": [{"run": "runs/a", "args": {"every": 4}, "started": "t1", "finished": "t2"}]})
    _write(tmp_path / "results" / "nometa" / "summary.json", {"groups": groups, "preservation": None})


def test_index_reads_runs_with_per_epoch_means_and_state(tmp_path):
    _tree(tmp_path)
    runs = {r["name"]: r for r in build_index(tmp_path)["runs"]}
    a = runs["a"]
    assert (a["state"], a["delivery"], a["projector"], a["seed"]) == ("done", "wide", "levels", 1)
    assert a["react_by_epoch"] == {"ctx": [5.0, 0.5], "net": [3.0, 1.5]}
    assert a["text_loss_by_epoch"] == [0.30000000000000004, 0.1]
    assert runs["b"]["state"] == "paused (resume.pt, no train_log.json)"
    assert runs["b"]["delivery"] == "sum"  # older runs without channel.json are sum + mlp


def test_index_links_evaluations_to_their_run(tmp_path):
    _tree(tmp_path)
    evals = {e["folder"]: e for e in build_index(tmp_path)["evaluations"]}
    q = evals["q"]
    assert q["run"] == "runs/a" and q["kind"] == "quick (every 4)"
    assert q["interrupt"]["reaction_rate"] == 0.9 and q["zeroed_interrupt_reaction"] == 0.0
    assert q["signals"]["ctx"] == 1.0
    assert evals["nometa"]["run"] is None  # not lost: listed, flagged as undescribed


def test_unfinished_full_evaluation_does_not_relabel_the_quick_summary(tmp_path):
    # summary.json on disk is the quick evaluation's until the full one finishes (paper audit, 2026-11-15).
    _tree(tmp_path)
    _write(tmp_path / "results" / "q" / "eval_meta.json",
           {"invocations": [{"run": "runs/a", "args": {"every": 4}, "started": "t1", "finished": "t2"},
                            {"run": "runs/a", "args": {"every": 1}, "started": "t3"}]})
    q = next(e for e in build_index(tmp_path)["evaluations"] if e["folder"] == "q")
    assert q["kind"] == "quick (every 4); full unfinished"


def test_write_index_writes_markdown_and_json(tmp_path):
    _tree(tmp_path)
    write_index(tmp_path)
    assert json.loads((tmp_path / "results" / "index.json").read_text(encoding="utf-8"))["runs"]
    text = (tmp_path / "results" / "index.md").read_text(encoding="utf-8")
    assert "runs/a" in text and "| q |" in text and "GENERATED" in text


def test_bg_evaluations_get_their_own_table(tmp_path):
    _tree(tmp_path)
    groups = {"on|all|all": {"n": 8, "reaction_rate": 0.875, "reaction_ci": [0.5, 0.98], "identification_rate": 1.0},
              "on|all|held_out": {"n": 4, "reaction_rate": 0.75}, "on|all|seen": {"n": 4, "reaction_rate": 1.0},
              "on|twin|all": {"n": 8, "false_alarm_rate": 0.0, "false_alarm_ci": [0.0, 0.32]},
              "zeroed|all|all": {"n": 8, "reaction_rate": 0.0}, "swap|all|all": {"n": 4, "identification_rate": 1.0}}
    _write(tmp_path / "results" / "bgq" / "summary.json", {"kind": "bg", "groups": groups,
           "resumption": {"math": {"difference": -0.02, "difference_ci": [-0.1, 0.05], "n_pairs": 10}}})
    _write(tmp_path / "results" / "bgq" / "eval_meta.json",
           {"invocations": [{"run": "runs/a", "args": {"every": 4}, "started": "t1", "finished": "t2"}]})
    index = write_index(tmp_path)
    text = (tmp_path / "results" / "index.md").read_text(encoding="utf-8")
    assert "## Background-task evaluations" in text and "| bgq |" in text.split("## Background-task evaluations")[1]
    assert "| bgq |" not in text.split("## Background-task evaluations")[0]
    bgq = next(e for e in index["evaluations"] if e["folder"] == "bgq")
    assert bgq["bg"]["reaction"]["reaction_rate"] == 0.875 and bgq["bg"]["zeroed"] == 0.0


def test_bg_runs_get_their_own_training_table(tmp_path):
    _tree(tmp_path)
    react = {"done": [9.0, 1.0, 0.5, 0.25], "error": [8.0, 2.0, 1.0, 1.0],
             "waiting": [7.0, 3.0, 0.5, 0.5], "stalled": [6.0, 4.0, 0.25, 0.25]}
    _write(tmp_path / "runs" / "bg" / "train_log.json",
           {"model": "m", "epochs": 2, "seed": 0, "minutes": 5.0, "final_loss": 0.02, "losses": [0.2, 0.1],
            "react_by_signal": react})
    write_index(tmp_path)
    text = (tmp_path / "results" / "index.md").read_text(encoding="utf-8")
    phase1, bg = text.split("## Training (background-task runs)")
    bg = bg.split("\n## ")[0]
    assert "| `bg` | 5.00 / 0.38 | 5.00 / 1.00 | 5.00 / 0.50 | 5.00 / 0.25 |" in bg
    assert "| `bg` |" not in phase1.split("## Training: mean decision loss")[1]
    assert "| `a` |" not in bg


def _spoof_cell(rate, **extra):
    return {"n": 10, "reactions": int(10 * rate), "rate": rate, "wilson": [0.0, rate + 0.2], "control_rate": 0.0,
            "mcnemar_p": 1.0, "pass": True, "is_control": False, **extra}


def test_spoof_evaluations_get_their_own_table(tmp_path):
    # prio.eval.spoof writes trigger / suppress / verdict, no "groups" (crashed the index, 2026-11-16).
    _tree(tmp_path)
    _write(tmp_path / "results" / "sp" / "summary.json", {
        "trigger": {"on|P1/F1": _spoof_cell(0.1, **{"pass": False}), "on|P1/none": _spoof_cell(0.0, is_control=True)},
        "suppress": {"on|P1/S2-F3": _spoof_cell(0.9, drop_points=1.0)},
        "verdict": {"on": {"pooled_reactions": 1, "pooled_n": 10, "pooled_rate": 0.1, "cells_pass": False,
                           "trigger_pass": False, "claim_dropped": False}}})
    _write(tmp_path / "results" / "sp" / "eval_meta.json",
           {"invocations": [{"run": "runs/a", "args": {"every": 1}, "started": "t1", "finished": "t2"}]})
    index = write_index(tmp_path)
    text = (tmp_path / "results" / "index.md").read_text(encoding="utf-8")
    assert "## Forged-status evaluations" in text
    section = text.split("## Forged-status evaluations")[1]
    assert "| sp | on | P1/F1 |" in section and "| sp | on | P1/S2-F3 |" in section
    assert "| sp |" not in text.split("## Forged-status evaluations")[0]
    sp = next(e for e in index["evaluations"] if e["folder"] == "sp")
    assert sp["run"] == "runs/a" and sp["spoof"]["verdict"]["on"]["pooled_rate"] == 0.1


def test_unknown_summary_format_is_listed_not_fatal(tmp_path):
    _tree(tmp_path)
    _write(tmp_path / "results" / "odd" / "summary.json", {"something": 1})
    index = write_index(tmp_path)
    odd = next(e for e in index["evaluations"] if e["folder"] == "odd")
    assert odd["format"] == "unknown"
    assert "| odd |" not in (tmp_path / "results" / "index.md").read_text(encoding="utf-8").split("## Reaction rate")[0]


def _table_row(label, k, n, bar=None, met=None):
    return {"label": label, "k": k, "n": n, "rate": k / n, "wilson": [0.1, 0.9], "bar": bar, "met": met, "note": ""}


def test_state_table_evaluations_get_their_own_section(tmp_path):
    # prio.table.evaluate writes kind "table" with the pre-registered lines and no "groups"
    # (R22 and R23 were listed as a format the index does not read until 2026-11-17).
    _tree(tmp_path)
    _write(tmp_path / "results" / "tb" / "summary.json", {
        "kind": "table", "domain": "npc", "families": 2, "records": 12,
        "lines": {"L1": {"criterion": "every `act` class correct >= 85%", "met": False, "note": "",
                         "rows": [_table_row("terror", 9, 10, ">= 85%", True), _table_row("thirst_dry", 4, 10, ">= 85%", False)]},
                  "L2": {"criterion": "`mention` classes pooled >= 80%", "met": True, "note": "",
                         "rows": [_table_row("pooled", 8, 10, ">= 80%", True), _table_row("beside it", 1, 10)]}},
        "beside": {"By signal": [_table_row("fear", 9, 10)]}})
    _write(tmp_path / "results" / "tb" / "eval_meta.json",
           {"invocations": [{"run": "runs/a", "args": {"every": 1}, "started": "t1", "finished": "t2"}]})
    index = write_index(tmp_path)
    text = (tmp_path / "results" / "index.md").read_text(encoding="utf-8")
    section = text.split("## State-table evaluations")[1].split("\n## ")[0]
    assert "| tb | npc | L1 | NOT met | thirst_dry | 4 / 10 | 40.0% [10.0, 90.0] | >= 85% | NO |" in section
    assert "| tb | npc | L2 | met | pooled | 8 / 10 | 80.0% [10.0, 90.0] | >= 80% | yes |" in section
    assert "| tb | npc | L2 | met | beside it | 1 / 10 | 10.0% [10.0, 90.0] | - | - |" in section
    assert "`results/tb`" not in text  # no longer under the formats the index does not read
    tb = next(e for e in index["evaluations"] if e["folder"] == "tb")
    assert tb["table"]["lines"]["L1"]["met"] is False and tb["table"]["families"] == 2 and "format" not in tb


def test_preservation_folders_get_their_own_section(tmp_path):
    # prio.eval.preservation writes preservation_summary.json and no summary.json
    # (the R11 folders were under "Other files" until 2026-11-17).
    _tree(tmp_path)

    def check(base, trained, difference, lo, hi, preserved):
        return {"n": 300, "base": base, "base_ci": [0.8, 0.9], "trained": trained, "trained_ci": [0.8, 0.9],
                "difference": difference, "ci": [lo, hi], "preserved": preserved}

    _write(tmp_path / "results" / "pres" / "preservation_summary.json",
           {"gsm8k": check(0.85, 0.86, 0.01, -0.03, 0.04, True), "explain": check(0.9, 0.8, -0.1, -0.2, 0.0, False), "beside": {}})
    _write(tmp_path / "results" / "pres" / "eval_meta.json",
           {"invocations": [{"run": "runs/a", "args": {"n": 300}, "started": "t0"},
                            {"run": "runs/a", "args": {"n": 300}, "started": "t1", "finished": "t2"}]})
    index = write_index(tmp_path)
    text = (tmp_path / "results" / "index.md").read_text(encoding="utf-8")
    section = text.split("## Capability preservation")[1].split("\n## ")[0]
    assert "| pres | `runs/a` | GSM8K accuracy | 300 | 85.0% | 86.0% | +1.0 [-3.0, +4.0] | preserved |" in section
    assert "| pres | `runs/a` | Explain format kept | 300 | 90.0% | 80.0% | -10.0 [-20.0, +0.0] | a cost |" in section
    assert "`results/pres/`" not in text  # no longer under the other files
    assert index["preservation"][0]["run"] == "runs/a" and index["preservation"][0]["finished"] == "t2"
