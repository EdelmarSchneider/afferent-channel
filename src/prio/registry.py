"""Index of every trained run and every evaluation, read from the files themselves.

`runs/<name>/` (train_log.json, channel.json, resume.pt) and `results/<folder>/` (summary.json,
eval_meta.json, throughput.json) -> `results/index.json` + `results/index.md`. The numbers in the
index come from the result files, never typed by hand. Entry point: `scripts/index_results.py`.
"""

import json
import time
from pathlib import Path

from prio.train import channel_record

SIGNALS = ("ctx", "think", "energy", "time", "net", "ram")
BG_EVENTS = ("done", "error", "waiting", "stalled")  # decision-loss keys of background-task runs


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _per_epoch(values: list[float], epochs: int) -> list[float]:
    n = len(values) // epochs
    return [sum(values[i * n:(i + 1) * n]) / n for i in range(epochs)] if n else []


def run_entry(run_dir: Path) -> dict:
    log = _read(run_dir / "train_log.json")
    channel = channel_record(run_dir)
    if log:
        state = "done"
    elif (run_dir / "resume.pt").exists():
        state = "paused (resume.pt, no train_log.json)"
    elif (run_dir / "channel.pt").exists():
        state = "stopped (weights of a finished epoch, no train_log.json)"
    else:
        state = "empty"
    entry = {"name": run_dir.name, "path": f"runs/{run_dir.name}", "state": state, **channel}
    if log:
        epochs = log["epochs"]
        entry.update({
            "model": log["model"], "seed": log["seed"], "epochs": epochs, "minutes": log["minutes"],
            "final_loss": log["final_loss"], "lr_proj": log.get("lr_proj", 1e-3),
            "skipped_batches": log.get("skipped_batches"),
            "react_by_epoch": {s: _per_epoch(v, epochs) for s, v in log.get("react_by_signal", {}).items()},
            "text_loss_by_epoch": _per_epoch(log["losses"], epochs),
            "args": log.get("args"),
        })
    return entry


def _group(groups: dict, key: str) -> dict | None:
    g = groups.get(key)
    if g is None:
        return None
    return {k: g.get(k) for k in ("n", "reaction_rate", "reaction_ci", "false_alarm_rate", "false_alarm_ci",
                                  "near_miss_alarm_rate", "identification_rate", "delay_median",
                                  "wrongful_call_rate", "wrongful_refusal_rate")}


def eval_entry(folder: Path) -> dict:
    summary = _read(folder / "summary.json")
    meta = _read(folder / "eval_meta.json") or {"invocations": []}
    invocations = meta["invocations"]
    # summary.json is written when an invocation finishes: label it by the last FINISHED one.
    finished = [i for i in invocations if i.get("finished")]
    last = finished[-1] if finished else (invocations[-1] if invocations else None)
    label = lambda inv: "full" if inv.get("args", {}).get("every") == 1 else f"quick (every {inv.get('args', {}).get('every')})"
    kind = None if last is None else label(last)
    if last is not None and invocations and invocations[-1] is not last and not invocations[-1].get("finished"):
        kind += f"; {label(invocations[-1])} unfinished"
    base = {"folder": folder.name, "path": f"results/{folder.name}", "run": (last or {}).get("run"), "kind": kind,
            "invocations": [{k: i.get(k) for k in ("run", "started", "finished", "note")} | {"every": i.get("args", {}).get("every")}
                            for i in invocations]}
    if "trigger" in summary:  # forged-status evaluation (prio.eval.spoof)
        return base | {"spoof": {k: summary.get(k) for k in ("trigger", "suppress", "verdict")}}
    if summary.get("kind") == "table":  # state-table evaluation (prio.table.evaluate): the pre-registered lines
        return base | {"table": {k: summary.get(k) for k in ("domain", "families", "records")} | {"lines": summary.get("lines") or {}}}
    if "groups" not in summary:  # a format this index does not read: listed, never fatal
        return base | {"format": "unknown"}
    groups = summary["groups"]
    rate = lambda key: (groups.get(key) or {}).get("reaction_rate")
    if summary.get("kind") == "bg":  # background-task evaluation (prio.bg.evaluate)
        return base | {
            "bg": {"reaction": groups.get("on|all|all"), "held_out": groups.get("on|all|held_out"),
                   "seen": groups.get("on|all|seen"), "false_alarm": groups.get("on|twin|all"),
                   "zeroed": rate("zeroed|all|all"),
                   "swap_identification": (groups.get("swap|all|all") or {}).get("identification_rate"),
                   "resumption": summary.get("resumption")},
        }
    return base | {
        "interrupt": _group(groups, "on|all interrupt|all"),
        "gating": _group(groups, "on|all gating|all"),
        "starts_past": _group(groups, "on|all starts_past|all"),
        "swap": _group(groups, "on|all swap|all"),
        "zeroed_interrupt_reaction": rate("zeroed|all interrupt|all"),
        "signals": {s: rate(f"on|{s}|all") for s in SIGNALS if f"on|{s}|all" in groups},
        "preservation": summary.get("preservation"),
        "throughput": _read(folder / "throughput.json"),
    }


def preservation_entry(folder: Path) -> dict:
    """A capability-preservation folder (`prio.eval.preservation`): both checks with their line."""
    summary = _read(folder / "preservation_summary.json")
    invocations = (_read(folder / "eval_meta.json") or {"invocations": []})["invocations"]
    finished = [i for i in invocations if i.get("finished")]
    last = finished[-1] if finished else (invocations[-1] if invocations else {})
    return {"folder": folder.name, "path": f"results/{folder.name}", "run": last.get("run"),
            "started": last.get("started"), "finished": last.get("finished"),
            "checks": {name: summary.get(name) for name in ("gsm8k", "explain")}}


def stream_entry(folder: Path) -> dict:
    """A folder of the R12 arms without a forced prefix (`prio.r12.stream`: arms free, live, Ls):
    the counts of its `stream_summary.json`. Folders `r12p-*` hold the pilot set."""
    summary = _read(folder / "stream_summary.json")
    invocations = (_read(folder / "eval_meta.json") or {"invocations": []})["invocations"]
    finished = [i for i in invocations if i.get("finished")]
    last = finished[-1] if finished else (invocations[-1] if invocations else {})
    event, twin, off = (summary.get(k) or {} for k in ("event", "twin", "off"))
    return {"folder": folder.name, "path": f"results/{folder.name}", "run": last.get("run"),
            "set": "pilot" if folder.name.startswith("r12p-") else "confirm",
            "arm": summary.get("arm"), "model": summary.get("model"), "started": last.get("started"),
            "finished": last.get("finished"), "batch_size": (last.get("args") or {}).get("batch_size"),
            "event": {k: event.get(k) for k in ("planned", "occurred", "hits", "reaction_rate", "reaction_ci", "premature",
                                                "crossing_off_k", "not_occurred")},
            "twin": {k: twin.get(k) for k in ("n", "false_alarms", "false_alarm_rate")},
            "off": {k: off.get(k) for k in ("n", "false_alarms", "false_alarm_rate")}}


def build_index(root: Path) -> dict:
    root = Path(root)
    runs = [run_entry(d) for d in sorted((root / "runs").iterdir()) if d.is_dir()] if (root / "runs").exists() else []
    results = root / "results"
    folders = sorted(d for d in results.iterdir() if d.is_dir()) if results.exists() else []
    evaluations = [eval_entry(d) for d in folders if (d / "summary.json").exists()]
    preservation = [preservation_entry(d) for d in folders
                    if (d / "preservation_summary.json").exists() and not (d / "summary.json").exists()]
    streams = [stream_entry(d) for d in folders
               if (d / "stream_summary.json").exists() and not (d / "summary.json").exists()]
    read = {e["folder"] for e in evaluations + preservation + streams}
    other = sorted(p.name + ("/" if p.is_dir() else "") for p in results.iterdir()
                   if p.name not in {"index.md", "index.json"} and p.name not in read) if results.exists() else []
    return {"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": runs, "evaluations": evaluations,
            "preservation": preservation, "r12_stream": streams, "other_results": other}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _rate_ci(g: dict | None, rate: str = "reaction_rate", ci: str = "reaction_ci") -> str:
    if not g or g.get(rate) is None:
        return "-"
    lo, hi = g.get(ci) or (None, None)
    return f"{_pct(g[rate])} [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else _pct(g[rate])


def to_markdown(index: dict) -> str:
    phase1 = [e for e in index["evaluations"] if "interrupt" in e]
    bg = [e for e in index["evaluations"] if "bg" in e]
    spoof = [e for e in index["evaluations"] if "spoof" in e]
    unknown = [e for e in index["evaluations"] if e.get("format") == "unknown"]
    out = [f"# Index of runs and evaluations", "",
           f"GENERATED by `scripts/index_results.py` on {index['generated']} from the files in `runs/` and `results/`. "
           "Do not edit by hand: rerun the script. What each experiment asked and found: the paper.", "",
           "## Trained runs", "",
           "| Run | State | Delivery | Projector | Lane width | Channel rate | Seed | Epochs | Minutes | Final text loss |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in index["runs"]:
        out.append(f"| `{r['path']}` | {r['state']} | {r['delivery']} | {r['projector']} | "
                   f"{r['lane_width'] if r['delivery'] != 'sum' else '-'} | {r.get('lr_proj', '-')} | {r.get('seed', '-')} | "
                   f"{r.get('epochs', '-')} | {r['minutes']:.1f} | {r['final_loss']:.3f} |" if r["state"] == "done" else
                   f"| `{r['path']}` | {r['state']} | {r['delivery']} | {r['projector']} | - | - | - | - | - | - |")
    done = [r for r in index["runs"] if r["state"] == "done"]
    bg_runs = [r for r in done if set(r["react_by_epoch"]) & set(BG_EVENTS)]
    for title, note, keys, rows in (
            ("Training: mean decision loss (`react`) per epoch",
             "6.67 = channel unused (base rate on `data/n6`); floors: limit signals ~0, `net` 1.37, `ram` 1.09.",
             SIGNALS, [r for r in done if r not in bg_runs]),
            ("Training (background-task runs): mean decision loss (`react`) per epoch, by event",
             "7.24 = channel unused (base rate on `data/bg3`).", BG_EVENTS, bg_runs)):
        if not rows:
            continue
        out += ["", f"## {title}", "", note, "",
                "| Run | " + " | ".join(f"`{s}`" for s in keys) + " | text loss |", "|---|" + "---|" * (len(keys) + 1)]
        for r in rows:
            cells = [" / ".join(f"{v:.2f}" for v in r["react_by_epoch"].get(s, [])) or "-" for s in keys]
            out.append(f"| `{r['name']}` | " + " | ".join(cells) + " | " +
                       " / ".join(f"{v:.3f}" for v in r["text_loss_by_epoch"]) + " |")
    out += ["", "## Evaluations (channel on unless noted)", "",
            "| Folder | Run | Kind | Finished | Interrupt reaction [95% CI] | False alarms [95% CI] | Near-miss alarms | "
            "Identification | Gating | Starts past | Swap | Zeroed (interrupt) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in phase1:
        i = e["interrupt"] or {}
        finished = e["invocations"][-1].get("finished") if e["invocations"] else None
        out.append(f"| {e['folder']} | `{e['run']}` | {e['kind'] or 'UNDESCRIBED (no eval_meta.json)'} | {finished or '-'} | "
                   f"{_rate_ci(e['interrupt'])} | {_rate_ci(e['interrupt'], 'false_alarm_rate', 'false_alarm_ci')} | "
                   f"{_pct(i.get('near_miss_alarm_rate'))} | {_pct(i.get('identification_rate'))} | "
                   f"{_pct((e['gating'] or {}).get('reaction_rate'))} | {_pct((e['starts_past'] or {}).get('reaction_rate'))} | "
                   f"{_pct((e['swap'] or {}).get('reaction_rate'))} | {_pct(e['zeroed_interrupt_reaction'])} |")
    out += ["", "## Reaction rate per signal (channel on)", "",
            "| Folder | " + " | ".join(f"`{s}`" for s in SIGNALS) + " |", "|---|" + "---|" * len(SIGNALS)]
    for e in phase1:
        out.append(f"| {e['folder']} | " + " | ".join(_pct(e["signals"].get(s)) for s in SIGNALS) + " |")
    preserved = [e for e in phase1 if e["preservation"]]
    if preserved:
        out += ["", "## Preservation (GSM8K)", "", "| Folder | n | Base | Trained | Difference [95% CI] |", "|---|---|---|---|---|"]
        for e in preserved:
            p = e["preservation"]
            lo, hi = p["difference_ci"]
            out.append(f"| {e['folder']} | {p['n']} | {_pct(p['base_accuracy'])} | {_pct(p['tuned_nominal_accuracy'])} | "
                       f"{100 * p['accuracy_difference']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    measured = [e for e in phase1 if e["throughput"]]
    if measured:
        out += ["", "## Throughput", "", "| Folder | Tokens/s on | Tokens/s off | Slowdown |", "|---|---|---|---|"]
        for e in measured:
            t = e["throughput"]
            out.append(f"| {e['folder']} | {t['median_on']:.1f} | {t['median_off']:.1f} | {_pct(t['overhead'])} |")
    multi = [e for e in index["evaluations"] if len(e["invocations"]) > 1 or any(i.get("note") for i in e["invocations"])]
    if multi:
        out += ["", "## Evaluation history (folders written more than once, or with a note)", ""]
        for e in multi:
            for inv in e["invocations"]:
                out.append(f"- {e['folder']}: `{inv['run']}` every {inv['every']}, {inv['started']} -> {inv.get('finished')}"
                           + (f"; {inv['note']}" if inv.get("note") else ""))
    if bg:
        out += ["", "## Background-task evaluations", "",
                "| Folder | Run | Kind | Reaction [95% CI] | Identification | Held-out reaction | Seen reaction | "
                "False alarms [95% CI] | Zeroed | Swap identification | Math resumption difference [95% CI] |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in bg:
            b = e["bg"]
            math = (b["resumption"] or {}).get("math") or {}
            diff = "-" if math.get("difference") is None else f"{100 * math['difference']:+.1f}"
            if math.get("difference_ci"):
                diff += " [{:+.1f}, {:+.1f}]".format(*(100 * v for v in math["difference_ci"]))
            out.append(f"| {e['folder']} | `{e['run']}` | {e['kind'] or 'UNDESCRIBED (no eval_meta.json)'} | "
                       f"{_rate_ci(b['reaction'])} | {_pct((b['reaction'] or {}).get('identification_rate'))} | "
                       f"{_pct((b['held_out'] or {}).get('reaction_rate'))} | {_pct((b['seen'] or {}).get('reaction_rate'))} | "
                       f"{_rate_ci(b['false_alarm'], 'false_alarm_rate', 'false_alarm_ci')} | {_pct(b['zeroed'])} | "
                       f"{_pct(b['swap_identification'])} | {diff} |")
    if spoof:
        out += ["", "## Forged-status evaluations", "",
                "Trigger cells: reactions to forged status text with a nominal channel (`none` = paired control). "
                "Suppression cells: true events with reassuring text. Bar: preregistration R2 (Appendix A of the paper).", "",
                "| Folder | Condition | Cell | n | Reactions | Rate [95% CI] | Control rate | Drop (points) | McNemar p | Pass |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for e in spoof:
            for arm in ("trigger", "suppress"):
                for key, c in (e["spoof"].get(arm) or {}).items():
                    condition, cell = key.split("|", 1)
                    lo, hi = c.get("wilson") or (None, None)
                    rate_ci = _pct(c.get("rate")) + (f" [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else "")
                    drop = "-" if c.get("drop_points") is None else f"{c['drop_points']:+.1f}"
                    p = "-" if c.get("mcnemar_p") is None else f"{c['mcnemar_p']:.3g}"
                    verdict = "-" if c.get("is_control") or c.get("pass") is None else ("yes" if c["pass"] else "NO")
                    out.append(f"| {e['folder']} | {condition} | {cell} | {c.get('n')} | {c.get('reactions')} | {rate_ci} | "
                               f"{_pct(c.get('control_rate'))} | {drop} | {p} | {verdict} |")
            for condition, v in (e["spoof"].get("verdict") or {}).items():
                out.append(f"- {e['folder']} `{condition}`: pooled {v.get('pooled_reactions')} / {v.get('pooled_n')} = "
                           f"{_pct(v.get('pooled_rate'))}; every cell passes: {v.get('cells_pass')}; "
                           f"claim dropped (a cell above 8%): {v.get('claim_dropped')}")
    tables = [e for e in index["evaluations"] if "table" in e]
    if tables:
        out += ["", "## State-table evaluations", "",
                "The lines of preregistration amendment A8 (`prio.table.evaluate`), channel on unless the row says "
                "otherwise. Notes of the rows and the tables reported beside the lines: each folder's `summary.md`.", "",
                "| Folder | Domain | Line | Line met | Row | k / n | Rate [95% Wilson] | Bar | Row met |",
                "|---|---|---|---|---|---|---|---|---|"]
        for e in tables:
            for name, stated in e["table"]["lines"].items():
                for r in stated.get("rows") or []:
                    lo, hi = r.get("wilson") or (None, None)
                    rate = _pct(r.get("rate")) + (f" [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else "")
                    met = "-" if r.get("met") is None else ("yes" if r["met"] else "NO")
                    out.append(f"| {e['folder']} | {e['table']['domain']} | {name} | {'met' if stated.get('met') else 'NOT met'} | "
                               f"{r.get('label')} | {r.get('k')} / {r.get('n')} | {rate} | {r.get('bar') or '-'} | {met} |")
    if index.get("preservation"):
        out += ["", "## Capability preservation (R11)", "",
                "Base = the same weights with adapter and channel off, through the same batched loop. Line "
                "(preregistration R11, amendment A4): preserved if the paired difference is >= -2 points and its "
                "95% CI lower bound >= -5; otherwise a cost.", "",
                "| Folder | Run | Check | n | Base | Trained | Difference, points [95% CI] | Line |",
                "|---|---|---|---|---|---|---|---|"]
        for e in index["preservation"]:
            for key, label in (("gsm8k", "GSM8K accuracy"), ("explain", "Explain format kept")):
                c = e["checks"].get(key)
                if c:
                    lo, hi = c["ci"]
                    out.append(f"| {e['folder']} | `{e['run']}` | {label} | {c['n']} | {_pct(c['base'])} | {_pct(c['trained'])} | "
                               f"{100 * c['difference']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                               f"{'preserved' if c['preserved'] else 'a cost'} |")
    if index.get("r12_stream"):
        out += ["", "## R12 arms without a forced prefix (free, live, Ls)", "",
                "Counts of each folder's `stream_summary.json` (`prio.r12.stream`). The pre-registered lines of R12 are computed "
                "by `prio.r12.lines` (`results/r12/r12_lines.md`), never from this table. Folders `r12p-*` hold the pilot set: "
                "never reported as results (the paired arm's pilot folders are the `r12p-paired-*` rows of the evaluations table).",
                "",
                "| Folder | Set | Arm | Model | Run | Finished | Event runs | Crossings that occurred | Hits | Reaction [95% CI] | "
                "Premature | Twin false alarms | Off false alarms | Crossings off token k |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in index["r12_stream"]:
            v, twin, off = e["event"], e["twin"], e["off"]
            out.append(f"| {e['folder']} | {'pilot, never reported' if e['set'] == 'pilot' else 'confirmatory'} | {e['arm']} | "
                       f"{e['model']} | `{e['run']}` | {e['finished'] or '-'} | {v['planned']} | {v['occurred']} | {v['hits']} | "
                       f"{_rate_ci(v)} | {v['premature']} | {twin['false_alarms']} / {twin['n']} | "
                       f"{off['false_alarms']} / {off['n']} | {v['crossing_off_k']} |")
    if unknown:
        out += ["", "## Evaluations in a format this index does not read", ""] + \
               [f"- `{e['path']}` (run `{e['run']}`, {e['kind'] or 'no eval_meta.json'})" for e in unknown]
    out += ["", "## Other files in `results/`", ""] + [f"- `results/{name}`" for name in index["other_results"]] + [""]
    return "\n".join(out)


def write_index(root: Path) -> dict:
    index = build_index(root)
    results = Path(root) / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (results / "index.md").write_text(to_markdown(index), encoding="utf-8")
    return index
