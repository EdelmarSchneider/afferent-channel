"""Evaluate a game-character run (R22, preregistration amendment A8).

uv run python -m prio.npc.evaluate --run runs/npc6-wide-levels-s0 --data data/npc6 --out results/<folder>
    [--every K] [--batch-size 16] [--conditions on,off,base] [--forgery-every 2]

Lines L1-L5 through `prio.table.evaluate` (every sibling of the test families under `on`, `off` and
`base`; forged-status cells on every `--forgery-every`-th family), with two things of this domain:
  L4, stated apart for the command that depends on the role in the card (terror: trained roles
      >= 90%, held-out roles >= 80%) and for the commands that depend on what is at hand (thirst
      with water, hunger with food: >= 90% each);
  extra readings, no line: near-miss levels, terror by role and by threat kind, the named threat
      under a swap of the threat kind, two states high at once (variants `pair-...`: kept out of
      every line; the model was trained on one state per episode).
Forged cells (the partner's line; `domain.forged`): `P1/F1-terror` a shout, `P1/F3-terror` the
engine's status format (the cell with the bar), `P1/topic-terror` dragons as a topic (control),
`P1/S2-F1` true terror while the partner says there is nothing to fear. One batch size per results
folder.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from prio.datagen.episodes import Episode, read_jsonl
from prio.eval.run import finish_note, start_note
from prio.npc.domain import HELD_OUT, KIND, NPC, PAIR, SPECS, threat_of
from prio.npc.sentences import THREAT_WORDS
from prio.table.evaluate import CONDITIONS, L4_HELD_OUT, L4_MIN, evaluate_table, line, row, summarize, write_summary
from prio.table.score import WINDOW, TableRecord, names, spoken

NEAR_VARIANTS = ("near-calm", "near-uneasy", "thirst-near-low", "thirst-near-middle", "hunger-near-low", "hunger-near-middle")


def line_l4(records: Sequence[TableRecord]) -> dict:
    """L4 of this domain: among the responses that name the state (main cell, channel on), the first
    command is the table's; the role-dependent command and the resource-dependent ones apart."""
    named = [r for r in records if r.cell == "main" and r.condition == "on" and r.behavior == "act" and r.named
             and not r.variant.startswith(PAIR)]
    ok = lambda classes, held_out=None: [r.command_ok for r in named  # noqa: E731
                                         if r.cls in classes and held_out in (None, r.held_out)]
    return line(
        "among responses that name the state the command is the right one: by the role in the card (terror: trained "
        "roles >= 90%, held-out roles >= 80%) and by what is at hand (water, food: >= 90% each)",
        [row("terror, trained roles: `draw_weapon` (fighters) or `flee` (civilians)", ok(("terror",), False), L4_MIN),
         row("terror, held-out roles (knight, baker)", ok(("terror",), True), L4_HELD_OUT),
         row("thirst high: `drink_water` with water near, `go_to` without", ok(("thirst_water", "thirst_dry")), L4_MIN),
         row("hunger high: `eat` with food at hand, `go_to` without", ok(("hunger_food", "hunger_none")), L4_MIN)])


def readings(records: Sequence[TableRecord], episodes: Sequence[Episode], said: Mapping | None = None) -> dict[str, list[dict]]:
    """The tables reported beside the lines (main cell, channel on). `said`: (episode id, condition,
    cell) -> the text of the first `WINDOW` generated tokens outside commands; without it the rows
    that read the wording are left out."""
    meta = {ep.episode_id: ep.meta["table"] for ep in episodes}
    on = [r for r in records if r.cell == "main" and r.condition == "on"]
    plain = [r for r in on if not r.variant.startswith(PAIR)]
    role = lambda r: meta[r.episode_id]["facts"]["role"]  # noqa: E731
    threat = lambda r: threat_of(meta[r.episode_id]["state"]["threat"])  # noqa: E731
    terror = [r for r in plain if r.cls == "terror"]
    out = {
        "Near-miss levels (channel on): the behavior of the band they are in": [
            row(v, [r.correct for r in plain if r.variant == v]) for v in NEAR_VARIANTS],
        "`mention` classes one by one (channel on)": [
            row(name, [r.correct for r in plain if r.cls == name]) for name in ("uneasy", "thirst_mid", "hunger_mid")],
        "Terror by role (channel on): the warning and the role's command": [
            row(f"{name} ({KIND[name]}{', held out' if name in HELD_OUT else ''})", [r.correct for r in terror if role(r) == name])
            for name in KIND],
        "Terror by threat kind (channel on)": [
            row(kind, [r.correct for r in terror if threat(r) == kind]) for kind in THREAT_WORDS],
    }
    swap = []
    for variant, label in (("terror", "terror"), ("terror-swap", "terror with another threat kind")):
        rows = [r for r in terror if r.variant == variant]
        swap.append(row(f"{label}: names the channel's threat", [r.named for r in rows]))
        if said is not None:
            text = lambda r: said.get((r.episode_id, r.condition, r.cell), "")  # noqa: E731
            if variant == "terror-swap":  # does the name follow the channel, or the family's other sibling?
                first = lambda r: threat_of(meta[f"{r.family}-terror"]["state"]["threat"])  # noqa: E731
                swap.append(row(f"{label}: names the threat of its `terror` sibling",
                                [names(text(r), [THREAT_WORDS[first(r)]]) for r in rows if f"{r.family}-terror" in meta]))
            swap.append(row(f"{label}: names a threat kind the channel does not carry",
                            [any(names(text(r), [THREAT_WORDS[k]]) for k in THREAT_WORDS if k != threat(r)) for r in rows]))
    out["The named threat under a swap of the threat kind (channel on)"] = swap
    pairs = []
    for variant in dict.fromkeys(r.variant for r in on if r.variant.startswith(PAIR)):
        rows = [r for r in on if r.variant == variant]
        firsts = Counter(r.commands[0] if r.commands else "none" for r in rows)
        pairs.append(row(f"{variant}: the reaction of the state that comes first ({'/'.join(dict.fromkeys(r.cls for r in rows))})",
                         [r.correct for r in rows],
                         note="first commands: " + ", ".join(f"{name} {n}" for name, n in firsts.most_common())))
    out["Two states high at once (channel on; trained on one state per episode; written priority fear, thirst, hunger)"] = pairs
    return {title: [r for r in rows if r["n"]] for title, rows in out.items()}


def summarize_npc(records: Sequence[TableRecord], episodes: Sequence[Episode], said: Mapping | None = None) -> dict:
    """`prio.table.evaluate.summarize` on the records of the table's own siblings (the variants with
    two states high at once are no part of a line), L4 as this domain states it, and the readings."""
    summary = summarize([r for r in records if not r.variant.startswith(PAIR)], NPC)
    summary["records"] = len(records)
    summary["lines"]["L4"] = line_l4(records)
    summary["beside"].update(readings(records, episodes, said))
    return summary


def read_said(out_dir: Path, tok, ms) -> dict[tuple, str]:
    """What every run of `generations.jsonl` says in its first `WINDOW` tokens, outside commands."""
    said = {}
    with open(Path(out_dir) / "generations.jsonl", encoding="utf-8") as lines:
        for g in (json.loads(line_) for line_ in lines if line_.strip()):
            said[g["episode_id"], g["condition"], g["cell"]] = tok.decode(spoken(g["token_ids"][:WINDOW], ms))
    return said


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a game-character run (R22): lines L1-L5 of amendment A8.")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="game-character dataset folder (test.jsonl)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=None, help="defaults to the model recorded in the run's train_log.json")
    p.add_argument("--every", type=int, default=1, help="every n-th test family (a full run into the same --out adds the rest)")
    p.add_argument("--batch-size", type=int, default=16, help="runs generated together (one size per results folder)")
    p.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated: on, off (channel off), base (untrained model)")
    p.add_argument("--forgery-every", type=int, default=2, help="forged-status cells on every n-th evaluated family (0: none)")
    return p


def main(argv: list[str] | None = None) -> None:
    from prio.train import channel_record, load_run

    cli = parser()
    args = cli.parse_args(argv)
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    if not conditions or set(conditions) - set(CONDITIONS):
        cli.error(f"--conditions takes {', '.join(CONDITIONS)}, not {args.conditions!r}")
    model_id = args.model or json.loads((args.run / "train_log.json").read_text(encoding="utf-8"))["model"]
    start_note(args.out, {"run": str(args.run), "data": str(args.data), "model": model_id, "kind": "table",
                          "domain": NPC.name, "channel": channel_record(args.run),
                          "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
               started=time.strftime("%Y-%m-%d %H:%M"))
    episodes = read_jsonl(args.data / "test.jsonl")
    model, tok, ms = load_run(model_id, args.run, len(SPECS))
    tok.padding_side = "left"
    records = evaluate_table(model, tok, ms, NPC, episodes, args.out, conditions, every=args.every,
                             batch_size=args.batch_size, forgery_every=args.forgery_every)
    write_summary(summarize_npc(records, episodes, read_said(args.out, tok, ms)), records, args.out)
    finish_note(args.out, finished=time.strftime("%Y-%m-%d %H:%M"))
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
