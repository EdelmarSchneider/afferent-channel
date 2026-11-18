"""Read background-task episodes by hand: for each event kind, the tasks and slots, the text around
the event with harness segments marked, the slot states at the event and after the harness answers,
the report, the closing line and (math) whether the final number is still the gold answer.
CPU only (tokenizer). Usage: uv run python scripts/inspect_bg.py data/bg3 2 [test|train]
"""

import sys
from pathlib import Path

from transformers import AutoTokenizer

from prio.bg.protocol import slot_columns, state_of
from prio.datagen.episodes import read_jsonl
from prio.eval.run import extract_number


def marked(tok, ep, a: int, b: int) -> str:
    """Decoded target_ids[a:b] with harness segments shown as [HARNESS: ...]."""
    out, i = [], a
    for s, e in sorted(ep.harness_spans or []):
        if e <= a or s >= b:
            continue
        out.append(tok.decode(ep.target_ids[i:max(i, s)]))
        out.append(" [HARNESS: " + tok.decode(ep.target_ids[max(s, a):min(e, b)]).strip()[:160] + "] ")
        i = min(e, b)
    out.append(tok.decode(ep.target_ids[i:b]))
    return "".join(out)


def main() -> None:
    data, per_kind = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    episodes = read_jsonl(data / f"{split}.jsonl")
    for kind in ("done", "error", "waiting", "stalled"):
        chosen = [e for e in episodes if e.kind == "bg_event" and e.signal == kind][::50][:per_kind]
        for ep in chosen:
            m, n = ep.meta, len(ep.prompt_ids)
            ev = m["event"]
            print("=" * 100)
            print(f"{ep.episode_id} {kind} phase={ep.phase} family={m['family']} held_out={m['held_out']} "
                  f"event slot={ev['slot']} policy={m['policy']}")
            print("tasks:", "; ".join(f"slot {t['slot']}: {t['verb']} {t['phrase']} ({t['end']})" for t in m["tasks"]))
            p, s = slot_columns(ev["slot"])
            first = n + m["answer_ends"][0]
            last = n + m["answer_ends"][-1]
            print("slot state t-1 / t / after 1st answer / after last answer:",
                  [state_of(ep.signals[i][s]) for i in (n + ep.t_cross - 1, n + ep.t_cross, first, min(last, len(ep.signals) - 1))],
                  "progress at t-1:", float(ep.signals[n + ep.t_cross - 1][p]))
            print("--- around the event:")
            print(repr(marked(tok, ep, max(0, ep.t_cross - 25), min(len(ep.target_ids), m["resume_start"] + 25))))
            a, b = m["report_span"]
            print("--- report:", repr(tok.decode(ep.target_ids[a:b])))
            print("--- end:", repr(tok.decode(ep.target_ids[-45:])))
            if m["family"] == "math":
                print("--- math final number:", extract_number(tok.decode(ep.target_ids)), "gold:", m["answer"])


if __name__ == "__main__":
    main()
