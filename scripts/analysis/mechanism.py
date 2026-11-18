"""R1 mechanism runs on existing weights (preregistration R1; B2 + B3, T3).

B2: minimal pairs (event vs the same trajectory with the crossing signal one level on the nominal
side at the decision position; prefix KV shared). Decision tokens: r = argmax with the event,
c = argmax with the control (pairs with r == c are counted, not scored). Metric: LD = logit(r) -
logit(c); retained = (LD_cond - LD_control) / (LD_intact - LD_control). Interventions at the
decision position only (prefix cached with the intact model):
  readers zeroed (all | attention | FFN | per matrix | per layer), writers zeroed (all | type |
  per layer), lanes cut from layer L up / below layer L, every reader and writer at once;
  prefix_off: prefix computed with the channel off (do earlier positions' lanes matter?);
  channel_off: channel off everywhere (sanity: should sit near the control level);
  greedy:k (phase 2): readers of the k layers with the largest single-layer effect, ranked on the
  other half of the pairs (cross-fitted). H1: k <= 25% of layers removes >= 80%. H2: zeroing all
  writers removes < 20%.
B3 patching: control run with the event run's residual at the input of layer L (main d dims |
  64 lanes), decision position: where the channel's information enters the model's own stream.
T3: v_ch(L) = mean(event - control) main-stream residual at the decision position; v_tx(L) =
  mean(residual at the last token of a past-limit status notice - same notice with a nominal
  value), appended after the decision token with the channel off, in the trained model and in the
  base model. Cosine per layer vs a sign-flip null on the channel pairs, and vs other signals'
  notices. Phase 2 adds v_tx (norm-matched to v_ch) or v_ch at layer L with the channel off and
  measures how much of LD it restores.
B3 probes (--probes): residuals at the decision position, the token before, random earlier
  positions and +2/+4/+8, channel on / off / base; logistic probes (past-limit, which signal,
  phase, reaction under way among past-limit positions, task family; Hewitt-Liang control task
  for past-limit) and ridge for each signal's value inside the nominal range; held-out tasks.

Outputs (--out): pairs.jsonl + phase2.jsonl (per pair, resumable), t3_state.pt (resume state of
T3, ~80 MB for the full test split: do not version it), summary.json + summary.md, meta.json;
probes/probes.json + probes.md.

Usage (project root, GPU, GPU lock held by hand; estimated ~45-50 min for the 655 pairs of data/n6
test with T3, + ~10 min for --probes; --every 2 halves the pair part; not yet timed on the real model):
  uv run python scripts/analysis/mechanism.py --run runs/n6-wide-levels-s0 --data data/n6 \
      --out results/mechanism/r1-wide-levels-s0 --device cuda [--probes] [--every N] [--no-t3]
Rerunning the same command resumes (pairs.jsonl, t3_state.pt, phase2.jsonl).
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mech_utils as mu  # noqa: E402

from prio.datagen.episodes import Episode, read_jsonl  # noqa: E402
from prio.signals import signal_set  # noqa: E402

PAIR_KINDS = ("event", "gating_event", "starts_past")
NOTICES = {  # template, past-limit value, nominal value (same token count)
    "ctx": ("[STATUS: context window {}% full]", "92", "40"),
    "think": ("[STATUS: thinking budget {}% left]", "8", "60"),
    "energy": ("[STATUS: battery {}%]", "8", "60"),
    "time": ("[STATUS: time used {}%]", "92", "40"),
    "net": ("[STATUS: network {}]", "down", "up"),
    "ram": ("[STATUS: free memory {}%]", "8", "60"),
}


# --- small helpers --------------------------------------------------------------------------------

def clean(x):
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python, floats rounded to 5 digits."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 5) if math.isfinite(x) else None
    return x


def _encode(tok, text: str) -> list[int]:
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tok.encode(text))


def notice_texts(spec) -> tuple[str, str]:
    """(past-limit notice, same notice with a nominal value)."""
    template, past, nominal = NOTICES.get(spec.name, ("[STATUS: " + spec.name + " {}%]", "8", "60"))
    return " " + template.format(past), " " + template.format(nominal)


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "ab") as f:
        f.write((json.dumps(clean(row)) + "\n").encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


# --- conditions -----------------------------------------------------------------------------------

def b2_conditions(iv: mu.Interventions) -> list[mu.Cond]:
    """Phase-1 conditions at the decision position (event signals unless patched)."""
    readers, writers, n = iv.readers, iv.writers, iv.n_layers
    sel = lambda sites, pred: frozenset(i for i, s in enumerate(sites) if pred(s))  # noqa: E731
    conds = [mu.Cond("readers:all", readers=sel(readers, lambda s: True)),
             mu.Cond("readers:attn", readers=sel(readers, lambda s: s.group == "attn")),
             mu.Cond("readers:ffn", readers=sel(readers, lambda s: s.group == "ffn"))]
    for proj in sorted({s.proj for s in readers}):
        conds.append(mu.Cond(f"readers:proj={proj}", readers=sel(readers, lambda s, p=proj: s.proj == p)))
    conds += [mu.Cond(f"readers:L{l}", readers=sel(readers, lambda s, l=l: s.layer == l)) for l in range(n)]
    conds += [mu.Cond(f"lanes_from:L{l}", readers=sel(readers, lambda s, l=l: s.layer >= l)) for l in range(1, n)]
    conds += [mu.Cond(f"lanes_upto:L{l}", readers=sel(readers, lambda s, l=l: s.layer < l)) for l in range(1, n)]
    if writers:
        conds += [mu.Cond("writers:all", writers=sel(writers, lambda s: True)),
                  mu.Cond("writers:attn", writers=sel(writers, lambda s: s.group == "attn")),
                  mu.Cond("writers:ffn", writers=sel(writers, lambda s: s.group == "ffn")),
                  mu.Cond("readers+writers:all", readers=sel(readers, lambda s: True), writers=sel(writers, lambda s: True))]
        conds += [mu.Cond(f"writers:L{l}", writers=sel(writers, lambda s, l=l: s.layer == l)) for l in range(n)]
    conds += [mu.Cond(f"patch_main:L{l}", signals="control", patch=(l, "main")) for l in range(n)]
    if writers:  # the residual carries the lanes only in the wide delivery
        conds += [mu.Cond(f"patch_lanes:L{l}", signals="control", patch=(l, "lanes")) for l in range(n)]
    return conds


def greedy_conditions(iv: mu.Interventions, order: list[int]) -> list[mu.Cond]:
    return [mu.Cond(f"greedy:k{k}", readers=frozenset(i for i, s in enumerate(iv.readers) if s.layer in order[:k]))
            for k in range(1, len(order) + 1)]


def _scores(iv, kv, pair, conds, r, c, batch, patch_source=None, add_vectors=None, enabled=True, adapter=True):
    out = {}
    for start in range(0, len(conds), batch):
        chunk = conds[start:start + batch]
        rows = np.stack([pair.control[-1] if cd.signals == "control" else pair.event[-1] for cd in chunk])
        logits = iv.last(kv, pair.ids[-1], rows, conds=chunk, patch_source=patch_source, add_vectors=add_vectors,
                         enabled=enabled, adapter=adapter)
        top = logits.argmax(-1)
        for k, cd in enumerate(chunk):
            out[cd.name] = {"ld": float(logits[k, r] - logits[k, c]), "top_r": bool(top[k] == r)}
    return out


def analyze_pair(iv: mu.Interventions, pair: mu.Pair, conds, batch: int = 16) -> dict:
    """Intact event / control, then every condition (split pairs only). Keys starting with '_'
    hold tensors for T3 and are not written."""
    kv = iv.prefix(pair.ids, pair.event)
    logits, rec = iv.last(kv, pair.ids[-1], np.stack([pair.event[-1], pair.control[-1]]), record=True)
    r, c = int(logits[0].argmax()), int(logits[1].argmax())
    out = {"episode_id": pair.episode_id, "task_id": pair.task_id, "kind": pair.kind, "signal": pair.signal,
           "phase": pair.phase, "family": pair.family, "length": len(pair.ids), "r": r, "c": c, "split": r != c,
           "ld_intact": float(logits[0, r] - logits[0, c]), "ld_control": float(logits[1, r] - logits[1, c]),
           "logprob_r_event": float(torch.log_softmax(logits[0], -1)[r]),
           "logprob_r_control": float(torch.log_softmax(logits[1], -1)[r]), "conditions": {}}
    out["_rec"] = rec
    if r == c:
        return out
    source = {layer: values[0] for layer, values in rec.items()}
    out["conditions"] = _scores(iv, kv, pair, conds, r, c, batch, patch_source=source)
    del kv
    kv_off = iv.prefix(pair.ids, pair.event, enabled=False)
    off = iv.last(kv_off, pair.ids[-1], np.stack([pair.event[-1], pair.control[-1]]))
    out["conditions"]["prefix_off:event"] = {"ld": float(off[0, r] - off[0, c]), "top_r": bool(off[0].argmax() == r)}
    out["conditions"]["prefix_off:control"] = {"ld": float(off[1, r] - off[1, c]), "top_r": bool(off[1].argmax() == r)}
    none = iv.last(kv_off, pair.ids[-1], pair.event[-1:], enabled=False)[0]
    out["conditions"]["channel_off"] = {"ld": float(none[r] - none[c]), "top_r": bool(none.argmax() == r)}
    out["_kv_off"] = kv_off
    return out


def greedy_order(records: list[dict], parity: int, n_layers: int) -> list[int]:
    """Layers by mean single-layer removal (1 - retained of readers:L), measured on the pairs of
    the OTHER parity (cross-fitting: a pair never ranks the layers it is scored with)."""
    removal = np.zeros(n_layers)
    for layer in range(n_layers):
        values = [1 - mu.retained(r["conditions"][f"readers:L{layer}"]["ld"], r["ld_intact"], r["ld_control"])
                  for r in records if r["split"] and r["index"] % 2 != parity and f"readers:L{layer}" in r["conditions"]]
        values = [v for v in values if np.isfinite(v)]
        removal[layer] = np.mean(values) if values else 0.0
    return [int(l) for l in np.argsort(-removal, kind="stable")]


# --- T3 -------------------------------------------------------------------------------------------

def t3_collect(iv, pair, tok, specs, rec, kv_off, base: bool = True) -> dict:
    """Per-layer main-stream vectors of one pair: the channel difference and the notice contrasts."""
    d = iv.d
    spec = specs[[s.name for s in specs].index(pair.signal)]
    past, nominal = (_encode(tok, t) for t in notice_texts(spec))
    out = {"ch": torch.stack([(rec[l][0, :d] - rec[l][1, :d]).float().cpu() for l in sorted(rec)])}
    models = [("trained", True, kv_off)]
    if base:
        models.append(("base", False, iv.prefix(pair.ids, pair.event, enabled=False, adapter=False)))
    for name, adapter, kv in models:
        def h(extra):
            ids = [pair.ids[-1]] + extra
            got = iv.continuation(kv, ids, np.repeat(pair.event[-1:], len(ids), axis=0), enabled=False, adapter=adapter)
            return torch.stack([got[l][:d].float().cpu() for l in sorted(got)])
        none, hp, hn = h([]), h(past), h(nominal)
        out[f"tx_{name}"] = hp - hn
        out[f"ins_{name}"] = hp - none
    return out


class T3State:
    """Per-pair channel differences (float16) and per-signal sums of the notice contrasts."""

    def __init__(self):
        self.done: set[str] = set()
        self.ch: dict[str, list[torch.Tensor]] = {}
        self.sums: dict[str, dict[str, torch.Tensor]] = {}
        self.counts: dict[str, int] = {}

    def add(self, episode_id: str, signal: str, vectors: dict) -> None:
        self.done.add(episode_id)
        self.ch.setdefault(signal, []).append(vectors["ch"].half())
        sums = self.sums.setdefault(signal, {})
        for key, value in vectors.items():
            if key != "ch":
                sums[key] = sums.get(key, 0) + value.double()
        self.counts[signal] = self.counts.get(signal, 0) + 1

    def means(self, key: str) -> dict[str, torch.Tensor]:
        return {s: (v[key] / self.counts[s]).float() for s, v in self.sums.items() if key in v}

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"done": sorted(self.done), "ch": self.ch, "sums": self.sums, "counts": self.counts}, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "T3State":
        state = cls()
        if path.exists():
            saved = torch.load(path, map_location="cpu")
            state.done, state.ch, state.sums, state.counts = set(saved["done"]), saved["ch"], saved["sums"], saved["counts"]
        return state


def t3_summary(state: T3State, n_perm: int = 200) -> dict:
    v_ch = {s: torch.stack(v).float().mean(0) for s, v in state.ch.items()}
    tx = {name: state.means(f"tx_{name}") for name in ("trained", "base")}
    ins = state.means("ins_trained")
    out = {"n_pairs": state.counts, "by_signal": {}}
    for s, diffs in state.ch.items():
        diffs = torch.stack(diffs).float().numpy()  # [n, L + 1, d]
        n_points = diffs.shape[1]
        row = {k: [] for k in ("cos_trained", "cos_base", "cos_insert", "null_mean", "null_sd", "z_trained",
                               "mismatched_mean", "mismatched_max", "shared")}
        for l in range(n_points):
            target = tx["trained"][s][l].numpy()
            cos = mu.cosine(v_ch[s][l].numpy(), target)
            null_mean, null_sd = mu.signflip_null(diffs[:, l], target, n_perm=n_perm, seed=l)
            others = [mu.cosine(v_ch[s][l].numpy(), tx["trained"][o][l].numpy()) for o in tx["trained"] if o != s]
            others = [o for o in others if np.isfinite(o)]
            row["cos_trained"].append(cos)
            row["cos_base"].append(mu.cosine(v_ch[s][l].numpy(), tx["base"][s][l].numpy()) if s in tx["base"] else None)
            row["cos_insert"].append(mu.cosine(v_ch[s][l].numpy(), ins[s][l].numpy()))
            row["null_mean"].append(null_mean)
            row["null_sd"].append(null_sd)
            z = (cos - null_mean) / null_sd if null_sd and null_sd > 0 and np.isfinite(cos) else float("nan")
            row["z_trained"].append(z)
            row["mismatched_mean"].append(float(np.mean(others)) if others else float("nan"))
            row["mismatched_max"].append(float(np.max(others)) if others else float("nan"))
            row["shared"].append(bool(np.isfinite(z) and z > 3 and (not others or cos > max(others))))
        out["by_signal"][s] = row
    return out


def t3_add_vectors(state: T3State) -> dict:
    """Per signal and layer: the trained model's notice direction scaled to the norm of the
    channel direction ("tx"), and the channel direction itself ("ch")."""
    v_ch = {s: torch.stack(v).float().mean(0) for s, v in state.ch.items()}
    tx = state.means("tx_trained")
    vectors = {}
    for s in v_ch:
        for l in range(v_ch[s].shape[0]):
            ch, t = v_ch[s][l], tx[s][l]
            vectors[f"ch:{s}:{l}"] = ch
            vectors[f"tx:{s}:{l}"] = t * (ch.norm() / t.norm()) if t.norm() > 0 else torch.zeros_like(t)
    return vectors


# --- summary --------------------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"n": 0}
    arr = np.asarray(values)
    return {"n": len(arr), "mean_retained": float(arr.mean()), "median_retained": float(np.median(arr)),
            "sem": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else None}


def summarize(records: list[dict], n_layers: int) -> dict:
    split = [r for r in records if r["split"]]
    names = sorted({n for r in split for n in r["conditions"]})

    def table(subset):
        out = {}
        for name in names:
            got = [r for r in subset if name in r["conditions"]]
            values = [mu.retained(r["conditions"][name]["ld"], r["ld_intact"], r["ld_control"]) for r in got]
            out[name] = {**_stats(values),
                         "flip_rate": float(np.mean([not r["conditions"][name]["top_r"] for r in got])) if got else None}
        return out

    conditions = table(split)
    mean = lambda name: conditions.get(name, {}).get("mean_retained")  # noqa: E731
    greedy = [mean(f"greedy:k{k}") for k in range(1, n_layers + 1)]
    k80 = next((k for k, v in enumerate(greedy, 1) if v is not None and v <= 0.2), None)
    writers_all = mean("writers:all")
    summary = {
        "n_pairs": len(records), "n_split": len(split),
        "split_by_kind": {k: sum(1 for r in split if r["kind"] == k) for k in sorted({r["kind"] for r in records})},
        "pairs_by_kind": {k: sum(1 for r in records if r["kind"] == k) for k in sorted({r["kind"] for r in records})},
        "conditions": conditions,
        "by_kind": {k: table([r for r in split if r["kind"] == k]) for k in sorted({r["kind"] for r in split})},
        "by_signal": {s: table([r for r in split if r["signal"] == s]) for s in sorted({r["signal"] for r in split})},
        "H1": {"greedy_mean_retained": greedy, "k_for_80pct": k80, "max_k": int(math.floor(0.25 * n_layers)),
               "pass": None if all(v is None for v in greedy) else bool(k80 is not None and k80 <= math.floor(0.25 * n_layers))},
        "H2": {"writers_all_removed": None if writers_all is None else 1 - writers_all,
               "pass": None if writers_all is None else bool(1 - writers_all < 0.2)},
        "patch_curves": {part: [mean(f"patch_{part}:L{l}") for l in range(n_layers)] for part in ("main", "lanes")},
        "single_layer_readers": [mean(f"readers:L{l}") for l in range(n_layers)],
        "single_layer_writers": [mean(f"writers:L{l}") for l in range(n_layers)],
        "lanes_from": [mean("readers:all")] + [mean(f"lanes_from:L{l}") for l in range(1, n_layers)],
        "lanes_upto": [None] + [mean(f"lanes_upto:L{l}") for l in range(1, n_layers)],
        "prefix_off": {k: v.get("prefix_off:event", {}).get("mean_retained")
                       for k, v in {"all": conditions, **{k: t for k, t in
                                                          {k: table([r for r in split if r["kind"] == k])
                                                           for k in sorted({r["kind"] for r in split})}.items()}}.items()},
        "channel_off": mean("channel_off"),
    }
    off = summary["channel_off"]
    summary["red_flag"] = None if off is None else (
        "channel off everywhere keeps most of the logit difference: contradicts the behavioral control"
        if off > 0.5 else None)
    return summary


def summary_markdown(s: dict, t3: dict | None) -> str:
    f = lambda x: "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))  # noqa: E731
    c = s["conditions"]
    lines = ["# R1: B2 / B3 / T3 on minimal pairs", "",
             f"Pairs {s['n_pairs']} ({s['pairs_by_kind']}); scored (r != c) {s['n_split']} ({s['split_by_kind']}).",
             "Retained = (LD_condition - LD_control) / (LD_intact - LD_control); 1 = intact, 0 = control level.", "",
             "## Hypotheses", "",
             f"- H1 (cumulative ablation of <= {s['H1']['max_k']} layers removes >= 80%): k for 80% = "
             f"{f(s['H1']['k_for_80pct'])} -> {f(s['H1']['pass'])}",
             f"- H2 (zeroing all writers removes < 20%): removed {f(s['H2']['writers_all_removed'])} -> {f(s['H2']['pass'])}",
             f"- Channel off everywhere: retained {f(s['channel_off'])}; red flag: {f(s['red_flag'])}",
             f"- Prefix computed with the channel off (only the decision position reads the lanes): {s['prefix_off']}", "",
             "## Grouped ablations", "", "| condition | mean retained | SEM | flip rate | n |", "|---|---|---|---|---|"]
    for name in sorted(n for n in c if not any(n.startswith(p) for p in
                                                ("readers:L", "writers:L", "lanes_", "patch_", "greedy:", "t3_"))):
        lines.append(f"| {name} | {f(c[name].get('mean_retained'))} | {f(c[name].get('sem'))} | "
                     f"{f(c[name].get('flip_rate'))} | {c[name].get('n')} |")
    n_layers = len(s["single_layer_readers"])
    lines += ["", "## Per layer", "",
              "| layer | readers:L | writers:L | lanes cut from L up | lanes cut below L | greedy k=L+1 | patch main:L | patch lanes:L |",
              "|---|---|---|---|---|---|---|---|"]
    for l in range(n_layers):
        lines.append(f"| {l} | {f(s['single_layer_readers'][l])} | {f(s['single_layer_writers'][l])} | "
                     f"{f(s['lanes_from'][l])} | {f(s['lanes_upto'][l])} | {f(s['H1']['greedy_mean_retained'][l])} | "
                     f"{f(s['patch_curves']['main'][l])} | {f(s['patch_curves']['lanes'][l])} |")
    t3c = {n: v for n, v in c.items() if n.startswith("t3_")}
    if t3:
        lines += ["", "## T3: channel direction vs text-notice direction (cosine per layer)", "",
                  "| signal | layer | cos trained | cos base | cos insert | null mean (SD) | z | mismatched mean / max | shared |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for sig, row in t3["by_signal"].items():
            for l in range(len(row["cos_trained"])):
                lines.append(f"| {sig} | {l} | {f(row['cos_trained'][l])} | {f(row['cos_base'][l])} | "
                             f"{f(row['cos_insert'][l])} | {f(row['null_mean'][l])} ({f(row['null_sd'][l])}) | "
                             f"{f(row['z_trained'][l])} | {f(row['mismatched_mean'][l])} / {f(row['mismatched_max'][l])} | "
                             f"{row['shared'][l]} |")
    if t3c:
        lines += ["", "## T3: adding a direction at layer L with the channel off (retained, flip rate)", "",
                  "| layer | add v_tx (norm of v_ch) | add v_ch |", "|---|---|---|"]
        lines.append(f"| none | {f(t3c.get('t3_off', {}).get('mean_retained'))} | - |")
        for l in range(n_layers):
            tx, ch = t3c.get(f"t3_tx:L{l}", {}), t3c.get(f"t3_ch:L{l}", {})
            lines.append(f"| {l} | {f(tx.get('mean_retained'))} ({f(tx.get('flip_rate'))}) | "
                         f"{f(ch.get('mean_retained'))} ({f(ch.get('flip_rate'))}) |")
    return "\n".join(lines) + "\n"


# --- pipeline -------------------------------------------------------------------------------------

def run_mechanism(model, tok, ms, specs, episodes: list[Episode], out: Path, batch: int = 16, t3: bool = True,
                  t3_base: bool = True, n_perm: int = 200, log_every: int = 25) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in (mu.make_pair(ep, specs, ms) for ep in episodes) if p is not None]
    pairs_path, phase2_path, state_path = out / "pairs.jsonl", out / "phase2.jsonl", out / "t3_state.pt"
    done = {r["episode_id"]: r for r in _read_jsonl(pairs_path)}
    state = T3State.load(state_path) if t3 else None
    started = time.time()
    with mu.Interventions(model) as iv:
        conds = b2_conditions(iv)
        # phase 1
        for index, pair in enumerate(pairs):
            need_b2 = pair.episode_id not in done
            need_t3 = t3 and pair.episode_id not in state.done
            if not (need_b2 or need_t3):
                continue
            result = analyze_pair(iv, pair, conds if need_b2 else [], batch) if need_b2 else None
            if need_b2:
                rec, kv_off = result.pop("_rec"), result.pop("_kv_off", None)
                result["index"] = index
                done[pair.episode_id] = result
            else:
                kv = iv.prefix(pair.ids, pair.event)
                _, rec = iv.last(kv, pair.ids[-1], np.stack([pair.event[-1], pair.control[-1]]), record=True)
                kv_off = None
            if need_t3:
                kv_off = kv_off if kv_off is not None else iv.prefix(pair.ids, pair.event, enabled=False)
                state.add(pair.episode_id, pair.signal, t3_collect(iv, pair, tok, specs, rec, kv_off, base=t3_base))
            if need_b2:  # written after T3 so a cut between the two redoes both
                _append_jsonl(pairs_path, result)
            if t3 and (index + 1) % log_every == 0:
                state.save(state_path)
            if log_every and (index + 1) % log_every == 0:
                print(f"phase 1: {index + 1}/{len(pairs)} pairs, {(time.time() - started) / 60:.1f} min", flush=True)
        if t3:
            state.save(state_path)
        records = [done[p.episode_id] for p in pairs]
        for k, r in enumerate(records):
            r["index"] = k
        # phase 2: cross-fitted greedy ablation; T3 additions
        phase2 = {r["episode_id"]: r for r in _read_jsonl(phase2_path)}
        orders = {parity: greedy_order(records, parity, iv.n_layers) for parity in (0, 1)}
        add_vectors = t3_add_vectors(state) if t3 else None
        for k, pair in enumerate(pairs):
            record = records[k]
            if not record["split"] or pair.episode_id in phase2:
                continue
            r, c = record["r"], record["c"]
            kv = iv.prefix(pair.ids, pair.event)
            got = _scores(iv, kv, pair, greedy_conditions(iv, orders[k % 2]), r, c, batch)
            del kv
            if t3:
                kv_off = iv.prefix(pair.ids, pair.event, enabled=False)
                s = pair.signal
                t3_conds = [mu.Cond("t3_off")]
                t3_conds += [mu.Cond(f"t3_tx:L{l}", add=(l, f"tx:{s}:{l}")) for l in range(iv.n_layers)]
                t3_conds += [mu.Cond(f"t3_ch:L{l}", add=(l, f"ch:{s}:{l}")) for l in range(iv.n_layers)]
                got.update(_scores(iv, kv_off, pair, t3_conds, r, c, batch, add_vectors=add_vectors, enabled=False))
            row = {"episode_id": pair.episode_id, "conditions": got, "order": orders[k % 2]}
            _append_jsonl(phase2_path, row)
            phase2[pair.episode_id] = row
            if log_every and (k + 1) % log_every == 0:
                print(f"phase 2: {k + 1}/{len(pairs)} pairs, {(time.time() - started) / 60:.1f} min", flush=True)
    for r in records:
        if r["episode_id"] in phase2:
            r["conditions"] = {**r["conditions"], **phase2[r["episode_id"]]["conditions"]}
    summary = summarize(records, iv.n_layers)
    summary["greedy_orders"] = orders
    summary["t3"] = t3_summary(state, n_perm=n_perm) if t3 else None
    _write(out / "summary.json", json.dumps(clean(summary), indent=1))
    _write(out / "summary.md", summary_markdown(clean(summary), clean(summary["t3"])))
    return summary


# --- B3 probes ------------------------------------------------------------------------------------

def _decision_position(ep: Episode) -> int | None:
    n = len(ep.prompt_ids)
    if ep.kind == "gating_event" and ep.t_decision is not None:
        return n + ep.t_decision - 1
    return None if ep.t_cross is None else n + ep.t_cross


def probe_rows(ep: Episode, specs, ms, rng, n_random: int = 3, after=(2, 4, 8)) -> list[dict]:
    """Positions to probe in one episode, with their labels."""
    n, total = len(ep.prompt_ids), len(ep.prompt_ids) + len(ep.target_ids)
    decision = _decision_position(ep)
    if decision is None or decision >= total:
        return []
    names = [s.name for s in specs]
    positions = {decision}
    if decision - 1 >= n:
        positions.add(decision - 1)
    if decision - 1 > n:
        positions.update(int(p) for p in rng.choice(np.arange(n, decision - 1), size=min(n_random, decision - 1 - n),
                                                    replace=False))
    positions.update(decision + a for a in after if decision + a < total)
    start = {"event": n + (ep.t_cross or 0) + 1, "starts_past": n + (ep.t_cross or 0) + 1,
             "gating_event": n + (ep.t_decision or 0)}.get(ep.kind)
    ids = list(ep.prompt_ids) + list(ep.target_ids)
    rows = []
    for p in sorted(positions):
        values = ep.signals[p]
        past = [spec.past_limit(float(v)) for spec, v in zip(specs, values)]
        which = ep.signal if ep.signal in names and past[names.index(ep.signal)] else (
            names[past.index(True)] if any(past) else "none")
        seen = ep.target_ids[: p - n + 1]
        thinking = ms.think_open in seen and ms.think_close not in seen
        rows.append({
            "episode_id": ep.episode_id, "task_id": ep.task_id, "kind": ep.kind, "family": ep.family,
            "position": int(p), "token": int(ids[p]), "past_any": int(any(past)), "which": which,
            "phase": "thinking" if thinking else "answer",
            "reaction": (int(start is not None and p >= start) if any(past) else None),
            "values": {name: float(v) for name, v in zip(names, values)},
            "value_ok": {spec.name: bool(not past[i] and 0.02 < float(values[i]) < 0.98 and spec.dynamics != "binary")
                         for i, spec in enumerate(specs)},
        })
    return rows


def _spread(items: list, n: int | None) -> list:
    if n is None or n >= len(items):
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


PROBE_CONDITIONS = {"on": (True, True), "off": (False, True), "base": (False, False)}


def run_probes(model, ms, specs, episodes: list[Episode], out: Path, n_random: int = 3, max_episodes: int | None = None,
               l2: float = 1.0, fit_device: str | None = None, seed: int = 0) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    eps = _spread([ep for ep in episodes if _decision_position(ep) is not None], max_episodes)
    meta, acts = [], {c: {} for c in PROBE_CONDITIONS}
    started = time.time()
    with mu.Interventions(model) as iv:
        d = iv.d
        fit_device = fit_device or str(iv.device)
        for k, ep in enumerate(eps):
            rows = probe_rows(ep, specs, ms, rng, n_random)
            if not rows:
                continue
            ids = list(ep.prompt_ids) + list(ep.target_ids)
            positions = [r["position"] for r in rows]
            for cond, (enabled, adapter) in PROBE_CONDITIONS.items():
                rec = iv.collect(ids, ep.signals[: len(ids)], positions, enabled=enabled, adapter=adapter)
                for layer, values in rec.items():
                    acts[cond].setdefault(layer, []).append(values.half())
            meta += rows
            if (k + 1) % 50 == 0:
                print(f"probes: {k + 1}/{len(eps)} episodes, {(time.time() - started) / 60:.1f} min", flush=True)
    test = mu.task_split([m["task_id"] for m in meta], seed=seed)
    tokens = [m["token"] for m in meta]
    names = [s.name for s in specs]

    def labels(key):
        return np.array([m[key] for m in meta], dtype=object)

    def fit_class(x, y, mask=None):
        mask = np.ones(len(y), bool) if mask is None else mask
        tr, te = mask & ~test, mask & test
        if len(set(y[tr])) < 2 or te.sum() < 2:
            return None
        return mu.fit_logistic(x[tr], y[tr].astype(str), x[te], y[te].astype(str), l2=l2, device=fit_device)

    result = {"n_positions": len(meta), "n_test": int(test.sum()), "n_episodes": len(eps), "conditions": {}}
    past_any, which, phase = labels("past_any"), labels("which"), labels("phase")
    family, reaction = labels("family"), labels("reaction")
    react_mask = np.array([r is not None for r in reaction])
    control = mu.control_labels(tokens, past_any.astype(int), seed=seed).astype(object)
    for cond in PROBE_CONDITIONS:
        layers = sorted(acts[cond])
        streams = {"main": slice(0, d)}
        width = acts[cond][layers[0]][0].shape[-1]
        if width > d:
            streams["lanes"] = slice(d, width)
        result["conditions"][cond] = {}
        for stream, part in streams.items():
            per_layer = []
            for layer in layers:
                x = torch.cat(acts[cond][layer]).float().numpy()[:, part]
                row = {"layer": layer,
                       "past_any": fit_class(x, past_any), "past_any_control_task": fit_class(x, control),
                       "which": fit_class(x, which), "phase": fit_class(x, phase),
                       "reaction": fit_class(x, reaction, react_mask), "family": fit_class(x, family)}
                for i, name in enumerate(names):
                    ok = np.array([m["value_ok"][name] for m in meta])
                    if (ok & ~test).sum() >= 5 and (ok & test).sum() >= 2:
                        v = np.array([m["values"][name] for m in meta])
                        row[f"value:{name}"] = mu.fit_ridge(x[ok & ~test], v[ok & ~test], x[ok & test], v[ok & test])
                per_layer.append(row)
            result["conditions"][cond][stream] = per_layer
    _write(out / "probes.json", json.dumps(clean(result), indent=1))
    _write(out / "probes.md", probes_markdown(clean(result), names))
    return result


def probes_markdown(result: dict, names: list[str]) -> str:
    f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    acc = lambda r: "-" if r is None else f"{f(r.get('acc_test'))} ({f(r.get('majority_test'))})"  # noqa: E731
    lines = ["# R1 / B3: linear probes per layer", "",
             f"{result['n_positions']} positions from {result['n_episodes']} episodes; held-out tasks: {result['n_test']} positions.",
             "Accuracy on held-out tasks (majority-class rate in brackets); value = ridge R^2 inside the nominal range.", ""]
    for cond, streams in result["conditions"].items():
        for stream, rows in streams.items():
            lines += [f"## {cond} / {stream}", "",
                      "| layer | past-limit | control task | which signal | phase | reaction | family | "
                      + " | ".join(f"value {n}" for n in names) + " |",
                      "|---|---|---|---|---|---|---|" + "---|" * len(names)]
            for r in rows:
                values = " | ".join(f(r.get(f"value:{n}", {}).get("r2_test")) if r.get(f"value:{n}") else "-" for n in names)
                lines.append(f"| {r['layer']} | {acc(r['past_any'])} | {acc(r['past_any_control_task'])} | "
                             f"{acc(r['which'])} | {acc(r['phase'])} | {acc(r['reaction'])} | {acc(r['family'])} | {values} |")
            lines.append("")
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="R1 mechanism runs (B2 ablations, B3 patching / probes, T3).")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="cuda (bf16, as the evaluation) | cpu (float32, slow)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--kinds", default=",".join(PAIR_KINDS))
    parser.add_argument("--every", type=int, default=1, help="every n-th pair (quick reading)")
    parser.add_argument("--batch", type=int, default=16, help="conditions per batched pass of the decision position")
    parser.add_argument("--no-t3", action="store_true")
    parser.add_argument("--no-t3-base", action="store_true", help="skip the base-model notice direction")
    parser.add_argument("--no-b2", action="store_true", help="probes only")
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--probe-episodes", type=int, default=400)
    args = parser.parse_args(argv)

    import prio.train as train_module
    from prio.modelspec import load_base

    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    n_signals = manifest["n_signals"]
    if args.device == "cpu":
        torch.set_num_threads(4)
        train_module.load_base = lambda model_id: load_base(model_id, dtype=torch.float32, device="cpu")
    model_id = mu.run_model_id(args.run, manifest)
    model, tok, ms = train_module.load_run(model_id, args.run, n_signals)
    specs = signal_set(n_signals)
    episodes = read_jsonl(args.data / f"{args.split}.jsonl")
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"invocations": []}
    entry = {"run": str(args.run).replace("\\", "/"), "data": str(args.data).replace("\\", "/"),
             "model": model_id, "channel": train_module.channel_record(args.run),
             "args": {k: str(v).replace("\\", "/") if isinstance(v, Path) else v for k, v in vars(args).items()},
             "started": datetime.now().strftime("%Y-%m-%d %H:%M")}
    meta["invocations"].append(entry)
    _write(meta_path, json.dumps(meta, indent=1))
    if not args.no_b2:
        kinds = set(args.kinds.split(","))
        chosen = [ep for ep in episodes if ep.kind in kinds][:: args.every]
        print(f"{args.run}: {len(chosen)} episodes of kinds {sorted(kinds)}", flush=True)
        run_mechanism(model, tok, ms, specs, chosen, args.out, batch=args.batch, t3=not args.no_t3,
                      t3_base=not args.no_t3_base)
    if args.probes:
        run_probes(model, ms, specs, episodes, args.out / "probes", max_episodes=args.probe_episodes)
    entry["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _write(meta_path, json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
