"""B1 (preregistration R1): what the level table and the readers / writers learned.
CPU, seconds per run; reads only the channel file (`channel.pt` or `channel.safetensors`) and the
training data for visit counts.

Per signal, on the 100 level vectors of the `levels` projector (table weight [lanes, 6 x 100]):
- neighbor change ||v[k+1] - v[k]||: is the largest step at the trained limit?
- projection on d_s = mean(past-limit vectors) - mean(nominal vectors): step or ramp
  (jump across the limit / range; R^2 of a step fit vs a linear fit);
- Spearman(level, PC1) inside the nominal range: an ordered value code?
- mean cosine within the past cluster, within the nominal one, across;
- drift from the initial table, reconstructed exactly from the run's seed (the projector is the
  first random draw after `torch.manual_seed(seed)` in train.py; never-visited levels must equal
  it: checked) vs how often the level was visited in training;
- the same on the lanes as the readers see them, unit(sum of 6 vectors + bias), averaged over
  real nominal contexts (the map is nonlinear);
- readers: ||C d_s|| per layer and matrix (C = reader columns, d_s unit-size crossing direction),
  vs the gain of a random direction; writers: norm, effective rank, overlap of their top output
  directions with d_s;
- across runs: orthogonal Procrustes on visited levels, matched cosine vs a permuted null, and
  correlation of the cosine-similarity matrices (basis-free).
Statistics use visited levels only (>= --min-visits training tokens); rarely visited vectors stay
near their random start and would be noise.

Usage (project root):
  uv run python scripts/analysis/level_geometry.py --runs runs/n6-wide-levels-s0 runs/n6-wide-levels-s1 \
      runs/n6-wide-levels-s2 runs/n6-lanes-levels-s0 --data data/n6 --out results/mechanism/b1
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from prio.channel import SignalProjector
from prio.release import load_channel_tensors
from prio.signals import SignalSpec, signal_set

BINS = 100
CENTERS = (np.arange(BINS) + 0.5) / BINS
_LAYER = re.compile(r"(?:^|-)(?:layers|h)-(\d+)-")


# --- table ----------------------------------------------------------------------------------------

def level_vectors(state: dict, n_signals: int = 6, bins: int = BINS) -> np.ndarray:
    """[N, bins, width]: the vector of each 1% level of each signal (columns of the table)."""
    weight = state["net.0.weight"].float().numpy()  # [width, N * bins]
    return weight.T.reshape(n_signals, bins, weight.shape[0])


def init_table(seed: int, n_signals: int = 6, width: int = 64, bins: int = BINS) -> np.ndarray:
    """The table a run started from: the first random draw after torch.manual_seed(seed)."""
    torch.manual_seed(seed)
    return nn.Linear(n_signals * bins, width).weight.detach().numpy()


def limit_step(spec: SignalSpec) -> int | None:
    """Index k of the step level k -> k+1 that crosses the limit; None for a binary signal."""
    if spec.dynamics == "binary":
        return None
    return int(round(spec.limit * BINS)) - 1


def past_levels(spec: SignalSpec) -> np.ndarray:
    if spec.dynamics == "binary":  # value 0 -> level 0, value 1 -> level 99
        return np.arange(BINS) == (0 if spec.past_limit(0.0) else BINS - 1)
    return np.array([spec.past_limit(float(c)) for c in CENTERS])


def visit_counts(rows_list, n_signals: int = 6, bins: int = BINS) -> np.ndarray:
    """[N, bins] training tokens per level (a value between two centers is shared, as in the code)."""
    counts = np.zeros((n_signals, bins))
    for rows in rows_list:
        pos = np.clip(np.round(np.asarray(rows, np.float64) * bins - 0.5, 4), 0, bins - 1)
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        hi = np.minimum(lo + 1, bins - 1)
        for s in range(n_signals):
            counts[s] += np.bincount(lo[:, s], weights=1 - frac[:, s], minlength=bins)
            counts[s] += np.bincount(hi[:, s], weights=frac[:, s], minlength=bins)
    return counts


# --- statistics -----------------------------------------------------------------------------------

def _rank(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for value in np.unique(x):  # ties: average rank
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Orthogonal R minimizing ||a R - b||."""
    u, _, vt = np.linalg.svd(a.T @ b)
    return u @ vt


def _unit_rows(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _mean_cos(a, b=None) -> float:
    a = _unit_rows(a)
    if b is None:
        if len(a) < 2:
            return float("nan")
        sim = a @ a.T
        return float((sim.sum() - np.trace(sim)) / (len(a) * (len(a) - 1)))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float((a @ _unit_rows(b).T).mean())


def _r2(y, design) -> float:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - design @ coef) ** 2) / ss) if ss > 0 else float("nan")


def crossing_direction(table: np.ndarray, spec: SignalSpec, visits=None, min_visits: float = 1) -> np.ndarray:
    visited = np.ones(BINS, bool) if visits is None else np.asarray(visits) >= min_visits
    past = past_levels(spec)
    return table[past & visited].mean(0) - table[~past & visited].mean(0)


def geometry(table: np.ndarray, spec: SignalSpec, visits=None, min_visits: float = 1) -> dict:
    """Step / ramp statistics of one signal's 100 level vectors (see the module docstring)."""
    visited = np.ones(BINS, bool) if visits is None else np.asarray(visits) >= min_visits
    past = past_levels(spec)
    out = {"n_visited": int(visited.sum()), "norms": np.linalg.norm(table, axis=1).tolist()}
    if not (past & visited).any() or not (~past & visited).any():
        return {**out, "note": "no visited level on one side of the limit"}
    d = crossing_direction(table, spec, visits, min_visits)
    proj = table @ (d / np.linalg.norm(d))
    out.update(
        projection=proj.tolist(),
        cos_within_past=_mean_cos(table[past & visited]),
        cos_within_nominal=_mean_cos(table[~past & visited]),
        cos_across=_mean_cos(table[past & visited], table[~past & visited]),
        crossing_norm=float(np.linalg.norm(d)),
    )
    if spec.dynamics == "binary":
        a, b = table[0], table[-1]
        out.update(binary_distance=float(np.linalg.norm(a - b)),
                   binary_cos=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        return out
    change = np.linalg.norm(np.diff(table, axis=0), axis=1)
    steps = np.flatnonzero(visited[:-1] & visited[1:])
    k = limit_step(spec)
    order = steps[np.argsort(-change[steps])]
    span = proj[visited].max() - proj[visited].min()
    nominal = np.flatnonzero(~past & visited)
    centered = table[nominal] - table[nominal].mean(0)
    pc1 = centered @ np.linalg.svd(centered, full_matrices=False)[2][0] if len(nominal) > 2 else np.zeros(len(nominal))
    lv = np.flatnonzero(visited)
    # The level a crossing token carries (first past level) and the one just before it.
    cross, before = (k + 1, k) if spec.bad_when == "high" else (k, k + 1)
    norms = np.linalg.norm(table, axis=1)
    median_norm = float(np.median(norms[visited]))
    out.update(
        crossing_level=int(cross), last_nominal_level=int(before),
        crossing_level_norm_ratio=float(norms[cross] / median_norm),
        last_nominal_level_norm_ratio=float(norms[before] / median_norm),
        crossing_level_projection_vs_other_past=float(proj[cross] / np.mean(proj[np.flatnonzero(past & visited & (np.arange(BINS) != cross))]))
        if (past & visited).sum() > 1 else float("nan"),
    )
    out.update(
        neighbor_change=change.tolist(),
        neighbor_cos=[float(x) for x in np.sum(_unit_rows(table[:-1]) * _unit_rows(table[1:]), axis=1)],
        largest_step=int(order[0]) if len(order) else None,
        largest_step_at_limit=bool(len(order) and order[0] == k),
        limit_step_rank=int(np.flatnonzero(order == k)[0]) + 1 if k in order else None,
        limit_step_ratio=float(change[k] / np.median(change[steps])) if k in steps else float("nan"),
        step_fraction=float(abs(proj[k + 1] - proj[k]) / span) if k in steps and span > 0 else float("nan"),
        step_r2=_r2(proj[lv], np.stack([np.ones(len(lv)), past[lv].astype(float)], 1)),
        ramp_r2=_r2(proj[lv], np.stack([np.ones(len(lv)), lv.astype(float)], 1)),
        spearman_pc1_nominal=spearman(nominal, pc1),
        spearman_projection_nominal=spearman(nominal, proj[nominal]),
    )
    return out


def reader_profile(state: dict, d) -> list[dict]:
    """Per reader matrix: gain ||C d^|| of the unit crossing direction, the RMS gain of a random
    unit direction (||C||_F / sqrt(lanes)) and their ratio."""
    d = np.asarray(d, np.float64)
    d = d / np.linalg.norm(d)
    rows = []
    for key, value in state.items():
        if not key.startswith("readers."):
            continue
        name = key.removeprefix("readers.")
        c = value.double().numpy()
        gain = float(np.linalg.norm(c @ d))
        rms = float(np.linalg.norm(c) / np.sqrt(c.shape[1]))
        rows.append({"layer": int(_LAYER.search(name).group(1)), "proj": name.split("-")[-1], "gain": gain,
                     "rms_gain": rms, "gain_ratio": gain / rms if rms > 0 else float("nan")})
    return sorted(rows, key=lambda r: (r["layer"], r["proj"]))


def writer_profile(state: dict, d, top: int = 3) -> list[dict]:
    """Per writer matrix (rows [lanes, in]): Frobenius norm, effective rank (participation ratio
    of singular values), and ||U_top^T d^||: the share of the crossing direction inside the top
    output directions (random expectation sqrt(top / lanes))."""
    d = np.asarray(d, np.float64)
    d = d / np.linalg.norm(d)
    rows = []
    for key, value in state.items():
        if not key.startswith("writers."):
            continue
        name = key.removeprefix("writers.")
        w = value.double().numpy()
        u, s, _ = np.linalg.svd(w, full_matrices=False)
        rows.append({"layer": int(_LAYER.search(name).group(1)), "proj": name.split("-")[-1],
                     "frob": float(np.linalg.norm(w)),
                     "effective_rank": float(s.sum() ** 2 / (s ** 2).sum()) if s.sum() > 0 else 0.0,
                     "overlap_top": float(np.linalg.norm(u[:, :top].T @ d))})
    return sorted(rows, key=lambda r: (r["layer"], r["proj"]))


def normalized_levels(state: dict, contexts: np.ndarray, s: int, n_signals: int = 6) -> np.ndarray:
    """[100, width]: mean over `contexts` (rows of signal values) of unit(projector output) with
    signal s set to each level center: the lanes as the readers see them."""
    proj = SignalProjector(n_signals, state["net.0.weight"].shape[0], zero_init=False, kind="levels")
    proj.load_state_dict({k: v.float() for k, v in state.items() if k.startswith("net.")})
    rows = np.repeat(contexts[None], BINS, axis=0).astype(np.float32)  # [100, M, N]
    rows[:, :, s] = CENTERS[:, None]
    with torch.no_grad():
        out = proj(torch.from_numpy(rows)).double().numpy()
    out = out / np.sqrt((out ** 2).mean(-1, keepdims=True) + 1e-6)
    return out.mean(1)


# --- runs -----------------------------------------------------------------------------------------

def run_seed(run: Path) -> int:
    log = run / "train_log.json"
    if log.exists():
        try:
            seed = json.loads(log.read_text(encoding="utf-8")).get("args", {}).get("seed")
            if seed is not None:
                return int(seed)
        except (json.JSONDecodeError, AttributeError):
            pass
    found = re.search(r"-s(\d+)$", run.name)
    if not found:
        raise ValueError(f"cannot tell the seed of {run}")
    return int(found.group(1))


def load_data_rows(data: Path, split: str = "train"):
    """Signal rows of every episode, streamed (the token lists are dropped)."""
    import base64

    with open(data / f"{split}.jsonl", encoding="utf-8") as lines:
        for line in lines:
            packed = json.loads(line)["signals"]
            yield np.frombuffer(base64.b64decode(packed["b64"]), dtype=np.float32).reshape(packed["shape"])


def _round(x, nd=4):
    if isinstance(x, float):
        return round(x, nd) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if isinstance(x, (np.floating,)):
        return _round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def analyze_run(run: Path, specs, visits, contexts, min_visits: float) -> tuple[dict, np.ndarray, dict]:
    state = load_channel_tensors(run)  # channel.pt or channel.safetensors
    record = json.loads((run / "channel.json").read_text(encoding="utf-8")) if (run / "channel.json").exists() else {}
    if record.get("projector") != "levels":
        raise ValueError(f"{run}: B1 needs a levels projector, got {record}")
    n = len(specs)
    table = level_vectors(state, n)
    width = table.shape[-1]
    seed = run_seed(run)
    init = init_table(seed, n, width).T.reshape(n, BINS, width)
    never = visits == 0
    out = {"run": str(run).replace("\\", "/"), "channel": record, "seed": seed, "width": width,
           "init_check": {"never_visited_levels": int(never.sum()),
                          "max_abs_diff_never_visited": float(np.abs(table[never] - init[never]).max())
                          if never.any() else None},
           "expected_init_norm": float(np.sqrt(width / 3) / np.sqrt(n * BINS)),
           "signals": {}}
    directions = {}
    for s, spec in enumerate(specs):
        g = geometry(table[s], spec, visits[s], min_visits)
        drift = np.linalg.norm(table[s] - init[s], axis=1)
        cos_init = np.sum(_unit_rows(table[s]) * _unit_rows(init[s]), axis=1)
        rare, frequent = visits[s] < 10, visits[s] >= 1000
        normed = normalized_levels(state, contexts, s, n)
        gn = geometry(normed, spec, visits[s], min_visits)
        d_unit = crossing_direction(normed, spec, visits[s], min_visits)
        directions[spec.name] = d_unit
        readers = reader_profile(state, d_unit)
        layers = sorted({r["layer"] for r in readers})
        per_layer = np.array([np.sqrt(sum(r["gain"] ** 2 for r in readers if r["layer"] == layer)) for layer in layers])
        ratio = np.array([np.mean([r["gain_ratio"] for r in readers if r["layer"] == layer]) for layer in layers])
        top = max(1, int(round(0.25 * len(layers))))
        order = np.argsort(-per_layer)
        writers = writer_profile(state, d_unit) if any(k.startswith("writers.") for k in state) else []
        writer_norm = [float(np.sqrt(sum(w["frob"] ** 2 for w in writers if w["layer"] == layer))) for layer in layers]
        out["signals"][spec.name] = {
            "table": g,
            "lanes_seen_by_readers": {k: gn.get(k) for k in (
                "largest_step", "largest_step_at_limit", "limit_step_rank", "limit_step_ratio", "step_fraction",
                "step_r2", "ramp_r2", "spearman_pc1_nominal", "spearman_projection_nominal", "cos_within_past",
                "cos_within_nominal", "cos_across", "binary_distance", "binary_cos", "projection", "crossing_norm")
                if k in gn},
            "drift": {"distance_from_init": drift.tolist(), "cos_to_init": cos_init.tolist(),
                      "spearman_log_visits_vs_drift": spearman(np.log1p(visits[s]), drift),
                      "mean_cos_to_init_rare": float(cos_init[rare].mean()) if rare.any() else None,
                      "mean_cos_to_init_frequent": float(cos_init[frequent].mean()) if frequent.any() else None,
                      "n_rare": int(rare.sum()), "n_frequent": int(frequent.sum())},
            "readers": {"per_layer_gain": per_layer.tolist(),
                        "top_quarter_layers": [int(layers[i]) for i in order[:top]],
                        "top_quarter_share_of_gain2": float((per_layer[order[:top]] ** 2).sum() / (per_layer ** 2).sum()),
                        "per_layer_gain_ratio": ratio.tolist(),
                        "gain_ratio_min_median_max": [float(ratio.min()), float(np.median(ratio)), float(ratio.max())],
                        "top_quarter_layers_by_ratio": [int(layers[i]) for i in np.argsort(-ratio)[:top]],
                        "per_matrix": readers},
            "writers": {"per_matrix": writers, "per_layer_frob": writer_norm,
                        "median_effective_rank": float(np.median([w["effective_rank"] for w in writers])) if writers else None,
                        "random_overlap": float(np.sqrt(3 / width))} if writers else None,
        }
    return out, table, directions


def compare_runs(name_a, table_a, dirs_a, name_b, table_b, dirs_b, specs, visits, min_visits, n_perm=20, seed=0):
    """Procrustes alignment of run a's lane space onto run b's, fit on every visited level."""
    visited = visits >= min_visits  # [N, 100]
    a, b = table_a[visited], table_b[visited]
    rot = procrustes(a, b)
    rng = np.random.default_rng(seed)
    null_rots = [procrustes(a, b[rng.permutation(len(b))]) for _ in range(n_perm)]
    out = {"a": name_a, "b": name_b, "signals": {}}
    for s, spec in enumerate(specs):
        va, vb = table_a[s][visited[s]], table_b[s][visited[s]]
        matched = float(np.mean(np.sum(_unit_rows(va @ rot) * _unit_rows(vb), axis=1)))
        null = [float(np.mean(np.sum(_unit_rows(va @ r) * _unit_rows(vb), axis=1))) for r in null_rots]
        sa, sb = _unit_rows(va) @ _unit_rows(va).T, _unit_rows(vb) @ _unit_rows(vb).T
        upper = np.triu_indices(len(va), 1)
        rsa = float(np.corrcoef(sa[upper], sb[upper])[0, 1]) if len(upper[0]) > 2 else float("nan")
        da, db = dirs_a[spec.name] @ rot, dirs_b[spec.name]
        out["signals"][spec.name] = {"aligned_matched_cos": matched, "null_mean": float(np.mean(null)),
                                     "null_sd": float(np.std(null)), "rsa_r": rsa,
                                     "aligned_crossing_direction_cos": float(da @ db / (np.linalg.norm(da) * np.linalg.norm(db))),
                                     "n_levels": int(visited[s].sum())}
    return out


def markdown(report: dict) -> str:
    lines = ["# B1: level-table geometry", "",
             f"Visited level = at least {report['min_visits']} training tokens (`{report['data']}`).", ""]
    lines += ["## Step at the limit (raw table | lanes as the readers see them)", "",
              "| run | signal | visited | largest step at limit | limit step / median | jump / range | step R^2 | ramp R^2 "
              "| Spearman PC1 nominal | cos past / nominal / across | readers' view: jump / range, Spearman PC1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))  # noqa: E731
    for run in report["runs"]:
        name = Path(run["run"]).name
        for sig, r in run["signals"].items():
            t, v = r["table"], r["lanes_seen_by_readers"]
            if "binary_distance" in t:
                step = f"binary: distance {f(t['binary_distance'])}, cos {f(t['binary_cos'])}"
                lines.append(f"| {name} | {sig} | {t['n_visited']} | {step} | - | - | - | - | - | "
                             f"{f(t.get('cos_within_past'))} / {f(t.get('cos_within_nominal'))} / {f(t.get('cos_across'))} | "
                             f"binary cos {f(v.get('binary_cos'))} |")
                continue
            lines.append(
                f"| {name} | {sig} | {t['n_visited']} | {t.get('largest_step_at_limit')} (k={t.get('largest_step')}, "
                f"rank {t.get('limit_step_rank')}) | {f(t.get('limit_step_ratio'))} | {f(t.get('step_fraction'))} | "
                f"{f(t.get('step_r2'))} | {f(t.get('ramp_r2'))} | {f(t.get('spearman_pc1_nominal'))} | "
                f"{f(t.get('cos_within_past'))} / {f(t.get('cos_within_nominal'))} / {f(t.get('cos_across'))} | "
                f"{f(v.get('step_fraction'))}, {f(v.get('spearman_pc1_nominal'))} |")
    lines += ["", "## Drift from the initial table", "",
              "| run | init check: max diff on never-visited levels | signal | cos to init, rare (<10 tokens) | "
              "cos to init, frequent (>=1000) | Spearman(log visits, drift) |", "|---|---|---|---|---|---|"]
    for run in report["runs"]:
        name = Path(run["run"]).name
        for sig, r in run["signals"].items():
            dr = r["drift"]
            lines.append(f"| {name} | {f(run['init_check']['max_abs_diff_never_visited'])} | {sig} | "
                         f"{f(dr['mean_cos_to_init_rare'])} (n={dr['n_rare']}) | "
                         f"{f(dr['mean_cos_to_init_frequent'])} (n={dr['n_frequent']}) | "
                         f"{f(dr['spearman_log_visits_vs_drift'])} |")
    lines += ["", "## Crossing level vs the rest (raw table)", "",
              "| run | signal | crossing level | its norm / median norm | level before it: norm / median | "
              "crossing-level projection / mean of other past levels |", "|---|---|---|---|---|---|"]
    for run in report["runs"]:
        name = Path(run["run"]).name
        for sig, r in run["signals"].items():
            t = r["table"]
            if "crossing_level" in t:
                lines.append(f"| {name} | {sig} | {t['crossing_level']} | {f(t['crossing_level_norm_ratio'])} | "
                             f"{f(t['last_nominal_level_norm_ratio'])} | {f(t['crossing_level_projection_vs_other_past'])} |")
    lines += ["", "## Readers: where the crossing direction is read (static gain)", "",
              "Gain = ||C d_s|| per layer (all reader matrices); ratio = gain / gain of a random unit direction.", "",
              "| run | signal | top-quarter layers by gain | their share of gain^2 | ratio min / median / max | "
              "top-quarter layers by ratio | writers: median eff. rank | writers: overlap of top-3 outputs with d_s (max layer) |",
              "|---|---|---|---|---|---|---|---|"]
    for run in report["runs"]:
        name = Path(run["run"]).name
        for sig, r in run["signals"].items():
            w = r["writers"]
            rd = r["readers"]
            best = max(w["per_matrix"], key=lambda x: x["overlap_top"]) if w else None
            lines.append(f"| {name} | {sig} | {rd['top_quarter_layers']} | {f(rd['top_quarter_share_of_gain2'])} | "
                         f"{' / '.join(f(x) for x in rd['gain_ratio_min_median_max'])} | {rd['top_quarter_layers_by_ratio']} | "
                         f"{f(w['median_effective_rank']) if w else '-'} | "
                         + (f"{f(best['overlap_top'])} (L{best['layer']} {best['proj']}; random {f(w['random_overlap'])})"
                            if best else "-") + " |")
    wide = [run for run in report["runs"] if run["signals"][next(iter(run["signals"]))]["writers"]]
    if wide:
        lines += ["", "## Writers: Frobenius norm per layer (started at zero)", "",
                  "| run | " + " | ".join(f"L{k}" for k in range(len(wide[0]["signals"]["ctx"]["writers"]["per_layer_frob"])))
                  + " |", "|---|" + "---|" * len(wide[0]["signals"]["ctx"]["writers"]["per_layer_frob"])]
        for run in wide:
            frob = run["signals"][next(iter(run["signals"]))]["writers"]["per_layer_frob"]
            lines.append(f"| {Path(run['run']).name} | " + " | ".join(f"{x:.1f}" for x in frob) + " |")
    if report["comparisons"]:
        lines += ["", "## Across runs (Procrustes on visited levels)", "",
                  "| a | b | signal | aligned matched cos | permuted null mean (SD) | RSA r | crossing direction cos |",
                  "|---|---|---|---|---|---|---|"]
        for c in report["comparisons"]:
            for sig, r in c["signals"].items():
                lines.append(f"| {Path(c['a']).name} | {Path(c['b']).name} | {sig} | {f(r['aligned_matched_cos'])} | "
                             f"{f(r['null_mean'])} ({f(r['null_sd'])}) | {f(r['rsa_r'])} | "
                             f"{f(r['aligned_crossing_direction_cos'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True, help="training data of the runs (visit counts, contexts)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-signals", type=int, default=6)
    parser.add_argument("--min-visits", type=float, default=50)
    parser.add_argument("--contexts", type=int, default=256, help="nominal rows for the readers' view")
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    specs = signal_set(args.n_signals)
    visits = np.zeros((args.n_signals, BINS))
    rng = np.random.default_rng(0)
    pool = []
    for rows in load_data_rows(args.data):
        visits += visit_counts([rows], args.n_signals)
        sampled = rows[::50]
        nominal = sampled[[not any(sp.past_limit(float(v)) for sp, v in zip(specs, row)) for row in sampled]]
        if len(nominal):
            pool.append(nominal[rng.integers(len(nominal))])
    contexts = np.stack(pool)[rng.permutation(len(pool))[: args.contexts]]
    report = {"data": str(args.data).replace("\\", "/"), "min_visits": args.min_visits,
              "visits": visits.tolist(), "n_contexts": int(len(contexts)), "runs": [], "comparisons": []}
    tables = {}
    for run in args.runs:
        result, table, dirs = analyze_run(run, specs, visits, contexts, args.min_visits)
        report["runs"].append(result)
        tables[str(run)] = (table, dirs)
        print(f"{run}: done (init check {result['init_check']})", flush=True)
    for (na, (ta, da)), (nb, (tb, db)) in itertools.combinations(tables.items(), 2):
        report["comparisons"].append(compare_runs(na.replace("\\", "/"), ta, da, nb.replace("\\", "/"), tb, db,
                                                  specs, visits, args.min_visits))
    args.out.mkdir(parents=True, exist_ok=True)
    report = _round(report)
    (args.out / "level_geometry.json").write_bytes(json.dumps(report, indent=1).encode("utf-8"))
    (args.out / "level_geometry.md").write_bytes(markdown(report).encode("utf-8"))
    print(f"wrote {args.out / 'level_geometry.json'} and .md", file=sys.stderr)


if __name__ == "__main__":
    main()
