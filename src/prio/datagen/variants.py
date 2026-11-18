"""Variant datasets for the text and inserted-embedding baselines.

Rewrites <src>/{train,test}.jsonl. Same base runs, cuts, teacher reactions, twins and weights as
the source; only what the harness writes differs. CPU, seconds.

Episode contract after an insertion (all indices into target_ids):
- ids go BEFORE source index p; their new range [a, b) is appended to `harness_spans`
  (`train.collate` masks those labels: context, never a target);
- `t_cross` = last token of the insertion that first shows the past-limit state ("anchor"), so
  `episodes.reaction_start` (t_cross + 1) and the forced prefix target_ids[:t_cross + 1] of
  `eval/run.py` include it with no change there; without an anchor `t_cross` only shifts;
- every other index at or after p (t_decision, t_diverge, old spans) shifts by the inserted length;
- one signal row per inserted token: the row of the token before the insertion, with the noticed
  signal set to the value the insertion shows; `signals.shape == (len(prompt) + len(target), N)`;
- `meta["variant"]`: {"name", "insertions": [{at, end, role, signal, shown, anchor}], ...};
- episode ids are kept, so a variant file pairs with its source by `episode_id`.

Variants:
- `notice` (B3): ` [status ctx=0.90]` (9 tokens) right after t_cross; recovery: a second notice
  after the first recovered token; value twins: `--twin-share` of the no-event twins get a
  below-limit notice of their sibling's signal at the sibling's position, so text before the
  decision differs only in the number; `--nominal-share` of the other nominal episodes get a
  below-limit notice at a random valid cut (the channel data's near-miss share).
- `insert` (B6b): the `notice` plan with one `--insert-id` position instead of the text.
- `periodic` (B4; `periodic-64` = `--periods 64`): the full status line after <think> and at the
  first valid cut at or after every multiple of k; k drawn per base run from `--periods`. The
  reaction moves to just after the first line at or after the crossing (same phase); dropped and
  counted when no line shows the crossing in time (or a stale line contradicts a recovery target).
- `prompt-state` (B2): state at the last prompt token + per-signal policy in the system prompt,
  via `tasks.build_prompt`; targets unchanged.
- test only (`--carrier notice|insert`): `swap` (event kind: another interrupt signal's past-limit
  notice at the cut, signal and value as in `eval/run.py`'s swap pairing; written as kind "swap"
  with the swapped signal), `near_miss` (below-limit notice of the episode's own signal at the
  cut), `quiet` (the source episodes unchanged); event + gating_event kinds.
- `policy` (B3z zero-shot arm): `--src` is a built notice folder; every test file there (test.jsonl,
  test_<pairing>.jsonl) copied with `policy_text` after the system prompt; no train split. Evaluate
  with `eval/run.py --mode harness --conditions base` (untrained model).

python -m prio.datagen.variants --src data/n6 --variant notice --out data/n6-notice
python -m prio.datagen.variants --src data/n6-notice --variant policy --out data/n6-notice-policy
"""

import argparse
import bisect
import json
import zlib
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from prio.datagen.episodes import Episode, read_jsonl, write_jsonl
from prio.datagen.teacher import valid_cut_points
from prio.signals import SignalSpec, signal_set
from prio.trajectories import _NEAR_HI, _NEAR_LO, RESOLUTION, sample_trajectory

MAX_LEN = 2048  # = build.MAX_LEN: longer episodes are dropped (and counted), never truncated
VARIANTS = ("notice", "insert", "periodic", "prompt-state", "swap", "near_miss", "quiet", "policy")
TEST_ONLY = ("swap", "near_miss", "quiet")
REACTION_KINDS = ("event", "starts_past", "gating_event")
PAIRED_KINDS = ("event", "gating_event")  # eval/run.py pairs these with the controls
USER_OPEN, TURN_END = "<|im_start|>user", "<|im_end|>"  # chat-template markers around the user message

_ACTIONS = {  # policy lines for the prompt (B2) and zero-shot (B3z) arms; same actions as teacher._NOTICES
    "ctx": "say in one short first-person sentence that your context is almost full, then call compact_context.",
    "think": "say in one short first-person sentence that your thinking budget is almost gone, "
             "then close your thinking and give the answer.",
    "energy": "say in one short first-person sentence that your battery is almost empty, "
              "then call save_state and then sleep.",
    "time": "say in one short first-person sentence that you are almost out of time, "
            "then give a brief final answer and stop.",
    "net": "say in one short first-person sentence that you are offline, then call local_lookup instead of web_search.",
    "ram": "say in one short first-person sentence that memory is almost exhausted, "
           "then call run_light instead of run_heavy.",
}


# --- values and texts -------------------------------------------------------------------------

def shown(spec: SignalSpec, value: float) -> str:
    """The value as the harness writes it: two decimals, on the same side of the limit as `value`.

    The pressure (value for "high", 1 - value for "low") is floored to its 1% step: "high" values
    round down, "low" values round up. ctx 0.905 -> 0.90 (past), 0.895 -> 0.89; think 0.095 ->
    0.10 (past), 0.105 -> 0.11. A plain round would show ctx 0.895 as 0.90: past the limit.
    """
    if spec.dynamics == "binary":
        return f"{float(value):.2f}"
    pressure = float(value) if spec.bad_when == "high" else 1.0 - float(value)
    step = min(max(int(np.floor(pressure * RESOLUTION + 1e-4)), 0), RESOLUTION)
    return f"{(step if spec.bad_when == 'high' else RESOLUTION - step) / RESOLUTION:.2f}"


def notice_text(spec: SignalSpec, value: float) -> str:
    """Event notice in the cost-table format, e.g. ` [status ctx=0.90]` (9 Qwen3 tokens)."""
    return f" [status {spec.name}={shown(spec, value)}]"


def status_line(specs, row) -> str:
    """Full status line, e.g. ` [status ctx=0.12 think=1.00 ... ram=0.71]` (39 Qwen3 tokens)."""
    return " [status " + " ".join(f"{s.name}={shown(s, v)}" for s, v in zip(specs, row)) + "]"


def below_limit_value(spec: SignalSpec, rng) -> float:
    """A value just on the safe side: the near-miss band of trajectories.py (0.5-4.5 points before
    the limit), as a 1% step center. Binary signals: their good value."""
    if spec.dynamics == "binary":
        return 1.0 if spec.bad_when == "low" else 0.0
    p_lim = spec.limit if spec.bad_when == "high" else 1.0 - spec.limit
    p = rng.uniform(p_lim - _NEAR_LO, p_lim - _NEAR_HI)
    p = (min(np.floor(p * RESOLUTION), RESOLUTION - 1) + 0.5) / RESOLUTION
    return float(p if spec.bad_when == "high" else 1.0 - p)


def policy_text(specs) -> str:
    """What the device lines mean and what to do past each limit (B2 prompt, B3z zero-shot arm)."""
    lines = []
    for s in specs:
        op = ">=" if s.bad_when == "high" else "<="
        if s.name.startswith("resource_"):
            k = s.name.split("_", 1)[1]
            action = (f"say in one short first-person sentence that resource {k} is running low, "
                      f"then call handle_resource with k={k}.")
        else:
            action = _ACTIONS[s.name]
        lines.append(f"If {s.name} {op} {s.limit:.2f}, {action}")
    return "Lines like [status name=value] come from the device and report its state. " + " ".join(lines)


def prompt_state_text(specs, row) -> str:
    """Appended to the system prompt by `prompt-state`: the state at request time plus the policy."""
    return f" Device status at the start of this turn:{status_line(specs, row)}. {policy_text(specs)}"


def encode(tok, text: str) -> list[int]:
    """Token ids of `text` alone, as a harness splices them (no special tokens)."""
    return tok(text, add_special_tokens=False).input_ids if callable(tok) else tok.encode(text)


class _CachedDecode:
    """`tok` with single-token decodes cached: valid_cut_points decodes every token of a run."""

    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def decode(self, ids, **kw):
        if len(ids) != 1:
            return self.tok.decode(ids, **kw)
        key = int(ids[0])
        if key not in self.cache:
            self.cache[key] = self.tok.decode([key])
        return self.cache[key]


# --- insertion mechanics ----------------------------------------------------------------------

@dataclass
class Insertion:
    at: int  # index into the source target_ids before which the ids go
    ids: list[int]
    row: np.ndarray  # signal row of every inserted token
    role: str  # crossing | recovery | value_twin | near_miss | swap | status
    signal: str | None = None
    shown: str | None = None
    anchor: bool = False  # first shows the past-limit state: t_cross moves to its last token


def insert(ep: Episode, insertions: list[Insertion], name: str, **meta) -> Episode:
    """`ep` with the insertions spliced in (contract in the module docstring)."""
    if ep.meta and "variant" in ep.meta:
        raise ValueError(f"{ep.episode_id} is already a variant ({ep.meta['variant']['name']})")
    order = sorted(insertions, key=lambda x: x.at)
    n_prompt, n = len(ep.prompt_ids), len(ep.target_ids)

    def shift(i: int) -> int:
        return i + sum(len(x.ids) for x in order if x.at <= i)

    def shift_span(a: int, b: int) -> list[int]:
        return [shift(a), b + sum(len(x.ids) for x in order if x.at < b)]

    target, rows, spans, records, prev = [], [ep.signals[:n_prompt]], [], [], 0
    t_cross = shift(ep.t_cross) if ep.t_cross is not None else None
    for x in order:
        if not 0 <= x.at <= n:
            raise ValueError(f"{ep.episode_id}: insertion at {x.at} outside a target of {n} tokens")
        target += ep.target_ids[prev:x.at]
        rows.append(ep.signals[n_prompt + prev:n_prompt + x.at])
        start = len(target)
        target += list(x.ids)
        rows.append(np.repeat(np.asarray(x.row, dtype=np.float32)[None], len(x.ids), axis=0))
        spans.append([start, len(target)])
        records.append({"at": start, "end": len(target), "role": x.role, "signal": x.signal, "shown": x.shown,
                        "anchor": x.anchor})
        if x.anchor:
            t_cross = len(target) - 1
        prev = x.at
    target += ep.target_ids[prev:]
    rows.append(ep.signals[n_prompt + prev:])
    signals = np.concatenate(rows).astype(np.float32)
    assert signals.shape == (n_prompt + len(target), ep.signals.shape[1])
    harness = sorted([shift_span(a, b) for a, b in ep.harness_spans or ()] + spans)
    return replace(
        ep, target_ids=target, signals=signals, t_cross=t_cross,
        t_decision=shift(ep.t_decision) if ep.t_decision is not None else None,
        t_diverge=shift(ep.t_diverge) if ep.t_diverge is not None else None,
        harness_spans=harness or None,
        focus_spans=[shift_span(a, b) for a, b in ep.focus_spans] if ep.focus_spans else ep.focus_spans,
        meta={**(ep.meta or {}), "variant": {"name": name, "insertions": records, **meta}},
    )


def _rng(seed: int, *keys) -> np.random.Generator:
    """Order-independent per-episode generator (re-runs and file order give the same draws)."""
    return np.random.default_rng([seed, *(zlib.crc32(str(k).encode("utf-8")) for k in keys)])


def _row(ep: Episode, at: int) -> np.ndarray:
    """Signal row of the token before target index `at` (the last prompt token for at = 0)."""
    return ep.signals[len(ep.prompt_ids) + at - 1].copy()


def _set(row: np.ndarray, index: int, value: float) -> np.ndarray:
    row = row.copy()
    row[index] = value
    return row


def _recovered_at(ep: Episode, spec: SignalSpec, index: int) -> int | None:
    """First target index after t_cross whose signal is no longer past the limit."""
    n_prompt = len(ep.prompt_ids)
    return next((j for j in range(ep.t_cross + 1, len(ep.target_ids))
                 if not spec.past_limit(float(ep.signals[n_prompt + j, index]))), None)


class Carrier:
    """Renders one noticed value: the text notice, or the single inserted position of B6b."""

    def __init__(self, tok, insert_id: int | None = None):
        self.tok, self.insert_id = tok, insert_id

    def __call__(self, spec: SignalSpec, value: float) -> list[int]:
        return [self.insert_id] if self.insert_id is not None else encode(self.tok, notice_text(spec, value))


def _noticed(carrier, specs, at, spec_index, value, row, role, anchor=False) -> Insertion:
    spec = specs[spec_index]
    return Insertion(at=at, ids=carrier(spec, value), row=_set(row, spec_index, value), role=role,
                     signal=spec.name, shown=shown(spec, value), anchor=anchor)


def _crossing(carrier, specs, ep: Episode, role="crossing") -> list[Insertion]:
    """Past-limit notice right after t_cross; recovery kinds: a second one after recovery."""
    names = [s.name for s in specs]
    index = names.index(ep.signal)
    spec = specs[index]
    row = _row(ep, ep.t_cross + 1)
    value = float(row[index])
    if not spec.past_limit(float(shown(spec, value))):
        raise ValueError(f"{ep.episode_id}: {ep.signal} = {value} at t_cross is not past its limit")
    out = [_noticed(carrier, specs, ep.t_cross + 1, index, value, row, role, anchor=True)]
    if ep.kind == "recovery":
        back = _recovered_at(ep, spec, index)
        if back is None:
            raise ValueError(f"{ep.episode_id}: recovery episode whose signal never recovers")
        row = _row(ep, back + 1)
        out.append(_noticed(carrier, specs, back + 1, index, float(row[index]), row, "recovery"))
    return out


# --- notice / insert --------------------------------------------------------------------------

def _is_twin(ep: Episode) -> bool:
    return ep.kind == "nominal" and ep.t_diverge is not None


def _siblings(episodes: list[Episode]) -> dict[str, Episode]:
    """task_id -> the reaction episode of a base run that has a no-event twin (train: one reaction
    per run; test runs may carry two, e.g. an event and a starts_past, but no twins)."""
    twinned = {ep.task_id for ep in episodes if _is_twin(ep)}
    out: dict[str, Episode] = {}
    for ep in episodes:
        if ep.kind in REACTION_KINDS and ep.task_id in twinned:
            if ep.task_id in out:
                raise ValueError(f"two reaction episodes for {ep.task_id}")
            out[ep.task_id] = ep
    return out


def _pick(candidates: list[Episode], share: float, seed: int, salt: str) -> set[str]:
    """Exactly round(share * n) episode ids, drawn without replacement."""
    if not candidates:
        return set()
    order = np.random.default_rng([seed, zlib.crc32(salt.encode("utf-8"))]).permutation(len(candidates))
    return {candidates[i].episode_id for i in order[:round(share * len(candidates))]}


def notice_split(episodes, specs, carrier, words, ms, *, seed, twin_share, nominal_share, name):
    stats = Counter()
    names = [s.name for s in specs]
    siblings = _siblings(episodes)
    twins = [e for e in episodes if _is_twin(e) and e.task_id in siblings]
    stats["twins_without_sibling"] = sum(_is_twin(e) and e.task_id not in siblings for e in episodes)
    value_twins = _pick(twins, twin_share, seed, "value_twins")
    nominals = [e for e in episodes if e.kind == "nominal" and e.t_diverge is None]
    near = _pick(nominals, nominal_share, seed, "near_miss")
    out = []
    for ep in episodes:
        rng = _rng(seed, ep.episode_id)
        if ep.kind in REACTION_KINDS or ep.kind in ("recovery", "gating_unused"):
            out.append(insert(ep, _crossing(carrier, specs, ep), name))
        elif ep.episode_id in value_twins:
            sibling = siblings[ep.task_id]
            index, at = names.index(sibling.signal), sibling.t_cross + 1
            value = below_limit_value(specs[index], rng)
            out.append(insert(ep, [_noticed(carrier, specs, at, index, value, _row(ep, at), "value_twin")], name))
        elif ep.episode_id in near:
            cut, index = _near_miss_cut(ep, specs, words, ms, rng)
            value = below_limit_value(specs[index], rng)
            out.append(insert(ep, [_noticed(carrier, specs, cut, index, value, _row(ep, cut), "near_miss")], name))
        else:
            out.append(ep)
    return out, stats


def _near_miss_cut(ep, specs, words, ms, rng) -> tuple[int, int]:
    """A random valid cut and a signal that can be near its limit there (`think` only while thinking)."""
    cuts = valid_cut_points(ep.target_ids, ms, words) or [len(ep.target_ids) // 2]
    cut = int(cuts[int(rng.integers(len(cuts)))])
    close = ep.target_ids.index(ms.think_close) if ms.think_close in ep.target_ids else -1
    allowed = [i for i, s in enumerate(specs) if s.name != "think" or cut < close]
    return cut, int(allowed[int(rng.integers(len(allowed)))])


# --- periodic ---------------------------------------------------------------------------------

def every_k(cuts: list[int], k: int, after: int) -> list[int]:
    """First valid cut at or after each multiple of k, strictly after `after`. A multiple already
    passed by the previous line gets no line of its own. `cuts` sorted."""
    points, last = [], after
    while True:
        i = bisect.bisect_left(cuts, max((last // k + 1) * k, last + 1))
        if i == len(cuts):
            return points
        points.append(cuts[i])
        last = cuts[i]


def _turn_start(target: list[int], ms) -> int:
    """The first line goes after <think> (where `starts_past` reactions begin), else before token 0."""
    return 1 if target and target[0] == ms.think_open else 0


def _lines(ep: Episode, points: list[int], specs, tok, anchor: int | None = None) -> list[Insertion]:
    out = []
    for p in points:
        row = _row(ep, p)
        out.append(Insertion(at=p, ids=encode(tok, status_line(specs, row)), row=row, role="status", anchor=p == anchor))
    return out


def _in_phase(q: int, phase: str | None, base: list[int], ms) -> bool:
    close = base.index(ms.think_close) if ms.think_close in base else len(base)
    return q < close if phase == "thinking" else q > close if phase == "answer" else True


def _anchor(points, ep: Episode, base, ms) -> tuple[int | None, str | None]:
    """First line that shows the crossing in time, or the reason there is none."""
    later = [p for p in points if p >= ep.t_cross + 1]
    if ep.kind == "gating_event":
        q = next((p for p in later if p < ep.t_decision), None)
        return (q, None) if q is not None else (None, "no_line_before_decision")
    q = later[0] if later else None
    return (q, None) if q is not None and _in_phase(q, ep.phase, base, ms) else (None, "no_line_in_phase")


def _too_long(ep: Episode) -> bool:
    return len(ep.prompt_ids) + len(ep.target_ids) > MAX_LEN


def _first_difference(a: list[int], b: list[int], start: int) -> int:
    return next((i for i in range(start, min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))


def periodic_split(episodes, specs, tok, words, ms, *, seed, periods):
    stats, kept, ks = Counter(), {}, Counter()
    names = [s.name for s in specs]
    twins = {e.task_id: e for e in episodes if _is_twin(e)}
    paired = set()

    def k_of(ep):
        return int(periods[int(_rng(seed, "k", ep.task_id).integers(len(periods)))])

    def points_of(target, k):
        start = _turn_start(target, ms)
        return [start] + every_k(valid_cut_points(target, ms, words), k, after=start)

    def keep(*eps):
        if any(_too_long(e) for e in eps):
            stats["too_long"] += len(eps)
            return
        for e in eps:
            kept[e.episode_id] = e
            ks[str(e.meta["variant"]["k"])] += 1

    for ep in episodes:
        k = k_of(ep)
        if ep.kind in REACTION_KINDS and ep.split == "train":  # the reaction moves: rebuilt with its twin
            twin = twins.get(ep.task_id)
            if twin is None:
                stats["no_twin"] += 1
                continue
            paired.add(twin.episode_id)
            pair, reason = _periodic_pair(ep, twin, specs, tok, words, ms, k, points_of)
            if reason:
                stats[reason] += 2
            else:
                keep(*pair)
            continue
        if _is_twin(ep):
            continue  # with its reaction episode
        points = points_of(ep.target_ids, k)
        anchor = None
        if ep.kind in REACTION_KINDS:  # test split: the target is the base run
            anchor, reason = _anchor(points, ep, ep.target_ids, ms)
            if reason:
                stats[reason] += 1
                continue
        elif ep.kind in ("recovery", "gating_unused"):
            index = names.index(ep.signal)
            back = _recovered_at(ep, specs[index], index) if ep.kind == "recovery" else None
            end = back if back is not None else len(ep.target_ids)
            showing = [p for p in points if ep.t_cross + 1 <= p <= end]  # row p - 1 past
            anchor = showing[0] if showing else None
            if ep.kind == "recovery":
                last = max(p for p in points if p < ep.t_decision)
                if ep.t_cross + 1 <= last <= end:
                    stats["stale_line_at_decision"] += 1
                    continue
        keep(insert(ep, _lines(ep, points, specs, tok, anchor), "periodic", k=k))
    stats["twins_without_reaction"] = sum(1 for e in episodes if _is_twin(e) and e.episode_id not in paired)
    return [kept[e.episode_id] for e in episodes if e.episode_id in kept], stats, ks


def _periodic_pair(event, twin, specs, tok, words, ms, k, points_of):
    """Train reaction + twin: lines at the twin's schedule; the reaction follows the first line that
    shows the crossing. Returns ((event', twin'), None) or (None, reason)."""
    base = twin.target_ids
    gating = event.kind == "gating_event"
    cut = event.t_decision if gating else event.t_cross + 1
    if event.target_ids[:cut] != base[:cut]:
        return None, "twin_mismatch"
    points = points_of(base, k)
    q, reason = _anchor(points, event, base, ms)
    if reason:
        return None, reason
    n_prompt = len(event.prompt_ids)
    if gating:
        moved, after = event, event.t_decision
    else:  # base text up to the line, then the same reaction; the crossing row is held over the moved stretch
        reaction = event.target_ids[cut:]
        rows = np.concatenate([event.signals[:n_prompt + cut],
                               np.repeat(event.signals[n_prompt + event.t_cross][None], q - cut, axis=0),
                               event.signals[n_prompt + cut:]])
        moved, after = replace(event, target_ids=base[:q] + reaction, signals=rows), q
    shared = [p for p in points if p < after or (not gating and p == q)]
    own = shared + every_k(valid_cut_points(moved.target_ids, ms, words), k, after=max(after, shared[-1]))
    out_event = insert(moved, _lines(moved, own, specs, tok, anchor=q), "periodic", k=k)
    out_twin = insert(twin, _lines(twin, points, specs, tok), "periodic", k=k)
    start = out_event.t_decision if gating else out_event.t_cross + 1
    diverge = _first_difference(out_event.target_ids, out_twin.target_ids, start)
    return (replace(out_event, t_diverge=diverge), replace(out_twin, t_diverge=diverge)), None


# --- prompt-state -----------------------------------------------------------------------------

def state_prompt(ep: Episode, tok, extra: str) -> list[int]:
    """The episode's prompt rebuilt by `tasks.build_prompt` with `extra` after the system prompt.
    The user message is read back from the prompt; the rebuild must reproduce the source prompt."""
    from prio.tasks import SYSTEM_PROMPT, Task, build_prompt

    text = tok.decode(ep.prompt_ids)
    start = text.rindex(USER_OPEN) + len(USER_OPEN)
    task = Task(ep.task_id, ep.family, text[start:text.index(TURN_END, start)].strip(), None, ep.split)
    if build_prompt(tok, task) != ep.prompt_ids:
        raise ValueError(f"{ep.episode_id}: the prompt is not rebuilt exactly from its user message")
    return build_prompt(tok, task, system=SYSTEM_PROMPT + extra)


def _resize_prompt_rows(ep: Episode, new_prompt: list[int]) -> np.ndarray:
    """Prompt rows for a changed prompt: shared head and tail keep theirs, new tokens copy the row
    where the change starts."""
    old, n_old, n_new = ep.prompt_ids, len(ep.prompt_ids), len(new_prompt)
    pre = next((i for i in range(min(n_old, n_new)) if old[i] != new_prompt[i]), min(n_old, n_new))
    suf = 0
    while suf < min(n_old, n_new) - pre and old[n_old - 1 - suf] == new_prompt[n_new - 1 - suf]:
        suf += 1
    middle = np.repeat(ep.signals[min(pre, n_old - 1)][None], n_new - pre - suf, axis=0)
    return np.concatenate([ep.signals[:pre], middle, ep.signals[n_old - suf:n_old]]).astype(np.float32)


def prompt_state_split(episodes, specs, tok):
    out = []
    for ep in episodes:
        if ep.meta and "variant" in ep.meta:
            raise ValueError(f"{ep.episode_id} is already a variant")
        row = ep.signals[len(ep.prompt_ids) - 1]
        prompt = state_prompt(ep, tok, prompt_state_text(specs, row))
        signals = np.concatenate([_resize_prompt_rows(ep, prompt), ep.signals[len(ep.prompt_ids):]])
        out.append(replace(ep, prompt_ids=prompt, signals=signals,
                           meta={**(ep.meta or {}), "variant": {"name": "prompt-state", "insertions": [],
                                                                "status": status_line(specs, row).strip()}}))
    return out, Counter()


# --- policy prompt (B3z) ----------------------------------------------------------------------

def policy_folder(src: Path, out: Path, tok) -> dict:
    """Copy every test file of the notice folder `src` into `out` with `policy_text` after the system
    prompt (targets, cuts and harness spans unchanged; prompt rows resized). Manifest: the source's
    own (so `eval/run.dataset_tasks` still finds the tasks) + `policy_prompt`, `policy_source`."""
    src, out = Path(src), Path(out)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("variant") != "notice":
        raise ValueError(f"{src}: the policy arm reads a notice folder (found {manifest.get('variant')})")
    out.mkdir(parents=True, exist_ok=True)
    text = None
    for path in sorted(src.glob("test*.jsonl")):
        episodes = read_jsonl(path)
        specs = signal_set(episodes[0].signals.shape[1]) if episodes else signal_set(6)
        text = policy_text(specs)
        rows = []
        for ep in episodes:
            prompt = state_prompt(ep, tok, " " + text)
            signals = np.concatenate([_resize_prompt_rows(ep, prompt), ep.signals[len(ep.prompt_ids):]])
            rows.append(replace(ep, prompt_ids=prompt, signals=signals))
        write_jsonl(out / path.name, rows)
    manifest = {**manifest, "policy_prompt": text, "policy_source": str(src)}
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    return manifest


# --- test-only files --------------------------------------------------------------------------

def swap_split(episodes, specs, carrier, ms, *, eval_seed):
    """The swap control of eval/run.py in text form: same signal choice (`_swap_signal`, by the
    episode's index in the source test file) and the past value of the same swapped trajectory."""
    from prio.eval.run import _horizon, _swap_signal

    names, out, stats = [s.name for s in specs], [], Counter()
    for index, ep in enumerate(episodes):
        if ep.kind != "event":
            continue
        n_prompt, prefix = len(ep.prompt_ids), ep.target_ids[:ep.t_cross + 1]
        closed_at = prefix.index(ms.think_close) if ms.think_close in prefix else None
        swap = _swap_signal(ep, specs, index, closed_at is None)
        if swap is None:
            stats["no_swap_signal"] += 1
            continue
        think_end = None if closed_at is None else n_prompt + closed_at + 1
        crossing = n_prompt + len(prefix) - 1
        values = sample_trajectory(specs, crossing + 1 + _horizon(False, swap), np.random.default_rng([eval_seed, index, 2]),
                                   event=names.index(swap), t_cross=crossing, think_end=think_end).values
        row, i = values[crossing], names.index(swap)
        x = Insertion(at=ep.t_cross + 1, ids=carrier(specs[i], float(row[i])), row=row, role="swap", signal=swap,
                      shown=shown(specs[i], float(row[i])), anchor=True)
        swapped = insert(ep, [x], "swap", original_signal=ep.signal, original_kind=ep.kind)
        out.append(replace(swapped, kind="swap", signal=swap))
    return out, stats


def near_miss_split(episodes, specs, carrier, *, seed):
    names, out = [s.name for s in specs], []
    for ep in episodes:
        if ep.kind not in PAIRED_KINDS:
            continue
        index, at = names.index(ep.signal), ep.t_cross + 1
        value = below_limit_value(specs[index], _rng(seed, "near_miss", ep.episode_id))
        out.append(insert(ep, [_noticed(carrier, specs, at, index, value, _row(ep, at), "near_miss", anchor=True)],
                          "near_miss"))
    return out, Counter()


def quiet_split(episodes):
    out = []
    for ep in episodes:
        if ep.kind in PAIRED_KINDS:
            if ep.meta and "variant" in ep.meta:
                raise ValueError(f"{ep.episode_id} is already a variant")
            out.append(replace(ep, meta={**(ep.meta or {}), "variant": {"name": "quiet", "insertions": []}}))
    return out, Counter()


# --- driver -----------------------------------------------------------------------------------

def parse_variant(variant: str, periods) -> tuple[str, tuple[int, ...]]:
    """`periodic-64` -> ("periodic", (64,)); other names pass through."""
    if variant.startswith("periodic-"):
        return "periodic", (int(variant.split("-", 1)[1]),)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; one of {VARIANTS} or periodic-<k>")
    return variant, tuple(int(k) for k in periods)


def _host(into: Path, src: Path, carrier: str | None, insert_id: int | None) -> tuple[dict, str, int | None]:
    """The manifest of the notice / insert folder that receives test-only files, and the carrier and
    INSERT_ID they must use (the folder's own; a different one given is refused)."""
    path = into / "manifest.json"
    host = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if host.get("variant") not in ("notice", "insert"):
        raise ValueError(f"{into}: test-only files go into a notice or insert variant folder (found {host.get('variant')})")
    if Path(host["source"]).resolve() != src.resolve():
        raise ValueError(f"{into} was built from source {host['source']}, not from {src}")
    own_carrier, own_id = host["variant"], host["options"].get("insert_id")
    if carrier not in (None, own_carrier) or (own_carrier == "insert" and insert_id not in (None, own_id)):
        raise ValueError(f"{into} holds the {own_carrier} carrier (insert id {own_id}): "
                         f"test-only files with carrier {carrier} / insert id {insert_id} would not pair with it")
    return host, own_carrier, own_id if own_carrier == "insert" else None


def build_variant(src: Path, out: Path | None, variant: str, tok, ms, *, seed: int = 0, twin_share: float = 0.5,
                  nominal_share: float = 1 / 3, insert_id: int | None = None, carrier: str | None = None,
                  periods=(64,), eval_seed: int = 0, into: Path | None = None) -> dict:
    """Write <out>/{train,test}.jsonl (the splits <src> has; test only for test-only variants) and
    <out>/manifest.json. Returns the manifest. Deterministic: a re-run writes the same bytes.

    `into` (test-only variants): write <into>/test_<variant>.jsonl instead, beside the event file of
    the notice / insert folder `into` built from the same source, where `eval/run.py --mode harness`
    finds it; carrier and INSERT_ID are that folder's; its manifest gets `test_only[<variant>]`.
    `carrier` None = "notice" (or the folder's, with `into`)."""
    src = Path(src)
    name, periods = parse_variant(variant, periods)
    if name == "policy":
        return policy_folder(src, out, tok)
    host = None
    if into is not None:
        if name not in TEST_ONLY:
            raise ValueError(f"--test-only-into takes a test-only variant {TEST_ONLY}, not {name!r}")
        out = Path(into)
        host, carrier, insert_id = _host(out, src, carrier, insert_id)
    carrier = carrier or "notice"
    out = Path(out)
    if src.resolve() == out.resolve():
        raise ValueError("--out must differ from --src")
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8")) if (src / "manifest.json").exists() else {}
    if "variant" in src_manifest:
        raise ValueError(f"{src} is already a variant ({src_manifest['variant']}): build from the source dataset")
    if name == "insert":
        carrier = "insert"
    if carrier == "insert" and insert_id is None:
        raise ValueError("the insert carrier needs --insert-id (the delivery's INSERT_ID)")
    render = Carrier(tok, insert_id if carrier == "insert" else None)
    words = _CachedDecode(tok)
    splits = ("test",) if name in TEST_ONLY else ("train", "test")
    out.mkdir(parents=True, exist_ok=True)
    # `tasks_requested` + `seed`: what eval/run.dataset_tasks needs to rebuild the test tasks (spoof, rules).
    kept = ("model", "n_signals", "seed", "tasks_requested")
    manifest = {"variant": name, "source": str(src),
                "source_manifest": {k: src_manifest.get(k) for k in kept if k in src_manifest},
                "options": {"seed": seed, "twin_share": twin_share, "nominal_share": nominal_share,
                            "insert_id": insert_id, "carrier": carrier, "periods": list(periods), "eval_seed": eval_seed},
                "episodes": {}, "kinds": {}, "insertions": {}, "harness_tokens": {}, "dropped": {}, "notes": {}}
    if name == "periodic":
        manifest["periods"] = {}
    for split in splits:
        path = src / f"{split}.jsonl"
        if not path.exists():
            continue
        episodes = read_jsonl(path)
        specs = signal_set(episodes[0].signals.shape[1]) if episodes else signal_set(6)
        if name in ("notice", "insert"):
            result, stats = notice_split(episodes, specs, render, words, ms, seed=seed, twin_share=twin_share,
                                         nominal_share=nominal_share, name=name)
        elif name == "periodic":
            result, stats, ks = periodic_split(episodes, specs, tok, words, ms, seed=seed, periods=periods)
            manifest["periods"][split] = dict(sorted(ks.items(), key=lambda kv: int(kv[0])))
        elif name == "prompt-state":
            result, stats = prompt_state_split(episodes, specs, tok)
        elif name == "swap":
            result, stats = swap_split(episodes, specs, render, ms, eval_seed=eval_seed)
        elif name == "near_miss":
            result, stats = near_miss_split(episodes, specs, render, seed=seed)
        else:
            result, stats = quiet_split(episodes)
        write_jsonl(out / (f"test_{name}.jsonl" if host is not None else f"{split}.jsonl"), result)
        records = [r for e in result for r in (e.meta or {}).get("variant", {}).get("insertions", [])]
        manifest["episodes"][split] = len(result)
        manifest["kinds"][split] = dict(Counter(e.kind for e in result))
        manifest["insertions"][split] = dict(Counter(r["role"] for r in records))
        manifest["harness_tokens"][split] = int(sum(r["end"] - r["at"] for r in records))
        manifest["dropped"][split] = {k: v for k, v in stats.items() if v and k not in ("twins_without_sibling",
                                                                                        "twins_without_reaction")}
        manifest["notes"][split] = {k: stats[k] for k in ("twins_without_sibling", "twins_without_reaction") if stats[k]}
    if host is not None:  # the folder keeps one manifest: its own, with the test-only files listed
        host.setdefault("test_only", {})[name] = {**manifest, "file": f"test_{name}.jsonl"}
        (out / "manifest.json").write_bytes((json.dumps(host, indent=1) + "\n").encode("utf-8"))
        return manifest
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    return manifest


def manifest_model(manifest: dict) -> str:
    """The tokenizer's model of a dataset manifest: its own `model`, or a variant's `source_manifest`'s."""
    return manifest.get("model") or manifest["source_manifest"]["model"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rewrite an episode set into a text / inserted-embedding variant.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--variant", required=True, help=f"one of {VARIANTS}, or periodic-<k>")
    parser.add_argument("--out", type=Path, default=None, help="output folder (not with --test-only-into)")
    parser.add_argument("--test-only-into", type=Path, default=None, metavar="VARIANT_DIR",
                        help="test-only variants: write test_<variant>.jsonl into this notice / insert folder (built "
                             "from the same --src), where eval/run.py --mode harness finds it; its carrier and insert id")
    parser.add_argument("--model", default=None, help="tokenizer; defaults to the model in <src>/manifest.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--twin-share", type=float, default=0.5, help="notice/insert: twins given a below-limit notice")
    parser.add_argument("--nominal-share", type=float, default=1 / 3,
                        help="notice/insert: other nominal episodes given a below-limit notice")
    parser.add_argument("--insert-id", type=int, default=None, help="insert / --carrier insert: the delivery's INSERT_ID")
    parser.add_argument("--carrier", choices=("notice", "insert"), default=None,
                        help="test-only variants (default notice; with --test-only-into: the folder's)")
    parser.add_argument("--periods", default="64", help="periodic: k values drawn per base run, e.g. 32,64,128,256")
    parser.add_argument("--eval-seed", type=int, default=0, help="swap: the --seed of eval/run.py, for the same values")
    args = parser.parse_args(argv)
    if (args.out is None) == (args.test_only_into is None):
        parser.error("give exactly one of --out and --test-only-into")

    from transformers import AutoTokenizer

    from prio.modelspec import resolve_spec

    model = args.model or manifest_model(json.loads((args.src / "manifest.json").read_text(encoding="utf-8")))
    tok = AutoTokenizer.from_pretrained(model)
    manifest = build_variant(args.src, args.out, args.variant, tok, resolve_spec(model, tok), seed=args.seed,
                             twin_share=args.twin_share, nominal_share=args.nominal_share, insert_id=args.insert_id,
                             carrier=args.carrier, periods=[int(k) for k in args.periods.split(",")],
                             eval_seed=args.eval_seed, into=args.test_only_into)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
