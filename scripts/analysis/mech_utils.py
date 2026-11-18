"""Shared pieces of the mechanism analysis (preregistration R1, with T3).

- Sites: the reader and writer matrices of a ChannelModel, found through the channel's own hooks
  (works with or without the LoRA wrapper), with layer index and group (attention | FFN).
- Interventions: context manager adding hooks that act per batch element, so one batched pass of
  a single position evaluates many conditions at once:
    reader masks  -> the lanes a reader sees are scaled to 0 for that element (exact: linear);
    writer masks  -> what a writer adds to the lanes is scaled to 0;
    patching      -> residual at the input of a decoder layer (main d dims | lanes | all) replaced
                     by a recorded one, last position only;
    adding        -> a vector added to the main stream at a layer input, last position only;
    recording     -> residual at every layer input + the last layer's output.
  The channel's code (prio/channel.py) is untouched; hooks are removed on exit.
- Cached-prefix single-position runner: prefix KV computed once per pair, then each condition
  recomputes only the decision position (minimal pairs differ only there).
- Minimal pairs: event vs the same trajectory with the crossing signal one level on the nominal
  side at the decision position.
- Probes: logistic (torch, L2, LBFGS) and ridge (closed form), held-out split by task.
No sklearn / scipy in the venv: numpy + torch only.
"""
import contextlib
import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import DynamicCache

from prio import channel as ch
from prio.datagen.episodes import Episode
from prio.signals import SignalSpec

_LAYER = re.compile(r"(?:^|-)(?:layers|h)-(\d+)-")


def run_model_id(run_dir, manifest: dict) -> str:
    """Base model of a run: the one its train_log.json records (a 0.6B run on data built with 1.7B), else the data's."""
    log = Path(run_dir) / "train_log.json"
    if log.exists():
        return json.loads(log.read_text(encoding="utf-8"))["model"]
    return manifest["model"]


# --- sites ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    key: str  # key in model.readers / model.writers
    layer: int
    group: str  # "attn" | "ffn"
    proj: str  # q_proj, k_proj, ..., c_attn, c_fc, o_proj, down_proj, c_proj
    module: nn.Module = field(compare=False, hash=False, repr=False)


def _site(key: str, module: nn.Module) -> Site:
    parts = key.split("-")
    found = _LAYER.search(key)
    if found is None:
        raise ValueError(f"no layer index in {key!r}")
    group = "attn" if any("attn" in p for p in parts[:-1]) else "ffn"
    return Site(key, int(found.group(1)), group, parts[-1], module)


def channel_sites(model) -> tuple[list[Site], list[Site]]:
    """(readers, writers) of a ChannelModel, sorted by layer then name."""
    readers, writers = [], []
    for _, module in model.base.named_modules():
        for hook in module._forward_hooks.values():
            func = getattr(hook, "func", None)
            if func is ch._read_lanes:
                readers.append(_site(hook.args[2], module))
            elif func is ch._write_lanes:
                writers.append(_site(hook.args[2], module))
    order = lambda s: (s.layer, s.group, s.proj)  # noqa: E731
    return sorted(readers, key=order), sorted(writers, key=order)


def decoder_layers(model) -> nn.ModuleList:
    for name, module in model.base.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) and name.rsplit(".", 1)[-1] in ("layers", "h"):
            return module
    raise ValueError("no decoder layer list (named 'layers' or 'h') in the base model")


# --- conditions -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Cond:
    """One intervention at the decision position. `signals` names the row the caller feeds
    ("event" | "control"); `readers` / `writers` are site indices zeroed; `patch` = (layer, part)
    with part in main | lanes | all, values taken from `patch_source[layer]`; `add` = (layer, key)
    adds `add_vectors[key]` to the main stream at that layer's input."""
    name: str
    signals: str = "event"
    readers: frozenset = frozenset()
    writers: frozenset = frozenset()
    patch: tuple | None = None
    add: tuple | None = None


INTACT = Cond("intact")


def retained(ld: float, ld_intact: float, ld_control: float) -> float:
    """Fraction of the event-vs-control logit difference left: 1 = intact, 0 = control level."""
    span = ld_intact - ld_control
    if not np.isfinite(span) or abs(span) < 1e-9:
        return float("nan")
    return float((ld - ld_control) / span)


# --- interventions --------------------------------------------------------------------------------

def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


@contextlib.contextmanager
def channel_state(model, enabled: bool = True, adapter: bool = True):
    """Channel on/off; `adapter=False` also disables the LoRA adapter (the base model)."""
    model.enabled = enabled
    try:
        if not adapter and hasattr(model.base, "disable_adapter"):
            with model.base.disable_adapter():
                yield
        else:
            yield
    finally:
        model.enabled = True


class Interventions:
    """Hooks for masks, patching, adding and recording; see the module docstring."""

    def __init__(self, model):
        self.model = model
        self.readers, self.writers = channel_sites(model)
        self.layers = decoder_layers(model)
        self.n_layers = len(self.layers)
        self.d = model.base.get_input_embeddings().embedding_dim
        self.device = model.projector.net[0].weight.device
        self._handles = []
        self._clear()

    # state of the pass in progress
    def _clear(self):
        self.reader_keep: list[torch.Tensor | None] = [None] * len(self.readers)
        self.writer_keep: list[torch.Tensor | None] = [None] * len(self.writers)
        self.patches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.adds: dict[int, torch.Tensor] = {}
        self.record = False
        self.positions: list[int] | None = None
        self.recorded: dict[int, torch.Tensor] = {}

    def __enter__(self):
        lanes = self.model._lanes
        for i, site in enumerate(self.readers):
            self._handles.append(site.module.register_forward_pre_hook(functools.partial(self._mask_lanes, lanes, i)))
            self._handles.append(site.module.register_forward_hook(functools.partial(self._unmask_lanes, lanes, i)))
        for i, site in enumerate(self.writers):
            self._handles.append(site.module.register_forward_hook(functools.partial(self._mask_written, lanes, i)))
        for idx, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_pre_hook(functools.partial(self._layer_input, idx),
                                                                 with_kwargs=True))
        self._handles.append(self.layers[-1].register_forward_hook(self._last_output))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._clear()
        self.model._lanes.values = self.model._lanes.written = None
        return False

    # hooks
    def _mask_lanes(self, lanes, i, module, args):
        keep = self.reader_keep[i]
        self._saved = lanes.values
        if keep is not None and lanes.values is not None:
            lanes.values = lanes.values * keep.view(-1, *([1] * (lanes.values.dim() - 1))).to(lanes.values.dtype)

    def _unmask_lanes(self, lanes, i, module, args, output):
        if self.reader_keep[i] is not None:
            lanes.values = self._saved
        self._saved = None

    def _mask_written(self, lanes, i, module, args, output):
        keep = self.writer_keep[i]
        if keep is not None and lanes.written is not None:
            lanes.written = lanes.written * keep.view(-1, *([1] * (lanes.written.dim() - 1))).to(lanes.written.dtype)

    def _take(self, hidden):
        rows = hidden[:, -1] if self.positions is None else hidden[:, self.positions]
        return rows.detach().clone()

    def _layer_input(self, idx, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        if self.record:
            self.recorded[idx] = self._take(hidden)
        patch, add = self.patches.get(idx), self.adds.get(idx)
        if patch is None and add is None:
            return None
        hidden = hidden.clone()
        last = hidden[:, -1]
        if patch is not None:
            mask, values = patch
            mask = mask[:, : last.shape[-1]].to(last.dtype)
            last = last * (1 - mask) + values[:, : last.shape[-1]].to(last.dtype) * mask
        if add is not None:
            last = last.clone()
            last[:, : self.d] = last[:, : self.d] + add.to(last.dtype)
        hidden[:, -1] = last
        if args:
            return (hidden,) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": hidden}

    def _last_output(self, module, args, output):
        if self.record:
            self.recorded[self.n_layers] = self._take(output[0] if isinstance(output, tuple) else output)

    # building a batch of conditions
    def _set(self, conds, patch_source, add_vectors):
        b = len(conds)
        dev = self.device
        for i in range(len(self.readers)):
            keep = torch.tensor([0.0 if i in c.readers else 1.0 for c in conds], device=dev)
            self.reader_keep[i] = None if bool((keep == 1).all()) else keep
        for i in range(len(self.writers)):
            keep = torch.tensor([0.0 if i in c.writers else 1.0 for c in conds], device=dev)
            self.writer_keep[i] = None if bool((keep == 1).all()) else keep
        width = None
        for layer in sorted({c.patch[0] for c in conds if c.patch}):
            values = patch_source[layer].to(dev)
            width = values.shape[-1]
            mask = torch.zeros(b, width, device=dev)
            for k, c in enumerate(conds):
                if c.patch and c.patch[0] == layer:
                    part = c.patch[1]
                    if part == "lanes" and width == self.d:
                        raise ValueError("lane patching needs the wide delivery (the residual carries no lanes)")
                    lo, hi = {"main": (0, self.d), "lanes": (self.d, width), "all": (0, width)}[part]
                    mask[k, lo:hi] = 1.0
            self.patches[layer] = (mask, values.reshape(1, -1).expand(b, -1))
        for layer in sorted({c.add[0] for c in conds if c.add}):
            add = torch.zeros(b, self.d, device=dev)
            for k, c in enumerate(conds):
                if c.add and c.add[0] == layer:
                    add[k] = add_vectors[c.add[1]].to(dev).float()
            self.adds[layer] = add

    # passes
    @torch.no_grad()
    def prefix(self, ids, rows, enabled: bool = True, adapter: bool = True):
        """KV cache of every position but the last (list of (keys, values) per layer)."""
        x = torch.tensor([list(ids[:-1])], device=self.device)
        s = torch.as_tensor(np.asarray(rows[:-1]), dtype=torch.float32, device=self.device)[None]
        with channel_state(self.model, enabled, adapter), _autocast(self.device):
            out = self.model(x, s, use_cache=True)
        return [(layer.keys, layer.values) for layer in out.past_key_values.layers]

    @torch.no_grad()
    def last(self, kv, last_id: int, rows, conds=None, record: bool = False, patch_source=None, add_vectors=None,
             enabled: bool = True, adapter: bool = True):
        """One batched pass of the decision position: element k gets `rows[k]` and `conds[k]`.
        Returns logits [B, V] (float32), and the recorded residuals {layer: [B, D]} if `record`."""
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, np.asarray(rows).shape[-1])
        b = len(rows)
        conds = conds or [INTACT] * b
        if len(conds) != b:
            raise ValueError(f"{len(conds)} conditions for {b} rows")
        cache = DynamicCache(ddp_cache_data=[(k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1)) for k, v in kv])
        x = torch.full((b, 1), int(last_id), device=self.device)
        s = torch.from_numpy(rows).to(self.device)[:, None]
        self._set(conds, patch_source, add_vectors)
        self.record = record
        try:
            with channel_state(self.model, enabled, adapter), _autocast(self.device):
                logits = self.model(x, s, past_key_values=cache, use_cache=True).logits[:, -1].float()
            recorded = dict(self.recorded)
        finally:
            self._clear()
        return (logits, recorded) if record else logits

    @torch.no_grad()
    def collect(self, ids, rows, positions, enabled: bool = True, adapter: bool = True):
        """Full forward pass; residual at every layer input (+ last output) at `positions`:
        {layer: [P, D]} on the CPU."""
        x = torch.tensor([list(ids)], device=self.device)
        s = torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=self.device)[None]
        self.record, self.positions = True, list(positions)
        try:
            with channel_state(self.model, enabled, adapter), _autocast(self.device):
                self.model(x, s)
            return {layer: r[0].cpu() for layer, r in self.recorded.items()}
        finally:
            self._clear()

    @torch.no_grad()
    def continuation(self, kv, ids, rows, enabled: bool = True, adapter: bool = True):
        """Feed several tokens after a cached prefix; residuals at the last of them {layer: [D]}."""
        b_kv = [(k, v) for k, v in kv]
        cache = DynamicCache(ddp_cache_data=b_kv)
        x = torch.tensor([list(ids)], device=self.device)
        s = torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=self.device)[None]
        self.record = True
        try:
            with channel_state(self.model, enabled, adapter), _autocast(self.device):
                self.model(x, s, past_key_values=cache, use_cache=True)
            return {layer: r[0] for layer, r in self.recorded.items()}
        finally:
            self._clear()


# --- minimal pairs --------------------------------------------------------------------------------

def nominal_side(spec: SignalSpec) -> float:
    """The 1% level just on the nominal side of the limit (bin center); binary: the nominal value."""
    if spec.dynamics == "binary":
        return float(spec.nominal)
    k = round(spec.limit * 100)
    return (k - 0.5) / 100 if spec.bad_when == "high" else (k + 0.5) / 100


@dataclass
class Pair:
    episode_id: str
    task_id: str
    kind: str
    signal: str
    phase: str | None
    family: str
    ids: list[int]  # prompt + forced target; the last token is the decision position
    event: np.ndarray  # [T, N]
    control: np.ndarray  # [T, N]; differs from `event` only at the last row, crossing signal


def make_pair(ep: Episode, specs, ms) -> Pair | None:
    """Minimal pair for an interrupt event / starts-past episode (decision at t_cross, the first
    token carrying the past-limit value) or a gating decision (last token before the tool call).
    None for other kinds, for a `think` event after thinking ended, or when the stored value at
    the decision position is not past the limit."""
    if ep.kind in ("event", "starts_past"):
        cut = ep.t_cross + 1
    elif ep.kind == "gating_event":
        cut = ep.t_decision
    else:
        return None
    names = [s.name for s in specs]
    i = names.index(ep.signal)
    spec = specs[i]
    prefix = ep.target_ids[:cut]
    if ep.signal == "think" and ms.think_close in prefix:
        return None
    ids = list(ep.prompt_ids) + list(prefix)
    rows = ep.signals[: len(ids)].astype(np.float32).copy()
    if ms.think_close in prefix and "think" in names:  # as the evaluation: nominal after </think>
        t = names.index("think")
        rows[len(ep.prompt_ids) + prefix.index(ms.think_close) + 1:, t] = specs[t].nominal
    if not spec.past_limit(float(rows[-1, i])):
        return None
    control = rows.copy()
    control[-1, i] = nominal_side(spec)
    return Pair(ep.episode_id, ep.task_id, ep.kind, ep.signal, ep.phase, ep.family, ids, rows, control)


# --- probes ---------------------------------------------------------------------------------------

def task_split(task_ids, seed: int = 0, test_fraction: float = 0.3) -> np.ndarray:
    """Boolean mask: True = held-out test, by task id (no task on both sides)."""
    unique = sorted(set(task_ids))
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(unique)[: max(1, int(round(len(unique) * test_fraction)))].tolist())
    return np.array([t in held for t in task_ids])


def _standardize(x_train, x_test):
    mean = x_train.mean(0, keepdims=True)
    std = x_train.std(0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def fit_logistic(x_train, y_train, x_test, y_test, l2: float = 1e-2, steps: int = 100, device: str = "cpu") -> dict:
    """Multinomial logistic regression (L2), full-batch LBFGS. Returns train/test accuracy and
    the majority-class test accuracy."""
    xt, xs = _standardize(np.asarray(x_train, np.float32), np.asarray(x_test, np.float32))
    classes = np.unique(np.concatenate([y_train, y_test]))
    index = {c: k for k, c in enumerate(classes)}
    yt = torch.tensor([index[c] for c in y_train], device=device)
    ys = torch.tensor([index[c] for c in y_test], device=device)
    xt_t, xs_t = torch.from_numpy(xt).to(device), torch.from_numpy(xs).to(device)
    w = torch.zeros(xt.shape[1], len(classes), device=device, requires_grad=True)
    b = torch.zeros(len(classes), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(xt_t @ w + b, yt) + l2 * w.pow(2).sum() / len(yt)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc_train = float(((xt_t @ w + b).argmax(1) == yt).float().mean())
        acc_test = float(((xs_t @ w + b).argmax(1) == ys).float().mean()) if len(ys) else float("nan")
    majority = classes[np.argmax([np.sum(np.asarray(y_train) == c) for c in classes])]
    return {"acc_train": acc_train, "acc_test": acc_test,
            "majority_test": float(np.mean(np.asarray(y_test) == majority)) if len(y_test) else float("nan"),
            "n_train": int(len(yt)), "n_test": int(len(ys)), "classes": int(len(classes))}


def fit_ridge(x_train, y_train, x_test, y_test, alpha: float = 10.0) -> dict:
    """Ridge regression in closed form (dual when features > samples). Test R^2."""
    xt, xs = _standardize(np.asarray(x_train, np.float64), np.asarray(x_test, np.float64))
    yt, ys = np.asarray(y_train, np.float64), np.asarray(y_test, np.float64)
    mu = yt.mean()
    n, d = xt.shape
    if d > n:
        coef = xt.T @ np.linalg.solve(xt @ xt.T + alpha * np.eye(n), yt - mu)
    else:
        coef = np.linalg.solve(xt.T @ xt + alpha * np.eye(d), xt.T @ (yt - mu))
    pred = xs @ coef + mu
    ss = np.sum((ys - ys.mean()) ** 2)
    return {"r2_test": float(1 - np.sum((ys - pred) ** 2) / ss) if ss > 0 else float("nan"),
            "n_train": int(n), "n_test": int(len(ys))}


def control_labels(token_ids, labels, seed: int = 0) -> np.ndarray:
    """Hewitt-Liang control task: each token id gets a fixed random label drawn from the real
    label distribution. A probe that scores well here memorizes token identity."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    pool = labels.copy()
    assigned = {}
    out = []
    for t in token_ids:
        if t not in assigned:
            assigned[t] = pool[rng.integers(len(pool))]
        out.append(assigned[t])
    return np.asarray(out)


# --- directions -----------------------------------------------------------------------------------

def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def signflip_null(diffs, target, n_perm: int = 200, seed: int = 0) -> tuple[float, float]:
    """Null for cos(mean(diffs), target): mean and SD of cos(mean(eps_i * diff_i), target) with
    random signs eps_i (the event / control labels of each pair swapped at random)."""
    diffs = np.asarray(diffs, np.float64)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        values.append(cosine((signs[:, None] * diffs).mean(0), target))
    return float(np.mean(values)), float(np.std(values))
