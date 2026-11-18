"""The interoceptive channel: the signal vector that was current when a token entered the model,
delivered with that token. Three deliveries: added into the token embedding ("sum"); carried in
dimensions of its own that every layer reads ("lanes"); carried in extra dimensions of the vector
that travels between the layers, which every layer reads and can write ("wide").

Baseline deliveries (paper baselines S2, B6a, B6b; program R6): no channel at all ("none", the
text baselines); the projection in place of the embedding of one position the harness inserts
("insert", soft notice); one learned vector added to the tokens the harness inserted ("segment",
role-marked text); fixed vectors added at one block while a signal is past its limit ("steer",
training-free)."""

import functools
import math
from pathlib import Path

import torch
from torch import nn

from prio.signals import signal_set

DELIVERIES = ("sum", "lanes", "wide", "none", "insert", "steer", "segment")
PROJECTORS = ("mlp", "levels", "levels+fourier")
# Options of the sum delivery: "rms" = each token's channel vector at unit size times a learned gain (R8).
SUM_NORMS = ("rms",)
# Deliveries whose input is the projection of the signals.
_PROJECTED = ("sum", "lanes", "wide", "insert")
# Angular frequencies of the "levels+fourier" code: 16, geometric from pi (period 2: one smooth
# ramp over [0, 1]) to 100 pi (period 0.02: two 1% levels).
FOURIER_FREQS = math.pi * 100.0 ** (torch.arange(16, dtype=torch.float32) / 15)
# Module names, per architecture: Qwen3 first, then GPT-2 (the test model).
# Matrices that read the residual stream inside a block: attention inputs and feed-forward input.
READERS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "c_attn", "c_fc")
# Matrices that write the residual stream: attention output and feed-forward output.
WRITERS = ("o_proj", "down_proj", "c_proj")
# The modules whose output the model adds to the residual stream.
SUBLAYERS = ("self_attn", "mlp", "attn")
# Normalizations inside a block that take the residual stream as input.
NORMS = ("input_layernorm", "post_attention_layernorm", "ln_1", "ln_2")
# The normalization after the last block, and the output head that reads it.
FINAL_NORMS = ("norm", "ln_f")
HEADS = ("lm_head",)
# Position embeddings that the model adds to the input embeddings.
PADDED = ("wpe",)
# The list of decoder blocks (delivery "steer"): Qwen3 `model.layers`, GPT-2 `transformer.h`.
BLOCKS = ("layers", "h")


class SignalProjector(nn.Module):
    """f: R^N -> R^d. With `zero_init` the output layer starts at zero, so an untrained channel
    changes nothing.

    Each scalar in [0, 1] is first expanded by piecewise-linear encoding into `bins` features.
    A raw scalar cannot resolve a limit: 0.89 and 0.90 are almost the same input. Here a value at
    the center of bin k gives features 0..k = 1 and the rest 0 (a thermometer code), so moving one
    bin flips exactly one feature; values between centers interpolate. `bins=0` feeds raw scalars
    (kept for the ablation).

    kind "mlp": two hidden layers on top of that code. kind "levels": no hidden layer and another
    code: one feature per 1% level, on only at the signal's current level (between two centers:
    shared by the two neighbors). The single linear layer is then a lookup table with one output
    vector per level of each signal, and the output is the sum of the N vectors in use plus a
    bias. A limit crossing swaps one whole vector for another and each is trained directly.
    Through the hidden layers a crossing is one feature in 100 x N and its effect grows slowly
    (phase 1: after four epochs a limit crossing moved the output 50 times less than a binary
    signal). A linear layer on the thermometer code does not work either: hundreds of vectors are
    in use at every token, move together at every step and swamp the embedding.

    kind "levels+fourier" (program R10): the "levels" code plus, per signal, 16 sin / cos pairs of
    the value at FOURIER_FREQS, into the same single linear layer. The pairs are scaled by 1/4,
    so each signal's smooth block has norm 1 like its level code. The one-hot part keeps sharp
    thresholds; the smooth part is an ordered value code (near values get near features), which
    a value readout or a limit given in words needs. Feature layout: the N x `bins` level code
    (as in "levels", so `net[0].weight[:, s * bins + k]` is still the vector of level k of signal
    s), then N x 32 smooth features.
    """

    def __init__(self, n_signals: int, d_model: int, hidden: int = 256, bins: int = 100, zero_init: bool = True,
                 kind: str = "mlp"):
        super().__init__()
        if kind not in PROJECTORS:
            raise ValueError(f"projector {kind!r}: expected one of {PROJECTORS}")
        self.n_signals = n_signals
        self.bins = bins
        self.kind = kind
        features = n_signals * max(bins, 1)
        if kind == "levels+fourier":
            features += n_signals * 2 * len(FOURIER_FREQS)
        if kind in ("levels", "levels+fourier"):
            self.net = nn.Sequential(nn.Linear(features, d_model))
        else:
            self.net = nn.Sequential(
                nn.Linear(features, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def encode(self, signals: torch.Tensor) -> torch.Tensor:  # [B, T, N] -> [B, T, N * bins (+ N * 32)]
        code = self._code(signals)
        if self.kind != "levels+fourier":
            return code
        angles = signals.unsqueeze(-1) * FOURIER_FREQS.to(device=signals.device, dtype=signals.dtype)
        smooth = torch.cat([angles.sin(), angles.cos()], dim=-1) / math.sqrt(len(FOURIER_FREQS))
        return torch.cat([code, smooth.flatten(-2)], dim=-1)

    def _code(self, signals: torch.Tensor) -> torch.Tensor:  # [B, T, N] -> [B, T, N * bins]
        if self.bins == 0:
            return signals
        steps = torch.arange(self.bins, device=signals.device, dtype=signals.dtype)
        if self.kind != "mlp":  # 1 at the current level's center, falling to 0 at the neighbors' centers
            position = (signals * self.bins - 0.5).clamp(0.0, self.bins - 1.0)
            return (1.0 - (position.unsqueeze(-1) - steps).abs()).clamp(min=0.0).flatten(-2)
        reached = ((signals.unsqueeze(-1) - steps / self.bins) * self.bins + 0.5).clamp(0.0, 1.0)
        return reached.flatten(-2)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:  # [B, T, N] -> [B, T, d]
        return self.net(self.encode(signals))


class _Lanes:
    """The lanes of the forward pass in progress, shared between the hooks.

    values: the lanes as the readers see them now. written: what a writer computed and its
    sub-layer has not yet appended to its output. widened: the pass carries the lanes inside the
    vector that travels between the layers (wide delivery).
    """

    def __init__(self):
        self.values: torch.Tensor | None = None
        self.written: torch.Tensor | None = None
        self.widened = False


def _named(name: str, names: tuple[str, ...]) -> bool:
    return name.rsplit(".", 1)[-1] in names


def _unit(x: torch.Tensor) -> torch.Tensor:
    """Each vector scaled to root-mean-square 1 (the model's own normalization, without a gain).
    Computed in float32: some activations of the real model are in the thousands."""
    wide = x.float()
    return (wide * wide.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()).to(x.dtype)


def _read_lanes(lanes: _Lanes, readers: nn.ParameterDict, key: str, module, args, output):
    """Forward hook of a reader: its output plus its lane columns times the lanes."""
    if lanes.values is None:
        return None
    columns = readers[key]
    return output + nn.functional.linear(lanes.values.to(columns.dtype), columns).to(output.dtype)


def _cut_lanes(lanes: _Lanes, width: int, module, args):
    """Pre-hook of a normalization (wide delivery): the lanes go around it and get their own.
    At unit size they become the current lanes, the module gets the model's own dimensions."""
    stream = args[0]
    if stream.shape[-1] == width:  # a pass without lanes
        lanes.values = None
        return None
    lanes.values = _unit(stream[..., width:])
    return (stream[..., :width],) + tuple(args[1:])


def _restore_lanes(lanes: _Lanes, module, args, output):
    """Forward hook of the final normalization: the vector keeps its width up to the output head."""
    return None if lanes.values is None else torch.cat([output, lanes.values.to(output.dtype)], dim=-1)


def _drop_lanes(width: int, module, args):
    """Pre-hook of the output head: it reads the model's own dimensions only."""
    stream = args[0]
    return None if stream.shape[-1] == width else (stream[..., :width],) + tuple(args[1:])


def _write_lanes(lanes: _Lanes, writers: nn.ParameterDict, key: str, module, args, output):
    """Forward hook of a writer: its lane rows times its input at unit size, kept for the
    sub-layer's hook. Unscaled, the few tokens whose activations are thousands of times larger
    than the rest would alone decide what the writers learn and what the lanes hold."""
    if lanes.values is not None:
        rows = writers[key]
        lanes.written = nn.functional.linear(_unit(args[0]).to(rows.dtype), rows)
    return None


def _append_written(lanes: _Lanes, module, args, output):
    """Forward hook of a sub-layer: its output grows by what its writer computed, so the model's
    own `stream + output` adds that to the lanes."""
    if lanes.written is None:
        return None
    written, lanes.written = lanes.written, None
    first = output[0] if isinstance(output, tuple) else output
    wide = torch.cat([first, written.to(first.dtype)], dim=-1)
    return (wide,) + tuple(output[1:]) if isinstance(output, tuple) else wide


def _pad_for_lanes(lanes: _Lanes, lane_width: int, module, args, output):
    """Forward hook of a position embedding that is added to the input: zeros for the lanes."""
    return nn.functional.pad(output, (0, lane_width)) if lanes.widened else None


class _Steering:
    """What the forward pass in progress adds to the output of the steered block ([B, T, d])."""

    def __init__(self):
        self.added: torch.Tensor | None = None


def _add_steering(steering: _Steering, module, args, output):
    """Forward hook of the steered block (delivery "steer")."""
    if steering.added is None:
        return None
    first = output[0] if isinstance(output, tuple) else output
    steered = first + steering.added.to(first.dtype)
    return (steered,) + tuple(output[1:]) if isinstance(output, tuple) else steered


def blocks(base) -> nn.ModuleList:
    """The decoder blocks of `base` (also through an adapter): the first ModuleList named like
    one of BLOCKS. `blocks(model.base)[model.steer_layer]` is the block the steer delivery hooks."""
    for name, module in base.named_modules():
        if isinstance(module, nn.ModuleList) and _named(name, BLOCKS):
            return module
    raise ValueError(f"no module list of the base is named like one of {BLOCKS}")


def insert_id(model, tokenizer) -> int:
    """INSERT_ID of the insert delivery: the first embedding row the tokenizer never produces,
    `len(tokenizer)`. Qwen3-1.7B: tokenizer ids end at 151,668, so 151,669 of 151,936 rows (its
    embeddings are tied: generation must ban this id). The tiny test GPT-2 (100 rows) with a
    tokenizer of n < 100 ids: n. ValueError when every row is a token (e.g. GPT-2 with its own
    50,257-token tokenizer: resize the embedding first)."""
    free, rows = len(tokenizer), model.get_input_embeddings().num_embeddings
    if free >= rows:
        raise ValueError(f"no free embedding row: the tokenizer has {free} ids, the embedding {rows} rows")
    return free


class ChannelModel(nn.Module):
    """Wraps a causal LM and gives every token the signal vector that was current when it entered.

    delivery "sum": the projection of the signals is added to the token's input embedding.
    delivery "lanes": the projection (`lane_width` values) stays out of the embedding. Every
    module of `base` named in READERS gets `lane_width` extra input columns (`readers`, zero at
    the start, so an untrained model equals the base) and adds columns x lanes to its output:
    extra dimensions of the residual stream that the harness writes, every layer reads and no
    layer changes.
    delivery "wide": the lanes are appended to the token embedding at unit size and travel with
    it from layer to layer (`[B, T, d + lane_width]`). Readers as in "lanes"; every module named
    in WRITERS gets `lane_width` extra output rows (`writers`, zero at the start), whose output
    its sub-layer (SUBLAYERS) appends to its own, so the model's additions add it to the lanes.
    The normalizations (NORMS, FINAL_NORMS) see the model's own d dimensions only; the lanes have
    a normalization of their own (`_unit`) before every read, and a writer reads its input at
    unit size. The output head (HEADS) never sees the lanes. With the writers at zero this is the
    "lanes" delivery with unit-size lanes.

    `projector`: the kind of SignalProjector ("mlp" | "levels" | "levels+fourier").

    `sum_norm` (delivery "sum" only, program R8): None = the projection is added as it is (zero at
    the start). "rms" = scale control: each token's projection at unit size (`_unit`) times a
    learned gain, `sum_scale` x `sum_gain`: `sum_scale` (fixed) = 0.1 x the mean RMS of the
    embedding rows, `sum_gain` (learned, channel rate) starts at 1, so the gain starts at 0.1 x the
    mean embedding RMS and Adam moves it in relative steps. The projector is not zero-initialized
    then (a zero vector has no direction).

    Baseline deliveries. They keep a placeholder `projector` (frozen, never applied) so that
    callers can still read `projector.n_signals` and the device from `projector.net[0]`.
    delivery "none": the channel is never applied and nothing of it trains (text baselines;
    plain LoRA).
    delivery "insert" (B6b): positions whose input id is `insert_id` get the projection of their
    signals (d values, not zero-initialized: a new position starts like a fresh embedding row)
    instead of their token embedding; every other position is plain text. Disabled: the plain
    embedding row of `insert_id`. `insert_id`: see `insert_id()`.
    delivery "segment" (R6): one learned vector (`segment`, d values, zero at the start) added to
    the embeddings of the positions where `forward`'s `harness_mask` is true (the tokens the
    harness inserted; `train.collate` builds the mask from the episode's harness spans). Without a
    mask no position gets it: a look-alike text from anyone else stays plain text. Signals unused.
    delivery "steer" (B6a, training-free): while signal s is past its limit (`past_limit` of
    `signal_set(n_signals)`), every token adds `steer_alpha` x `steer_vectors[s]` to the output of
    block `steer_layer` (0-based, `blocks()`; HF `hidden_states[steer_layer + 1]`). Vectors start
    at zero; set them with `set_steering`. Not meant for training (the added vectors are cleared
    after each pass, so activation checkpointing would recompute without them).

    Build the ChannelModel before any adapter wraps the base.

    "lanes" only: after a forward pass that can be back-propagated the lanes stay set, because
    activation checkpointing reruns the layers during the backward pass. Do not call `base`
    directly between such a pass and the next one. ("wide" has no such limit: the lanes are part
    of the activations.)
    """

    def __init__(self, base, n_signals: int, bins: int = 100, delivery: str = "sum", lane_width: int = 64,
                 projector: str = "mlp", sum_norm: str | None = None, insert_id: int | None = None,
                 steer_layer: int | None = None):
        super().__init__()
        if delivery not in DELIVERIES:
            raise ValueError(f"delivery {delivery!r}: expected one of {DELIVERIES}")
        if sum_norm is not None and (delivery != "sum" or sum_norm not in SUM_NORMS):
            raise ValueError(f"sum_norm {sum_norm!r}: expected one of {SUM_NORMS}, with the sum delivery only")
        if (insert_id is None) == (delivery == "insert"):
            raise ValueError("insert_id: required by the insert delivery and used by it only")
        if (steer_layer is None) == (delivery == "steer"):
            raise ValueError("steer_layer: required by the steer delivery and used by it only")
        self.base = base
        self.delivery = delivery
        self.lane_width = lane_width
        self.projector_kind = projector
        self.sum_norm = sum_norm
        self.insert_id = insert_id
        self.steer_layer = steer_layer
        self.enabled = True
        embed = base.get_input_embeddings()
        device = embed.weight.device
        self.readers = nn.ParameterDict()
        self.writers = nn.ParameterDict()
        self._lanes = _Lanes()
        self._steering = _Steering()
        if delivery == "sum":
            self.projector = SignalProjector(n_signals, embed.embedding_dim, bins=bins, kind=projector,
                                             zero_init=sum_norm is None)
            if sum_norm is not None:
                rows = embed.weight.detach().split(8192)  # float32 in chunks: the whole table would take 1.2 GB
                rms = torch.cat([chunk.float().pow(2).mean(dim=-1).sqrt() for chunk in rows]).mean()
                self.register_buffer("sum_scale", 0.1 * rms)
                self.sum_gain = nn.Parameter(torch.ones((), device=device))
        elif delivery == "insert":
            if not 0 <= insert_id < embed.num_embeddings:
                raise ValueError(f"insert_id {insert_id}: the embedding has rows 0-{embed.num_embeddings - 1}")
            self.projector = SignalProjector(n_signals, embed.embedding_dim, bins=bins, zero_init=False, kind=projector)
        elif delivery not in _PROJECTED:
            if projector not in PROJECTORS:
                raise ValueError(f"projector {projector!r}: expected one of {PROJECTORS}")
            self.projector = SignalProjector(n_signals, 1, hidden=1, bins=0).requires_grad_(False)
            if delivery == "segment":
                self.segment = nn.Parameter(torch.zeros(embed.embedding_dim, device=device))
            elif delivery == "steer":
                layers = blocks(base)
                if not 0 <= steer_layer < len(layers):
                    raise ValueError(f"steer_layer {steer_layer}: the base has blocks 0-{len(layers) - 1}")
                specs = signal_set(n_signals)
                self.register_buffer("steer_vectors", torch.zeros(n_signals, embed.embedding_dim, device=device))
                self.register_buffer("steer_alpha", torch.ones((), device=device))
                self.register_buffer("steer_limits", torch.tensor([s.limit for s in specs], device=device),
                                     persistent=False)
                self.register_buffer("steer_high", torch.tensor([s.bad_when == "high" for s in specs], device=device),
                                     persistent=False)
                layers[steer_layer].register_forward_hook(functools.partial(_add_steering, self._steering))
        else:
            self.projector = SignalProjector(n_signals, lane_width, bins=bins, zero_init=False, kind=projector)
            for name, module in base.named_modules():
                key = name.replace(".", "-")
                if _named(name, READERS):
                    out_features = getattr(module, "out_features", None) or module.nf  # nn.Linear | GPT-2 Conv1D
                    self.readers[key] = nn.Parameter(torch.zeros(out_features, lane_width))
                    module.register_forward_hook(functools.partial(_read_lanes, self._lanes, self.readers, key))
                if delivery != "wide":
                    continue
                if _named(name, WRITERS):
                    in_features = getattr(module, "in_features", None) or module.weight.shape[0]  # Conv1D: [in, out]
                    self.writers[key] = nn.Parameter(torch.zeros(lane_width, in_features))
                    module.register_forward_hook(functools.partial(_write_lanes, self._lanes, self.writers, key))
                elif _named(name, SUBLAYERS):
                    module.register_forward_hook(functools.partial(_append_written, self._lanes))
                elif _named(name, NORMS + FINAL_NORMS):
                    module.register_forward_pre_hook(functools.partial(_cut_lanes, self._lanes, embed.embedding_dim))
                    if _named(name, FINAL_NORMS):
                        module.register_forward_hook(functools.partial(_restore_lanes, self._lanes))
                elif _named(name, HEADS):
                    module.register_forward_pre_hook(functools.partial(_drop_lanes, embed.embedding_dim))
                elif _named(name, PADDED):
                    module.register_forward_hook(functools.partial(_pad_for_lanes, self._lanes, lane_width))
            if len(self.readers) == 0:
                raise ValueError(f"{delivery} delivery: no module of the base is named like one of {READERS}")
            if delivery == "wide" and len(self.writers) == 0:
                raise ValueError(f"wide delivery: no module of the base is named like one of {WRITERS}")
        self.projector.to(embed.weight.device)
        self.readers.to(embed.weight.device)
        self.writers.to(embed.weight.device)

    def forward(self, input_ids, signals, attention_mask=None, past_key_values=None, labels=None, use_cache=False,
                position_ids=None, logits_to_keep=None, harness_mask=None):
        """`position_ids`, `logits_to_keep`: passed to the base (batched generation with left
        padding; logits of the last position only). None = the base's own default.
        `harness_mask` ([B, T] bool, like `input_ids`): true at the tokens the harness inserted.
        Read by the segment delivery only; None = no such token."""
        if signals.shape[:2] != input_ids.shape or signals.shape[-1] != self.projector.n_signals:
            raise ValueError(
                f"signals {tuple(signals.shape)} do not match input_ids {tuple(input_ids.shape)} "
                f"x {self.projector.n_signals} signals"
            )
        if harness_mask is not None and harness_mask.shape != input_ids.shape:
            raise ValueError(f"harness_mask {tuple(harness_mask.shape)} does not match input_ids {tuple(input_ids.shape)}")
        emb = self.base.get_input_embeddings()(input_ids)
        self._lanes.values = self._lanes.written = None
        self._steering.added = None
        if self.enabled and self.delivery in _PROJECTED:
            weight = self.projector.net[0].weight
            projected = self.projector(signals.to(device=weight.device, dtype=weight.dtype))
            if self.delivery == "sum" and self.sum_norm is None:
                emb = emb + projected.to(emb.dtype)
            elif self.delivery == "sum":
                emb = emb + (self.sum_scale * self.sum_gain * _unit(projected)).to(emb.dtype)
            elif self.delivery == "lanes":
                self._lanes.values = projected
            elif self.delivery == "insert":
                emb = torch.where((input_ids == self.insert_id).unsqueeze(-1), projected.to(emb.dtype), emb)
            else:
                emb = torch.cat([emb, _unit(projected).to(emb.dtype)], dim=-1)
                self._lanes.widened = True
        elif self.enabled and self.delivery == "segment" and harness_mask is not None:
            emb = emb + harness_mask.unsqueeze(-1).to(emb.dtype) * self.segment.to(emb.dtype)
        elif self.enabled and self.delivery == "steer":
            limits = self.steer_limits
            values = signals.to(device=limits.device, dtype=limits.dtype)
            past = torch.where(self.steer_high, values >= limits, values <= limits)
            self._steering.added = self.steer_alpha * (past.to(limits.dtype) @ self.steer_vectors)
        try:
            return self.base(
                inputs_embeds=emb,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                **({} if logits_to_keep is None else {"logits_to_keep": logits_to_keep}),
            )
        finally:
            self._lanes.widened = False
            self._steering.added = None
            if self.delivery == "wide" or (self._lanes.values is not None and not self._lanes.values.requires_grad):
                self._lanes.values = None

    def channel_parameters(self) -> list[nn.Parameter]:
        """The parts of the channel that train at the channel's own rate: the projector and, in
        the "lanes" delivery, the reader columns. In the "wide" delivery the reader columns and
        writer rows are new columns and rows of the model's matrices, fed by unit-size inputs
        like the adapter: they train at the adapter's rate. Sum with `sum_norm`: also the gain.
        "insert": the projector. "segment": the segment vector. "none", "steer": nothing."""
        if self.delivery in ("none", "steer"):
            return []
        if self.delivery == "segment":
            return [self.segment]
        if self.delivery in ("wide", "insert"):
            return list(self.projector.parameters())
        gain = [self.sum_gain] if self.sum_norm is not None else []
        return list(self.projector.parameters()) + list(self.readers.parameters()) + gain

    def set_steering(self, vectors: torch.Tensor, alpha: float) -> None:
        """Delivery "steer" (B6a): v_s of every signal, [n_signals, d], and the factor alpha. The
        scale is the caller's (e.g. unit-size v_s and alpha = k x the residual RMS at that block)."""
        if self.delivery != "steer":
            raise ValueError(f"set_steering: the {self.delivery} delivery has no steering vectors")
        if tuple(vectors.shape) != tuple(self.steer_vectors.shape):
            raise ValueError(f"steering vectors {tuple(vectors.shape)}: expected {tuple(self.steer_vectors.shape)}")
        with torch.no_grad():
            self.steer_vectors.copy_(vectors)
            self.steer_alpha.fill_(float(alpha))

    def _extras(self) -> dict[str, torch.Tensor]:
        """The channel's tensors outside the projector, readers and writers, by their key in
        channel.pt. None in the sum, lanes and wide deliveries without options: their files keep
        the keys they always had."""
        if self.sum_norm is not None:
            return {"sum_norm.gain": self.sum_gain, "sum_norm.scale": self.sum_scale}
        if self.delivery == "segment":
            return {"segment.vector": self.segment}
        if self.delivery == "steer":
            return {"steer.vectors": self.steer_vectors, "steer.alpha": self.steer_alpha}
        return {}

    def channel_state(self) -> dict[str, torch.Tensor]:
        """The tensors of channel.pt by key (also those of the released channel.safetensors)."""
        state = dict(self.projector.state_dict())
        for part in ("readers", "writers"):
            state.update({f"{part}.{key}": value for key, value in getattr(self, part).state_dict().items()})
        state.update({key: value.detach() for key, value in self._extras().items()})
        return state

    def save_channel(self, path: Path) -> None:
        torch.save(self.channel_state(), path)

    def load_channel(self, path: Path) -> None:
        self.load_channel_state(torch.load(path, map_location=self.projector.net[0].weight.device, weights_only=True), path)

    def load_channel_state(self, state: dict[str, torch.Tensor], path: Path | str = "the channel state") -> None:
        """`channel_state()` of a saved channel (consumed); `path` names it in errors."""
        state = dict(state)
        for part in ("readers", "writers"):
            prefix = part + "."
            getattr(self, part).load_state_dict(
                {key.removeprefix(prefix): state.pop(key) for key in list(state) if key.startswith(prefix)})
        with torch.no_grad():
            for key, tensor in self._extras().items():
                if key not in state:
                    raise ValueError(f"{path} has no {key!r}: it was saved by another channel configuration")
                tensor.copy_(state.pop(key))
        self.projector.load_state_dict(state)
