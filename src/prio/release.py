"""Public API of the released models: load a trained afferent channel and generate with a signal schedule.

A released run folder holds:
  lora/                  the LoRA adapter (PEFT format) for the base model
  channel.json           how the channel is delivered (delivery, lane width, projector)
  channel.safetensors    the channel's tensors; its metadata lists the signals in channel order
Run folders written by training hold `channel.pt` instead of `channel.safetensors`; both load
from a local folder (`channel.pt` with `torch.load(..., weights_only=True)`). A Hugging Face repo
is read through `channel.safetensors` only.

Typical use:

    from prio.release import load_channel_model, generate_with_signals
    lm = load_channel_model("path/to/operating-state")       # or a Hugging Face repo id + subfolder
    prompt = lm.chat_prompt("Explain step by step how a heat pump works.")
    out = generate_with_signals(lm, prompt, lambda step: {"ctx": 0.955 if step >= 8 else 0.705})
    print(out.text)

Loading and generation go through the same code the evaluations used (`ChannelModel`,
`prio.generate.generate`, the evaluation's `follow_thinking` and stop tokens).
"""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
import torch

from prio.channel import _PROJECTED, FOURIER_FREQS, ChannelModel
from prio.generate import generate
from prio.modelspec import ModelSpec, load_base, template_kwargs
from prio.signals import SignalSpec, signal_set
from prio.train import channel_record

BASE_MODEL = "Qwen/Qwen3-1.7B"
MODEL_REPO = "DominuZ/afferent-channel-qwen3-1.7b"
DATA_REPO = "DominuZ/afferent-channel-data"
CHANNEL_FILE = "channel.safetensors"
LEGACY_FILE = "channel.pt"
# Subfolder of the model repo -> run folder name used by the experiment commands (docs/REPRODUCE.md).
RELEASED_RUNS = {
    "operating-state": "n6-wide-levels-s0",
    "operating-state-seed1": "n6-wide-levels-s1",
    "operating-state-seed2": "n6-wide-levels-s2",
    "background-jobs": "bg3s-wide-levels-s0",
    "messenger": "n7msg-wide-levels-s0",
    "game-character": "npc6-wide-levels-s0",
    "text-notice-baseline": "n6-notice-none-s0",
}
RUN_FILES = ("lora/*", "channel.json", CHANNEL_FILE, "train_log.json")  # files of one released run
_BINS = 100  # SignalProjector's default code width per signal (every released run uses it)
# Keys a release channel.json may carry that ChannelModel does not take.
_RECORD_ONLY = ("n_signals", "signals", "signal_set", "base_model")


@dataclass(frozen=True)
class Signal:
    """One input of the channel. Values are in [0, 1] and are read at 1% resolution (pass level
    centers such as 0.905 for "90%"). `nominal`: the value of the signal spec when nothing applies.
    `resting`: a value inside the range seen in training, used for a signal a schedule leaves out
    (some nominal values, e.g. energy at 1.0, never occur in the training data, and the levels
    projector has an untrained vector for every level never seen). `limit` / `bad_when`: the value
    at or past which the trained reaction is due ("high": value >= limit; "low": value <= limit);
    None for inputs without one. `kind`: "interrupt" (react now) or "gating" (react only when about
    to use the affected tool). `levels`: for a categorical input, the value of each category."""
    name: str
    nominal: float
    limit: float | None = None
    bad_when: str | None = None
    kind: str | None = None
    levels: dict[str, float] | None = None
    description: str = ""
    resting: float | None = None

    @property
    def default(self) -> float:
        return self.nominal if self.resting is None else self.resting

    def past_limit(self, value: float) -> bool | None:
        if self.limit is None or self.bad_when is None:
            return None
        return value >= self.limit if self.bad_when == "high" else value <= self.limit


def from_spec(spec: SignalSpec, description: str = "", levels: Mapping[str, float] | None = None,
              resting: float | None = None) -> Signal:
    return Signal(spec.name, spec.nominal, spec.limit, spec.bad_when, spec.kind,
                  dict(levels) if levels is not None else None, description, resting)


_DESCRIPTIONS = {
    "ctx": "context window in use (high is bad)",
    "think": "thinking budget left (low is bad)",
    "energy": "battery left (low is bad)",
    "time": "deadline: share of the time budget used (high is bad)",
    "net": "network: 1 = up, 0 = down",
    "ram": "free memory (low is bad)",
    "cell": "cellular link: 1 = up, 0 = down",
}
# Resting values (level centers) well inside the training range of each signal and away from its limit.
_RESTING = {"ctx": 0.305, "think": 1.0, "energy": 0.805, "time": 0.205, "net": 1.0, "ram": 0.805, "cell": 1.0,
            "fear": 0.205, "thirst": 0.205, "hunger": 0.205, "threat": 0.055, "water_near": 1.0, "food_near": 1.0}
SIGNAL_SETS = ("operating-state", "messenger", "game-character", "background-tasks")


def release_signals(name: str) -> tuple[Signal, ...]:
    """The signals of a released model family, in channel order, built from the specs the data
    builders used. Names: SIGNAL_SETS."""
    if name == "operating-state":
        return tuple(from_spec(s, _DESCRIPTIONS.get(s.name, ""), resting=_RESTING[s.name]) for s in signal_set(6))
    if name == "messenger":
        from prio.msg.domain import SPECS
        return tuple(from_spec(s, _DESCRIPTIONS.get(s.name, ""), resting=_RESTING[s.name]) for s in SPECS)
    if name == "game-character":
        from prio.npc.domain import EDGES, SPECS, THREATS
        felt = {k: f"{k} (high is bad): middle band from {lo:.2f}, high band from {hi:.2f}" for k, (lo, hi) in EDGES.items()}
        felt["fear"] = f"fear (high is bad): uneasy from {EDGES['fear'][0]:.2f}, terror from {EDGES['fear'][1]:.2f}"
        text = {**felt, "threat": "what threatens the character: one level per kind, set together with fear",
                "water_near": "water within reach: 1 = yes, 0 = no", "food_near": "food at hand: 1 = yes, 0 = no"}
        return tuple(from_spec(s, text[s.name], THREATS if s.name == "threat" else None, _RESTING[s.name])
                     for s in SPECS)
    if name == "background-tasks":
        from prio.bg.protocol import N_SLOTS, STATE_VALUE
        out = []
        for k in range(1, N_SLOTS + 1):
            out.append(Signal(f"slot{k}_progress", 0.0, description=f"progress of the task watched in slot {k}"))
            out.append(Signal(f"slot{k}_state", STATE_VALUE["off"], levels=dict(STATE_VALUE),
                              description=f"state of the task watched in slot {k}: one level per state"))
        return tuple(out)
    raise ValueError(f"signal set {name!r}: expected one of {SIGNAL_SETS}")


# --- channel files -------------------------------------------------------------------------------

def save_channel_safetensors(state: Mapping[str, torch.Tensor], path: Path, signals: Sequence[Signal],
                             signal_set_name: str | None = None) -> None:
    """Write the channel's tensors (`ChannelModel.channel_state()` or a loaded channel.pt) as
    safetensors; the metadata records the signals (JSON) and their number."""
    from safetensors.torch import save_file

    tensors = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    metadata = {"format": "pt", "n_signals": str(len(signals)), "signals": json.dumps([asdict(s) for s in signals])}
    if signal_set_name is not None:
        metadata["signal_set"] = signal_set_name
    save_file(tensors, str(path), metadata=metadata)


def read_channel_file(path: Path, device: str | torch.device = "cpu") -> tuple[dict[str, torch.Tensor], tuple[Signal, ...] | None]:
    """(tensors, signals) of channel.safetensors, or (tensors, None) of a channel.pt."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device=str(device)) as f:
            meta = f.metadata() or {}
            state = {key: f.get_tensor(key) for key in f.keys()}
        signals = tuple(Signal(**s) for s in json.loads(meta["signals"])) if "signals" in meta else None
        if signals is not None and "n_signals" in meta and int(meta["n_signals"]) != len(signals):
            raise ValueError(f"{path}: metadata names {len(signals)} signals but says n_signals={meta['n_signals']}")
        return state, signals
    return torch.load(path, map_location=device, weights_only=True), None


def channel_file(run_dir: Path) -> Path:
    """The channel file of a run folder: channel.safetensors (published) or channel.pt (written by
    training). Both hold the same tensors."""
    run_dir = Path(run_dir)
    for name in (CHANNEL_FILE, LEGACY_FILE):
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"{run_dir}: neither {CHANNEL_FILE} nor {LEGACY_FILE}")


def load_channel_tensors(run_dir: Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """The channel's tensors of a run folder, from whichever channel file it holds."""
    return read_channel_file(channel_file(run_dir), device)[0]


def export_run(run_dir: Path, signals: Sequence[Signal], signal_set_name: str | None = None) -> Path:
    """Write run_dir/channel.safetensors from run_dir/channel.pt (kept) and check the round trip:
    same keys, dtypes, shapes and values. Returns the new file's path."""
    run_dir = Path(run_dir)
    state, _ = read_channel_file(run_dir / LEGACY_FILE)
    non_tensors = sorted(k for k, v in state.items() if not torch.is_tensor(v))
    if non_tensors:
        raise ValueError(f"{run_dir / LEGACY_FILE}: non-tensor entries {non_tensors}")
    n = n_signals_of(state, channel_record(run_dir))
    if n != len(signals):
        raise ValueError(f"{run_dir}: the channel has {n} signals, {len(signals)} given")
    out = run_dir / CHANNEL_FILE
    save_channel_safetensors(state, out, signals, signal_set_name)
    back, named = read_channel_file(out)
    if set(back) != set(state) or any(back[k].dtype != state[k].dtype or not torch.equal(back[k], state[k]) for k in state):
        raise ValueError(f"{out}: the round trip changed the tensors")
    if named != tuple(signals):
        raise ValueError(f"{out}: the round trip changed the signal metadata")
    return out


def n_signals_of(state: Mapping[str, torch.Tensor], record: Mapping[str, Any]) -> int:
    """Number of signals of a saved channel, from its first projector layer."""
    width = int(state["net.0.weight"].shape[1])
    if record["delivery"] not in _PROJECTED:  # placeholder projector: one input per signal
        return width
    per = _BINS + (2 * len(FOURIER_FREQS) if record["projector"] == "levels+fourier" else 0)
    if width % per:
        raise ValueError(f"projector input of {width} features is not a multiple of {per}")
    return width // per


# --- loading -------------------------------------------------------------------------------------

def resolve_run(source: str | Path, subfolder: str | None = None, revision: str | None = None,
                token: str | bool | None = None) -> Path:
    """A local run folder, or the folder downloaded from a Hugging Face model repo (`source` =
    repo id, `subfolder` = the run inside it). From a repo only `lora/`, `channel.json` and
    `channel.safetensors` are downloaded (never a pickle). Private repos need a token
    (`hf auth login` or `token=`)."""
    local = Path(source) / (subfolder or "")
    if local.exists():
        return local
    if Path(source).exists():
        raise FileNotFoundError(f"{local} does not exist")
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    patterns = [f"{prefix}lora/*", f"{prefix}channel.json", f"{prefix}{CHANNEL_FILE}"]
    run = Path(snapshot_download(repo_id=str(source), revision=revision, token=token, allow_patterns=patterns))
    run = run / (subfolder or "")
    if not (run / "channel.json").exists() or not (run / CHANNEL_FILE).exists():
        raise FileNotFoundError(f"{source}: no released run (channel.json and {CHANNEL_FILE}) at {prefix or 'the repo root'}")
    return run


def place_runs(staging: Path, runs_dir: Path, subfolders: Sequence[str]) -> list[Path]:
    """Move downloaded model-repo subfolders (`staging/<subfolder>`) to `runs_dir/<run name>`
    (RELEASED_RUNS). A run folder that already exists is left as it is."""
    staging, runs_dir = Path(staging), Path(runs_dir)
    placed = []
    for sub in subfolders:
        dest = runs_dir / RELEASED_RUNS[sub]
        if not dest.exists():
            src = staging / sub
            if not (src / "channel.json").exists() or not (src / CHANNEL_FILE).exists():
                raise FileNotFoundError(f"{src}: not a downloaded run (channel.json and {CHANNEL_FILE} expected)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        placed.append(dest)
    return placed


def fetch_runs(runs_dir: Path | str = "runs", subfolders: Sequence[str] | None = None, *, repo_id: str = MODEL_REPO,
               revision: str | None = None, token: str | bool | None = None) -> list[Path]:
    """Download released runs from the model repo into `runs_dir/<run name>`, the folders the
    experiment commands read (`runs/n6-wide-levels-s0`, ...). Default: every released run.
    Runs already present are skipped. Downloads go through `runs_dir/_download` (removed when empty)."""
    runs_dir = Path(runs_dir)
    wanted = list(subfolders) if subfolders else list(RELEASED_RUNS)
    unknown = [s for s in wanted if s not in RELEASED_RUNS]
    if unknown:
        raise ValueError(f"not a released run: {unknown}; expected some of {list(RELEASED_RUNS)}")
    todo = [s for s in wanted if not (runs_dir / RELEASED_RUNS[s]).exists()]
    if todo:
        from huggingface_hub import snapshot_download

        staging = runs_dir / "_download"
        snapshot_download(repo_id=repo_id, revision=revision, token=token, local_dir=str(staging),
                          allow_patterns=[f"{s}/{p}" for s in todo for p in RUN_FILES])
        place_runs(staging, runs_dir, todo)
        shutil.rmtree(staging / ".cache", ignore_errors=True)  # download metadata of the moved files
        with contextlib.suppress(OSError):
            staging.rmdir()
    return [runs_dir / RELEASED_RUNS[s] for s in wanted]


def fetch_data(root: Path | str = ".", folders: Sequence[str] = ("data", "results"), *, repo_id: str = DATA_REPO,
               revision: str | None = None, token: str | bool | None = None) -> Path:
    """Download folders of the dataset repo (default: all of `data/` and `results/`) into `root`,
    at the paths the experiment commands read. The dataset card itself is not downloaded."""
    from huggingface_hub import snapshot_download

    patterns = [f"{f.strip('/')}/*" for f in folders]
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, token=token, local_dir=str(root),
                      allow_patterns=patterns)
    return Path(root)


@dataclass
class ChannelLM:
    """A loaded model: base + adapter + channel, its tokenizer, the special tokens and the signals."""
    model: ChannelModel
    tokenizer: Any
    spec: ModelSpec
    signals: tuple[Signal, ...]
    run_dir: Path

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.signals)

    def resting(self) -> dict[str, float]:
        """The row a schedule starts from: every signal at its resting value."""
        return {s.name: s.default for s in self.signals}

    def row(self, values: Mapping[str, float] | Sequence[float]) -> list[float]:
        """One row in channel order: a mapping (unnamed signals at their resting value) or a full sequence."""
        if isinstance(values, Mapping):
            unknown = sorted(set(values) - set(self.names))
            if unknown:
                raise ValueError(f"unknown signals {unknown}: this model reads {list(self.names)}")
            return [float(values.get(s.name, s.default)) for s in self.signals]
        row = [float(v) for v in values]
        if len(row) != len(self.signals):
            raise ValueError(f"a row of {len(row)} values for {len(self.signals)} signals {list(self.names)}")
        return row

    def chat_prompt(self, user: str, system: str | None = None, tools: list[dict] | None = None) -> list[int]:
        return chat_prompt(self.tokenizer, user, system, tools)

    def generate(self, prompt, schedule, max_new_tokens: int = 200, **kwargs) -> "Generation":
        return generate_with_signals(self, prompt, schedule, max_new_tokens, **kwargs)


def load_channel_model(source: str | Path, model_id: str = BASE_MODEL, device: str = "cpu", dtype=None, *,
                       subfolder: str | None = None, signals: Sequence[Signal] | Sequence[str] | None = None,
                       revision: str | None = None, token: str | bool | None = None) -> ChannelLM:
    """Load base model + LoRA adapter + channel from a run folder (local, or `source` = Hugging Face
    repo id with `subfolder`). `dtype`: default bfloat16 on CUDA, float32 on CPU. `signals`: names
    (or Signal objects) in channel order when the channel file carries none (default: the
    operating-state set)."""
    run = resolve_run(source, subfolder, revision, token)
    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    base, tok, ms = load_base(model_id, dtype=dtype, device=device)
    record = channel_record(run)
    for key in _RECORD_ONLY:
        record.pop(key, None)
    adapter = record.pop("adapter", True)
    path = channel_file(run)
    state, named = read_channel_file(path, base.get_input_embeddings().weight.device)
    n = n_signals_of(state, record)
    model = ChannelModel(base, n, **record)
    if adapter:
        from peft import PeftModel

        model.base = PeftModel.from_pretrained(base, run / "lora")
    model.load_channel_state(state, path)
    chosen = _pick_signals(signals, named, n)
    return ChannelLM(model.eval(), tok, ms, chosen, run)


def _pick_signals(given, named, n: int) -> tuple[Signal, ...]:
    if given is not None:
        out = tuple(s if isinstance(s, Signal) else Signal(str(s), 0.0) for s in given)
    elif named is not None:
        out = named
    else:
        out = release_signals("operating-state") if n == 6 else tuple(
            from_spec(s, _DESCRIPTIONS.get(s.name, "")) for s in signal_set(n))
    if len(out) != n:
        raise ValueError(f"{len(out)} signal names for a channel of {n} signals")
    return out


# --- prompts and generation ----------------------------------------------------------------------

def chat_prompt(tok, user: str, system: str | None = None, tools: list[dict] | None = None) -> list[int]:
    """Token ids of a chat prompt up to the start of the assistant turn, thinking on, as in training.
    Defaults: the operating-state system prompt and tools (`prio.tasks`, `prio.reactions.TOOLS`).
    A game character passes its card as `system` and `prio.npc.domain.NPC_TOOLS` as `tools`."""
    from prio.reactions import TOOLS
    from prio.tasks import system_prompt

    messages = [{"role": "system", "content": system_prompt(tok) if system is None else system},
                {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tools=TOOLS if tools is None else tools, add_generation_prompt=True,
                                   enable_thinking=True, tokenize=False, **template_kwargs(tok))
    return tok(text, add_special_tokens=False).input_ids


@dataclass
class Generation:
    text: str  # the generated tokens, special tokens included (<think>, <tool_call>, ...)
    token_ids: list[int]
    signals: np.ndarray  # [len(token_ids), N]: the row each generated token carried
    stop_reason: str  # "eos" | "max_new_tokens"


Row = Union[Mapping[str, float], Sequence[float]]
Schedule = Union[Callable[[int], Row], Mapping[str, float], np.ndarray, Sequence[Row]]


def _rows_of(lm: ChannelLM, schedule: Schedule, n: int) -> np.ndarray:
    """`n` rows in channel order from a schedule: callable(step), one constant mapping, or rows
    (the last one repeats)."""
    if callable(schedule):
        rows = [lm.row(schedule(step)) for step in range(n)]
    elif isinstance(schedule, Mapping):
        rows = [lm.row(schedule)] * n
    else:
        given = [lm.row(r) for r in schedule]
        if not given:
            raise ValueError("an empty schedule")
        rows = [given[min(i, len(given) - 1)] for i in range(n)]
    return np.asarray(rows, dtype=np.float32).reshape(n, len(lm.signals))


def generate_with_signals(lm: ChannelLM, prompt: str | Sequence[int], schedule: Schedule, max_new_tokens: int = 200,
                          *, prompt_signals: Row | np.ndarray | None = None, follow_thinking: bool = True,
                          channel: bool = True, temperature: float = 0.0, top_p: float = 1.0,
                          seed: int | None = None) -> Generation:
    """Generate with the signals of every new token taken from `schedule`.

    `schedule(step)` gives the values that generated token `step` (0-based) carries as it enters
    the model; the model can react from token step + 1 on (a mapping: unnamed signals at their
    resting value, `Signal.resting`). A constant mapping or a sequence of rows also works. `prompt`: token ids, or a
    user message wrapped by `lm.chat_prompt`. `prompt_signals`: the prompt's rows (one row for every
    prompt token, or [T0, N]); default: the schedule's row of step 0. `follow_thinking`: once the
    model closes its thinking, `think` takes its nominal value, as in the evaluations. `channel=False`
    switches the channel off (the adapter stays). Greedy by default."""
    tok, ms, model = lm.tokenizer, lm.spec, lm.model
    from prio.eval.run import follow_thinking as follow, stop_ids

    prompt_ids = lm.chat_prompt(prompt) if isinstance(prompt, str) else [int(t) for t in prompt]
    new = _rows_of(lm, schedule, max_new_tokens)
    if prompt_signals is None:
        head = np.repeat(new[:1], len(prompt_ids), axis=0)
    elif isinstance(prompt_signals, np.ndarray) and prompt_signals.ndim == 2:
        head = prompt_signals.astype(np.float32)
        if head.shape != (len(prompt_ids), len(lm.signals)):
            raise ValueError(f"prompt_signals {head.shape} for {len(prompt_ids)} tokens x {len(lm.signals)} signals")
    else:
        head = np.repeat(np.asarray([lm.row(prompt_signals)], dtype=np.float32), len(prompt_ids), axis=0)
    values = np.concatenate([head, new])
    think = lm.names.index("think") if follow_thinking and "think" in lm.names else None
    nominal = lm.signals[think].nominal if think is not None else 1.0
    signal_fn = follow(values, len(prompt_ids), think, nominal, ms.think_close)
    device = model.base.get_input_embeddings().weight.device
    model.enabled = channel
    try:
        # bfloat16 autocast on CUDA, as the evaluations ran; nothing touches CUDA on another device
        cast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
        with cast:
            gen = generate(model, torch.tensor([prompt_ids], device=device),
                           torch.from_numpy(values[None, :len(prompt_ids)]).to(device), signal_fn, max_new_tokens,
                           stop_ids(tok, ms), temperature=temperature, top_p=top_p, seed=seed)
    finally:
        model.enabled = True
    return Generation(tok.decode(gen.token_ids, skip_special_tokens=False), list(gen.token_ids), gen.signals,
                      gen.stop_reason)
