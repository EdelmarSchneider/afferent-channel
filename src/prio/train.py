"""Fine-tuning: LoRA on the base model plus the signal projector. Base weights stay frozen."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from prio.channel import DELIVERIES, PROJECTORS, SUM_NORMS, ChannelModel, insert_id
from prio.datagen.episodes import Episode, reaction_start, read_jsonl
from prio.modelspec import load_base

IGNORE = -100


def collate(episodes: list[Episode], pad_id: int, n_signals: int, max_len: int = 2048,
            reaction_weight: float = 1.0) -> dict:
    """Right-padded batch. Loss only on target tokens. Over-long episodes are dropped, never
    truncated (truncation could cut the reaction away); `dropped` counts them. `weights` gives
    the reaction tokens `reaction_weight`, and so does the one token of a no-event twin where
    it parts from its reaction sibling; every other token weighs 1. `decision` marks that token
    in both: 1 where the reaction shows, 2 where the twin carries on, 0 elsewhere. `signal` names
    each row's signal (None for an episode without one). `harness_mask` (bool) is true at the
    tokens of the episode's harness spans (written by the harness, not by the model): no loss
    there, and the segment delivery marks them."""
    kept = []
    for ep in episodes:
        length = len(ep.prompt_ids) + len(ep.target_ids)
        if ep.signals.shape != (length, n_signals):
            raise ValueError(f"{ep.episode_id}: signals {ep.signals.shape} for {length} tokens x {n_signals} signals")
        if length <= max_len:
            kept.append(ep)
    width = max((len(ep.prompt_ids) + len(ep.target_ids) for ep in kept), default=0)
    batch = {
        "input_ids": torch.full((len(kept), width), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(kept), width), dtype=torch.long),
        "signals": torch.zeros((len(kept), width, n_signals), dtype=torch.float32),
        "labels": torch.full((len(kept), width), IGNORE, dtype=torch.long),
        "weights": torch.ones((len(kept), width), dtype=torch.float32),
        "decision": torch.zeros((len(kept), width), dtype=torch.long),
        "harness_mask": torch.zeros((len(kept), width), dtype=torch.bool),
        "signal": [ep.signal for ep in kept],
        "dropped": len(episodes) - len(kept),
    }
    for row, ep in enumerate(kept):
        n_prompt, length = len(ep.prompt_ids), len(ep.prompt_ids) + len(ep.target_ids)
        batch["input_ids"][row, :length] = torch.tensor(ep.prompt_ids + ep.target_ids)
        batch["attention_mask"][row, :length] = 1
        batch["signals"][row, :length] = torch.from_numpy(ep.signals)
        batch["labels"][row, n_prompt:length] = torch.tensor(ep.target_ids)
        start = reaction_start(ep)
        if start is not None:
            batch["weights"][row, n_prompt + start:length] = reaction_weight
        if ep.t_diverge is not None:
            batch["weights"][row, n_prompt + ep.t_diverge] = reaction_weight
            # bg events and state-table reactions (prio.table) weight their reaction by focus spans
            reacts = start is not None or ep.kind in ("bg_event", "table_act", "table_mention")
            batch["decision"][row, n_prompt + ep.t_diverge] = 1 if reacts else 2
        for a, b in ep.focus_spans or ():
            batch["weights"][row, n_prompt + a:n_prompt + b] = reaction_weight
        for a, b in ep.harness_spans or ():  # written by the harness: context, not a target
            batch["labels"][row, n_prompt + a:n_prompt + b] = IGNORE
            batch["harness_mask"][row, n_prompt + a:n_prompt + b] = True
    return batch


@torch.no_grad()
def decision_losses(logits: torch.Tensor, labels: torch.Tensor, decision: torch.Tensor) -> dict[str, list[float]]:
    """Cross-entropy on the decision tokens: "react" where a reaction shows, "hold" where its
    twin carries on. The text before them is the same, so both fall only if the channel is used."""
    out: dict[str, list[float]] = {"react": [], "hold": []}
    for name, mark in (("react", 1), ("hold", 2)):
        keep = decision[:, 1:] == mark
        if keep.any():
            out[name] = torch.nn.functional.cross_entropy(
                logits[:, :-1][keep].float(), labels[:, 1:][keep], reduction="none").tolist()
    return out


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over target tokens (position t predicts token t + 1), weighted per token.

    Only target positions go through the float32 softmax: over a 152k vocabulary that is the
    memory hog of training.
    """
    keep = labels[:, 1:] != IGNORE
    per_token = torch.nn.functional.cross_entropy(logits[:, :-1][keep].float(), labels[:, 1:][keep], reduction="none")
    w = weights[:, 1:][keep]
    return (per_token * w).sum() / w.sum()


def add_lora(model: ChannelModel, r: int = 16, alpha: int = 32, dropout: float = 0.05,
             checkpointing: bool = False) -> None:
    """`checkpointing`: recompute each layer's activations in the backward pass instead of storing
    them. Same gradients, far less memory, more time. Needed at 1.7B for episodes over ~1400 tokens."""
    if checkpointing:
        model.base.activation_checkpointing_enable()
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules="all-linear")
    model.base = get_peft_model(model.base, config)


def _write_whole(state: dict, path: Path) -> None:
    """Write beside the target and rename: a cut during the write leaves the previous file whole."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    torch.save(state, partial)
    os.replace(partial, path)


def fit(
    model: ChannelModel,
    episodes: list[Episode],
    pad_id: int,
    *,
    epochs: int = 3,
    lr_lora: float = 2e-4,
    lr_proj: float = 1e-3,
    batch_size: int = 2,
    grad_accum: int = 8,
    seed: int = 0,
    device: str = "cuda",
    max_len: int = 2048,
    reaction_weight: float = 20.0,
    log_every: int = 0,
    on_epoch_end=None,
    history: dict | None = None,
    state_path: Path | None = None,
    pause_after: int | None = None,
    snapshot_minutes: float | None = None,
) -> list[float]:
    """Train every parameter that requires grad. Returns the (weighted) loss of each batch.

    A batch that runs out of GPU memory is skipped. `on_epoch_end(epoch)` is called after each
    epoch (the CLI saves the run there, so a crash loses at most one epoch). `history`, if
    given, receives the loss of every decision token ("react", "hold"; see `decision_losses`),
    the react losses per signal ("react_by_signal") and the number of skipped batches: the early
    sign of whether the channel is being learned. The channel's own parameters (projector, lane
    reader columns) train at `lr_proj`, everything else at `lr_lora`. `history["minutes"]` is the
    training time so far. The model gets each batch's `harness_mask` (read by the segment delivery).

    `state_path`: after every epoch, everything needed to go on is written there (trained
    parameters, optimizer, schedule, random states, losses, history). If the file exists when
    `fit` starts, training continues after the last finished epoch: same model construction and
    same arguments required, `ValueError` otherwise. The caller deletes the file when the run is
    saved. `pause_after` = n returns at the end of epoch n if the run started before it (the
    schedule stays that of the full run); `history["epochs_done"]` tells how far it got.
    `snapshot_minutes`: also write the state inside an epoch, after the first optimizer step that
    ends at least that many minutes after the last write, so a killed run loses minutes, not an
    epoch (0 = after every optimizer step).
    """
    history = {} if history is None else history
    history.update(react=[], hold=[], react_by_signal={}, skipped=0, minutes=0.0, epochs_done=0)
    n_signals = model.projector.n_signals
    proj = model.channel_parameters()
    proj_ids = {id(p) for p in proj}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in proj_ids]
    optimizer = torch.optim.AdamW([{"params": rest, "lr": lr_lora}, {"params": proj, "lr": lr_proj}], weight_decay=0.0)
    steps_per_epoch = math.ceil(math.ceil(len(episodes) / batch_size) / grad_accum)
    total, warmup = steps_per_epoch * epochs, max(1, round(0.03 * steps_per_epoch * epochs))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))),
    )
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    amp = device.startswith("cuda")
    losses: list[float] = []
    settings = {"episodes": len(episodes), "epochs": epochs, "lr_lora": lr_lora, "lr_proj": lr_proj,
                "batch_size": batch_size, "grad_accum": grad_accum, "seed": seed, "max_len": max_len,
                "reaction_weight": reaction_weight}
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    first_epoch = 0
    if state_path is not None and Path(state_path).exists():
        state = torch.load(state_path, map_location=device)
        changed = sorted(k for k in settings if state["settings"].get(k) != settings[k])
        if {n: tuple(p.shape) for n, p in trainable.items()} != {n: tuple(p.shape) for n, p in state["parameters"].items()}:
            changed.append("trainable parameters")
        if changed:
            raise ValueError(f"{state_path} was written by a run with other settings ({', '.join(changed)}): "
                             "delete it to start from scratch")
        with torch.no_grad():
            for name, p in trainable.items():
                p.copy_(state["parameters"][name])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        rng.bit_generator.state = state["order_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())  # `map_location` moved the saved states to the device
        if amp and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([one.cpu() for one in state["cuda_rng"]])
        losses = state["losses"]
        history.update(state["history"])
        first_epoch = history["epochs_done"] = state["epochs_done"]
        batches_done = state.get("batches_done", 0)
    else:
        batches_done = 0
    skipped = history["skipped"]
    started = time.time() - 60 * history["minutes"]
    last_write = time.time()

    def write_state(epochs_done: int, batches: int, order_rng: dict) -> None:
        """`order_rng`: the state of the order generator at the start of epoch `epochs_done`."""
        history["minutes"] = (time.time() - started) / 60
        history["epochs_done"] = epochs_done
        _write_whole({
            "settings": settings, "epochs_done": epochs_done, "batches_done": batches,
            "parameters": {name: p.detach() for name, p in trainable.items()},
            "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
            "order_rng": order_rng, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if amp else None,
            "losses": losses, "history": history,
        }, state_path)

    model.train()
    for epoch in range(first_epoch, epochs):
        epoch_rng = rng.bit_generator.state
        order = rng.permutation(len(episodes))
        starts = range(0, len(order), batch_size)
        for n_batch, i in enumerate(starts, start=1):
            if epoch == first_epoch and n_batch <= batches_done:
                continue  # done before the run was interrupted
            batch = collate([episodes[j] for j in order[i:i + batch_size]], pad_id, n_signals, max_len, reaction_weight)
            if batch["input_ids"].shape[0] > 0:
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        logits = model(batch["input_ids"].to(device), batch["signals"].to(device),
                                       attention_mask=batch["attention_mask"].to(device),
                                       harness_mask=batch["harness_mask"].to(device)).logits
                    loss = weighted_loss(logits, batch["labels"].to(device), batch["weights"].to(device))
                    (loss / grad_accum).backward()
                    losses.append(loss.item())
                    found = decision_losses(logits, batch["labels"].to(device), batch["decision"].to(device))
                    for name, values in found.items():
                        history[name] += values
                    reacting = (batch["decision"] == 1).any(dim=1).nonzero().flatten().tolist()
                    for row, value in zip(reacting, found["react"]):
                        history["react_by_signal"].setdefault(batch["signal"][row], []).append(value)
                except torch.OutOfMemoryError:
                    skipped += 1
                    history["skipped"] = skipped
                    logits = loss = None
                    optimizer.zero_grad(set_to_none=True)
                    if amp:
                        torch.cuda.empty_cache()
                    print(f"out of memory on a batch of width {batch['input_ids'].shape[1]}: skipped ({skipped} so far)",
                          flush=True)
            if n_batch % grad_accum == 0 or n_batch == len(starts):
                torch.nn.utils.clip_grad_norm_(rest + proj, 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                if (state_path is not None and snapshot_minutes is not None and n_batch < len(starts)
                        and time.time() - last_write >= 60 * snapshot_minutes):
                    write_state(epoch, n_batch, epoch_rng)
                    last_write = time.time()
            if log_every and n_batch % log_every == 0:
                recent = float(np.mean(losses[-log_every:]))
                react, hold = (float(np.mean(history[k][-log_every // 4:])) if history[k] else float("nan")
                               for k in ("react", "hold"))
                by_signal = " ".join(f"{name} {float(np.mean(values[-20:])):.2f}"
                                     for name, values in sorted(history["react_by_signal"].items(), key=lambda kv: str(kv[0])))
                print(f"epoch {epoch + 1}/{epochs} batch {n_batch}/{len(starts)} loss {recent:.4f} "
                      f"decision react {react:.3f} hold {hold:.3f} | react by signal (last 20): {by_signal} | "
                      f"elapsed {(time.time() - started) / 60:.1f} min", flush=True)
        history["minutes"] = (time.time() - started) / 60
        history["epochs_done"] = epoch + 1
        if state_path is not None:
            write_state(epoch + 1, 0, rng.bit_generator.state)
            last_write = time.time()
        if on_epoch_end is not None:
            on_epoch_end(epoch)
        if pause_after is not None and epoch + 1 == pause_after and first_epoch < pause_after:
            break
    model.eval()
    return losses


def save_run(model: ChannelModel, out_dir: Path) -> None:
    """Adapter + channel only: small, and enough to rebuild the trained model. `channel.json`
    records how the channel is delivered and which projector it has, plus, only where they
    apply: `sum_norm`, `insert_id`, `steer_layer`, and `"adapter": false` for a model without
    an adapter (a steering run: no `lora/`, never the whole base model)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"delivery": model.delivery, "lane_width": model.lane_width, "projector": model.projector_kind}
    for key in ("sum_norm", "insert_id", "steer_layer"):
        if getattr(model, key) is not None:
            record[key] = getattr(model, key)
    if isinstance(model.base, PeftModel):
        model.base.save_pretrained(out_dir / "lora")
    else:
        record["adapter"] = False
    model.save_channel(out_dir / "channel.pt")
    (out_dir / "channel.json").write_text(json.dumps(record), encoding="utf-8")


def channel_record(out_dir: Path) -> dict:
    """How a saved run delivers its channel. A run without `channel.json` was saved before lane
    delivery existed: sum at input; a record without `projector` was saved before the levels
    projector existed: hidden-layer projector. Optional keys: see `save_run`."""
    record = Path(out_dir) / "channel.json"
    saved = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    return {"delivery": "sum", "lane_width": 64, "projector": "mlp", **saved}


def load_run(model_id: str, out_dir: Path, n_signals: int):
    """Return (trained ChannelModel in eval mode, tokenizer, ModelSpec). The run folder holds
    channel.pt (written by training) or channel.safetensors (published)."""
    out_dir = Path(out_dir)
    base, tok, ms = load_base(model_id)
    record = channel_record(out_dir)
    adapter = record.pop("adapter", True)
    model = ChannelModel(base, n_signals, **record)
    if adapter:
        model.base = PeftModel.from_pretrained(base, out_dir / "lora")
    if (out_dir / "channel.pt").exists():
        model.load_channel(out_dir / "channel.pt")
    else:  # a published run: channel.safetensors only (same tensors)
        from prio.release import CHANNEL_FILE, read_channel_file

        state, _ = read_channel_file(out_dir / CHANNEL_FILE)
        model.load_channel_state(state, out_dir / CHANNEL_FILE)
    return model.eval(), tok, ms


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LoRA + signal projector on an episode set.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n-signals", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--reaction-weight", type=float, default=20.0)
    parser.add_argument("--delivery", choices=DELIVERIES, default="sum",
                        help="sum: signals added to the token embedding; lanes: dimensions of their own, read by every layer; "
                             "wide: lanes inside the vector that travels between layers, read and written by every layer; "
                             "none: no channel (text baselines, plain LoRA); insert: the projection replaces the embedding "
                             "of the inserted positions (id = first free embedding row); segment: a learned vector on the "
                             "harness-inserted tokens; steer: training-free, not trained here")
    parser.add_argument("--lane-width", type=int, default=64)
    parser.add_argument("--projector", choices=PROJECTORS, default="mlp",
                        help="mlp: two hidden layers; levels: one learned vector per 1%% level of each signal; "
                             "levels+fourier: levels plus 16 sin/cos pairs of the value per signal")
    parser.add_argument("--sum-norm", choices=SUM_NORMS, default=None,
                        help="sum delivery only. rms: each token's channel vector at unit size times a learned gain "
                             "that starts at 0.1 x the mean embedding RMS")
    parser.add_argument("--lr-proj", type=float, default=1e-3,
                        help="learning rate of the channel's own parameters (projector, lane readers)")
    parser.add_argument("--snapshot-minutes", type=float, default=10.0,
                        help="also write resume.pt inside an epoch, this many minutes apart (a killed run loses minutes)")
    parser.add_argument("--pause-after", type=int, default=None,
                        help="stop at the end of this epoch and keep resume.pt; the same command without it continues")
    parser.add_argument("--no-grad-checkpointing", action="store_true",
                        help="store activations instead of recomputing them: faster, much more memory")
    args = parser.parse_args(argv)
    if args.delivery == "steer":
        parser.error("steer is training-free: build the ChannelModel, call set_steering and save_run")
    if args.sum_norm is not None and args.delivery != "sum":
        parser.error("--sum-norm needs --delivery sum")

    base, tok, _ = load_base(args.model)
    torch.manual_seed(args.seed)  # projector and adapter initialization are part of the run
    model = ChannelModel(base, args.n_signals, delivery=args.delivery, lane_width=args.lane_width,
                         projector=args.projector, sum_norm=args.sum_norm,
                         insert_id=insert_id(base, tok) if args.delivery == "insert" else None)
    add_lora(model, checkpointing=not args.no_grad_checkpointing)
    episodes = read_jsonl(args.data / "train.jsonl")
    state_path = args.out / "resume.pt"  # written after every epoch, deleted when the run is saved
    if state_path.exists():
        print(f"continuing the interrupted run in {args.out}", flush=True)
    history: dict = {}
    losses = fit(model, episodes, tok.pad_token_id, epochs=args.epochs, batch_size=args.batch_size,
                 grad_accum=args.grad_accum, seed=args.seed, device=str(model.projector.net[0].weight.device),
                 max_len=args.max_len, history=history, lr_proj=args.lr_proj,
                 reaction_weight=args.reaction_weight, log_every=100,
                 on_epoch_end=lambda epoch: save_run(model, args.out), state_path=state_path,
                 pause_after=args.pause_after, snapshot_minutes=args.snapshot_minutes)
    if history["epochs_done"] < args.epochs:
        print(f"paused {args.out} after epoch {history['epochs_done']} of {args.epochs} "
              f"({history['minutes']:.1f} min): run the same command to continue", flush=True)
        return
    save_run(model, args.out)
    log = {"model": args.model, "episodes": len(episodes), "epochs": args.epochs, "seed": args.seed,
           "minutes": history["minutes"], "final_loss": float(np.mean(losses[-50:])), "losses": losses,
           "decision_react": history["react"], "decision_hold": history["hold"],
           "react_by_signal": history["react_by_signal"], "delivery": args.delivery,
           "projector": args.projector, "lr_proj": args.lr_proj, "skipped_batches": history["skipped"],
           "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.out / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    state_path.unlink(missing_ok=True)
    print(f"saved {args.out}: final loss {log['final_loss']:.4f} in {log['minutes']:.1f} min")


if __name__ == "__main__":
    main()
