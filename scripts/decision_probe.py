"""Is a saved run using the channel? Loss on the decision token of training reactions, per signal,
with the channel on and off, and on the twins' hold token. On << off = the channel is used.
Runs on CPU by default, so it can run while the GPU trains (it slows that training by ~30%).

Usage: uv run python scripts/decision_probe.py runs/n6-s0 data/n6 [per_group] [device]
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import prio.train as train_module
from prio.datagen.episodes import reaction_start, read_jsonl
from prio.modelspec import load_base

run, data = Path(sys.argv[1]), Path(sys.argv[2])
per_group = int(sys.argv[3]) if len(sys.argv) > 3 else 6
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
if device == "cpu":
    torch.set_num_threads(12)
    train_module.load_base = lambda model_id: load_base(model_id, dtype=torch.float32, device="cpu")
train = read_jsonl(data / "train.jsonl")
model, tok, ms = train_module.load_run(manifest["model"], run, train[0].signals.shape[1])
print(f"{run}: delivery {model.delivery}, projector {model.projector_kind}", flush=True)

rng = np.random.default_rng(1)
groups = defaultdict(list)
for ep in train:
    if ep.t_diverge is None:
        continue
    name = (ep.signal, ep.kind, ep.phase) if reaction_start(ep) is not None else ("twin", "hold", "")
    groups[name].append(ep)


@torch.no_grad()
def decision_loss(ep, enabled: bool) -> float:
    cut = len(ep.prompt_ids) + ep.t_diverge
    ids = torch.tensor([(ep.prompt_ids + ep.target_ids)[:cut]], device=device)
    model.enabled = enabled
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
        logits = model(ids, torch.from_numpy(ep.signals[:cut]).unsqueeze(0).to(device)).logits[0, -1].float()
    return -torch.log_softmax(logits, -1)[ep.target_ids[ep.t_diverge]].item()


started = time.time()
for name in sorted(groups):
    eps = groups[name]
    picked = [eps[i] for i in rng.permutation(len(eps))[: per_group * (3 if name[0] == "twin" else 1)]]
    on = [decision_loss(ep, True) for ep in picked]
    off = [decision_loss(ep, False) for ep in picked]
    print(f"{name}: n={len(picked)} of {len(eps)} | channel on: mean {np.mean(on):.2f} median {np.median(on):.2f} "
          f"| channel off: mean {np.mean(off):.2f} median {np.median(off):.2f} "
          f"| {(time.time() - started) / 60:.1f} min", flush=True)
