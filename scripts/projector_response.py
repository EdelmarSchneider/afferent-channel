"""Does a trained projector answer a limit crossing? Needs only the run's channel file (seconds, CPU).

For each signal: move it one 1% step at a time, the other signals at random mid values, and
measure how far the projector's output moves at each step. A learned limit shows as the step
across the limit standing far above the median step. For the binary signal: the distance between
up and down.

Usage: uv run python scripts/projector_response.py runs/n6-s0 [more runs ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from prio.channel import SignalProjector
from prio.release import load_channel_tensors
from prio.signals import signal_set


def report(run: Path, n_signals: int = 6) -> None:
    state = load_channel_tensors(run)  # channel.pt or channel.safetensors
    state = {k: v.float() for k, v in state.items() if not k.startswith(("readers.", "writers."))}
    record = run / "channel.json"
    saved = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    kind = saved.get("projector", "mlp")
    last = max(k for k in state if k.endswith(".weight"))
    proj = SignalProjector(n_signals, state[last].shape[0], zero_init=False, kind=kind)
    proj.load_state_dict(state)
    proj.eval()
    specs = signal_set(n_signals)
    names = [s.name for s in specs]
    binary = [i for i, s in enumerate(specs) if s.dynamics == "binary"]
    rng = np.random.default_rng(0)
    centers = torch.tensor((np.arange(100) + 0.5) / 100, dtype=torch.float32)

    def mid_states(n: int) -> torch.Tensor:
        states = torch.tensor(rng.uniform(0.2, 0.8, size=(n, n_signals)), dtype=torch.float32)
        states[:, binary] = 1.0
        return states

    print(f"{run}: delivery {saved.get('delivery', 'sum')}, projector {kind}, output width {state[last].shape[0]}")
    with torch.no_grad():
        out = proj(mid_states(2000).unsqueeze(0))[0]
        print(f"  output norm at random mid states: median {out.norm(dim=-1).median():.3f} | "
              f"distance between two such states: median {(out[:1000] - out[1000:]).norm(dim=-1).median():.3f}")
        for i, spec in enumerate(specs):
            if i in binary:
                up = mid_states(1).unsqueeze(0)
                down = up.clone()
                down[..., i] = 0.0
                print(f"  {spec.name}: distance between up and down {(proj(up) - proj(down)).norm().item():.3f}")
                continue
            steps = []
            for _ in range(20):  # 20 settings of the other signals
                grid = mid_states(1).repeat(100, 1)
                grid[:, i] = centers
                o = proj(grid.unsqueeze(0))[0]
                steps.append((o[1:] - o[:-1]).norm(dim=-1).numpy())  # step k: bin k -> bin k + 1
            steps = np.mean(steps, axis=0)
            k = int(round(spec.limit * 100)) - 1  # 0.895 -> 0.905 is step 89; 0.105 -> 0.095 is step 9
            order = np.argsort(-steps)
            print(f"  {spec.name} (limit {spec.limit}, bad when {spec.bad_when}): step across the limit {steps[k]:.4f} "
                  f"| median step {np.median(steps):.4f} | ratio {steps[k] / np.median(steps):.0f} "
                  f"| rank {int(np.where(order == k)[0][0]) + 1}/99 "
                  f"| largest steps {[(int(j), round(float(steps[j]), 4)) for j in order[:3]]}")


for path in sys.argv[1:]:
    report(Path(path))
