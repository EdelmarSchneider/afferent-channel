"""Convert runs/<run>/channel.pt -> runs/<run>/channel.safetensors for the released runs (CPU).

channel.pt holds tensors only; the new file holds the same tensors (checked bit for bit after
writing) plus metadata: the signals in channel order (name, nominal value, limit, levels) and their
number. channel.pt and channel.json stay as they are.

    uv run python scripts/export_channel_safetensors.py              # every release run
    uv run python scripts/export_channel_safetensors.py n6-wide-levels-s0
"""

import argparse
import sys
import time
from pathlib import Path

from prio.release import export_run, release_signals

ROOT = Path(__file__).resolve().parents[1]
# Run folder -> its signal set (prio.release.SIGNAL_SETS).
RELEASE_RUNS = {
    "n6-wide-levels-s0": "operating-state",
    "n6-wide-levels-s1": "operating-state",
    "n6-wide-levels-s2": "operating-state",
    "bg3s-wide-levels-s0": "background-tasks",
    "n7msg-wide-levels-s0": "messenger",
    "npc6-wide-levels-s0": "game-character",
    "n6-notice-none-s0": "operating-state",  # text-notice baseline: no channel, the signal list is nominal only
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", default=list(RELEASE_RUNS), help="run folder names under runs/")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)
    unknown = [r for r in args.runs if r not in RELEASE_RUNS]
    if unknown:
        parser.error(f"not a release run: {unknown}")
    for run in args.runs:
        started = time.time()
        folder = args.runs_dir / run
        out = export_run(folder, release_signals(RELEASE_RUNS[run]), RELEASE_RUNS[run])
        size = out.stat().st_size / 2**20
        print(f"{run}: {out.name} {size:.1f} MiB, round trip identical ({time.time() - started:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
