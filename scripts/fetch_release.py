"""Download the released runs and data from Hugging Face into the folders the experiment commands read.

    uv run python scripts/fetch_release.py runs                       # every released run -> runs/<run name>/ (about 1.6 GB)
    uv run python scripts/fetch_release.py runs operating-state       # one run (a model-repo subfolder)
    uv run python scripts/fetch_release.py data                       # data/ and results/ (about 1.7 GB)
    uv run python scripts/fetch_release.py data data/n6 results/phase1  # only these folders

Model-repo subfolders and the run folders they fill (the names docs/REPRODUCE.md uses):
  operating-state -> runs/n6-wide-levels-s0, operating-state-seed1 / -seed2 -> runs/n6-wide-levels-s1 / -s2,
  background-jobs -> runs/bg3s-wide-levels-s0, messenger -> runs/n7msg-wide-levels-s0,
  game-character -> runs/npc6-wide-levels-s0, text-notice-baseline -> runs/n6-notice-none-s0.
Run folders that already exist are skipped. While the repositories are private, log in first
(`uv run hf auth login`).
"""

import argparse
import sys
from pathlib import Path

from prio.release import RELEASED_RUNS, fetch_data, fetch_runs

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("runs", "data"), help="runs: trained runs; data: datasets and evaluation outputs")
    parser.add_argument("items", nargs="*",
                        help=f"runs: model-repo subfolders ({', '.join(RELEASED_RUNS)}); data: folders such as data/n6")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this repository)")
    parser.add_argument("--revision", help="a branch, tag or commit of the Hugging Face repo")
    args = parser.parse_args(argv)
    if args.what == "runs":
        unknown = [s for s in args.items if s not in RELEASED_RUNS]
        if unknown:
            parser.error(f"not a released run: {unknown}; expected some of {list(RELEASED_RUNS)}")
        for run in fetch_runs(args.root / "runs", args.items or None, revision=args.revision):
            print(f"{run.relative_to(args.root).as_posix()}: ready")
    else:
        folders = args.items or ["data", "results"]
        fetch_data(args.root, folders, revision=args.revision)
        print(f"downloaded: {', '.join(folders)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
