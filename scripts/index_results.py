"""Rewrite results/index.md + results/index.json from the files in runs/ and results/. CPU, seconds.
Run after every training or evaluation. Usage: uv run python scripts/index_results.py [project_root]
"""

import sys
from pathlib import Path

from prio.registry import write_index

if __name__ == "__main__":
    index = write_index(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
    print(f"indexed {len(index['runs'])} runs, {len(index['evaluations'])} evaluations -> results/index.md")
