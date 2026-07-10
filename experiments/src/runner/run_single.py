from __future__ import annotations

import asyncio
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runner.config import load_single_config  # noqa: E402
from runner.execution import run_single_experiment  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiments/src/runner/run_single.py <config.yaml>")

    config = load_single_config(sys.argv[1])
    artifacts = asyncio.run(run_single_experiment(config))
    print(f"Tracked run artifacts: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
