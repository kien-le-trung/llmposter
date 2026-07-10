from __future__ import annotations

import asyncio
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runner.config import load_sweep_config  # noqa: E402
from runner.execution import run_single_experiment  # noqa: E402
from runner.sweep import expand_sweep  # noqa: E402


async def run_sweep(relative_path: str) -> None:
    sweep_config = load_sweep_config(relative_path)
    variants = expand_sweep(sweep_config)
    for variant in variants:
        artifacts = await run_single_experiment(variant)
        print(f"Tracked sweep variant {variant.name}: {artifacts.output_dir}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiments/src/runner/run_sweep.py <config.yaml>")

    asyncio.run(run_sweep(sys.argv[1]))


if __name__ == "__main__":
    main()
