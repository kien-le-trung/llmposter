from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STRATEGIES_PATH = Path(__file__).with_name("strategies.json")


def load_strategy_group(group_name: str) -> tuple[dict[str, str], ...]:
    raw_groups = json.loads(STRATEGIES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_groups, dict):
        raise ValueError("strategies.json must contain an object")

    raw_group = raw_groups.get(group_name)
    if not isinstance(raw_group, list):
        raise ValueError(f"strategies.json must contain a {group_name!r} list")

    return tuple(_parse_strategy(group_name, index, strategy) for index, strategy in enumerate(raw_group))


def _parse_strategy(group_name: str, index: int, strategy: Any) -> dict[str, str]:
    if not isinstance(strategy, dict):
        raise ValueError(f"{group_name} strategy {index} must be an object")

    name = strategy.get("name")
    prompt = strategy.get("prompt")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{group_name} strategy {index} must contain a name")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{group_name} strategy {index} must contain a prompt")

    return {"name": name.strip(), "prompt": prompt.strip()}


NON_IMPOSTER_CLUE_STRATEGIES = load_strategy_group("non_imposter")
IMPOSTER_CLUE_STRATEGIES = load_strategy_group("imposter")
