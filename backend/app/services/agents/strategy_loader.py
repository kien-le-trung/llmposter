from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

STRATEGIES_PATH = Path(__file__).resolve().parents[2] / "prompts" / "strategies.json"
PROMPT_TECHNIQUES = ("zero_shot", "few_shot", "reasoning_guided", "meta")
DEFAULT_PROMPT_TECHNIQUE = "few_shot"

IMPOSTER_STRATEGY_WEIGHTS_BY_PREVIOUS_CLUE_COUNT: dict[int, dict[str, int]] = {
    0: {
        "Abstraction": 50,
        "Adjacent association": 50,
    },
    1: {
        "Abstraction": 40,
        "Adjacent association": 40,
        "Ride previous clues": 10,
        "Contextual guess": 10,
    },
    2: {
        "Abstraction": 25,
        "Adjacent association": 25,
        "Ride previous clues": 20,
        "Contextual guess": 20,
        "Cluster matching": 10,
    },
    3: {
        "Abstraction": 10,
        "Adjacent association": 10,
        "Ride previous clues": 30,
        "Contextual guess": 30,
        "Cluster matching": 20,
    },
    4: {
        "Abstraction": 5,
        "Adjacent association": 5,
        "Ride previous clues": 25,
        "Contextual guess": 25,
        "Cluster matching": 40,
    },
}


def normalize_prompt_technique(technique: str | None) -> str:
    if technique is None:
        return DEFAULT_PROMPT_TECHNIQUE

    normalized = technique.lower()
    if normalized not in PROMPT_TECHNIQUES:
        return DEFAULT_PROMPT_TECHNIQUE

    return normalized


def load_strategy_group(
    group_name: str,
    technique: str | None = None,
) -> tuple[dict[str, str], ...]:
    raw_groups = json.loads(STRATEGIES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_groups, dict):
        raise ValueError("strategies.json must contain an object")

    raw_group = raw_groups.get(group_name)
    if not isinstance(raw_group, list):
        raise ValueError(f"strategies.json must contain a {group_name!r} list")

    technique = normalize_prompt_technique(technique)

    return tuple(
        _parse_strategy(group_name, index, strategy, technique)
        for index, strategy in enumerate(raw_group)
    )


def load_non_imposter_clue_strategies(
    technique: str | None = None,
) -> tuple[dict[str, str], ...]:
    return _without_technique(load_strategy_group("non_imposter", technique))


def load_imposter_clue_strategies(
    technique: str | None = None,
) -> tuple[dict[str, str], ...]:
    return _without_technique(load_strategy_group("imposter", technique))


def assign_non_imposter_clue_strategy(technique: str | None = None) -> dict[str, str]:
    return random.choice(load_strategy_group("non_imposter", technique))


def assign_imposter_clue_strategy(
    technique: str | None = None,
    previous_clue_count: int = 0,
) -> dict[str, str]:
    strategies = load_strategy_group("imposter", technique)
    strategy_weights = _imposter_strategy_weights_for_previous_clues(previous_clue_count)
    return _weighted_strategy_choice(strategies, strategy_weights)


def _imposter_strategy_weights_for_previous_clues(previous_clue_count: int) -> dict[str, int]:
    return IMPOSTER_STRATEGY_WEIGHTS_BY_PREVIOUS_CLUE_COUNT[
        max(0, min(previous_clue_count, 4))
    ]


def _without_technique(strategies: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
    return tuple(
        {"name": strategy["name"], "prompt": strategy["prompt"]}
        for strategy in strategies
    )


def _weighted_strategy_choice(
    strategies: tuple[dict[str, str], ...],
    strategy_weights: dict[str, int],
) -> dict[str, str]:
    weighted_strategies = [
        (strategy, strategy_weights[strategy["name"]])
        for strategy in strategies
        if strategy["name"] in strategy_weights and strategy_weights[strategy["name"]] > 0
    ]
    if not weighted_strategies:
        return random.choice(strategies)

    return random.choices(
        [strategy for strategy, _weight in weighted_strategies],
        weights=[weight for _strategy, weight in weighted_strategies],
        k=1,
    )[0]


def _parse_strategy(
    group_name: str,
    index: int,
    strategy: Any,
    technique: str,
) -> dict[str, str]:
    if not isinstance(strategy, dict):
        raise ValueError(f"{group_name} strategy {index} must be an object")

    name = strategy.get("name")
    prompts = strategy.get("prompts")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{group_name} strategy {index} must contain a name")
    if not isinstance(prompts, dict):
        raise ValueError(f"{group_name} strategy {index} must contain a prompts object")

    _validate_prompt_variants(group_name, index, prompts)
    prompt = prompts.get(technique) or prompts[DEFAULT_PROMPT_TECHNIQUE]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"{group_name} strategy {index} must contain a non-empty {technique!r} prompt"
        )

    return {"name": name.strip(), "prompt": prompt.strip(), "technique": technique}


def _validate_prompt_variants(group_name: str, index: int, prompts: dict[str, Any]) -> None:
    for technique in PROMPT_TECHNIQUES:
        prompt = prompts.get(technique)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"{group_name} strategy {index} must contain a non-empty "
                f"{technique!r} prompt"
            )
