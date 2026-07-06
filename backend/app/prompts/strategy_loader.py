from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.config import settings

STRATEGIES_PATH = Path(__file__).with_name("strategies.json")
PROMPT_TECHNIQUES = ("zero_shot", "few_shot", "reasoning_guided", "meta")
DEFAULT_PROMPT_TECHNIQUE = "few_shot"


def normalize_prompt_technique(technique: str | None) -> str:
    if technique in PROMPT_TECHNIQUES:
        return technique
    return DEFAULT_PROMPT_TECHNIQUE


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

    selected_technique = normalize_prompt_technique(technique or settings.clue_prompt_technique)
    return tuple(
        _parse_strategy(group_name, index, strategy, selected_technique)
        for index, strategy in enumerate(raw_group)
    )


def load_non_imposter_clue_strategies(
    technique: str | None = None,
) -> tuple[dict[str, str], ...]:
    return load_strategy_group("non_imposter", technique)


def load_imposter_clue_strategies(
    technique: str | None = None,
) -> tuple[dict[str, str], ...]:
    return load_strategy_group("imposter", technique)


class ConfiguredStrategyGroup(Sequence[dict[str, str]]):
    """Sequence facade that resolves strategies from the active prompt technique.

    Existing game code can keep using random.choice(CONSTANT), while each item is
    rendered from the currently configured CLUE_PROMPT_TECHNIQUE.
    """

    def __init__(self, group_name: str) -> None:
        self.group_name = group_name

    def __getitem__(self, index: int) -> dict[str, str]:
        return load_strategy_group(self.group_name)[index]

    def __len__(self) -> int:
        raw_groups = json.loads(STRATEGIES_PATH.read_text(encoding="utf-8"))
        raw_group = raw_groups.get(self.group_name)
        if not isinstance(raw_group, list):
            raise ValueError(f"strategies.json must contain a {self.group_name!r} list")
        return len(raw_group)


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

    return {"name": name.strip(), "prompt": prompt.strip()}


def _validate_prompt_variants(group_name: str, index: int, prompts: dict[str, Any]) -> None:
    for technique in PROMPT_TECHNIQUES:
        prompt = prompts.get(technique)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"{group_name} strategy {index} must contain a non-empty "
                f"{technique!r} prompt"
            )


NON_IMPOSTER_CLUE_STRATEGIES = ConfiguredStrategyGroup("non_imposter")
IMPOSTER_CLUE_STRATEGIES = ConfiguredStrategyGroup("imposter")
