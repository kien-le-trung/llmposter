from app.services.agents.strategy_loader import (
    DEFAULT_PROMPT_TECHNIQUE,
    PROMPT_TECHNIQUES,
    load_imposter_clue_strategies,
    load_non_imposter_clue_strategies,
    normalize_prompt_technique,
)


def test_default_prompt_technique_is_few_shot() -> None:
    assert DEFAULT_PROMPT_TECHNIQUE == "few_shot"
    assert normalize_prompt_technique(None) == "few_shot"
    assert normalize_prompt_technique("unknown") == "few_shot"


def test_all_non_imposter_prompt_techniques_load() -> None:
    for technique in PROMPT_TECHNIQUES:
        strategies = load_non_imposter_clue_strategies(technique)
        assert strategies
        assert all(strategy["name"] and strategy["prompt"] for strategy in strategies)


def test_all_imposter_prompt_techniques_load() -> None:
    for technique in PROMPT_TECHNIQUES:
        strategies = load_imposter_clue_strategies(technique)
        assert strategies
        assert all(strategy["name"] and strategy["prompt"] for strategy in strategies)


def test_prompt_changes_across_techniques() -> None:
    zero_shot = load_non_imposter_clue_strategies("zero_shot")[0]["prompt"]
    few_shot = load_non_imposter_clue_strategies("few_shot")[0]["prompt"]
    reasoning = load_non_imposter_clue_strategies("reasoning_guided")[0]["prompt"]
    meta = load_non_imposter_clue_strategies("meta")[0]["prompt"]

    assert len({zero_shot, few_shot, reasoning, meta}) == 4


def test_strategy_loaders_keep_sequence_shape() -> None:
    non_imposter_strategies = load_non_imposter_clue_strategies()
    imposter_strategies = load_imposter_clue_strategies()

    assert len(non_imposter_strategies) > 0
    assert len(imposter_strategies) > 0
    assert set(non_imposter_strategies[0]) == {"name", "prompt"}
    assert set(imposter_strategies[0]) == {"name", "prompt"}
