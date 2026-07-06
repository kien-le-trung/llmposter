INSTRUCTION_CLUE_SYSTEM_PROMPT = (
    "You write short clues for an imposter word game. "
    "Be subtle, strategic, and return valid JSON only."
)

INSTRUCTION_BATCHED_CLUE_SYSTEM_PROMPT = (
    "You write short clues for several players in an imposter word game. "
    "Be subtle, strategic, and return valid JSON only."
)


def build_instruction_clue_system_prompt(
    secret_word: str | None,
    imposter_hint: str | None = None,
) -> str:
    return INSTRUCTION_CLUE_SYSTEM_PROMPT


def build_instruction_batched_clue_system_prompt() -> str:
    return INSTRUCTION_BATCHED_CLUE_SYSTEM_PROMPT


def build_instruction_clue_user_prompt(
    secret_word: str | None,
    imposter_hint: str | None,
    previous_clues: list[tuple[str, str]],
    strategy: dict[str, str] | None = None,
) -> str:
    previous_clue_block = _format_previous_clues(previous_clues)

    if secret_word is None:
        strategy_prompt = _format_strategy_prompt(strategy)
        return (
            "You are the imposter. You do not know the secret word.\n"
            "Infer the shared theme from the hint and previous clues, then blend in.\n"
            f"Hint: {imposter_hint or 'common everyday thing'}\n"
            f"{previous_clue_block}"
            f"{strategy_prompt}"
            "Constraints:\n"
            "- The clue must be 2 to 5 words.\n"
            "- Do not copy a previous clue.\n"
            "- Return JSON with exactly this shape: {\"clue\":\"your clue\"}\n"
        )

    strategy_prompt = _format_strategy_prompt(strategy)
    return (
        f"{strategy_prompt}"
        f"Word: {secret_word}\n"
        f"{previous_clue_block}"
        "Output requirements:\n"
        "- The phrase must be 2 to 5 words.\n"
        "- Do not use the word itself.\n"
        "- Return JSON with exactly this shape: {\"clue\":\"your phrase\"}\n"
    )


def build_instruction_batched_clue_user_prompt(
    secret_word: str,
    player_names: list[str],
    strategies_by_player_name: dict[str, dict[str, str]] | None = None,
    previous_clues: list[tuple[str, str]] | None = None,
) -> str:
    players = "\n".join(f"- {player_name}" for player_name in player_names)
    strategy_block = _format_strategy_assignments(
        player_names,
        strategies_by_player_name,
    )
    clue_placeholders = ",".join(
        f'"{player_name}":"your clue"' for player_name in player_names
    )

    return (
        "Each listed player knows the secret word.\n"
        f"Word: {secret_word}\n"
        f"{_format_previous_clues(previous_clues or [])}"
        "Players:\n"
        f"{players}\n"
        f"{strategy_block}"
        "Output requirements:\n"
        "- Each phrase must be 2 to 5 words.\n"
        "- Include every listed player exactly once.\n"
        "- Return JSON only.\n"
        "Required JSON shape:\n"
        f'{{"clues":{{{clue_placeholders}}}}}'
    )


def _format_strategy_assignments(
    player_names: list[str],
    strategies_by_player_name: dict[str, dict[str, str]] | None,
) -> str:
    if not strategies_by_player_name:
        return ""

    strategy_lines = []
    for player_name in player_names:
        strategy = strategies_by_player_name.get(player_name)
        if strategy is None:
            continue

        strategy_lines.append(
            f"{player_name} strategy - {strategy['name']}:\n{strategy['prompt']}"
        )

    if not strategy_lines:
        return ""

    return "Player-specific prompts:\n" + "\n\n".join(strategy_lines) + "\n"


def _format_strategy_prompt(strategy: dict[str, str] | None) -> str:
    if strategy is None:
        return "Write a clue for the secret word.\n"

    return f"{strategy['prompt']}\n"


def _format_previous_clues(previous_clues: list[tuple[str, str]]) -> str:
    if not previous_clues:
        return "Previous clues: none\n"

    clue_lines = [f"{player_name}: {clue}" for player_name, clue in previous_clues]
    return "Previous clues:\n" + "\n".join(clue_lines) + "\n"
