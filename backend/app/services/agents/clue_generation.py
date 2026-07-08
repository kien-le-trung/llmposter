import re

from app.services.agents.strategy_loader import assign_imposter_clue_strategy


def build_instruction_clue_user_prompt(
    secret_word: str | None,
    imposter_hint: str | None,
    previous_clues: list[tuple[str, str]],
    strategy: dict[str, str] | None = None,
) -> str:
    if not previous_clues:
        previous_clue_block = "Previous clues: none\n"
    else:
        previous_clue_block = "Previous clues:\n" + "\n".join(
            f"{player_name}: {clue}" for player_name, clue in previous_clues
        ) + "\n"

    if secret_word is None:
        if strategy is None:
            strategy = assign_imposter_clue_strategy(
                None,
                previous_clue_count=len(previous_clues),
            )
        strategy_prompt = strategy["prompt"]
        return (
            "You are the imposter. You do not know the secret word.\n"
            f"Hint: {imposter_hint or 'common everyday thing'}\n"
            f"{previous_clue_block}"
            f"{strategy_prompt}"
            "Constraints:\n"
            "- Return JSON with exactly this shape: {\"clue\":\"your clue\"}\n"
        )

    strategy_prompt = strategy["prompt"]
    return (
        f"{strategy_prompt}"
        f"Word: {secret_word}\n"
        f"{previous_clue_block}"
        "Output requirements:\n"
        "- Do not use the word itself.\n"
        "- Return JSON with exactly this shape: {\"clue\":\"your phrase\"}\n"
    )


def build_instruction_batched_clue_user_prompt(
    secret_word: str,
    player_names: list[str],
    strategies_by_player_name: dict[str, dict[str, str]] | None = None,
    previous_clues: list[tuple[str, str]] | None = None,
) -> str:
    if not previous_clues:
        previous_clue_block = "Previous clues: none\n"
    else:
        previous_clue_block = "Previous clues:\n" + "\n".join(
            f"{player_name}: {clue}" for player_name, clue in previous_clues
        ) + "\n"

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
        f"{previous_clue_block}"
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


def clean_clue_response(
    response_text: str,
    secret_word: str | None = None,
    fallback_hint: str | None = None,
) -> str:
    clue = _first_output_line(response_text)
    clue = re.sub(r"^(clue|answer|output)\s*:\s*", "", clue, flags=re.IGNORECASE)
    clue = clue.strip().strip("\"'` .,!?;:")

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", clue)
    if len(words) > 5:
        clue = " ".join(words[:5])
    elif words:
        clue = " ".join(words)
    else:
        clue = ""

    if clue.lower() in {"2 to 5 word clue", "short clue", "your clue"}:
        clue = _fallback_clue(secret_word, fallback_hint)

    if secret_word and _contains_word(clue, secret_word):
        clue = _fallback_clue(secret_word, fallback_hint)

    if not clue:
        clue = _fallback_clue(secret_word, fallback_hint)

    return clue


def clean_batched_clue_response(
    response: dict[str, str] | str,
    player_names: list[str],
    secret_word: str,
    fallback_hint: str | None = None,
) -> dict[str, str]:
    parsed_clues = (
        response if isinstance(response, dict) else _parse_batched_clues(response, player_names)
    )
    return {
        player_name: clean_clue_response(
            parsed_clues.get(player_name, ""),
            secret_word=secret_word,
            fallback_hint=fallback_hint,
        )
        for player_name in player_names
    }


def _parse_batched_clues(response_text: str, player_names: list[str]) -> dict[str, str]:
    clues: dict[str, str] = {}
    normalized_names = {_normalize(player_name): player_name for player_name in player_names}

    for line in response_text.splitlines():
        if ":" not in line:
            continue

        raw_name, raw_clue = line.split(":", 1)
        player_name = normalized_names.get(_normalize(raw_name))
        if player_name is not None and player_name not in clues:
            clues[player_name] = raw_clue.strip()

    return clues


def _first_output_line(response_text: str) -> str:
    for line in response_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _fallback_clue(secret_word: str | None, fallback_hint: str | None) -> str:
    source = fallback_hint or "related idea"
    forbidden_words = set(re.findall(r"[A-Za-z0-9]+", secret_word or "", flags=re.IGNORECASE))
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", source)
        if word.lower() not in {"or", "and", "the", "a", "an"}
        and word.lower() not in {forbidden.lower() for forbidden in forbidden_words}
    ]

    if len(words) >= 2:
        return " ".join(words[:5])

    if words:
        return f"{words[0]} clue"

    return "related clue"
