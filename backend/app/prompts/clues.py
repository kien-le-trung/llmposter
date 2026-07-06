import re

CLUE_SYSTEM_PROMPT = (
    "You complete short clue examples for a word game. "
    "Output only the clue text."
)
BATCHED_CLUE_SYSTEM_PROMPT = (
    "You complete short clue examples for several word game players. "
    "Output valid JSON only."
)

CLUE_EXAMPLES = """Examples:
Secret word: apple
Clue: red orchard fruit

Secret word: bridge
Clue: crosses a river

Secret word: guitar
Clue: strings on stage"""

IMPOSTER_CLUE_EXAMPLES = """Examples:
Hint: Fruit, orchard, or red
Clue: sweet red snack

Hint: Crossing, river, or structure
Clue: over the water

Hint: Music, strings, or stage
Clue: concert sound"""


def build_clue_system_prompt(
    secret_word: str | None,
    imposter_hint: str | None = None,
) -> str:
    return CLUE_SYSTEM_PROMPT


def build_batched_clue_system_prompt() -> str:
    return BATCHED_CLUE_SYSTEM_PROMPT


def build_clue_user_prompt(
    secret_word: str | None,
    imposter_hint: str | None,
    previous_clues: list[tuple[str, str]],
) -> str:
    previous_clue_block = _format_previous_clues(previous_clues)

    if secret_word is None:
        hint = imposter_hint or "common everyday thing"
        return (
            "Task: write one short clue that sounds related to the hint.\n"
            "Rules:\n"
            "- 2 to 5 words\n"
            "- return JSON with one key: clue\n\n"
            f"{IMPOSTER_CLUE_EXAMPLES}\n\n"
            f"Hint: {hint}\n"
            f"{previous_clue_block}"
            "JSON:"
        )

    return (
        "Task: write one short clue for the secret word.\n"
        "Rules:\n"
        "- 2 to 5 words\n"
        "- do not say the secret word\n"
        "- return JSON with one key: clue\n\n"
        f"{CLUE_EXAMPLES}\n\n"
        f"Secret word: {secret_word}\n"
        f"{previous_clue_block}"
        "JSON:"
    )


def build_batched_clue_user_prompt(
    secret_word: str,
    player_names: list[str],
) -> str:
    players = "\n".join(f"- {player_name}" for player_name in player_names)
    clue_placeholders = ",".join(
        f'"{player_name}":"2 to 5 word clue"' for player_name in player_names
    )

    return (
        "Task: write one short clue for each player.\n"
        "Rules:\n"
        "- each clue is 2 to 5 words\n"
        "- do not say the secret word\n"
        "- return JSON with one top-level key: clues\n"
        "- clues must contain every player name shown\n\n"
        f"Secret word: {secret_word}\n"
        "Players:\n"
        f"{players}\n"
        "Return this exact JSON shape, replacing only clue text:\n"
        f'{{"clues":{{{clue_placeholders}}}}}'
    )


def clean_clue_response(
    response_text: str,
    secret_word: str | None = None,
    fallback_hint: str | None = None,
) -> str:
    clue = _first_output_line(response_text)
    clue = re.sub(r"^(clue|answer|output)\s*:\s*", "", clue, flags=re.IGNORECASE)
    clue = clue.strip().strip("\"'` .,!?:;")

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


def _format_previous_clues(previous_clues: list[tuple[str, str]]) -> str:
    if not previous_clues:
        return "Previous clues: none\n\n"

    clue_lines = [f"{player_name}: {clue}" for player_name, clue in previous_clues]
    return "Previous clues:\n" + "\n".join(clue_lines) + "\n\n"


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
