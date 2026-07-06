import re

VOTE_SYSTEM_PROMPT = (
    "You choose one allowed answer for a word game vote. "
    "Output valid JSON only."
)


def build_vote_system_prompt(voter_name: str, candidate_names: list[str]) -> str:
    return VOTE_SYSTEM_PROMPT


def build_vote_user_prompt(
    candidate_names: list[str],
    clue_lines: list[tuple[str, str]],
) -> str:
    formatted_clues = "\n".join(
        f"{player_name} = \"{clue}\"" for player_name, clue in clue_lines
    )
    allowed_answers = "\n".join(candidate_names)

    return (
        "Task: choose the clue that seems least related to the others.\n\n"
        "Candidates:\n"
        f"{formatted_clues}\n\n"
        "Allowed answers:\n"
        f"{allowed_answers}\n\n"
        'JSON: {"vote":"<one allowed answer>"}'
    )


def clean_vote_response(response_text: str, candidate_names: list[str]) -> str:
    vote = _first_output_line(response_text)
    vote = re.sub(r"^(vote|answer|output)\s*:\s*", "", vote, flags=re.IGNORECASE)
    vote = vote.strip().strip("\"'` .,!?:;")

    exact_vote = _match_candidate(vote, candidate_names)
    if exact_vote is not None:
        return exact_vote

    return candidate_names[0] if candidate_names else vote


def _first_output_line(response_text: str) -> str:
    for line in response_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _match_candidate(vote: str, candidate_names: list[str]) -> str | None:
    normalized_vote = _normalize(vote)
    if any(alias in normalized_vote for alias in ("human", "player")) and "You" in candidate_names:
        return "You"

    for candidate_name in candidate_names:
        if normalized_vote == _normalize(candidate_name):
            return candidate_name

    for candidate_name in candidate_names:
        candidate_parts = [_normalize(part) for part in candidate_name.split()]
        if normalized_vote in candidate_parts:
            return candidate_name

    for candidate_name in candidate_names:
        if _normalize(candidate_name) in normalized_vote:
            return candidate_name

    return None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
