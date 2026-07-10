from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from backend_handler.api_client import BackendExperimentResult

from .common import _average, _rate, extract_clue_responses, extract_players


def evaluate_prompt(results: list[BackendExperimentResult]) -> dict[str, Any]:
    clues_by_round = _clues_by_round(results)
    clues = [clue for round_clues in clues_by_round.values() for clue in round_clues]
    return {
        "generation_failed_rate": _rate(
            [result.status == "generation_failed" for result in results]
        ),
        "duplicate_clue_rate": _duplicate_clue_rate(clues_by_round),
        "secret_word_leak_rate": _secret_word_leak_rate(results),
        "empty_clue_rate": _rate([not clue.strip() for clue in clues]),
        "average_clue_word_count": _average(
            [len(clue.split()) for clue in clues if clue.strip()]
        ),
        "average_clues_per_round": _average(
            [len(round_clues) for round_clues in clues_by_round.values()]
        ),
        "missing_clue_round_rate": _missing_clue_round_rate(results),
    }


def _clues_by_round(results: list[BackendExperimentResult]) -> dict[str, list[str]]:
    clues_by_round: dict[str, list[str]] = defaultdict(list)
    for index, result in enumerate(results):
        round_id = result.round_id or f"missing-round-{index}"
        for response in extract_clue_responses(result):
            clue = response.get("agent_response", "")
            clues_by_round[round_id].append(str(clue))
    return dict(clues_by_round)


def _duplicate_clue_rate(clues_by_round: dict[str, list[str]]) -> float:
    duplicate_flags: list[bool] = []
    for clues in clues_by_round.values():
        counts = Counter(_normalize_text(clue) for clue in clues if _normalize_text(clue))
        duplicate_flags.extend(count > 1 for count in counts.values())
    return _rate(duplicate_flags)


def _secret_word_leak_rate(results: list[BackendExperimentResult]) -> float:
    flags: list[bool] = []
    for result in results:
        secret_word = _normalize_text(result.request.secret_word)
        for response in extract_clue_responses(result):
            clue = _normalize_text(str(response.get("agent_response", "")))
            flags.append(bool(secret_word and secret_word in clue))
    return _rate(flags)


def _missing_clue_round_rate(results: list[BackendExperimentResult]) -> float:
    flags: list[bool] = []
    for result in results:
        if result.round_payload is None:
            continue

        expected_agent_count = sum(
            1
            for player in extract_players(result)
            if player.get("id") != "human"
        )
        if expected_agent_count == 0:
            continue

        flags.append(len(extract_clue_responses(result)) < expected_agent_count)
    return _rate(flags)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())
