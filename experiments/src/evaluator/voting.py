from __future__ import annotations

from collections import Counter
from typing import Any

from backend_handler.api_client import BackendExperimentResult

from .common import (
    _average,
    _rate,
    extract_players,
    player_count,
    resolve_imposter_player_id,
    resolve_player_id_by_name,
)


def evaluate_voting(results: list[BackendExperimentResult]) -> dict[str, Any]:
    group_flags = _group_detection_flags(results)
    agent_flags = _agent_vote_detection_flags(results)
    agent_only_group_flags, tie_flags = _agent_only_group_detection_and_ties(results)
    random_chance = _average(
        [1.0 / count for result in results if (count := player_count(result))]
    )
    agent_only_detection = _rate(agent_only_group_flags)
    return {
        "vote_submission_success_rate": _rate(
            [result.vote_payload is not None for result in results]
        ),
        "group_detection_rate": _rate(group_flags),
        "agent_vote_detection_rate": _rate(agent_flags),
        "agent_only_group_detection_rate": agent_only_detection,
        "random_chance_detection_rate": random_chance,
        "detection_lift_over_random": agent_only_detection - random_chance,
        "imposter_win_rate": _rate(_imposter_win_flags(results)),
        "detection_rate_by_imposter_position": _detection_rate_by_imposter_position(
            results
        ),
        "vote_tie_rate": _rate(tie_flags),
    }


def _group_detection_flags(results: list[BackendExperimentResult]) -> list[bool]:
    flags: list[bool] = []
    for result in results:
        if result.vote_payload is None:
            continue

        imposter_id = resolve_imposter_player_id(result)
        voted_id = _resolve_group_voted_player_id(result)
        if imposter_id is not None and voted_id is not None:
            flags.append(voted_id == imposter_id)
    return flags


def _agent_vote_detection_flags(results: list[BackendExperimentResult]) -> list[bool]:
    flags: list[bool] = []
    for result in results:
        imposter_id = resolve_imposter_player_id(result)
        if result.vote_payload is None or imposter_id is None:
            continue

        for vote in _agent_votes(result):
            voted_id = resolve_player_id_by_name(result, vote.get("voted_for"))
            if voted_id is not None:
                flags.append(voted_id == imposter_id)
    return flags


def _agent_only_group_detection_and_ties(
    results: list[BackendExperimentResult],
) -> tuple[list[bool], list[bool]]:
    detection_flags: list[bool] = []
    tie_flags: list[bool] = []

    for result in results:
        imposter_id = resolve_imposter_player_id(result)
        if result.vote_payload is None or imposter_id is None:
            continue

        voted_ids = [
            voted_id
            for vote in _agent_votes(result)
            if (voted_id := resolve_player_id_by_name(result, vote.get("voted_for")))
        ]
        if not voted_ids:
            continue

        counts = Counter(voted_ids)
        highest_total = max(counts.values())
        leaders = [
            player_id
            for player_id, vote_total in counts.items()
            if vote_total == highest_total
        ]
        tie_flags.append(len(leaders) > 1)
        if len(leaders) == 1:
            detection_flags.append(leaders[0] == imposter_id)

    return detection_flags, tie_flags


def _imposter_win_flags(results: list[BackendExperimentResult]) -> list[bool]:
    flags: list[bool] = []
    for result in results:
        if result.vote_payload is not None and "imposter_won" in result.vote_payload:
            flags.append(bool(result.vote_payload["imposter_won"]))
    return flags


def _detection_rate_by_imposter_position(
    results: list[BackendExperimentResult],
) -> dict[str, float]:
    flags_by_position: dict[str, list[bool]] = {}
    for result in results:
        imposter_id = resolve_imposter_player_id(result)
        voted_id = _resolve_group_voted_player_id(result)
        if imposter_id is None or voted_id is None:
            continue

        position = _player_position(result, imposter_id)
        if position is None:
            continue

        flags_by_position.setdefault(str(position), []).append(voted_id == imposter_id)

    return {
        position: _rate(flags)
        for position, flags in sorted(
            flags_by_position.items(),
            key=lambda item: int(item[0]),
        )
    }


def _resolve_group_voted_player_id(result: BackendExperimentResult) -> str | None:
    if result.vote_payload is None:
        return None

    raw_player_id = result.vote_payload.get("group_voted_player_id")
    if isinstance(raw_player_id, str) and _player_position(result, raw_player_id) is not None:
        return raw_player_id

    return resolve_player_id_by_name(
        result,
        result.vote_payload.get("group_voted_player_name"),
    )


def _agent_votes(result: BackendExperimentResult) -> list[dict[str, Any]]:
    if result.vote_payload is None:
        return []

    raw_votes = result.vote_payload.get("agent_votes")
    if not isinstance(raw_votes, list):
        return []

    return [vote for vote in raw_votes if isinstance(vote, dict)]


def _player_position(
    result: BackendExperimentResult,
    player_id: str,
) -> int | None:
    for position, player in enumerate(extract_players(result)):
        if player.get("id") == player_id:
            return position
    return None
