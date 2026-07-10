from __future__ import annotations

from collections import Counter
from typing import Any

from backend_handler.api_client import BackendExperimentResult


def evaluate_common(results: list[BackendExperimentResult]) -> dict[str, Any]:
    statuses = Counter(result.status for result in results)
    http_errors = [
        artifact
        for result in results
        for artifact in result.request_artifacts
        if artifact.status_code is not None and artifact.status_code >= 400
    ]
    return {
        "total_cases": len(results),
        "successful_cases": sum(1 for result in results if result.success),
        "failed_cases": sum(1 for result in results if not result.success),
        "success_rate": _rate([result.success for result in results]),
        "average_latency_ms": _average([result.latency_ms for result in results]),
        "p50_latency_ms": _percentile(
            [result.latency_ms for result in results],
            0.50,
        ),
        "p95_latency_ms": _percentile(
            [result.latency_ms for result in results],
            0.95,
        ),
        "timeout_rate": _rate([_is_timeout(result.error) for result in results]),
        "http_error_rate": len(http_errors) / len(results) if results else 0.0,
        "backend_generation_failed_rate": _rate(
            [result.status == "generation_failed" for result in results]
        ),
        "error_counts_by_type": _error_counts_by_type(results),
        "status_counts": dict(statuses),
    }


def extract_clue_responses(result: BackendExperimentResult) -> list[dict[str, Any]]:
    if result.round_payload is None:
        return []

    turns = result.round_payload.get("turns")
    if not isinstance(turns, list) or not turns:
        return []

    first_turn = turns[0]
    if not isinstance(first_turn, dict):
        return []

    responses = first_turn.get("responses")
    if not isinstance(responses, list):
        return []

    return [
        response
        for response in responses
        if isinstance(response, dict) and response.get("agent_id") != "human"
    ]


def extract_players(result: BackendExperimentResult) -> list[dict[str, Any]]:
    if result.round_payload is None:
        return []

    raw_players = result.round_payload.get("playing_order")
    if not isinstance(raw_players, list):
        return []

    return [player for player in raw_players if isinstance(player, dict)]


def resolve_imposter_player_id(result: BackendExperimentResult) -> str | None:
    if result.vote_payload is None:
        return None

    imposter_name = result.vote_payload.get("imposter_was")
    if not isinstance(imposter_name, str):
        return None

    for player in extract_players(result):
        if player.get("name") == imposter_name:
            player_id = player.get("id")
            return str(player_id) if player_id is not None else None

    return None


def resolve_player_id_by_name(
    result: BackendExperimentResult,
    player_name: Any,
) -> str | None:
    if not isinstance(player_name, str):
        return None

    for player in extract_players(result):
        if player.get("name") == player_name:
            player_id = player.get("id")
            return str(player_id) if player_id is not None else None

    return None


def player_count(result: BackendExperimentResult) -> int:
    return len(extract_players(result))


def _average(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: list[float | int], percentile: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(float(value) for value in values)
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for flag in flags if flag) / len(flags))


def _is_timeout(error: str | None) -> bool:
    return error is not None and "timeout" in error.lower()


def _error_counts_by_type(results: list[BackendExperimentResult]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        if result.success:
            continue
        counts[_classify_error(result)] += 1
    return dict(counts)


def _classify_error(result: BackendExperimentResult) -> str:
    for artifact in result.request_artifacts:
        if artifact.status_code is not None and artifact.status_code >= 400:
            return f"http_{artifact.status_code}"

    if result.status == "generation_failed":
        return "generation_failed"
    if _is_timeout(result.error):
        return "timeout"
    if result.error:
        return result.error.split(":", 1)[0].strip() or "unknown_error"
    return "unknown_error"
