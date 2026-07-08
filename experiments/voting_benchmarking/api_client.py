from __future__ import annotations

import time
import asyncio
from typing import Any

import httpx

from schemas import (
    AgentVoteRecord,
    BenchmarkCase,
    ClueRecord,
    PlayerRecord,
    RoundArtifactRecord,
    VoteCountRecord,
    VoteRecord,
)

TERMINAL_STATUSES = {"ready_to_vote", "complete", "generation_failed"}
HUMAN_PLAYER_ID = "human"
HUMAN_PLAYER_NAME = "You"
VOTING_ALGORITHM = "embedding_distance_v1"
HUMAN_VOTE_STRATEGY = "first_agent_placeholder"


class VotingBenchmarkApiClient:
    def __init__(self, backend_url: str, timeout_seconds: float = 60.0) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def run_case(
        self,
        case: BenchmarkCase,
        technique: str,
        run_id: str,
    ) -> tuple[RoundArtifactRecord, list[ClueRecord], VoteRecord | None]:
        started = time.perf_counter()
        round_id: str | None = None
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                create_response = await client.post(
                    f"{self.backend_url}/rounds",
                    json={
                        "secret_word": case.secret_word,
                        "imposter_hint": case.imposter_hint,
                        "prompt_technique": technique,
                    },
                )
                create_response.raise_for_status()
                round_payload = create_response.json()
                round_id = str(round_payload["id"])

                round_payload = await self._complete_round_if_needed(
                    client,
                    round_id,
                    case.human_clue,
                )

                vote_payload = await self._submit_placeholder_vote(client, round_payload)

            latency_ms = (time.perf_counter() - started) * 1000.0
            players = _extract_playing_order(round_payload)
            imposter_player_id = _resolve_imposter_player_id(vote_payload, players)
            round_record = RoundArtifactRecord(
                run_id=run_id,
                technique=technique,
                round_id=round_id,
                secret_word=case.secret_word,
                imposter_hint=case.imposter_hint,
                human_clue=case.human_clue,
                status=str(round_payload.get("status", "unknown")),
                success=True,
                latency_ms=latency_ms,
                playing_order=players,
                imposter_player_id=imposter_player_id,
                imposter_player_name=str(vote_payload.get("imposter_was", "")),
                imposter_kind=_player_kind(imposter_player_id, players),
                num_players=len(players),
            )
            clue_records = _extract_clues(round_payload, round_record)
            vote_record = _extract_vote_record(
                vote_payload,
                round_record,
                _first_agent_player_id(players),
            )
            return round_record, clue_records, vote_record
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return (
                RoundArtifactRecord(
                    run_id=run_id,
                    technique=technique,
                    round_id=round_id,
                    secret_word=case.secret_word,
                    imposter_hint=case.imposter_hint,
                    human_clue=case.human_clue,
                    status="error",
                    success=False,
                    latency_ms=latency_ms,
                    playing_order=[],
                    error=str(exc),
                ),
                [],
                None,
            )

    async def _complete_round_if_needed(
        self,
        client: httpx.AsyncClient,
        round_id: str,
        human_clue: str,
    ) -> dict[str, Any]:
        deadline = time.perf_counter() + self.timeout_seconds
        while time.perf_counter() < deadline:
            round_response = await client.get(f"{self.backend_url}/rounds/{round_id}")
            round_response.raise_for_status()
            round_payload = round_response.json()
            status = round_payload.get("status")

            if status == "awaiting_human_clue":
                clue_response = await client.post(
                    f"{self.backend_url}/rounds/{round_id}/clue",
                    json={"clue": human_clue},
                )
                clue_response.raise_for_status()
                round_payload = clue_response.json()
                status = round_payload.get("status")

            if status in TERMINAL_STATUSES:
                return round_payload

            await asyncio.sleep(0.25)

        raise TimeoutError(f"Round {round_id} did not finish within {self.timeout_seconds}s")

    async def _submit_placeholder_vote(
        self,
        client: httpx.AsyncClient,
        round_payload: dict[str, Any],
    ) -> dict[str, Any]:
        first_agent_id = _first_agent_player_id(_extract_playing_order(round_payload))
        if first_agent_id is None:
            raise ValueError("Round has no agent player to use for placeholder human vote")

        round_id = str(round_payload["id"])
        vote_response = await client.post(
            f"{self.backend_url}/rounds/{round_id}/vote",
            json={"agent_id": first_agent_id},
        )
        vote_response.raise_for_status()
        return vote_response.json()


def _extract_playing_order(round_payload: dict[str, Any]) -> list[PlayerRecord]:
    raw_players = round_payload.get("playing_order")
    if not isinstance(raw_players, list):
        return []

    players: list[PlayerRecord] = []
    for position, raw_player in enumerate(raw_players):
        if not isinstance(raw_player, dict):
            continue
        players.append(
            PlayerRecord(
                player_id=str(raw_player.get("id", "")),
                player_name=str(raw_player.get("name", "")),
                player_kind=str(raw_player.get("kind", "")),
                position=position,
            )
        )
    return players


def _extract_clues(
    round_payload: dict[str, Any],
    round_record: RoundArtifactRecord,
) -> list[ClueRecord]:
    records: list[ClueRecord] = []
    turns = round_payload.get("turns")
    if not isinstance(turns, list):
        return records

    players_by_id = {player.player_id: player for player in round_record.playing_order}
    human_position = next(
        (
            player.position
            for player in round_record.playing_order
            if player.player_id == HUMAN_PLAYER_ID
        ),
        None,
    )
    responses = _opening_turn_responses(turns)
    last_position = len(round_record.playing_order) - 1
    for response in responses:
        player_id = str(response.get("agent_id", ""))
        player = players_by_id.get(player_id)
        if player is None:
            continue
        role = None
        if round_record.imposter_player_id is not None:
            role = "imposter" if player_id == round_record.imposter_player_id else "non_imposter"
        records.append(
            ClueRecord(
                run_id=round_record.run_id,
                technique=round_record.technique,
                round_id=str(round_payload.get("id", "")),
                secret_word=round_record.secret_word,
                imposter_hint=round_record.imposter_hint,
                player_id=player_id,
                player_name=str(response.get("agent_name", player.player_name)),
                player_kind=player.player_kind,
                role=role,
                position=player.position,
                clue=str(response.get("agent_response", "")),
                inference_mode=str(response.get("inference_mode", "")),
                previous_clue_count=player.position,
                is_first=player.position == 0,
                is_last=player.position == last_position,
                is_before_human=(
                    human_position is not None and player.position < human_position
                ),
                is_after_human=(
                    human_position is not None and player.position > human_position
                ),
            )
        )
    return records


def _opening_turn_responses(turns: list[Any]) -> list[dict[str, Any]]:
    if not turns or not isinstance(turns[0], dict):
        return []
    responses = turns[0].get("responses")
    if not isinstance(responses, list):
        return []
    return [response for response in responses if isinstance(response, dict)]


def _extract_vote_record(
    vote_payload: dict[str, Any],
    round_record: RoundArtifactRecord,
    human_voted_player_id: str | None,
) -> VoteRecord:
    players_by_name = {
        player.player_name: player.player_id for player in round_record.playing_order
    }
    players_by_id = {player.player_id: player for player in round_record.playing_order}
    group_voted_player_id = _resolve_player_id(
        vote_payload.get("group_voted_player_id"),
        vote_payload.get("group_voted_player_name"),
        players_by_name,
        players_by_id,
    )
    agent_votes = [
        _extract_agent_vote(raw_vote, round_record, players_by_name, players_by_id)
        for raw_vote in vote_payload.get("agent_votes", [])
        if isinstance(raw_vote, dict)
    ]
    vote_counts = [
        _extract_vote_count(raw_count, round_record)
        for raw_count in vote_payload.get("vote_counts", [])
        if isinstance(raw_count, dict)
    ]
    return VoteRecord(
        run_id=round_record.run_id,
        technique=round_record.technique,
        round_id=round_record.round_id or "",
        voting_algorithm=VOTING_ALGORITHM,
        human_vote_strategy=HUMAN_VOTE_STRATEGY,
        human_voted_player_id=human_voted_player_id,
        group_voted_player_id=group_voted_player_id,
        group_voted_player_name=vote_payload.get("group_voted_player_name"),
        group_voted_is_imposter=_is_imposter(group_voted_player_id, round_record),
        imposter_won=bool(vote_payload.get("imposter_won", False)),
        round_winner=str(vote_payload.get("round_winner", "")),
        agent_votes=agent_votes,
        vote_counts=vote_counts,
    )


def _extract_agent_vote(
    raw_vote: dict[str, Any],
    round_record: RoundArtifactRecord,
    players_by_name: dict[str, str],
    players_by_id: dict[str, PlayerRecord],
) -> AgentVoteRecord:
    voted_for_name = str(raw_vote.get("voted_for", ""))
    voted_for_player_id = _resolve_player_id(
        None,
        voted_for_name,
        players_by_name,
        players_by_id,
    )
    return AgentVoteRecord(
        voter_agent_id=str(raw_vote.get("voter_agent_id", "")),
        voter_agent_name=str(raw_vote.get("voter_agent_name", "")),
        voted_for_player_id=voted_for_player_id,
        voted_for_player_name=voted_for_name,
        voted_for_is_imposter=_is_imposter(voted_for_player_id, round_record),
        inference_mode=str(raw_vote.get("inference_mode", "")),
    )


def _extract_vote_count(
    raw_count: dict[str, Any],
    round_record: RoundArtifactRecord,
) -> VoteCountRecord:
    player_id = str(raw_count.get("player_id", ""))
    return VoteCountRecord(
        player_id=player_id,
        player_name=str(raw_count.get("player_name", "")),
        votes=int(raw_count.get("votes", 0)),
        is_imposter=_is_imposter(player_id, round_record),
    )


def _resolve_imposter_player_id(
    vote_payload: dict[str, Any],
    players: list[PlayerRecord],
) -> str | None:
    imposter_name = str(vote_payload.get("imposter_was", ""))
    for player in players:
        if player.player_name == imposter_name:
            return player.player_id
    return None


def _resolve_player_id(
    raw_player_id: Any,
    raw_player_name: Any,
    players_by_name: dict[str, str],
    players_by_id: dict[str, PlayerRecord],
) -> str | None:
    if isinstance(raw_player_id, str) and raw_player_id in players_by_id:
        return raw_player_id
    if isinstance(raw_player_name, str):
        return players_by_name.get(raw_player_name)
    return None


def _is_imposter(
    player_id: str | None,
    round_record: RoundArtifactRecord,
) -> bool | None:
    if player_id is None or round_record.imposter_player_id is None:
        return None
    return player_id == round_record.imposter_player_id


def _first_agent_player_id(players: list[PlayerRecord]) -> str | None:
    for player in players:
        if player.player_kind == "agent":
            return player.player_id
    return None


def _player_kind(player_id: str | None, players: list[PlayerRecord]) -> str | None:
    for player in players:
        if player.player_id == player_id:
            return player.player_kind
    return None
