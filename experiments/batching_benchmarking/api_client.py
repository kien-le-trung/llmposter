from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from schemas import BenchmarkCase, RoundLatencyRecord


TERMINAL_STATUSES = {"ready_to_vote", "complete", "generation_failed"}


class BatchingBenchmarkApiClient:
    def __init__(self, backend_url: str, timeout_seconds: float = 60.0) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def run_case(
        self,
        case: BenchmarkCase,
        mode: str,
        prompt_technique: str,
    ) -> RoundLatencyRecord:
        started = time.perf_counter()
        round_id: str | None = None
        create_latency_ms: float | None = None
        continuation_latency_ms: float | None = None
        poll_count = 0

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                create_started = time.perf_counter()
                create_response = await client.post(
                    f"{self.backend_url}/rounds",
                    json={
                        "secret_word": case.secret_word,
                        "imposter_hint": case.imposter_hint,
                        "prompt_technique": prompt_technique,
                    },
                )
                create_latency_ms = (time.perf_counter() - create_started) * 1000.0
                create_response.raise_for_status()
                round_payload = create_response.json()
                round_id = str(round_payload["id"])

                if round_payload.get("status") == "awaiting_human_clue":
                    round_payload, continuation_latency_ms = await self._submit_benchmark_clue(
                        client,
                        round_id,
                        case,
                    )

                round_payload, poll_count, polled_continuation_latency_ms = await self._poll_until_terminal(
                    client,
                    round_id,
                    round_payload,
                    case,
                )
                if polled_continuation_latency_ms is not None:
                    continuation_latency_ms = polled_continuation_latency_ms

            total_latency_ms = (time.perf_counter() - started) * 1000.0
            status = str(round_payload.get("status", "unknown"))
            return RoundLatencyRecord(
                mode=mode,
                prompt_technique=prompt_technique,
                round_id=round_id,
                secret_word=case.secret_word,
                imposter_hint=case.imposter_hint,
                status=status,
                success=status in {"ready_to_vote", "complete"},
                total_latency_ms=total_latency_ms,
                create_latency_ms=create_latency_ms,
                continuation_latency_ms=continuation_latency_ms,
                poll_count=poll_count,
                generated_agent_clue_count=_generated_agent_clue_count(round_payload),
                playing_order=_playing_order(round_payload),
            )
        except Exception as exc:
            total_latency_ms = (time.perf_counter() - started) * 1000.0
            return RoundLatencyRecord(
                mode=mode,
                prompt_technique=prompt_technique,
                round_id=round_id,
                secret_word=case.secret_word,
                imposter_hint=case.imposter_hint,
                status="error",
                success=False,
                total_latency_ms=total_latency_ms,
                create_latency_ms=create_latency_ms,
                continuation_latency_ms=continuation_latency_ms,
                poll_count=poll_count,
                error=str(exc),
            )

    async def _poll_until_terminal(
        self,
        client: httpx.AsyncClient,
        round_id: str,
        round_payload: dict[str, Any],
        case: BenchmarkCase,
    ) -> tuple[dict[str, Any], int, float | None]:
        poll_count = 0
        continuation_latency_ms: float | None = None
        deadline = time.perf_counter() + self.timeout_seconds
        while time.perf_counter() < deadline:
            if round_payload.get("status") in TERMINAL_STATUSES:
                return round_payload, poll_count, continuation_latency_ms

            if round_payload.get("status") == "awaiting_human_clue":
                round_payload, continuation_latency_ms = await self._submit_benchmark_clue(
                    client,
                    round_id,
                    case,
                )
                continue

            await asyncio.sleep(0.25)
            poll_count += 1
            round_response = await client.get(f"{self.backend_url}/rounds/{round_id}")
            round_response.raise_for_status()
            round_payload = round_response.json()

        round_payload["status"] = "timeout"
        return round_payload, poll_count, continuation_latency_ms

    async def _submit_benchmark_clue(
        self,
        client: httpx.AsyncClient,
        round_id: str,
        case: BenchmarkCase,
    ) -> tuple[dict[str, Any], float]:
        continuation_started = time.perf_counter()
        clue_response = await client.post(
            f"{self.backend_url}/rounds/{round_id}/clue",
            json={"clue": case.human_clue},
        )
        continuation_latency_ms = (time.perf_counter() - continuation_started) * 1000.0
        clue_response.raise_for_status()
        return clue_response.json(), continuation_latency_ms


def _generated_agent_clue_count(round_payload: dict[str, Any]) -> int:
    turns = round_payload.get("turns")
    if not isinstance(turns, list):
        return 0

    count = 0
    for turn in turns:
        responses = turn.get("responses") if isinstance(turn, dict) else None
        if not isinstance(responses, list):
            continue
        count += sum(
            1
            for response in responses
            if isinstance(response, dict) and response.get("agent_id") != "human"
        )

    return count


def _playing_order(round_payload: dict[str, Any]) -> list[str]:
    players = round_payload.get("playing_order")
    if not isinstance(players, list):
        return []

    return [
        str(player.get("id", ""))
        for player in players
        if isinstance(player, dict)
    ]
