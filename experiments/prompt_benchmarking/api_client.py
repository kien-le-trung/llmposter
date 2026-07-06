from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from schemas import BenchmarkCase, ClueBenchmarkRecord, RoundBenchmarkRecord

TERMINAL_STATUSES = {"ready_to_vote", "complete", "generation_failed"}


class PromptBenchmarkApiClient:
    def __init__(self, backend_url: str, timeout_seconds: float = 60.0) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def run_case(
        self,
        case: BenchmarkCase,
        technique: str,
    ) -> tuple[RoundBenchmarkRecord, list[ClueBenchmarkRecord]]:
        started = time.perf_counter()
        round_id: str | None = None
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                create_response = await client.post(
                    f"{self.backend_url}/rounds",
                    json={
                        "secret_word": case.secret_word,
                        "imposter_hint": case.imposter_hint,
                    },
                )
                create_response.raise_for_status()
                round_payload = create_response.json()
                round_id = round_payload["id"]

                round_payload = await self._complete_round_if_needed(
                    client,
                    round_id,
                    case.human_clue,
                )

            latency_ms = (time.perf_counter() - started) * 1000.0
            status = str(round_payload.get("status", "unknown"))
            success = status == "ready_to_vote"
            return (
                RoundBenchmarkRecord(
                    technique=technique,
                    round_id=round_id,
                    secret_word=case.secret_word,
                    imposter_hint=case.imposter_hint,
                    status=status,
                    latency_ms=latency_ms,
                    success=success,
                    error=None if success else f"terminal status: {status}",
                ),
                _extract_clues(round_payload, technique, case),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return (
                RoundBenchmarkRecord(
                    technique=technique,
                    round_id=round_id,
                    secret_word=case.secret_word,
                    imposter_hint=case.imposter_hint,
                    status="error",
                    latency_ms=latency_ms,
                    success=False,
                    error=str(exc),
                ),
                [],
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


def _extract_clues(
    round_payload: dict[str, Any],
    technique: str,
    case: BenchmarkCase,
) -> list[ClueBenchmarkRecord]:
    records: list[ClueBenchmarkRecord] = []
    turns = round_payload.get("turns")
    if not isinstance(turns, list):
        return records

    for turn in turns:
        responses = turn.get("responses") if isinstance(turn, dict) else None
        if not isinstance(responses, list):
            continue
        for response in responses:
            if not isinstance(response, dict):
                continue
            if response.get("agent_id") == "human":
                continue
            records.append(
                ClueBenchmarkRecord(
                    technique=technique,
                    round_id=str(round_payload.get("id", "")),
                    secret_word=case.secret_word,
                    imposter_hint=case.imposter_hint,
                    agent_id=str(response.get("agent_id", "")),
                    agent_name=str(response.get("agent_name", "")),
                    clue=str(response.get("agent_response", "")),
                    inference_mode=str(response.get("inference_mode", "")),
                )
            )
    return records
