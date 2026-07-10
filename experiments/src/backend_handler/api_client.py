from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import time
from typing import Any

import httpx


TERMINAL_ROUND_STATUSES = {"ready_to_vote", "complete", "generation_failed"}


@dataclass(frozen=True)
class BackendExperimentRequest:
    secret_word: str
    imposter_hint: str
    prompt_technique: str | None = None
    include_human: bool = False
    submit_vote: bool = True
    case_id: str | None = None
    repetition_index: int | None = None

    def to_round_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "secret_word": self.secret_word,
            "imposter_hint": self.imposter_hint,
            "include_human": self.include_human,
        }
        if self.prompt_technique is not None:
            payload["prompt_technique"] = self.prompt_technique
            
        return payload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendRequestArtifact:
    method: str
    path: str
    request_json: dict[str, Any] | None
    status_code: int | None
    response_json: Any | None
    response_text: str | None
    latency_ms: float
    started_at: str
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.status_code is not None and self.status_code < 400

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendExperimentResult:
    request: BackendExperimentRequest
    success: bool
    round_id: str | None
    status: str
    latency_ms: float
    round_payload: dict[str, Any] | None = None
    vote_payload: dict[str, Any] | None = None
    request_artifacts: list[BackendRequestArtifact] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["request"] = self.request.to_dict()
        data["request_artifacts"] = [
            artifact.to_dict() for artifact in self.request_artifacts
        ]
        return data


class BackendExperimentApiClient:
    def __init__(
        self,
        backend_url: str,
        timeout_seconds: float = 60.0,
        request_timeout_seconds: float = 20.0,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    async def health_check(self) -> BackendRequestArtifact:
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            artifact, _ = await self._request_json(client, "GET", "/health")
            return artifact

    async def list_agents(self) -> tuple[BackendRequestArtifact, list[dict[str, Any]]]:
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            artifact, payload = await self._request_json(client, "GET", "/agents")
            agents = payload if isinstance(payload, list) else []
            return artifact, [agent for agent in agents if isinstance(agent, dict)]

    async def run_request(
        self,
        request: BackendExperimentRequest,
    ) -> BackendExperimentResult:
        started = time.perf_counter()
        artifacts: list[BackendRequestArtifact] = []
        round_id: str | None = None
        round_payload: dict[str, Any] | None = None
        vote_payload: dict[str, Any] | None = None

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                create_artifact, create_payload = await self._request_json(
                    client,
                    "POST",
                    "/rounds",
                    json=request.to_round_payload(),
                )
                artifacts.append(create_artifact)
                self._raise_for_artifact(create_artifact)
                if not isinstance(create_payload, dict):
                    raise ValueError("Create round response must be a JSON object")

                round_payload = create_payload
                round_id = str(round_payload["id"])
                round_payload, poll_artifacts = await self.wait_for_round(
                    client,
                    round_id,
                )
                artifacts.extend(poll_artifacts)

                status = str(round_payload.get("status", "unknown"))
                if request.submit_vote and status == "ready_to_vote":
                    vote_payload, vote_artifact = await self.submit_agent_only_vote(
                        client,
                        round_id,
                    )
                    artifacts.append(vote_artifact)

            latency_ms = (time.perf_counter() - started) * 1000.0
            status = str(round_payload.get("status", "unknown")) if round_payload else "unknown"
            success = status == "ready_to_vote" or status == "complete"
            if request.submit_vote:
                success = success and vote_payload is not None
            return BackendExperimentResult(
                request=request,
                success=success,
                round_id=round_id,
                status=status,
                latency_ms=latency_ms,
                round_payload=round_payload,
                vote_payload=vote_payload,
                request_artifacts=artifacts,
                error=None if success else f"terminal status: {status}",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return BackendExperimentResult(
                request=request,
                success=False,
                round_id=round_id,
                status="error",
                latency_ms=latency_ms,
                round_payload=round_payload,
                vote_payload=vote_payload,
                request_artifacts=artifacts,
                error=_format_exception(exc),
            )

    async def wait_for_round(
        self,
        client: httpx.AsyncClient,
        round_id: str,
    ) -> tuple[dict[str, Any], list[BackendRequestArtifact]]:
        artifacts: list[BackendRequestArtifact] = []
        deadline = time.perf_counter() + self.timeout_seconds

        while time.perf_counter() < deadline:
            artifact, payload = await self._request_json(
                client,
                "GET",
                f"/rounds/{round_id}",
            )
            artifacts.append(artifact)
            self._raise_for_artifact(artifact)
            if not isinstance(payload, dict):
                raise ValueError("Get round response must be a JSON object")

            status = payload.get("status")
            if status == "awaiting_human_clue":
                raise ValueError(
                    "Round unexpectedly requested a human clue; "
                    "human clue flow is not wired for experiments yet"
                )
            if status in TERMINAL_ROUND_STATUSES:
                return payload, artifacts

            await asyncio.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"Round {round_id} did not finish within {self.timeout_seconds}s")

    async def submit_agent_only_vote(
        self,
        client: httpx.AsyncClient,
        round_id: str,
    ) -> tuple[dict[str, Any], BackendRequestArtifact]:
        artifact, payload = await self._request_json(
            client,
            "POST",
            f"/rounds/{round_id}/vote",
            json={},
        )
        self._raise_for_artifact(artifact)
        if not isinstance(payload, dict):
            raise ValueError("Vote response must be a JSON object")
        return payload, artifact

    async def submit_human_clue(
        self,
        client: httpx.AsyncClient,
        round_id: str,
        clue: str,
    ) -> tuple[dict[str, Any], BackendRequestArtifact]:
        artifact, payload = await self._request_json(
            client,
            "POST",
            f"/rounds/{round_id}/clue",
            json={"clue": clue},
        )
        self._raise_for_artifact(artifact)
        if not isinstance(payload, dict):
            raise ValueError("Submit clue response must be a JSON object")
        return payload, artifact

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> tuple[BackendRequestArtifact, Any | None]:
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        response_json: Any | None = None
        response_text: str | None = None
        status_code: int | None = None
        error: str | None = None

        try:
            response = await client.request(
                method,
                f"{self.backend_url}{path}",
                json=json,
            )
            status_code = response.status_code
            response_text = response.text
            if response.content:
                try:
                    response_json = response.json()
                except ValueError:
                    response_json = None
            if response.status_code >= 400:
                error = f"HTTP {response.status_code} from {method} {path}"
        except httpx.HTTPError as exc:
            error = _format_exception(exc)

        latency_ms = (time.perf_counter() - started) * 1000.0
        artifact = BackendRequestArtifact(
            method=method,
            path=path,
            request_json=json,
            status_code=status_code,
            response_json=response_json,
            response_text=response_text,
            latency_ms=latency_ms,
            started_at=started_at,
            error=error,
        )
        return artifact, response_json

    def _raise_for_artifact(self, artifact: BackendRequestArtifact) -> None:
        if artifact.success:
            return

        detail = artifact.response_text.strip() if artifact.response_text else artifact.error
        raise BackendApiClientError(detail or f"{artifact.method} {artifact.path} failed")


class BackendApiClientError(RuntimeError):
    pass


def _format_exception(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__
