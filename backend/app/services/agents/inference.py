import asyncio
import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.services.metrics import measure_stage

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


@dataclass(frozen=True)
class AgentConfig:
    id: str
    name: str
    role: str
    system_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    version: str


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    agent: AgentConfig
    response_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class InferenceResult:
    text: str
    inference_mode: str


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: list[list[float]]
    inference_mode: str


class InferenceServiceError(Exception):
    """Raised when the model inference service cannot return a usable response."""


class InferenceClient:
    def __init__(self, settings, max_concurrent_requests: int | None = None) -> None:
        self.settings = settings
        self.llm_config = settings.llm_config
        limit = max_concurrent_requests or self.llm_config.max_concurrent_requests
        if limit is None:
            limit = 1 if "localhost:8888" in self.llm_config.chat_url else 100
        self._semaphore = asyncio.Semaphore(limit)
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "InferenceClient":
        self._http_client = httpx.AsyncClient(timeout=20.0)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        if self.settings.inference_mode == "fake":
            return EmbeddingResult(
                embeddings=[_fake_embedding(text) for text in texts],
                inference_mode="fake",
            )

        payload = {
            "model": self.settings.embedding_model_name,
            "input": texts,
        }
        model_server_url = self.settings.embedding_model_server_url

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                with measure_stage(
                    "embedding_http",
                    {"model": self.settings.embedding_model_name, "count": str(len(texts))},
                ):
                    response = await client.post(
                        f"{model_server_url}/api/embed",
                        json=payload,
                    )
                response.raise_for_status()
                with measure_stage("embedding_response_parse"):
                    data = response.json()

            embeddings = data.get("embeddings")
            if embeddings is None and "embedding" in data:
                embeddings = [data["embedding"]]
            if not isinstance(embeddings, list):
                raise ValueError("missing embeddings")

            return EmbeddingResult(
                embeddings=[list(map(float, embedding)) for embedding in embeddings],
                inference_mode="remote",
            )
        except httpx.TimeoutException as exc:
            raise InferenceServiceError("Embedding model request timed out") from exc
        except httpx.RequestError as exc:
            raise InferenceServiceError("Embedding model server is unavailable") from exc
        except httpx.HTTPStatusError as exc:
            response_text = exc.response.text.strip()
            detail = f": {response_text}" if response_text else ""
            raise InferenceServiceError(
                f"Embedding model returned HTTP {exc.response.status_code}{detail}"
            ) from exc
        except ValueError as exc:
            raise InferenceServiceError("Embedding model returned invalid JSON") from exc
        except TypeError as exc:
            raise InferenceServiceError("Embedding model returned an unexpected response") from exc

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        if self.settings.inference_mode == "fake":
            return self._fake_generate(request)

        headers = {"Content-Type": "application/json"}
        if self.llm_config.api_key_env is not None:
            api_key = self.settings.get_env_value(self.llm_config.api_key_env)
            if not api_key:
                raise InferenceServiceError(
                    f"{self.llm_config.api_key_env} is not configured"
                )
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "messages": [
                {"role": "system", "content": request.agent.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": (
                request.agent.temperature
            ),
            "top_p": (
                request.agent.top_p
            ),
            "max_tokens": self._resolve_max_tokens(request),
            "model": self.llm_config.model,
        }
        try:
            client = self._http_client
            owns_client = client is None
            if client is None:
                owned_client = httpx.AsyncClient(timeout=20.0)
                client = await owned_client.__aenter__()
            try:
                with measure_stage(
                    "model_http",
                    {"model": self.llm_config.model, "agent": request.agent.id},
                ):
                    async with self._semaphore:
                        response = await client.post(
                            url=self.llm_config.chat_url,
                            headers=headers,
                            json=payload,
                        )
                response.raise_for_status()
                with measure_stage("model_response_parse"):
                    data = response.json()
            finally:
                if owns_client:
                    await owned_client.__aexit__(None, None, None)

            choice = data["choices"][0]
            text = choice["message"].get("content")
            if not isinstance(text, str) or not text.strip():
                finish_reason = choice.get("finish_reason", "unknown")
                raise InferenceServiceError(
                    "Model provider returned empty assistant content "
                    f"(finish_reason={finish_reason})"
                )
            return InferenceResult(text=text, inference_mode=self.llm_config.provider)
        except httpx.TimeoutException as exc:
            raise InferenceServiceError("Model provider request timed out") from exc
        except httpx.RequestError as exc:
            raise InferenceServiceError("Model provider is unavailable") from exc
        except httpx.HTTPStatusError as exc:
            response_text = exc.response.text.strip()
            detail = f": {response_text}" if response_text else ""
            raise InferenceServiceError(
                f"Model provider returned HTTP {exc.response.status_code}{detail}"
            ) from exc
        except ValueError as exc:
            raise InferenceServiceError("Model provider returned invalid JSON") from exc
        except KeyError as exc:
            raise InferenceServiceError("Model provider returned an unexpected response") from exc
        except IndexError as exc:
            raise InferenceServiceError("Model provider returned no choices") from exc

    async def generate_structured(
        self,
        request: InferenceRequest,
        response_model: type[StructuredModel],
        max_attempts: int = 3,
        validate: Callable[[StructuredModel], None] | None = None,
    ) -> tuple[StructuredModel, InferenceResult]:
        response_schema = response_model.model_json_schema()
        prompt = self._build_structured_prompt(request.prompt, response_schema)
        last_error = "unknown validation error"

        with measure_stage(
            "structured_generation",
            {"model": request.agent.id, "max_attempts": str(max_attempts)},
        ):
            for attempt in range(max_attempts):
                retry_prompt = prompt
                if attempt > 0:
                    retry_prompt = (
                        f"{prompt}\n\n"
                        "Previous output failed schema validation. "
                        f"Reason: {last_error}. Return valid JSON only."
                    )

                with measure_stage("structured_attempt", {"attempt": str(attempt + 1)}):
                    try:
                        result = await self.generate(
                            InferenceRequest(
                                prompt=retry_prompt,
                                agent=request.agent,
                                response_schema=response_schema,
                            )
                        )
                    except InferenceServiceError as exc:
                        if "empty assistant content" not in str(exc):
                            raise
                        last_error = str(exc)
                        continue

                try:
                    with measure_stage("schema_validation", {"attempt": str(attempt + 1)}):
                        json_text = _extract_json_object(result.text)
                        structured_response = response_model.model_validate_json(json_text)
                        if validate is not None:
                            validate(structured_response)
                    return structured_response, result
                except (ValueError, ValidationError) as exc:
                    last_error = str(exc)

        raise InferenceServiceError(
            f"Model returned invalid structured output after {max_attempts} attempts"
        )

    def _fake_generate(self, request: InferenceRequest) -> InferenceResult:
        if request.response_schema is not None:
            properties = request.response_schema.get("properties", {})
            if "clues" in properties:
                return InferenceResult(
                    text=json.dumps(
                        {
                            "clues": {
                                "Agent A": "fake clue",
                                "Agent B": "fake clue",
                                "Agent C": "fake clue",
                                "Agent D": "fake clue",
                            }
                        }
                    ),
                    inference_mode="fake",
                )

            if "clue" in properties:
                return InferenceResult(
                    text=json.dumps({"clue": "fake clue"}),
                    inference_mode="fake",
                )

            if "vote" in properties:
                return InferenceResult(
                    text=json.dumps({"vote": _choose_fake_vote(request.prompt)}),
                    inference_mode="fake",
                )

        return InferenceResult(
            text=(
                f"{request.agent.name} received: {request.prompt} "
                "This is fake inference output from the backend boundary."
            ),
            inference_mode="fake",
        )

    def _build_structured_prompt(self, prompt: str, response_schema: dict[str, Any]) -> str:
        return (
            f"{prompt}\n\n"
            "Return valid JSON only. Do not include markdown or explanation.\n"
            "JSON schema:\n"
            f"{json.dumps(response_schema, separators=(',', ':'))}"
        )

    def _resolve_max_tokens(self, request: InferenceRequest) -> int:
        if request.response_schema is None:
            return max(
                request.agent.max_tokens,
                self.llm_config.min_max_tokens,
            )

        return max(
            request.agent.max_tokens,
            self.llm_config.min_max_tokens,
            self.llm_config.structured_max_tokens,
        )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise ValueError("No JSON object found")

    return stripped[start_index : end_index + 1]


def _choose_fake_vote(prompt: str) -> str:
    allowed_answers_marker = "Allowed answers:\n"
    if allowed_answers_marker not in prompt:
        return "You"

    allowed_answers_block = prompt.split(allowed_answers_marker, 1)[1].split("\n\n", 1)[0]
    allowed_answers = [
        line.strip()
        for line in allowed_answers_block.splitlines()
        if line.strip()
    ]
    if "Agent A" in allowed_answers:
        return "Agent A"

    for allowed_answer in allowed_answers:
        if allowed_answer != "You":
            return allowed_answer

    return allowed_answers[0] if allowed_answers else "You"


def _fake_embedding(text: str) -> list[float]:
    digest = sha256(text.strip().lower().encode("utf-8")).digest()
    return [((byte / 255.0) * 2.0) - 1.0 for byte in digest[:16]]
