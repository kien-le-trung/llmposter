import asyncio
import json
import random
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import LLMConfig
from app.services.metrics import measure_stage

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 3


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
        self.grok_config = getattr(settings, "grok_config", None)
        primary_config = self._provider_configs()[0]
        limit = max_concurrent_requests or primary_config.max_concurrent_requests
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

        api_key = self.settings.get_env_value("GEMINI_API_KEY")
        if not api_key:
            raise InferenceServiceError("GEMINI_API_KEY is not configured")

        model_name = self.settings.embedding_model_name
        model_server_url = self.settings.embedding_model_server_url
        if not model_server_url:
            raise InferenceServiceError("EMBEDDING_MODEL_SERVER_URL is not configured")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        payload = {
            "requests": [
                {
                    "model": f"models/{model_name}",
                    "taskType": "SEMANTIC_SIMILARITY",
                    "outputDimensionality": 768,
                    "content": {
                        "parts": [
                            {"text": text},
                        ],
                    },
                }
                for text in texts
            ],
        }

        embedding_url = (
            f"{model_server_url.rstrip('/')}/models/"
            f"{model_name}:batchEmbedContents"
        )

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                with measure_stage(
                    "embedding_http",
                    {"model": self.settings.embedding_model_name, "count": str(len(texts))},
                ):
                    response = await client.post(
                        embedding_url,
                        headers=headers,
                        json=payload,
                    )
                response.raise_for_status()
                with measure_stage("embedding_response_parse"):
                    data = response.json()

            embedding_objects = data.get("embeddings")
            if not isinstance(embedding_objects, list):
                raise ValueError("missing embeddings")

            embeddings = []
            for embedding_object in embedding_objects:
                if not isinstance(embedding_object, dict):
                    raise TypeError("unexpected embedding object")

                values = embedding_object.get("values")
                if not isinstance(values, list):
                    raise ValueError("missing embedding values")

                embeddings.append([float(value) for value in values])

            if len(embeddings) != len(texts):
                raise ValueError("wrong number of embeddings")

            return EmbeddingResult(
                embeddings=embeddings,
                inference_mode="gemini",
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

        errors = []
        for llm_config in self._provider_configs():
            try:
                return await self._generate_with_config(request, llm_config)
            except InferenceServiceError as exc:
                errors.append(f"{llm_config.provider}: {exc}")
        raise InferenceServiceError("; ".join(errors))

    def _provider_configs(self) -> list[LLMConfig]:
        # Keep explicit local/experiment providers unchanged. Existing OpenRouter
        # configuration becomes the fallback without requiring edits to .env.
        if self.grok_config is not None and self.llm_config.provider == "openrouter":
            return [self.grok_config, self.llm_config]
        return [self.llm_config]

    async def _generate_with_config(
        self, request: InferenceRequest, llm_config: LLMConfig,
    ) -> InferenceResult:
        headers = {"Content-Type": "application/json"}
        if llm_config.api_key_env is not None:
            api_key = self.settings.get_env_value(llm_config.api_key_env)
            if not api_key:
                raise InferenceServiceError(
                    f"{llm_config.api_key_env} is not configured"
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
            "max_tokens": self._resolve_max_tokens(request, llm_config),
            "model": llm_config.model,
        }

        last_error: Exception | None = None
        for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
            retry_reason = "unknown"
            try:
                client = self._http_client
                owns_client = client is None
                if client is None:
                    owned_client = httpx.AsyncClient(timeout=20.0)
                    client = await owned_client.__aenter__()
                try:
                    with measure_stage(
                        "model_http",
                        {
                            "model": llm_config.model,
                            "agent": request.agent.id,
                            "attempt": str(attempt),
                        },
                    ):
                        async with self._semaphore:
                            response = await client.post(
                                url=llm_config.chat_url,
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
                return InferenceResult(
                    text=text,
                    inference_mode=llm_config.provider,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                retry_reason = "timeout"
            except httpx.RequestError as exc:
                last_error = exc
                retry_reason = "request_error"
            except ValueError as exc:
                last_error = exc
                retry_reason = "invalid_json"
            except (TypeError, AttributeError) as exc:
                last_error = exc
                retry_reason = "unexpected_response"
            except KeyError as exc:
                last_error = exc
                retry_reason = "unexpected_response"
            except IndexError as exc:
                last_error = exc
                retry_reason = "missing_choice"
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                    # Permanent HTTP failure: fail immediately.
                    raise InferenceServiceError(
                        f"Model provider returned HTTP {exc.response.status_code}"
                    ) from exc
                last_error = exc
                retry_reason = f"http_{exc.response.status_code}"

            if attempt < DEFAULT_MAX_ATTEMPTS:
                delay = _retry_delay(attempt)

                # If there's Retry-After from remote server, prioritize it
                if isinstance(last_error, httpx.HTTPStatusError):
                    retry_after = _retry_after_seconds(last_error.response)
                    if retry_after is not None:
                        delay = retry_after

                delay = min(delay, 10.0)
                with measure_stage(
                    "model_retry_backoff",
                    {
                        "model": llm_config.model,
                        "agent": request.agent.id,
                        "failed_attempt": str(attempt),
                        "next_attempt": str(attempt + 1),
                        "reason": retry_reason,
                        "delay_seconds": f"{delay:.3f}",
                    },
                ):
                    await asyncio.sleep(delay)

        if isinstance(last_error, httpx.TimeoutException):
            raise InferenceServiceError(
                f"Model provider request timed out after {DEFAULT_MAX_ATTEMPTS} attempts"
            ) from last_error

        if isinstance(last_error, httpx.RequestError):
            raise InferenceServiceError(
                f"Model provider was unavailable after {DEFAULT_MAX_ATTEMPTS} attempts"
            ) from last_error

        if isinstance(last_error, httpx.HTTPStatusError):
            raise InferenceServiceError(
                f"Model provider returned HTTP "
                f"{last_error.response.status_code} after {DEFAULT_MAX_ATTEMPTS} attempts"
            ) from last_error

        raise InferenceServiceError(
            f"Model provider returned an invalid response after {DEFAULT_MAX_ATTEMPTS} attempts"
        ) from last_error


    async def generate_structured(
        self,
        request: InferenceRequest,
        response_model: type[StructuredModel],
        max_attempts: int = 3,
        validate: Callable[[StructuredModel], None] | None = None,
    ) -> tuple[StructuredModel, InferenceResult]:
        configs = self._provider_configs()
        if self.settings.inference_mode == "fake" or len(configs) < 2:
            return await self._generate_structured(request, response_model, max_attempts, validate)

        errors = []
        for llm_config in configs:
            try:
                return await self._generate_structured(
                    request, response_model, max_attempts, validate, llm_config=llm_config,
                )
            except InferenceServiceError as exc:
                errors.append(f"{llm_config.provider}: {exc}")
        raise InferenceServiceError("; ".join(errors))

    async def _generate_structured(
        self,
        request: InferenceRequest,
        response_model: type[StructuredModel],
        max_attempts: int,
        validate: Callable[[StructuredModel], None] | None,
        llm_config: LLMConfig | None = None,
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
                        structured_request = InferenceRequest(
                            prompt=retry_prompt,
                            agent=request.agent,
                            response_schema=response_schema,
                        )
                        if llm_config is None:
                            result = await self.generate(structured_request)
                        else:
                            result = await self._generate_with_config(structured_request, llm_config)
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

            if "comment" in properties:
                return InferenceResult(
                    text='{"comment":"Could you explain how your clue fits?"}', inference_mode="fake",
                )
            if "target_id" in properties:
                context, _ = json.JSONDecoder().raw_decode(
                    request.prompt.split("GAME_CONTEXT:\n", 1)[1]
                )
                target = next(p["id"] for p in context["players"] if p["id"] != context["your_id"])
                return InferenceResult(text=json.dumps({"target_id": target}), inference_mode="fake")

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

    def _resolve_max_tokens(
        self, request: InferenceRequest, llm_config: LLMConfig,
    ) -> int:
        if request.response_schema is None:
            return max(
                request.agent.max_tokens,
                llm_config.min_max_tokens,
            )

        return max(
            request.agent.max_tokens,
            llm_config.min_max_tokens,
            llm_config.structured_max_tokens,
        )

def _retry_delay(attempt: int) -> float:
    base_delay = 0.5 * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, 0.25)
    return base_delay + jitter


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        return None


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
