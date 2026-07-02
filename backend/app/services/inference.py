from dataclasses import dataclass

import httpx


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


@dataclass(frozen=True)
class InferenceResult:
    text: str
    inference_mode: str


class InferenceServiceError(Exception):
    """Raised when the model inference service cannot return a usable response."""


class InferenceClient:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        if self.settings.inference_mode == "fake":
            return self._fake_generate(request)

        return await self._remote_generate(request)

    def _fake_generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            text=(
                f"{request.agent.name} received: {request.prompt} "
                "This is fake inference output from the backend boundary."
            ),
            inference_mode="fake",
        )

    async def _remote_generate(self, request: InferenceRequest) -> InferenceResult:
        payload = {
            "messages": [
                {"role": "system", "content": request.agent.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.agent.temperature,
            "top_p": request.agent.top_p,
            "max_tokens": request.agent.max_tokens,
            "model": self.settings.model_name,
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{self.settings.model_server_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            text = data["choices"][0]["message"]["content"]
            return InferenceResult(text=text, inference_mode="remote")
        except httpx.TimeoutException as exc:
            raise InferenceServiceError("Model server request timed out") from exc
        except httpx.RequestError as e:
            raise InferenceServiceError("Model server is unavailable") from e
        except httpx.HTTPStatusError as e:
            raise InferenceServiceError(
                f"Model server returned HTTP {e.response.status_code}"
            ) from e
        except ValueError as e:
            raise InferenceServiceError("Model server returned invalid JSON") from e
        except KeyError as e:
            raise InferenceServiceError("Model server returned an unexpected response") from e
        except IndexError as e:
            raise InferenceServiceError("Model server returned no choices") from e
