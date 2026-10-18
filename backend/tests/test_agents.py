import json
import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
import pytest

from app.core.config import DEFAULT_GROK_CONFIG, LLMConfig, Settings
from app.main import create_app
from app.services.agents.clue_generation import (
    build_instruction_clue_user_prompt,
)
from app.services.agents.strategy_loader import (
    load_imposter_clue_strategies,
)
from app.services.agents.inference import (
    AgentConfig,
    InferenceClient,
    InferenceRequest,
    InferenceResult,
    InferenceServiceError,
)


class StructuredTestResponse(BaseModel):
    clue: str = Field(min_length=1)


def build_test_agent() -> AgentConfig:
    return AgentConfig(
        id="agent_test",
        name="Agent Test",
        role="candidate",
        system_prompt="system",
        temperature=0.1,
        top_p=1.0,
        max_tokens=24,
        version="test",
    )


def build_test_settings(
    llm_config: LLMConfig | None = None,
    env_values: dict[str, str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        inference_mode="remote",
        llm_config=llm_config
        or LLMConfig(
            name="OpenRouter test",
            provider="openrouter",
            chat_url="https://openrouter.ai/api/v1/chat/completions",
            model="openrouter/free",
            api_key_env="OPENROUTER_API_KEY",
            min_max_tokens=128,
            structured_max_tokens=512,
        ),
        get_env_value=lambda name: (env_values or {}).get(name),
    )


def test_list_agents() -> None:
    client = TestClient(create_app())

    response = client.get("/agents")

    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_instruction_clue_prompt_uses_constraints_without_examples() -> None:
    prompt = build_instruction_clue_user_prompt(
        secret_word="satellite",
        imposter_hint=None,
        previous_clues=[],
        strategy={
            "name": "Indirect association",
            "prompt": "Write a phrase that is indirectly related to the secret word.",
        },
    )

    assert "Write a phrase that is indirectly related to the secret word." in prompt
    assert "Examples:" not in prompt
    assert "Do not use the word itself." in prompt
    assert '{"clue":"your phrase"}' in prompt


def test_instruction_imposter_clue_prompt_can_include_strategy() -> None:
    prompt = build_instruction_clue_user_prompt(
        secret_word=None,
        imposter_hint="space",
        previous_clues=[("Agent A", "moon orbit")],
        strategy={
            "name": "Ride previous clues",
            "prompt": "Use previous clues without copying them.",
        },
    )

    assert "You are the imposter." in prompt
    assert "Hint: space" in prompt
    assert "Agent A: moon orbit" in prompt
    assert "Use previous clues without copying them." in prompt
    assert '{"clue":"your clue"}' in prompt


def test_imposter_strategies_are_loaded_from_json() -> None:
    strategies = load_imposter_clue_strategies()
    strategy_names = {strategy["name"] for strategy in strategies}

    assert strategy_names == {
        "Ride previous clues",
        "Abstraction",
        "Cluster matching",
        "Contextual guess",
        "Adjacent association",
    }
    assert all(strategy["prompt"] for strategy in strategies)


def test_structured_generation_retries_until_schema_is_valid(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_generate(self, request):
        calls.append(request.prompt)
        if len(calls) == 1:
            return InferenceResult(text="not json", inference_mode="fake")
        return InferenceResult(text='{"clue":"valid clue"}', inference_mode="fake")

    monkeypatch.setattr(InferenceClient, "generate", fake_generate)
    client = InferenceClient(settings=build_test_settings())
    agent = build_test_agent()

    structured_response, result = asyncio.run(
        client.generate_structured(
            InferenceRequest(prompt="Return a clue.", agent=agent),
            StructuredTestResponse,
        )
    )

    assert structured_response.clue == "valid clue"
    assert result.inference_mode == "fake"
    assert len(calls) == 2
    assert "Previous output failed schema validation" in calls[1]


def test_generate_posts_to_openrouter(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "openrouter reply"}}]}

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, *, url: str, headers: dict[str, str], json: dict):
            calls.append({"url": url, "headers": headers, "json": json})
            return FakeResponse()

    monkeypatch.setattr("app.services.agents.inference.httpx.AsyncClient", FakeAsyncClient)
    client = InferenceClient(settings=build_test_settings(env_values={"OPENROUTER_API_KEY": "test-key"}))

    result = asyncio.run(
        client.generate(InferenceRequest(prompt="Return a clue.", agent=build_test_agent()))
    )

    assert result.text == "openrouter reply"
    assert result.inference_mode == "openrouter"
    assert calls == [
        {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            "json": {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "Return a clue."},
                ],
                "temperature": 0.1,
                "top_p": 1.0,
                "max_tokens": 128,
                "model": "openrouter/free",
            },
        }
    ]


def test_generate_requires_openrouter_api_key() -> None:
    client = InferenceClient(
        settings=build_test_settings(env_values={})
    )

    with pytest.raises(InferenceServiceError, match="OPENROUTER_API_KEY"):
        asyncio.run(
            client.generate(InferenceRequest(prompt="Return a clue.", agent=build_test_agent()))
        )


def test_generate_uses_larger_token_budget_for_openrouter_structured_calls(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"clue":"space signal"}'}}]}

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, *, url: str, headers: dict[str, str], json: dict):
            calls.append(json)
            return FakeResponse()

    monkeypatch.setattr("app.services.agents.inference.httpx.AsyncClient", FakeAsyncClient)
    client = InferenceClient(settings=build_test_settings(env_values={"OPENROUTER_API_KEY": "test-key"}))

    asyncio.run(
        client.generate(
            InferenceRequest(
                prompt="Return a clue.",
                agent=build_test_agent(),
                response_schema=StructuredTestResponse.model_json_schema(),
            )
        )
    )

    assert calls[0]["max_tokens"] == 512
    assert "response_format" not in calls[0]


def test_generate_rejects_empty_openrouter_content(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": None, "reasoning_details": []},
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, *, url: str, headers: dict[str, str], json: dict):
            return FakeResponse()

    monkeypatch.setattr("app.services.agents.inference.httpx.AsyncClient", FakeAsyncClient)
    client = InferenceClient(settings=build_test_settings(env_values={"OPENROUTER_API_KEY": "test-key"}))

    with pytest.raises(InferenceServiceError, match="empty assistant content"):
        asyncio.run(
            client.generate(InferenceRequest(prompt="Return a clue.", agent=build_test_agent()))
        )


def test_generate_omits_auth_for_local_config(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "local reply"}}]}

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, *, url: str, headers: dict[str, str], json: dict):
            calls.append({"url": url, "headers": headers, "json": json})
            return FakeResponse()

    monkeypatch.setattr("app.services.agents.inference.httpx.AsyncClient", FakeAsyncClient)
    client = InferenceClient(
        settings=build_test_settings(
            llm_config=LLMConfig(
                name="Local",
                provider="openai_compatible",
                chat_url="http://localhost:8888/v1/chat/completions",
                model="Qwen/Qwen2.5-3B-Instruct",
                api_key_env=None,
                min_max_tokens=64,
                structured_max_tokens=256,
            )
        )
    )

    result = asyncio.run(
        client.generate(InferenceRequest(prompt="Return a clue.", agent=build_test_agent()))
    )

    assert result.text == "local reply"
    assert result.inference_mode == "openai_compatible"
    assert calls[0]["url"] == "http://localhost:8888/v1/chat/completions"
    assert calls[0]["headers"] == {"Content-Type": "application/json"}
    assert calls[0]["json"]["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert calls[0]["json"]["max_tokens"] == 64


def grok_test_settings(env_values=None):
    settings = build_test_settings(env_values=env_values)
    settings.grok_config = DEFAULT_GROK_CONFIG
    return settings


def run_mock_generation(settings, handler, *, structured=False):
    import httpx

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = InferenceClient(settings)
            client._http_client = http_client
            request = InferenceRequest(prompt="Return a clue.", agent=build_test_agent())
            if structured:
                return await client.generate_structured(request, StructuredTestResponse)
            return await client.generate(request)

    return asyncio.run(run())


def test_grok_is_primary_and_uses_its_own_key():
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer grok-test-key"
        payload = json.loads(request.content)
        assert payload["model"] == DEFAULT_GROK_CONFIG.model
        assert payload["messages"][1]["content"] == "Return a clue."
        return httpx.Response(200, json={"choices": [{"message": {"content": "grok reply"}}]})

    settings = grok_test_settings({"GROK_API_KEY": "grok-test-key"})
    result = run_mock_generation(settings, handler)

    assert result == InferenceResult(text="grok reply", inference_mode="grok")
    assert len(calls) == 1
    assert str(calls[0].url) == DEFAULT_GROK_CONFIG.chat_url


@pytest.mark.parametrize("failure", ["missing_key", "unauthorized", "rate_limit", "timeout",
                                     "invalid_json", "empty_content", "malformed"])
def test_grok_failure_falls_back_to_openrouter(monkeypatch, failure):
    import httpx

    calls = []

    async def no_sleep(delay):
        pass

    monkeypatch.setattr("app.services.agents.inference.asyncio.sleep", no_sleep)

    def handler(request):
        calls.append(request)
        if request.url.host == "api.x.ai":
            if failure == "unauthorized":
                return httpx.Response(401, text="provider error")
            if failure == "rate_limit":
                return httpx.Response(429)
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            if failure == "invalid_json":
                return httpx.Response(200, text="not json")
            if failure == "empty_content":
                return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
            return httpx.Response(200, json={"choices": [None]})
        assert request.url.host == "openrouter.ai"
        assert request.headers["authorization"] == "Bearer router-test-key"
        assert json.loads(request.content)["model"] == "openrouter/free"
        return httpx.Response(200, json={"choices": [{"message": {"content": "fallback"}}]})

    keys = {"OPENROUTER_API_KEY": "router-test-key"}
    if failure != "missing_key":
        keys["GROK_API_KEY"] = "grok-test-key"
    result = run_mock_generation(grok_test_settings(keys), handler)

    assert result == InferenceResult(text="fallback", inference_mode="openrouter")
    assert calls[-1].url.host == "openrouter.ai"
    expected_primary_calls = 0 if failure == "missing_key" else (
        1 if failure in {"unauthorized", "empty_content"} else 3
    )
    assert len(calls) == expected_primary_calls + 1


def test_invalid_grok_structured_output_falls_back():
    import httpx

    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        content = '{"clue":"valid clue"}' if request.url.host == "openrouter.ai" else '{"wrong":1}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = grok_test_settings({"GROK_API_KEY": "grok-test-key",
                                   "OPENROUTER_API_KEY": "router-test-key"})
    response, result = run_mock_generation(settings, handler, structured=True)

    assert response.clue == "valid clue"
    assert result.inference_mode == "openrouter"
    assert hosts == ["api.x.ai"] * 3 + ["openrouter.ai"]


def test_both_providers_fail_without_exposing_response_body():
    import httpx

    def handler(request):
        return httpx.Response(401, text="sensitive provider response")

    settings = grok_test_settings({"GROK_API_KEY": "grok-test-key",
                                   "OPENROUTER_API_KEY": "router-test-key"})
    with pytest.raises(InferenceServiceError) as error:
        run_mock_generation(settings, handler)

    message = str(error.value)
    assert "grok: Model provider returned HTTP 401" in message
    assert "openrouter: Model provider returned HTTP 401" in message
    assert "sensitive" not in message
    assert "test-key" not in message


def test_settings_enable_grok_by_default():
    settings = Settings(_env_file=None)
    assert settings.grok_config == DEFAULT_GROK_CONFIG
    assert settings.llm_config.provider == "openrouter"


def test_grok_does_not_override_explicit_local_provider():
    settings = grok_test_settings()
    settings.llm_config = settings.llm_config.model_copy(update={"provider": "openai_compatible"})
    assert InferenceClient(settings)._provider_configs() == [settings.llm_config]


def test_structured_http_failure_tries_each_provider_once():
    import httpx

    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "api.x.ai":
            return httpx.Response(401)
        content = '{"clue":"valid clue"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = grok_test_settings({"GROK_API_KEY": "grok-test-key",
                                   "OPENROUTER_API_KEY": "router-test-key"})
    response, result = run_mock_generation(settings, handler, structured=True)

    assert response.clue == "valid clue"
    assert result.inference_mode == "openrouter"
    assert hosts == ["api.x.ai", "openrouter.ai"]


def test_fake_mode_skips_both_providers():
    def handler(request):
        pytest.fail("Fake inference must not make HTTP requests")

    settings = grok_test_settings()
    settings.inference_mode = "fake"
    response, result = run_mock_generation(settings, handler, structured=True)

    assert response.clue == "fake clue"
    assert result.inference_mode == "fake"
