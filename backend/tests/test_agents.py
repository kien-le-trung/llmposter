import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
import pytest

from app.core.config import LLMConfig
from app.main import create_app
from app.services.agents import (
    IMPOSTER_CLUE_STRATEGIES,
    build_batched_clue_user_prompt,
    build_clue_user_prompt,
    build_instruction_batched_clue_user_prompt,
    build_instruction_clue_user_prompt,
    build_vote_user_prompt,
    clean_batched_clue_response,
    clean_clue_response,
    clean_vote_response,
)
from app.services.inference import (
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


def test_clue_prompt_uses_few_shot_completion_shape() -> None:
    prompt = build_clue_user_prompt(
        secret_word="satellite",
        imposter_hint=None,
        previous_clues=[("Agent A", "circles earth")],
    )

    assert "Examples:" in prompt
    assert "Secret word: satellite" in prompt
    assert "Agent A: circles earth" in prompt
    assert prompt.endswith("JSON:")


def test_batched_clue_prompt_uses_named_output_lines() -> None:
    prompt = build_batched_clue_user_prompt(
        secret_word="satellite",
        player_names=["Agent A", "Agent B"],
    )

    assert "Task: write one short clue for each player." in prompt
    assert "Secret word: satellite" in prompt
    assert '"Agent A":"2 to 5 word clue"' in prompt
    assert '"Agent B":"2 to 5 word clue"' in prompt


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
    strategy_names = {strategy["name"] for strategy in IMPOSTER_CLUE_STRATEGIES}

    assert strategy_names == {
        "Ride previous clues",
        "Abstraction",
        "Cluster matching",
        "Contextual guess",
        "Adjacent association",
    }
    assert all(strategy["prompt"] for strategy in IMPOSTER_CLUE_STRATEGIES)


def test_instruction_batched_clue_prompt_uses_named_json_shape() -> None:
    prompt = build_instruction_batched_clue_user_prompt(
        secret_word="satellite",
        player_names=["Agent A", "Agent B"],
    )

    assert "Each listed player knows the secret word." in prompt
    assert "Examples:" not in prompt
    assert '"Agent A":"your clue"' in prompt
    assert '"Agent B":"your clue"' in prompt


def test_instruction_batched_clue_prompt_can_include_player_strategies() -> None:
    prompt = build_instruction_batched_clue_user_prompt(
        secret_word="satellite",
        player_names=["Agent A", "Agent B"],
        strategies_by_player_name={
            "Agent A": {
                "name": "Indirect association",
                "prompt": "Write an indirect relation.",
            },
            "Agent B": {
                "name": "Side effect",
                "prompt": "Write a result of the word.",
            },
        },
    )

    assert "Player-specific prompts:" in prompt
    assert "Agent A strategy - Indirect association:\nWrite an indirect relation." in prompt
    assert "Agent B strategy - Side effect:\nWrite a result of the word." in prompt
    assert '"Agent A":"your clue"' in prompt


def test_vote_prompt_uses_allowed_answers_shape() -> None:
    prompt = build_vote_user_prompt(
        candidate_names=["Agent A", "You"],
        clue_lines=[("Agent A", "space signal"), ("You", "red fruit")],
    )

    assert "Candidates:" in prompt
    assert 'Agent A = "space signal"' in prompt
    assert "Allowed answers:\nAgent A\nYou" in prompt
    assert prompt.endswith('JSON: {"vote":"<one allowed answer>"}')


def test_clue_cleanup_caps_words_and_removes_secret_word() -> None:
    clue = clean_clue_response(
        "Clue: satellite orbiting above the earth with signals",
        secret_word="satellite",
        fallback_hint="Space, signals, or orbit",
    )

    assert clue == "space signals orbit"


def test_batched_clue_cleanup_maps_named_lines() -> None:
    clues = clean_batched_clue_response(
        "Agent A: circles earth\nAgent B: satellite signal",
        player_names=["Agent A", "Agent B"],
        secret_word="satellite",
        fallback_hint="Space, signals, or orbit",
    )

    assert clues["Agent A"] == "circles earth"
    assert clues["Agent B"] == "space signals orbit"


def test_vote_cleanup_matches_candidate_names() -> None:
    assert clean_vote_response("Vote: agent a.", ["Agent A", "You"]) == "Agent A"
    assert clean_vote_response("I choose human", ["Agent A", "You"]) == "You"


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

    monkeypatch.setattr("app.services.inference.httpx.AsyncClient", FakeAsyncClient)
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

    monkeypatch.setattr("app.services.inference.httpx.AsyncClient", FakeAsyncClient)
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

    monkeypatch.setattr("app.services.inference.httpx.AsyncClient", FakeAsyncClient)
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

    monkeypatch.setattr("app.services.inference.httpx.AsyncClient", FakeAsyncClient)
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
