import asyncio
import json
from time import perf_counter

import httpx
import pytest

from app.core.config import LLMConfig, Settings
from app.services import latency
from app.services.agents.clue_agent import ClueDecision
from app.services.agents.inference import InferenceClient, InferenceRequest
from app.services.agents.runtime_agents import list_static_agent_configs


@pytest.fixture
def events(tmp_path):
    path = tmp_path / "events.jsonl"
    latency.configure(path, "test-run")
    yield path
    latency.configure(None)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_telemetry_is_content_free_and_task_contexts_are_isolated(events):
    async def task(round_id):
        with latency.scope(round_id=round_id, prompt="private prompt", memory="private memory"):
            await asyncio.sleep(0)
            with latency.span("work"):
                latency.emit("marker", api_key="secret", generated_text="private reply")

    async def run():
        await asyncio.gather(task("one"), task("two"))

    asyncio.run(run())
    captured = rows(events)
    assert {row["round_id"] for row in captured} == {"one", "two"}
    assert all(row["run_id"] == "test-run" for row in captured)
    assert not any(word in events.read_text() for word in ("private", "secret", "api_key"))
    for row in captured:
        if row["event"] == "work":
            assert row["duration_ms"] == (row["end_ns"] - row["start_ns"]) / 1e6


def test_client_limit_produces_serial_and_parallel_waves(events):
    async def batch(limit):
        active = peak = 0

        async def respond(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.08)
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": '{"clue":"distant signal"}'}}],
                    "provider": "test", "model": "test-model",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                })
            finally:
                active -= 1

        settings = Settings(_env_file=None, llm_config=LLMConfig(
            name="test", provider="local", model="test", chat_url="http://stub/chat",
            max_concurrent_requests=limit,
        ))
        client = InferenceClient(settings)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = InferenceClient(settings, http_client=http)

            async def call(index):
                with latency.scope(round_id=f"limit-{limit}", phase="clues", turn_id=str(index)):
                    return await client.generate_structured(InferenceRequest(
                        prompt="test", agent=list_static_agent_configs()[0],
                    ), ClueDecision)

            started = perf_counter()
            results = await asyncio.gather(*(call(i) for i in range(4)))
            assert all(result.clue == "distant signal" for result, _ in results)
            return peak, perf_counter() - started

    serial_peak, serial_elapsed = asyncio.run(batch(1))
    parallel_peak, parallel_elapsed = asyncio.run(batch(4))
    assert serial_peak == 1
    assert parallel_peak == 4
    assert serial_elapsed > parallel_elapsed * 1.8
    captured = rows(events)
    http_events = [r for r in captured if r["event"] == "provider_http"]
    assert len(http_events) == 8
    assert all(r["duration_ms"] >= 60 for r in http_events)
    assert len({r["request_id"] for r in http_events}) == 8
    waits = [r for r in captured if r["event"] == "client_semaphore_wait"
             and r["round_id"] == "limit-1"]
    assert max(r["duration_ms"] for r in waits) > 150
    assert all(r["prompt_tokens"] == 10 for r in captured if r["event"] == "provider_usage")


@pytest.mark.parametrize("failure", ["rate_limit", "timeout", "invalid"])
def test_retry_telemetry_preserves_attempts_and_validation(events, monkeypatch, failure):
    from app.services.agents import inference
    monkeypatch.setattr(inference, "_retry_delay", lambda attempt: 0)

    async def run():
        count = 0

        async def respond(request):
            nonlocal count
            count += 1
            if count == 1:
                if failure == "timeout":
                    raise httpx.ReadTimeout("not exported")
                if failure == "rate_limit":
                    return httpx.Response(429, headers={"Retry-After": "0"})
                content = "{}"
            else:
                content = '{"clue":"distant signal"}'
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        settings = Settings(_env_file=None, llm_config=LLMConfig(
            name="test", provider="local", model="test", chat_url="http://stub/chat",
        ))
        client = InferenceClient(settings)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = InferenceClient(settings, http_client=http)
            with latency.scope(round_id="retry", turn_id="clue:a", phase="clues"):
                decision, _ = await client.generate_structured(InferenceRequest(
                    prompt="private prompt", agent=list_static_agent_configs()[0],
                ), ClueDecision)
            assert decision.clue == "distant signal"
            assert count == 2

    asyncio.run(run())
    captured = rows(events)
    assert len([r for r in captured if r["event"] == "provider_http"]) == 2
    assert any(r["event"] == ("validation_retry" if failure == "invalid" else "retry_backoff")
               for r in captured)
    assert "private prompt" not in events.read_text()


def test_cancelled_waiter_does_not_leak_capacity(events):
    async def run():
        semaphore = asyncio.Semaphore(0)

        async def wait():
            with latency.scope(round_id="cancelled", turn_id="turn"):
                async with latency.acquire(semaphore, "capacity_wait"):
                    pytest.fail("Cancelled waiter must not acquire")

        task = asyncio.create_task(wait())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert semaphore.locked()

    asyncio.run(run())
    assert rows(events)[0]["outcome"] == "cancelled"


@pytest.mark.parametrize("mode", ["silence", "deadline"])
def test_discussion_telemetry_keeps_silence_and_expired_turns(events, monkeypatch, mode):
    from datetime import UTC, datetime, timedelta

    from app.services.agents import discussion as chat
    from app.services.agents.debate_agent import DebateDecision, DebateMemory
    from app.services.game.state import RoundState

    async def run():
        agents = list_static_agent_configs()[:2]
        state = RoundState(
            id="timed-round", secret_word="satellite", imposter_hint="orbit",
            imposter_player_id=agents[0].id, status="discussion",
            playing_order=[a.id for a in agents], player_names_by_id={a.id: a.name for a in agents},
            prompt_technique="few_shot", current_player_index=2, turns=[],
            created_at=datetime.now(UTC), include_human=False,
        )
        state.discussion.deadline = datetime.now(UTC) + timedelta(seconds=.06)

        async def enter(self):
            return self  # Avoid TLS-context startup; this test isolates the scheduler.

        async def decision(*args):
            if mode == "deadline":
                await asyncio.Event().wait()
            return {"decision": DebateDecision(comment=None, memory=DebateMemory()),
                    "inference_mode": "test", "prompt_characters": 0}

        monkeypatch.setattr(chat, "run_turn", decision)
        settings = Settings(_env_file=None, discussion_max_concurrent_turns=1)
        chat.ensure_discussion_tasks(state, agents, settings)
        await asyncio.wait_for(state.discussion._expiry_task, 2)
        assert state.status == "ready_to_vote"
        await chat.shutdown_tasks()

    asyncio.run(run())
    captured = rows(events)
    turns = [row for row in captured if row["event"] == "discussion_turn"]
    assert len(turns) == 2
    assert {row["outcome"] for row in turns} == ({"silence"} if mode == "silence" else {"expired"})
    assert not any(row["event"] == "message_published" for row in captured)
    assert all(row["round_id"] == "timed-round" for row in turns)
