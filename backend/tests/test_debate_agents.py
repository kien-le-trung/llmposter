"""Debate graph and coordinator invariants, with controlled inference and clocks."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from langsmith.run_helpers import get_tracing_context

from app.core.config import Settings
from app.services.agents import debate_agent as graph
from app.services.agents import discussion as chat
from app.services.agents.inference import InferenceClient, InferenceResult, InferenceServiceError
from app.services.agents.runtime_agents import list_static_agent_configs
from app.services.game import engine, state as game_state


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add(self, row):
        pass

    def commit(self):
        pass


def config(**overrides):
    return Settings(_env_file=None, inference_mode="fake", **overrides)


def make_state(round_id="debate-test"):
    agents = list_static_agent_configs()
    players = ["human", *[a.id for a in agents]]
    state = game_state.RoundState(
        id=round_id, secret_word="satellite", imposter_hint="orbit",
        imposter_player_id=agents[0].id, status="discussion", playing_order=players,
        player_names_by_id={"human": "You", **{a.id: a.name for a in agents}},
        prompt_technique="few_shot", current_player_index=len(players), turns=[],
        created_at=datetime.now(UTC),
        discussion=chat.DiscussionState(
            deadline=datetime.now(UTC) + timedelta(seconds=90),
        ),
    )
    return state


def publish(state, text="public", player="human"):
    chat.publish_comment(state, player, text, "human", FakeSession())


def turn_result(comment=None, summary="updated"):
    return {"decision": graph.DebateDecision(
        comment=comment, memory=graph.DebateMemory(summary=summary),
    ), "inference_mode": "fake", "prompt_characters": 100}


async def eventually(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(chat, "SessionLocal", FakeSession)


def test_snapshot_is_detached_bounded_and_player_specific():
    state = make_state()
    agents = list_static_agent_configs()
    for i in range(40):
        publish(state, text=f"message {i + 1}")
    memory = graph.AgentMemory(processed_through=10)
    snapshot = graph.capture_snapshot(state, agents[0].id, "one", memory)
    context = json.loads(snapshot.context_json)
    assert "your_word" not in context
    assert context["your_hint"] == "orbit"
    assert context["your_role"] == "imposter"
    assert [m["sequence"] for m in context["recent_messages"]] == list(range(3, 11))
    assert [m["sequence"] for m in context["new_messages"]] == list(range(11, 27))
    assert snapshot.batch_endpoint == 26
    assert snapshot.catching_up
    assert "player_name" in context["new_messages"][0]
    memory.memory.summary = "later change"
    publish(state, text="arrived later")
    assert "later change" not in snapshot.context_json
    assert "arrived later" not in snapshot.context_json
    player_context = json.loads(graph.capture_snapshot(
        state, agents[1].id, "two", graph.AgentMemory(),
    ).context_json)
    assert player_context["your_word"] == "satellite"
    assert "your_hint" not in player_context


@pytest.mark.parametrize("comment", [None, "A useful comment"])
def test_graph_uses_one_structured_call_and_keeps_memory_proposed(monkeypatch, comment):
    async def run():
        state = make_state()
        agent = list_static_agent_configs()[0]
        memory = graph.AgentMemory()
        snapshot = graph.capture_snapshot(state, agent.id, "one", memory)
        calls = []
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        async def generate(self, request, response_model, validate):
            assert get_tracing_context()["enabled"] is False
            calls.append(request)
            decision = turn_result(comment)["decision"]
            validate(decision)
            return decision, InferenceResult(text="", inference_mode="fake")

        monkeypatch.setattr(InferenceClient, "generate_structured", generate)
        client = InferenceClient(config())
        result = await graph.run_turn(snapshot, agent, client)
        assert len(calls) == 1
        assert result["decision"].comment == comment
        assert result["prompt_characters"] == len(calls[0].prompt)
        assert memory.version == 0 and memory.memory.summary == ""
        assert not state.discussion.messages
    asyncio.run(run())


def test_real_graph_fake_inference_and_catch_up_batches():
    async def run():
        state = make_state()
        agent = list_static_agent_configs()[0]
        for i in range(35):
            publish(state, str(i))
        runtime = chat.AgentRuntime()
        seen = []
        client = InferenceClient(config())
        for index in range(3):
            snapshot = graph.capture_snapshot(state, agent.id, str(index), runtime.memory)
            seen.extend(m["sequence"] for m in json.loads(snapshot.context_json)["new_messages"])
            result = await graph.run_turn(snapshot, agent, client)
            if index < 2:
                assert result["decision"].comment is None
            runtime.generation_id = snapshot.generation_id
            assert chat.commit_turn(state, runtime, snapshot, result) in {"silence", "published"}
        assert seen == list(range(1, 36))
        assert runtime.memory.processed_through == 35
        assert len(state.discussion.messages) == 36
    asyncio.run(run())


def test_memory_validation_bounds_and_evidence():
    state = make_state()
    agent = list_static_agent_configs()[0]
    publish(state)
    snapshot = graph.capture_snapshot(state, agent.id, "one", graph.AgentMemory())
    for sequence in (0, 2):
        decision = graph.DebateDecision(comment=None, memory=graph.DebateMemory(
            observations=[graph.MemoryEntry(text="Human said something", player_id="human",
                                            evidence=[sequence])],
        ))
        with pytest.raises(ValueError, match="received messages"):
            graph.validate_decision(snapshot, decision)
    with pytest.raises(ValueError, match="participant"):
        graph.validate_decision(snapshot, graph.DebateDecision(
            comment=None, memory=graph.DebateMemory(
                suspicions=[graph.MemoryEntry(text="Unknown player", player_id="unknown")],
            ),
        ))
    with pytest.raises(ValueError, match="evidence"):
        graph.validate_decision(snapshot, graph.DebateDecision(
            comment=None, memory=graph.DebateMemory(
                observations=[graph.MemoryEntry(text="Claim", player_id="human")],
            ),
        ))
    with pytest.raises(ValueError, match="attributed author"):
        graph.validate_decision(snapshot, graph.DebateDecision(
            comment=None, memory=graph.DebateMemory(observations=[
                graph.MemoryEntry(text="Wrong attribution", player_id=agent.id, evidence=[1]),
            ]),
        ))
    with pytest.raises(ValidationError):
        graph.DebateMemory(summary="x" * 1201)
    with pytest.raises(ValidationError):
        graph.MemoryEntry(text="x" * 201, player_id="human")
    with pytest.raises(ValidationError):
        graph.DebateMemory(questions=[graph.MemoryEntry(text="why", player_id="human")] * 7)


def test_commit_accepts_older_snapshot_rejects_duplicate_and_isolates_memory():
    async def run():
        state = make_state()
        other_round = make_state("other")
        agent = list_static_agent_configs()[0]
        runtime = chat.AgentRuntime(generation_id="one")
        snapshot = graph.capture_snapshot(state, agent.id, "one", runtime.memory)
        publish(state, "arrived during inference")
        assert chat.commit_turn(state, runtime, snapshot, turn_result("reply")) == "published"
        assert runtime.memory.version == 1
        assert runtime.memory.processed_through == 0
        assert chat.commit_turn(state, runtime, snapshot, turn_result("duplicate")) == (
            "superseded_or_duplicate"
        )
        assert len(state.discussion.messages) == 2
        assert chat.AgentRuntime().memory.version == 0
        assert not other_round.discussion._agents
        game_state.ROUNDS[state.id] = state
        try:
            response = (await engine.get_round(state.id)).model_dump_json()
            assert "updated" not in response
            assert "processed_through" not in response
        finally:
            game_state.ROUNDS.pop(state.id)
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["superseded", "version", "expired", "persistence"])
def test_rejected_or_failed_publication_never_commits_memory(monkeypatch, failure):
    async def run():
        state = make_state()
        agent = list_static_agent_configs()[0]
        runtime = chat.AgentRuntime(generation_id="one")
        snapshot = graph.capture_snapshot(state, agent.id, "one", runtime.memory)
        if failure == "superseded":
            runtime.generation_id = "two"
        elif failure == "version":
            snapshot = replace(snapshot, memory_version=99)
        elif failure == "expired":
            state.discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
        else:
            class BrokenSession(FakeSession):
                def commit(self):
                    raise RuntimeError("database failure")
            monkeypatch.setattr(chat, "SessionLocal", BrokenSession)
        if failure == "persistence":
            with pytest.raises(RuntimeError):
                chat.commit_turn(state, runtime, snapshot, turn_result("reply"))
        else:
            assert chat.commit_turn(state, runtime, snapshot, turn_result("reply")) != "published"
        assert runtime.memory.version == 0
        assert runtime.memory.memory.summary == ""
        assert not state.discussion.messages
    asyncio.run(run())


def test_scheduler_fixed_debounce_cooldown_and_silence():
    state = make_state()
    agent = list_static_agent_configs()[0]
    runtime = chat.AgentRuntime(initial=False, last_completed=100)
    state.discussion._agents[agent.id] = runtime
    state.discussion._last_message_at = 100
    settings = config()
    assert chat.next_turn_delay(state, agent.id, runtime, settings, 101) == 9
    publish(state)
    runtime.pending_since = 101
    assert chat.next_turn_delay(state, agent.id, runtime, settings, 101) == 1
    # A second arrival must not extend the first notification's debounce deadline.
    publish(state)
    assert runtime.pending_since == 101
    assert chat.next_turn_delay(state, agent.id, runtime, settings, 102) == 0
    runtime.pending_since = None
    runtime.memory.processed_through = 2
    publish(state, player=agent.id)
    assert runtime.pending_since is None
    state.discussion._last_message_at = 105
    assert chat.next_turn_delay(state, agent.id, runtime, settings, 106) == 9
    runtime.retry = True
    assert chat.next_turn_delay(state, agent.id, runtime, settings, 101) == 1


def test_concurrency_is_shared_across_rounds_and_deadline_cancels_waiters(monkeypatch):
    async def run():
        settings = config(discussion_max_concurrent_turns=1)
        agent = list_static_agent_configs()[0]
        first, second = make_state("one"), make_state("two")
        started = asyncio.Event()
        cancelled = asyncio.Event()
        calls = []

        async def controlled(snapshot, agent, client):
            calls.append(snapshot.round_id)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(chat, "run_turn", controlled)
        chat.ensure_discussion_tasks(first, [agent], settings)
        await asyncio.wait_for(started.wait(), 2)
        chat.ensure_discussion_tasks(second, [agent], settings)
        await asyncio.sleep(0.01)
        assert calls == ["one"]
        assert second.discussion._agents[agent.id].generation_id is None
        second.discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
        await asyncio.wait_for(chat.close_at_deadline(second), 2)
        assert all(task.done() for task in second.discussion._ai_tasks.values())
        first.discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
        await asyncio.wait_for(chat.close_at_deadline(first), 2)
        assert cancelled.is_set()
        assert all(task.done() for task in first.discussion._ai_tasks.values())
        assert not first.discussion.messages and not second.discussion.messages
        await chat.shutdown_tasks()
        assert not chat.TASKS
    asyncio.run(run())


def test_failure_retries_and_silence_timer_reconsiders(monkeypatch, caplog):
    caplog.set_level("INFO", logger=chat.__name__)
    async def run():
        state = make_state()
        agent = list_static_agent_configs()[0]
        settings = config(discussion_cooldown_seconds=0.01, discussion_silence_seconds=0.02)
        calls = []
        done = asyncio.Event()

        async def controlled(snapshot, agent, client):
            calls.append(snapshot)
            if len(calls) == 1:
                raise InferenceServiceError("private provider error")
            if len(calls) == 3:
                done.set()
            return turn_result()

        monkeypatch.setattr(chat, "run_turn", controlled)
        chat.ensure_discussion_tasks(state, [agent], settings)
        try:
            await asyncio.wait_for(done.wait(), 2)
            await eventually(lambda: state.discussion._agents[agent.id].memory.version == 2)
            assert calls[0].memory_version == calls[1].memory_version == 0
            assert calls[2].memory_version == 1
            assert not state.discussion.messages
            records = [r.debate for r in caplog.records if hasattr(r, "debate")]
            assert records[0]["outcome"] == "failure"
            assert records[1]["outcome"] == "silence"
            assert "queue_wait_ms" in records[1] and "inference_ms" in records[1]
            assert records[1]["prompt_characters"] == 100
            assert "private provider error" not in caplog.text
            assert "updated" not in caplog.text
        finally:
            await chat.shutdown_tasks()
    asyncio.run(run())


def test_graph_propagates_failure_without_mutating_input(monkeypatch):
    async def run():
        state = make_state()
        agent = list_static_agent_configs()[0]
        memory = graph.AgentMemory()
        snapshot = graph.capture_snapshot(state, agent.id, "one", memory)

        async def fail(*args, **kwargs):
            raise InferenceServiceError("provider unavailable")

        monkeypatch.setattr(InferenceClient, "generate_structured", fail)
        client = InferenceClient(config())
        with pytest.raises(InferenceServiceError):
            await graph.run_turn(snapshot, agent, client)
        assert memory == graph.AgentMemory()
    asyncio.run(run())


@pytest.mark.parametrize("field,value", [
    ("discussion_debounce_seconds", 0), ("discussion_cooldown_seconds", -1),
    ("discussion_silence_seconds", 0), ("discussion_max_concurrent_turns", 0),
])
def test_scheduling_settings_require_positive_bounds(field, value):
    with pytest.raises(ValidationError):
        config(**{field: value})



def test_arrivals_during_inference_are_coalesced_without_skipping(monkeypatch):
    async def run():
        state = make_state()
        agent = list_static_agent_configs()[0]
        settings = config(discussion_debounce_seconds=0.01,
                          discussion_cooldown_seconds=0.01,
                          discussion_silence_seconds=60)
        snapshots = []
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()
        release_second = asyncio.Event()

        async def controlled(snapshot, agent, client):
            snapshots.append(snapshot)
            if len(snapshots) == 1:
                first_started.set()
                await release_first.wait()
            else:
                second_started.set()
                await release_second.wait()
            return turn_result("My comment")

        monkeypatch.setattr(chat, "run_turn", controlled)
        chat.ensure_discussion_tasks(state, [agent], settings)
        try:
            await asyncio.wait_for(first_started.wait(), 2)
            async with state.discussion._lock:
                publish(state, "One")
                publish(state, "Two")
                publish(state, "Three")
            assert state.discussion._agents[agent.id].memory.version == 0
            release_first.set()
            await asyncio.wait_for(second_started.wait(), 2)
            assert snapshots[0].conversation_sequence == 0
            assert snapshots[1].processed_through == 0
            assert snapshots[1].batch_endpoint == 4  # Three human messages plus its own reply.
            new = json.loads(snapshots[1].context_json)["new_messages"]
            assert [message["sequence"] for message in new] == [1, 2, 3, 4]
            release_second.set()
            await eventually(lambda: state.discussion._agents[agent.id].memory.version == 2)
            await asyncio.sleep(0.03)
            assert len(snapshots) == 2  # Its second comment does not trigger a third request.
            assert state.discussion._agents[agent.id].memory.processed_through == 4
        finally:
            await chat.shutdown_tasks()
    asyncio.run(run())


def test_recent_statements_stay_bounded_and_new_rounds_start_empty():
    async def run():
        state = make_state()
        agent = list_static_agent_configs()[0]
        runtime = chat.AgentRuntime()
        for i in range(12):
            snapshot = graph.capture_snapshot(state, agent.id, str(i), runtime.memory)
            runtime.generation_id = str(i)
            chat.commit_turn(state, runtime, snapshot, turn_result(f"comment {i}"))
        assert runtime.memory.recent_statements == [f"comment {i}" for i in range(4, 12)]
        assert len(state.discussion.messages) == 12
        assert chat.AgentRuntime().memory == graph.AgentMemory()
    asyncio.run(run())
