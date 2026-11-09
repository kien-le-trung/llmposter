import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.services.game import engine
from app.services.game import schemas, state as game_state
from app.core.config import Settings
from app.db.models import DiscussionMessageModel
from app.db.session import Base, get_db
from app.main import create_app
from app.services.agents import discussion as chat
from app.services.agents.debate_agent import DebateDecision, DebateMemory
from app.services.agents.inference import InferenceClient, InferenceResult, InferenceServiceError
from app.services.agents.runtime_agents import list_static_agent_configs
from app.services.voting import voting


class FakeSession:
    saved = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add(self, model):
        self.saved.append(model)

    def commit(self):
        pass


def settings():
    return Settings(_env_file=None, inference_mode="fake", word_selection_mode="fixed")


def make_state():
    agents = list_static_agent_configs()
    ids = ["human", *[agent.id for agent in agents]]
    state = game_state.RoundState(
        id="discussion-test", secret_word="satellite", imposter_hint="orbit",
        imposter_player_id=agents[0].id, status="generating_clues", playing_order=ids,
        player_names_by_id={"human": "You", **{a.id: a.name for a in agents}},
        prompt_technique="few_shot", current_player_index=0, turns=[],
        created_at=datetime.now(UTC), human_clue="circles earth",
        discussion=chat.DiscussionState(),
    )
    state.completed_clues = {
        pid: schemas.AgentTurnResponse(agent_id=pid, agent_name=state.player_names_by_id[pid],
                                      agent_response="circles earth", inference_mode="fake")
        for pid in ids
    }
    engine.reveal_available_clues(state)
    return state


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    game_state.ROUNDS.clear()
    FakeSession.saved = []
    monkeypatch.setattr(engine, "SessionLocal", FakeSession)
    monkeypatch.setattr(chat, "SessionLocal", FakeSession)


@pytest.fixture
def client():
    app = create_app(settings())
    app.dependency_overrides[get_db] = lambda: FakeSession()
    with TestClient(app) as client:
        yield client


def install_state():
    state = make_state()
    game_state.ROUNDS[state.id] = state
    return state


def test_open_only_after_final_clue_and_deadline_does_not_reset():
    state = make_state()
    deadline = state.discussion.deadline
    assert state.status == "discussion"
    assert 89 < (deadline - datetime.now(UTC)).total_seconds() <= 90
    engine.reveal_available_clues(state)
    assert state.discussion.deadline == deadline
    state.current_player_index = 0
    state.turns = []
    state.completed_clues.pop("human")
    engine.reveal_available_clues(state)
    assert state.status == "awaiting_human_clue"


def test_all_rounds_require_discussion(client, monkeypatch):
    monkeypatch.setattr(engine.random, "shuffle", lambda ids: None)
    response = client.post("/rounds", json={"discussion_enabled": False})
    round_id = response.json()["id"]
    assert "enabled" not in response.json()["discussion"]
    client.post(f"/rounds/{round_id}/clue", json={"clue": "space object"})
    assert client.get(f"/rounds/{round_id}").json()["status"] == "discussion"
    assert client.post(f"/rounds/{round_id}/vote", json={"agent_id": "agent_a"}).status_code == 409
    ai = client.post("/rounds", json={"include_human": False}).json()
    assert client.get(f'/rounds/{ai["id"]}').json()["status"] == "discussion"
    assert client.post(f'/rounds/{ai["id"]}/discussion', json={"text": "hello"}).status_code == 409
    assert client.post(f'/rounds/{ai["id"]}/clue', json={"clue": "hello"}).status_code == 409


def test_human_can_send_without_permission(client):
    state = install_state()
    url = f"/rounds/{state.id}/discussion"
    for text in ["First comment", "Second comment"]:
        assert client.post(url, json={"text": text}).status_code == 200
    assert [m.sequence for m in state.discussion.messages] == [1, 2]
    assert all(m.player_id == "human" for m in state.discussion.messages)
    assert client.post(url, json={"text": " "}).status_code == 422
    assert client.post(url, json={"text": "x" * 501}).status_code == 422
    assert client.post(url, json={"action": "raise"}).status_code == 422
    state.discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
    assert client.post(url, json={"text": "late"}).status_code == 409
    assert len(state.discussion.messages) == 2


def test_read_enforces_deadline_and_hides_private_state(client):
    state = install_state()
    state.imposter_player_id = "human"
    state.discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
    response = client.get(f"/rounds/{state.id}")
    assert response.json()["status"] == "ready_to_vote"
    assert "satellite" not in response.text
    assert "raised_player_ids" not in response.text
    assert "active_speaker_id" not in response.text
    assert "imposter_player_id" not in response.text


def test_context_allowlist_for_both_roles():
    state = make_state()
    imposter = json.loads(chat.player_context(state, "agent_a"))
    ordinary = json.loads(chat.player_context(state, "agent_b"))
    assert imposter["your_hint"] == "orbit"
    assert "satellite" not in json.dumps(imposter)
    assert "your_word" not in imposter
    assert ordinary["your_word"] == "satellite"
    assert "orbit" not in json.dumps(ordinary)
    for context in [imposter, ordinary]:
        assert "imposter_player_id" not in context
        assert "completed_clues" not in context
        assert "voted_agent_id" not in context
        assert len(context["clues"]) == 5


def test_timer_closes_without_browser_requests():
    async def run():
        state = make_state()
        state.discussion.deadline = datetime.now(UTC) + timedelta(milliseconds=10)
        await chat.close_at_deadline(state)
        assert state.status == "ready_to_vote"

    asyncio.run(run())


def test_discussion_votes_use_public_transcript_and_validate_targets(monkeypatch):
    async def run():
        state = make_state()
        chat.publish_comment(state, "human", "That explanation makes sense.", "human", FakeSession())
        agents = list_static_agent_configs()
        prompts = []

        async def generate(self, request, response_model, validate=None, **kwargs):
            context = json.loads(request.prompt.split("GAME_CONTEXT:\n")[1])
            prompts.append(context)
            for invalid in [request.agent.id, "unknown"]:
                with pytest.raises(ValueError):
                    validate(chat.DiscussionVote(target_id=invalid))
            validate(chat.DiscussionVote(target_id="human"))
            return chat.DiscussionVote(target_id="human"), InferenceResult(text="", inference_mode="openrouter")

        monkeypatch.setattr(InferenceClient, "generate_structured", generate)
        result = await voting._build_agent_votes(state, agents, settings())
        assert len(result) == 4
        assert all(v.inference_mode == "openrouter" and v.voted_for == "human" for v in result)
        assert all(p["discussion"][0]["text"] == "That explanation makes sense." for p in prompts)
        assert all("votes" not in p and "human_vote" not in p for p in prompts)

    asyncio.run(run())


def test_failed_vote_is_retryable_and_keeps_round_open(client, monkeypatch):
    state = install_state()
    state.status = "ready_to_vote"

    async def fail(*args, **kwargs):
        raise InferenceServiceError("Both providers failed")

    monkeypatch.setattr(InferenceClient, "generate_structured", fail)
    response = client.post(f"/rounds/{state.id}/vote", json={"agent_id": "agent_a"})
    assert response.status_code == 503
    assert state.status == "ready_to_vote"
    assert not state.discussion._voting


def test_fake_discussion_votes_complete_round_with_existing_tally(client):
    state = install_state()
    chat.publish_comment(state, "human", "Please explain that clue.", "human", FakeSession())
    state.discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
    response = client.post(f"/rounds/{state.id}/vote", json={"agent_id": "agent_a"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["agent_votes"]) == 4
    assert all(vote["inference_mode"] == "fake" for vote in body["agent_votes"])
    assert body["group_voted_player_id"] == "human"
    assert state.status == "complete"


def test_published_messages_persist_in_real_database():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    state = make_state()
    with Session(engine) as db:
        chat.publish_comment(state, "human", "Can you explain?", "human", db)
        message = db.scalar(select(DiscussionMessageModel))
        assert message.text == "Can you explain?"
        assert message.sequence == 1
        assert message.round_id == state.id




@pytest.mark.parametrize("mode", ["comment", "silence", "failure", "late"])
def test_ai_loop_comments_silence_failure_and_expiry(monkeypatch, mode):
    async def run():
        state = make_state()
        agent = list_static_agent_configs()[0]
        contexts = []
        config = settings()
        config.discussion_cooldown_seconds = 0.01
        config.discussion_debounce_seconds = 0.001
        config.discussion_silence_seconds = 0.02

        async def generate(self, request, response_model, validate=None, **kwargs):
            contexts.append(request.prompt)
            if len(contexts) == 1:
                chat.publish_comment(state, "human", "Explain?", "human", FakeSession())
                if mode == "failure":
                    raise InferenceServiceError("unavailable")
                if mode == "late":
                    state.discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
            else:
                state.discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
            decision = DebateDecision(
                comment=None if mode == "silence" else "My response", memory=DebateMemory(),
            )
            validate(decision)
            return decision, InferenceResult(text="", inference_mode="openrouter")

        monkeypatch.setattr(InferenceClient, "generate_structured", generate)
        await asyncio.wait_for(chat.ai_comment_loop(state, agent, config), timeout=2)
        assert state.status == "ready_to_vote"
        assert [m.player_id for m in state.discussion.messages] == (
            ["human", agent.id] if mode == "comment" else ["human"]
        )
        memory = state.discussion._agents[agent.id].memory
        assert memory.version == (1 if mode in {"comment", "silence"} else 0)
        if mode != "late":
            assert "Explain?" in contexts[1]
        else:
            assert len(contexts) == 1
    asyncio.run(run())


def test_independent_loops_allow_human_send_and_shutdown(monkeypatch):
    async def run():
        state = install_state()
        agents = list_static_agent_configs()
        started = set()
        all_started = asyncio.Event()
        release = asyncio.Event()
        in_flight = set()

        async def generate(self, request, response_model, **kwargs):
            assert request.agent.id not in in_flight
            in_flight.add(request.agent.id)
            started.add(request.agent.id)
            if len(started) == len(agents):
                all_started.set()
            await release.wait()
            in_flight.remove(request.agent.id)
            return DebateDecision(comment="A comment", memory=DebateMemory()), InferenceResult(
                text="", inference_mode="fake",
            )

        monkeypatch.setattr(InferenceClient, "generate_structured", generate)
        chat.ensure_discussion_tasks(state, agents, settings())
        first_tasks = dict(state.discussion._ai_tasks)
        chat.ensure_discussion_tasks(state, agents, settings())
        assert state.discussion._ai_tasks == first_tasks
        # HTTP client construction can take several seconds on Windows.
        await asyncio.wait_for(all_started.wait(), timeout=10)
        response = await engine.submit_discussion_comment(
            state.id, schemas.SubmitDiscussionComment(text="Human speaks now"), FakeSession(),
        )
        assert response.discussion.messages[0].player_id == "human"
        release.set()
        # Let all independent responses publish before stopping the loops.
        for _ in range(200):
            if len(state.discussion.messages) == 5:
                break
            await asyncio.sleep(0.005)
        assert len(state.discussion.messages) == 5
        assert [m.sequence for m in state.discussion.messages] == list(range(1, 6))
        await chat.shutdown_tasks()
        assert all(task.done() for task in first_tasks.values())

    asyncio.run(run())


def test_vote_persistence_failure_rolls_back_and_can_retry(client, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError

    state = install_state()
    state.status = "ready_to_vote"
    rolled_back = []
    original = FakeSession.commit

    def fail(self):
        raise SQLAlchemyError("write failed")

    monkeypatch.setattr(FakeSession, "commit", fail)
    monkeypatch.setattr(FakeSession, "rollback", lambda self: rolled_back.append(True), raising=False)
    response = client.post(f"/rounds/{state.id}/vote", json={"agent_id": "agent_a"})
    assert response.status_code == 503
    assert state.status == "ready_to_vote"
    assert state.voted_agent_id is None
    assert not state.discussion._voting
    assert rolled_back == [True]
    monkeypatch.setattr(FakeSession, "commit", original)
    assert client.post(f"/rounds/{state.id}/vote", json={"agent_id": "agent_a"}).status_code == 200


def test_concurrent_vote_is_rejected_until_first_commits(monkeypatch):
    async def run():
        state = install_state()
        state.status = "ready_to_vote"
        started, release = asyncio.Event(), asyncio.Event()
        original = voting._build_agent_votes

        async def blocked(*args, **kwargs):
            started.set()
            await release.wait()
            return await original(*args)

        monkeypatch.setattr(voting, "_build_agent_votes", blocked)
        payload = schemas.VoteRequest(agent_id="agent_a")
        task = asyncio.create_task(engine.vote(state.id, payload, FakeSession(), settings()))
        try:
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(engine.GameError, match="Voting already"):
                await engine.vote(state.id, payload, FakeSession(), settings())
            release.set()
            await task
            assert state.status == "complete"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_tied_vote_has_no_elimination():
    agents = list_static_agent_configs()[:2]
    votes = [voting.AgentVoteResponse(
        voter_agent_id=a.id, voter_agent_name=a.name,
        voted_for=agents[1 - i].id, inference_mode="fake",
    ) for i, a in enumerate(agents)]
    counts, target = voting._tally_round_votes(None, votes, {a.id: a.name for a in agents})
    assert target is None
    assert [count.votes for count in counts] == [1, 1]


def test_removed_fields_are_absent_from_openapi(client):
    components = client.get("/openapi.json").json()["components"]["schemas"]
    assert "discussion_enabled" not in components["CreateRoundRequest"]["properties"]
    assert "enabled" not in components["DiscussionState"]["properties"]
    assert "human_clue" not in components["VoteRequest"]["properties"]


@pytest.fixture(autouse=True)
def application_lifespans(monkeypatch):
    from contextlib import ExitStack
    original = TestClient
    with ExitStack() as stack:
        def managed(*args, **kwargs):
            return stack.enter_context(original(*args, **kwargs))
        monkeypatch.setitem(globals(), "TestClient", managed)
        yield
