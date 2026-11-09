import asyncio
from datetime import UTC, datetime

import pytest

from app.core.config import Settings
from app.services.agents.inference import InferenceServiceError
from app.services.agents.runtime_agents import list_static_agent_configs
from app.services.game import engine, schemas, state


class FakeSession:
    def __init__(self):
        self.saved = []
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add(self, model):
        self.saved.append(model)

    def commit(self):
        self.commits += 1


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    state.ROUNDS.clear()
    state.ACTIVE_GENERATION_ROUNDS.clear()
    monkeypatch.setattr(engine, "SessionLocal", FakeSession)
    yield
    state.ROUNDS.clear()
    state.ACTIVE_GENERATION_ROUNDS.clear()


def settings():
    return Settings(_env_file=None, inference_mode="fake",
                    word_selection_mode="fixed")


def make_state(order, imposter):
    agents = list_static_agent_configs()
    result = state.RoundState(
        id="engine-test", secret_word="satellite", imposter_hint="orbit",
        imposter_player_id=imposter, status="generating_clues", playing_order=order,
        player_names_by_id={"human": "You", **{a.id: a.name for a in agents}},
        prompt_technique="few_shot", current_player_index=0, turns=[],
        created_at=datetime.now(UTC),
    )
    state.ROUNDS[result.id] = result
    return result


def test_create_lookup_and_human_clue_schedule_without_fastapi(monkeypatch):
    async def run():
        monkeypatch.setattr(engine.random, "shuffle", lambda ids: None)
        monkeypatch.setattr(engine, "choice", lambda ids: "human")
        jobs = []
        db = FakeSession()
        config = settings()
        response = await engine.create_round(
            None, config, db, schedule=lambda *args: jobs.append(args),
        )
        assert response.status == "generating_clues"
        assert response.visible_word is None
        assert jobs[0][0] is engine.continue_clue_generation
        assert jobs[0][1] == response.id
        await jobs[0][0](*jobs[0][1:])
        assert (await engine.get_round(response.id)).status == "awaiting_human_clue"
        result = await engine.submit_clue(
            response.id, schemas.SubmitClueRequest(clue="  orbiting object  "), db, config,
            schedule=lambda *args: jobs.append(args),
        )
        assert result.status == "discussion"
        assert len(jobs) == 2
        assert jobs[1][0] is engine.continue_clue_generation
        assert state.ROUNDS[result.id].human_clue == "orbiting object"
        assert db.saved[0].clue == "orbiting object"
        assert db.commits == 1
    asyncio.run(run())


def test_semantic_errors_without_http_objects(monkeypatch):
    async def run():
        with pytest.raises(engine.GameError, match="Unknown round") as failure:
            await engine.get_round("missing")
        assert failure.value.kind == "not_found"
        config = settings()
        current = make_state(["human"], "human")
        with pytest.raises(engine.GameError, match="not waiting") as failure:
            await engine.submit_clue(
                current.id, schemas.SubmitClueRequest(clue="test"), FakeSession(), config,
                schedule=lambda *args: None,
            )
        assert failure.value.kind == "conflict"
        monkeypatch.setattr(engine, "list_runtime_agent_configs", lambda *args: [])
        with pytest.raises(engine.GameError, match="No agents") as failure:
            await engine.create_round(None, config, FakeSession(), schedule=lambda *args: None)
        assert failure.value.kind == "configuration"
    asyncio.run(run())


def test_imposter_waits_and_clues_reveal_in_order(monkeypatch):
    async def run():
        agents = list_static_agent_configs()[:3]
        first, imposter, last = [a.id for a in agents]
        current = make_state([first, imposter, last], imposter)
        release_first = asyncio.Event()
        last_started = asyncio.Event()
        imposter_started = asyncio.Event()
        calls = []

        async def generate(round_state, agent, preceding, config, client=None):
            calls.append(agent.id)
            if agent.id == first:
                await release_first.wait()
            elif agent.id == last:
                last_started.set()
            else:
                assert [r.agent_id for r in preceding] == [first]
                assert current.turns[0].responses[0].agent_id == first
                imposter_started.set()
            return schemas.AgentTurnResponse(
                agent_id=agent.id, agent_name=agent.name,
                agent_response="a clue", inference_mode="fake",
            )

        monkeypatch.setattr(engine, "generate_agent_clue", generate)
        task = asyncio.create_task(engine.continue_clue_generation(current.id, agents, settings()))
        try:
            await asyncio.wait_for(last_started.wait(), timeout=2)
            assert not imposter_started.is_set()
            assert not current.turns or not current.turns[0].responses
            # An overlapping continuation must not start a second generation.
            await engine.continue_clue_generation(current.id, agents, settings())
            assert not imposter_started.is_set()
            release_first.set()
            await asyncio.wait_for(task, timeout=2)
            assert imposter_started.is_set()
            assert [r.agent_id for r in current.turns[0].responses] == current.playing_order
            assert sorted(calls) == sorted(current.playing_order)
            assert current.status == "discussion"
            assert current.id not in state.ACTIVE_GENERATION_ROUNDS
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_generation_failure_cancels_pending_work_and_clears_guard(monkeypatch):
    async def run():
        agents = list_static_agent_configs()[:2]
        current = make_state([a.id for a in agents] + ["human"], "human")
        pending_started = asyncio.Event()
        pending_cancelled = asyncio.Event()

        async def generate(round_state, agent, preceding, config, client=None):
            if agent.id == agents[0].id:
                await pending_started.wait()
                raise InferenceServiceError("failed")
            pending_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                pending_cancelled.set()

        monkeypatch.setattr(engine, "generate_agent_clue", generate)
        await asyncio.wait_for(
            engine.continue_clue_generation(current.id, agents, settings()), timeout=2,
        )
        assert pending_cancelled.is_set()
        assert current.status == "generation_failed"
        assert current.id not in state.ACTIVE_GENERATION_ROUNDS
    asyncio.run(run())


def test_round_names_are_unique_stable_and_independent_of_config(monkeypatch):
    async def run():
        agents = list_static_agent_configs()
        monkeypatch.setattr(engine.random, "sample", lambda pool, count: list(pool))
        first = await engine.create_round(None, settings(), FakeSession(), schedule=lambda *args: None)
        names = state.ROUNDS[first.id].player_names_by_id.copy()
        assert names["human"] == "Human"
        assert set(names) == {"human", *[a.id for a in agents]}
        assert len(set(names.values())) == len(names)
        assert all(names[a.id] != a.name for a in agents)
        assert {p.id: p.name for p in first.playing_order} == names
        monkeypatch.setattr(engine.random, "sample", lambda pool, count: list(reversed(pool)))
        second = await engine.create_round(None, settings(), FakeSession(), schedule=lambda *args: None)
        assert state.ROUNDS[second.id].player_names_by_id != names
        assert {p.id: p.name for p in (await engine.get_round(first.id)).playing_order} == names
        # A stale clue label cannot overwrite the authoritative roster.
        state.ROUNDS[first.id].turns = [schemas.TurnResponse(
            id="old", sequence=1, user_prompt="clue", created_at=datetime.now(UTC),
            responses=[schemas.AgentTurnResponse(agent_id=agents[0].id, agent_name="stale",
                       agent_response="a clue", inference_mode="fake")],
        )]
        assert engine.get_player_names_by_id_from_round(state.ROUNDS[first.id]) == names
    asyncio.run(run())


def test_round_names_flow_through_prompts_clues_discussion_votes_and_persistence(monkeypatch):
    import json
    from app.services.agents import debate_agent, discussion
    from app.services.agents.inference import InferenceClient, InferenceResult

    async def run():
        agents = list_static_agent_configs()
        current = make_state(["human", *[a.id for a in agents]], agents[0].id)
        current.player_names_by_id = {"human": "Human", **{
            a.id: f"RoundName{i}" for i, a in enumerate(agents)
        }}
        names = current.player_names_by_id
        requests = []

        async def generate(self, request, response_model, validate=None):
            requests.append(request)
            if response_model.__name__ == "ClueDecision":
                result = response_model(clue="distant object")
            else:
                context = json.loads(request.prompt.split("GAME_CONTEXT:\n")[1])
                assert context["your_name"] == names[context["your_id"]]
                assert {p["id"]: p["name"] for p in context["players"]} == names
                result = response_model(target_id="human")
            if validate:
                validate(result)
            return result, InferenceResult(text="", inference_mode="fake")

        monkeypatch.setattr(InferenceClient, "generate_structured", generate)
        previous = schemas.AgentTurnResponse(
            agent_id="human", agent_name="stale", agent_response="circles earth",
            inference_mode="human",
        )
        clue = await engine.generate_agent_clue(current, agents[0], [previous], settings(), client=InferenceClient(settings()))
        assert "Human: circles earth" in requests[0].prompt
        assert "stale" not in requests[0].prompt
        assert clue.agent_id == agents[0].id and clue.agent_name == names[agents[0].id]
        identity = json.loads(requests[0].prompt.splitlines()[1])
        assert identity["your_name"] == names[agents[0].id]
        assert {p["id"]: p["name"] for p in identity["players"]} == names
        current.turns = [schemas.TurnResponse(id="clues", sequence=1, user_prompt="clue",
                         responses=[clue], created_at=datetime.now(UTC))]
        current.discussion.deadline = datetime.now(UTC)
        db = FakeSession()
        discussion.publish_comment(current, "human", "Hello", "human", db)
        assert current.discussion.messages[0].player_name == "Human"
        assert db.saved[0].player_id == "human" and db.saved[0].player_name == "Human"
        context = json.loads(debate_agent.capture_snapshot(
            current, agents[0].id, "generation", debate_agent.AgentMemory(),
        ).context_json)
        assert context["your_name"] == names[agents[0].id]
        assert {p["id"]: p["name"] for p in context["players"]} == names
        assert context["clues"][0]["player_name"] == names[agents[0].id]
        current.status = "ready_to_vote"
        monkeypatch.setattr(engine, "list_runtime_agent_configs", lambda *args: agents)
        monkeypatch.setattr(engine, "get_runtime_agent_config", lambda *args: agents[0])
        result = await engine.vote(current.id, schemas.VoteRequest(agent_id=agents[0].id), db, settings())
        assert result.imposter_was == names[agents[0].id]
        assert result.voted_agent_name == names[agents[0].id]
        assert result.group_voted_player_name == "Human"
        assert all(v.voted_for == "human" and v.voter_agent_name == names[v.voter_agent_id]
                   for v in result.agent_votes)
        agent_rows = [row for row in db.saved if getattr(row, "voter_id", None) in {a.id for a in agents}]
        assert len(agent_rows) == len(agents)
        assert all(row.voter_name == names[row.voter_id] and row.voted_for_id == "human"
                   and row.voted_for_name == "Human" for row in agent_rows)
        assert all(count.player_name == names[count.player_id] for count in result.vote_counts)
    asyncio.run(run())
