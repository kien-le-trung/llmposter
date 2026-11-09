from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from app.services.agents import clue_agent
import pytest

from app.services.game import engine
from app.services.game import state as game_state
from app.core.config import Settings
from app.db.session import get_db
from app.main import create_app
from app.services.agents.inference import InferenceResult
from app.services.agents.runtime_agents import list_static_agent_configs as list_agent_configs
from app.services.voting import voting


def setup_function() -> None:
    game_state.ROUNDS.clear()


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def add(self, model) -> None:
        pass

    def commit(self) -> None:
        pass


def get_fake_db():
    with FakeSession() as db:
        yield db


@pytest.fixture(autouse=True)
def use_fake_session(monkeypatch) -> None:
    monkeypatch.setattr(engine, "SessionLocal", FakeSession)
    monkeypatch.setattr(engine.chat, "SessionLocal", FakeSession)


def build_test_settings(**overrides) -> Settings:
    settings = Settings(_env_file=None)
    settings.inference_mode = "fake"
    settings.word_selection_mode = "fixed"
    settings.fixed_secret_word = "satellite"
    settings.fixed_imposter_hint = "orbit"
    settings.clue_prompt_technique = "few_shot"
    for name, value in overrides.items():
        setattr(settings, name, value)

    return settings


def build_test_client(**settings_overrides) -> TestClient:
    app = create_app(build_test_settings(**settings_overrides))
    app.dependency_overrides[get_db] = get_fake_db
    return TestClient(app)


def set_playing_order(monkeypatch, player_ids: list[str]) -> None:
    def use_order(order: list[str]) -> None:
        order[:] = player_ids

    monkeypatch.setattr(engine.random, "shuffle", use_order)


@pytest.mark.parametrize("name", ["", "   ", "x" * 41])
def test_create_round_rejects_invalid_human_name(name):
    response = build_test_client().post("/rounds", json={"human_name": name})
    assert response.status_code == 422
    assert not game_state.ROUNDS


def test_entered_name_reaches_round_roster_and_clue(monkeypatch):
    monkeypatch.setattr(engine.random, "sample", lambda pool, count: list(pool))
    monkeypatch.setattr(engine.random, "shuffle", lambda order: None)
    monkeypatch.setattr(engine, "choice", lambda ids: "human")
    client = build_test_client()
    # Alex would otherwise be assigned to the first agent.
    response = client.post("/rounds", json={"human_name": "  aLeX  "})
    assert response.status_code == 201
    body = response.json()
    names = {p["id"]: p["name"] for p in body["playing_order"]}
    assert names["human"] == "aLeX"
    assert len({n.casefold() for n in names.values()}) == len(names)
    round_id = body["id"]
    assert game_state.ROUNDS[round_id].player_names_by_id == names
    response = client.post(f"/rounds/{round_id}/clue", json={"clue": "orbiting object"})
    assert response.status_code == 200
    human = next(r for turn in response.json()["turns"] for r in turn["responses"]
                 if r["agent_id"] == "human")
    assert human["agent_name"] == "aLeX"


def test_create_round_selects_random_word_when_enabled(monkeypatch) -> None:
    agent_ids = [agent.id for agent in list_agent_configs()]
    monkeypatch.setattr(engine, "select_random_word", lambda: ("forest", "Trees and shade"))
    set_playing_order(monkeypatch, [*agent_ids, game_state.HUMAN_PLAYER_ID])
    client = build_test_client(word_selection_mode="random")

    response = client.post("/rounds", json={})

    assert response.status_code == 201
    body = response.json()
    assert game_state.ROUNDS[body["id"]].secret_word == "forest"
    assert game_state.ROUNDS[body["id"]].imposter_hint == "Trees"


def test_create_round_payload_word_overrides_random_mode(monkeypatch) -> None:
    agent_ids = [agent.id for agent in list_agent_configs()]
    monkeypatch.setattr(engine, "select_random_word", lambda: ("forest", "Trees and shade"))
    set_playing_order(monkeypatch, [*agent_ids, game_state.HUMAN_PLAYER_ID])
    client = build_test_client(word_selection_mode="random")

    response = client.post(
        "/rounds",
        json={"secret_word": "satellite", "imposter_hint": "Space and signals"},
    )

    assert response.status_code == 201
    body = response.json()
    assert game_state.ROUNDS[body["id"]].secret_word == "satellite"
    assert game_state.ROUNDS[body["id"]].imposter_hint == "Space"


def test_create_round_hides_word_when_human_is_imposter(monkeypatch) -> None:
    agent_ids = [agent.id for agent in list_agent_configs()]
    monkeypatch.setattr(engine, "choice", lambda player_ids: game_state.HUMAN_PLAYER_ID)
    set_playing_order(monkeypatch, [*agent_ids, game_state.HUMAN_PLAYER_ID])
    client = build_test_client()

    response = client.post(
        "/rounds",
        json={"secret_word": "satellite", "imposter_hint": "Space and signals"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["user_role"] == "imposter"
    assert body["visible_word"] is None
    assert body["imposter_hint"] == "Space"


def test_create_round_reveals_word_when_human_is_player(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    agent_ids = [agent.id for agent in list_agent_configs()]
    monkeypatch.setattr(engine, "choice", lambda player_ids: first_agent_id)
    set_playing_order(monkeypatch, [*agent_ids, game_state.HUMAN_PLAYER_ID])
    client = build_test_client()

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    body = response.json()
    assert body["user_role"] == "player"
    assert body["visible_word"] == "satellite"
    assert body["imposter_hint"] is None


def test_create_round_rejects_unknown_prompt_technique() -> None:
    client = build_test_client()

    response = client.post(
        "/rounds",
        json={"secret_word": "satellite", "prompt_technique": "missing"},
    )

    assert response.status_code == 422
    assert "Unknown prompt technique" in response.json()["detail"]


def test_prompt_technique_override_is_stored_and_reused_after_human_clue(
    monkeypatch,
) -> None:
    captured_techniques: list[str | None] = []
    agents = list_agent_configs()

    def record_non_imposter_strategy(technique=None):
        captured_techniques.append(technique)
        return {"name": "Recorded", "prompt": "Write a recorded clue.\n"}

    async def record_prompt(self, request):
        return InferenceResult(text='{"clue":"clue"}', inference_mode="fake")

    monkeypatch.setattr(clue_agent, "assign_non_imposter_clue_strategy", record_non_imposter_strategy)
    monkeypatch.setattr(engine.InferenceClient, "generate", record_prompt)
    monkeypatch.setattr(engine, "choice", lambda player_ids: game_state.HUMAN_PLAYER_ID)
    set_playing_order(monkeypatch, [game_state.HUMAN_PLAYER_ID, *[agent.id for agent in agents]])
    client = build_test_client()

    create_response = client.post(
        "/rounds",
        json={"secret_word": "satellite", "prompt_technique": "meta"},
    )
    round_id = create_response.json()["id"]
    response = client.post(f"/rounds/{round_id}/clue", json={"clue": "human clue"})

    assert response.status_code == 200
    body = response.json()
    assert body["prompt_technique"] == "meta"
    assert game_state.ROUNDS[round_id].prompt_technique == "meta"
    assert captured_techniques == ["meta", "meta", "meta", "meta"]


def test_get_round() -> None:
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.get(f"/rounds/{round_id}")

    assert response.status_code == 200
    assert response.json()["id"] == round_id


def test_get_unknown_round_returns_404() -> None:
    client = build_test_client()

    response = client.get("/rounds/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown round"


def test_group_vote_eliminates_imposter_when_human_vote_matches(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    agent_ids = [agent.id for agent in list_agent_configs()]

    async def vote_for_first_agent(round_state, agents, settings, **kwargs):
        return [
            voting.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=round_state.player_names_by_id[agent.id],
                voted_for=first_agent_id,
                inference_mode="fake",
            )
            for agent in agents
        ]

    monkeypatch.setattr(engine, "choice", lambda player_ids: first_agent_id)
    monkeypatch.setattr(voting, "_build_agent_votes", vote_for_first_agent)
    set_playing_order(monkeypatch, [*agent_ids, game_state.HUMAN_PLAYER_ID])
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]
    clue_response = client.post(f"/rounds/{round_id}/clue", json={"clue": "circles earth"})
    assert clue_response.status_code == 200

    game_state.ROUNDS[round_id].discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": first_agent_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["voted_agent_id"] == first_agent_id
    assert body["secret_word"] == "satellite"
    assert len(body["agent_votes"]) == len(list_agent_configs())
    assert body["group_voted_player_id"] == first_agent_id
    assert body["imposter_won"] is False
    assert body["round_winner"] == "players"
    assert game_state.ROUNDS[round_id].status == "complete"


def test_agent_only_round_votes_without_human_placeholder(monkeypatch) -> None:
    agents = list_agent_configs()
    imposter_agent = agents[0]

    async def vote_for_imposter(round_state, agents, settings, **kwargs):
        return [
            voting.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=round_state.player_names_by_id[agent.id],
                voted_for=imposter_agent.id,
                inference_mode="fake",
            )
            for agent in agents
        ]

    monkeypatch.setattr(engine, "choice", lambda player_ids: imposter_agent.id)
    monkeypatch.setattr(voting, "_build_agent_votes", vote_for_imposter)
    set_playing_order(monkeypatch, [agent.id for agent in agents])
    client = build_test_client()

    create_response = client.post(
        "/rounds",
        json={"secret_word": "satellite", "include_human": False},
    )
    assert create_response.status_code == 201
    round_id = create_response.json()["id"]
    body = client.get(f"/rounds/{round_id}").json()
    assert all(player["kind"] == "agent" for player in body["playing_order"])
    assert body["status"] == "discussion"
    game_state.ROUNDS[round_id].discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)

    response = client.post(f"/rounds/{round_id}/vote", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["voted_agent_id"] is None
    assert body["group_voted_player_id"] == imposter_agent.id
    assert body["vote_counts"] == [
        {
            "player_id": imposter_agent.id,
            "player_name": game_state.ROUNDS[round_id].player_names_by_id[imposter_agent.id],
            "votes": len(agents),
        }
    ]
    assert body["round_winner"] == "players"


def test_llm_votes_never_embed(monkeypatch):
    async def fail_embed(*args, **kwargs):
        raise AssertionError("Live voting must not embed clues")

    monkeypatch.setattr(engine.InferenceClient, "embed", fail_embed)
    client = build_test_client()
    response = client.post("/rounds", json={"include_human": False})
    round_id = response.json()["id"]
    game_state.ROUNDS[round_id].discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
    response = client.post(f"/rounds/{round_id}/vote", json={})
    assert response.status_code == 200
    assert all(v["inference_mode"] == "fake" for v in response.json()["agent_votes"])


def test_group_vote_decides_round_outcome(monkeypatch) -> None:
    agents = list_agent_configs()
    imposter_agent = agents[0]
    human_vote_agent = agents[1]
    agent_ids = [agent.id for agent in agents]

    async def vote_for_imposter(round_state, agents, settings, **kwargs):
        return [
            voting.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=round_state.player_names_by_id[agent.id],
                voted_for=imposter_agent.id,
                inference_mode="fake",
            )
            for agent in agents
        ]

    monkeypatch.setattr(engine, "choice", lambda player_ids: imposter_agent.id)
    monkeypatch.setattr(voting, "_build_agent_votes", vote_for_imposter)
    set_playing_order(monkeypatch, [*agent_ids, game_state.HUMAN_PLAYER_ID])
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]
    clue_response = client.post(f"/rounds/{round_id}/clue", json={"clue": "circles earth"})
    assert clue_response.status_code == 200

    game_state.ROUNDS[round_id].discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": human_vote_agent.id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["group_voted_player_id"] == imposter_agent.id
    assert body["group_voted_player_name"] == game_state.ROUNDS[round_id].player_names_by_id[imposter_agent.id]
    assert body["imposter_won"] is False
    assert body["round_winner"] == "players"
    assert body["vote_counts"][0]["player_id"] == imposter_agent.id
    assert body["vote_counts"][0]["votes"] == len(agents)


def test_imposter_wins_when_group_votes_out_non_imposter(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    agent_ids = [agent.id for agent in list_agent_configs()]

    async def vote_for_first_agent(round_state, agents, settings, **kwargs):
        return [
            voting.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=round_state.player_names_by_id[agent.id],
                voted_for=first_agent_id,
                inference_mode="fake",
            )
            for agent in agents
        ]

    monkeypatch.setattr(engine, "choice", lambda player_ids: game_state.HUMAN_PLAYER_ID)
    monkeypatch.setattr(voting, "_build_agent_votes", vote_for_first_agent)
    set_playing_order(monkeypatch, [*agent_ids, game_state.HUMAN_PLAYER_ID])
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]
    clue_response = client.post(f"/rounds/{round_id}/clue", json={"clue": "circles earth"})
    assert clue_response.status_code == 200

    game_state.ROUNDS[round_id].discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": first_agent_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["imposter_was"] == game_state.ROUNDS[round_id].player_names_by_id["human"]
    assert len(body["agent_votes"]) == len(list_agent_configs())
    assert body["imposter_won"] is True
    assert body["round_winner"] == "imposter"


def test_vote_unknown_round_returns_404() -> None:
    client = build_test_client()

    response = client.post(
        "/rounds/missing/vote",
        json={"agent_id": "agent_a", "human_clue": "circles earth"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown round"


def test_vote_unknown_agent_returns_404(monkeypatch) -> None:
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    game_state.ROUNDS[round_id].discussion.deadline = datetime.now(UTC) - timedelta(seconds=1)
    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": "missing", "human_clue": "circles earth"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown agent"


@pytest.fixture(autouse=True)
def application_lifespans(monkeypatch):
    from contextlib import ExitStack
    original = TestClient
    with ExitStack() as stack:
        def managed(*args, **kwargs):
            return stack.enter_context(original(*args, **kwargs))
        monkeypatch.setitem(globals(), "TestClient", managed)
        yield
