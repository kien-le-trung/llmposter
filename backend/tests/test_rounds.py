import asyncio

from fastapi.testclient import TestClient

from app.api.routes import rounds
from app.main import create_app
from app.services.agents import list_agent_configs
from app.services.inference import InferenceResult, InferenceServiceError


def setup_function() -> None:
    rounds.ROUNDS.clear()
    rounds.settings.inference_mode = "fake"
    rounds.settings.word_selection_mode = "fixed"
    rounds.settings.fixed_secret_word = "satellite"
    rounds.settings.fixed_imposter_hint = "Space, signals, or orbit"


def test_create_round_generates_agent_opening_clues(monkeypatch) -> None:
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    client = TestClient(create_app())

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["status"] == "active"
    assert body["user_role"] in {"player", "imposter"}
    assert "secret_word" not in body
    assert len(body["turns"]) == 1
    assert len(body["turns"][0]["responses"]) == len(list_agent_configs())
    assert rounds.ROUNDS[body["id"]].secret_word == "satellite"
    assert rounds.ROUNDS[body["id"]].imposter_player_id


def test_create_round_selects_random_word_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(rounds.settings, "word_selection_mode", "random")
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "random_choice", lambda word_bank: ("forest", "Trees and shade"))
    client = TestClient(create_app())

    response = client.post("/rounds", json={})

    assert response.status_code == 201
    body = response.json()
    assert rounds.ROUNDS[body["id"]].secret_word == "forest"


def test_create_round_hides_word_when_human_is_imposter(monkeypatch) -> None:
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    client = TestClient(create_app())

    response = client.post(
        "/rounds",
        json={"secret_word": "satellite", "imposter_hint": "Space and signals"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["user_role"] == "imposter"
    assert body["visible_word"] is None
    assert body["imposter_hint"] == "Space and signals"


def test_create_round_reveals_word_when_human_is_player(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "choice", lambda player_ids: first_agent_id)
    client = TestClient(create_app())

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    body = response.json()
    assert body["user_role"] == "player"
    assert body["visible_word"] == "satellite"
    assert body["imposter_hint"] is None


def test_opening_clues_are_generated_with_previous_clues(monkeypatch) -> None:
    prompts: list[str] = []

    async def record_prompt(self, request):
        prompts.append(request.prompt)
        return InferenceResult(
            text=f"clue {len(prompts)}",
            inference_mode="fake",
        )

    monkeypatch.setattr(rounds.InferenceClient, "generate", record_prompt)
    client = TestClient(create_app())

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    assert prompts[0] == rounds.OPENING_PROMPT
    assert "Agent A: clue 1" in prompts[1]
    assert "Agent A: clue 1" in prompts[2]
    assert "Agent B: clue 2" in prompts[2]
    assert "Agent C: clue 3" not in prompts[2]


def test_get_round() -> None:
    client = TestClient(create_app())
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.get(f"/rounds/{round_id}")

    assert response.status_code == 200
    assert response.json()["id"] == round_id


def test_get_unknown_round_returns_404() -> None:
    client = TestClient(create_app())

    response = client.get("/rounds/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown round"


def test_create_round_returns_503_when_inference_fails(monkeypatch) -> None:
    async def raise_inference_error(self, request):
        raise InferenceServiceError("Model server is unavailable")

    monkeypatch.setattr(rounds.InferenceClient, "generate", raise_inference_error)
    client = TestClient(create_app())

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Model server is unavailable"


def test_group_vote_eliminates_imposter_when_human_vote_matches(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "choice", lambda player_ids: first_agent_id)
    client = TestClient(create_app())
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": first_agent_id, "human_clue": "circles earth"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["voted_agent_id"] == first_agent_id
    assert body["secret_word"] == "satellite"
    assert len(body["agent_votes"]) == len(list_agent_configs())
    assert body["group_voted_player_id"] == first_agent_id
    assert body["imposter_won"] is False
    assert body["round_winner"] == "players"
    assert rounds.ROUNDS[round_id].status == "complete"


def test_agent_vote_candidate_order_rotates(monkeypatch) -> None:
    system_prompts: list[str] = []

    async def record_system_prompt(self, request):
        system_prompts.append(request.agent.system_prompt)
        return InferenceResult(text="You", inference_mode="fake")

    agents = list_agent_configs()
    round_state = rounds.RoundState(
        id="round_1",
        secret_word="satellite",
        imposter_hint="Space, signals, or orbit",
        imposter_player_id=agents[0].id,
        status="active",
        turns=[
            rounds.TurnResponse(
                id="turn_1",
                sequence=1,
                user_prompt=rounds.OPENING_PROMPT,
                responses=[],
                created_at=rounds.datetime.now(rounds.UTC),
            )
        ],
        created_at=rounds.datetime.now(rounds.UTC),
    )

    monkeypatch.setattr(rounds.InferenceClient, "generate", record_system_prompt)

    asyncio.run(rounds.generate_agent_votes(round_state, agents, "circles earth"))

    assert "Agent B, Agent C, Agent D, You" in system_prompts[0]
    assert "Agent C, Agent D, Agent A, You" in system_prompts[1]


def test_group_vote_decides_round_outcome(monkeypatch) -> None:
    agents = list_agent_configs()
    imposter_agent = agents[0]
    human_vote_agent = agents[1]

    async def vote_for_imposter(round_state, agents, human_clue):
        return [
            rounds.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=agent.name,
                voted_for=imposter_agent.name,
                inference_mode="fake",
            )
            for agent in agents
        ]

    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "choice", lambda player_ids: imposter_agent.id)
    monkeypatch.setattr(rounds, "generate_agent_votes", vote_for_imposter)
    client = TestClient(create_app())
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": human_vote_agent.id, "human_clue": "circles earth"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["group_voted_player_id"] == imposter_agent.id
    assert body["group_voted_player_name"] == imposter_agent.name
    assert body["imposter_won"] is False
    assert body["round_winner"] == "players"
    assert body["vote_counts"][0]["player_id"] == imposter_agent.id
    assert body["vote_counts"][0]["votes"] == len(agents)


def test_imposter_wins_when_group_votes_out_non_imposter(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    client = TestClient(create_app())
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": first_agent_id, "human_clue": "circles earth"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["imposter_was"] == "You"
    assert len(body["agent_votes"]) == len(list_agent_configs())
    assert body["imposter_won"] is True
    assert body["round_winner"] == "imposter"


def test_vote_unknown_round_returns_404() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/rounds/missing/vote",
        json={"agent_id": "agent_a", "human_clue": "circles earth"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown round"


def test_vote_unknown_agent_returns_404(monkeypatch) -> None:
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    client = TestClient(create_app())
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": "missing", "human_clue": "circles earth"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown agent"
