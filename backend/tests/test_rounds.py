from fastapi.testclient import TestClient

from app.api.routes import rounds
from app.core.config import Settings
from app.main import create_app
from app.services.agents.inference import InferenceResult, InferenceServiceError
from app.services.agents.runtime_agents import list_static_agent_configs as list_agent_configs
from app.services.voting import voting


def setup_function() -> None:
    rounds.ROUNDS.clear()
    voting.ROUND_EMBEDDINGS.clear()
    voting.EMBEDDING_CACHE.clear()


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
    return TestClient(create_app(build_test_settings(**settings_overrides)))


def set_playing_order(monkeypatch, player_ids: list[str]) -> None:
    def use_order(order: list[str]) -> None:
        order[:] = player_ids

    monkeypatch.setattr(rounds.random, "shuffle", use_order)


def test_create_round_generates_agent_opening_clues(monkeypatch) -> None:
    agent_ids = [agent.id for agent in list_agent_configs()]
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
    client = build_test_client()

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["status"] == "generating_clues"
    assert body["user_role"] in {"player", "imposter"}
    assert "secret_word" not in body
    assert [player["id"] for player in body["playing_order"]] == [
        *agent_ids,
        rounds.HUMAN_PLAYER_ID,
    ]
    assert body["current_player_id"] == agent_ids[1]
    assert len(body["turns"]) == 1
    assert len(body["turns"][0]["responses"]) == 1
    assert rounds.ROUNDS[body["id"]].secret_word == "satellite"
    assert rounds.ROUNDS[body["id"]].imposter_player_id


def test_create_round_selects_random_word_when_enabled(monkeypatch) -> None:
    agent_ids = [agent.id for agent in list_agent_configs()]
    monkeypatch.setattr(rounds, "select_random_word", lambda: ("forest", "Trees and shade"))
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
    client = build_test_client(word_selection_mode="random")

    response = client.post("/rounds", json={})

    assert response.status_code == 201
    body = response.json()
    assert rounds.ROUNDS[body["id"]].secret_word == "forest"
    assert rounds.ROUNDS[body["id"]].imposter_hint == "Trees"


def test_create_round_payload_word_overrides_random_mode(monkeypatch) -> None:
    agent_ids = [agent.id for agent in list_agent_configs()]
    monkeypatch.setattr(rounds, "select_random_word", lambda: ("forest", "Trees and shade"))
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
    client = build_test_client(word_selection_mode="random")

    response = client.post(
        "/rounds",
        json={"secret_word": "satellite", "imposter_hint": "Space and signals"},
    )

    assert response.status_code == 201
    body = response.json()
    assert rounds.ROUNDS[body["id"]].secret_word == "satellite"
    assert rounds.ROUNDS[body["id"]].imposter_hint == "Space"


def test_create_round_hides_word_when_human_is_imposter(monkeypatch) -> None:
    agent_ids = [agent.id for agent in list_agent_configs()]
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
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
    monkeypatch.setattr(rounds, "choice", lambda player_ids: first_agent_id)
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
    client = build_test_client()

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    body = response.json()
    assert body["user_role"] == "player"
    assert body["visible_word"] == "satellite"
    assert body["imposter_hint"] is None


def test_opening_clues_stop_when_human_is_first(monkeypatch) -> None:
    prompts: list[str] = []

    async def record_prompt(self, request):
        prompts.append(request.prompt)
        return InferenceResult(text='{"clue":"space signal"}', inference_mode="fake")

    monkeypatch.setattr(rounds.InferenceClient, "generate", record_prompt)
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    set_playing_order(monkeypatch, [rounds.HUMAN_PLAYER_ID, *[agent.id for agent in list_agent_configs()]])
    client = build_test_client()

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "awaiting_human_clue"
    assert body["current_player_id"] == rounds.HUMAN_PLAYER_ID
    assert body["turns"][0]["responses"] == []
    assert prompts == []


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
        if '"clues"' in request.prompt:
            return InferenceResult(
                text=(
                    '{"clues":{'
                    f'"{agents[0].name}":"clue 1",'
                    f'"{agents[1].name}":"clue 2",'
                    f'"{agents[2].name}":"clue 3",'
                    f'"{agents[3].name}":"clue 4"'
                    "}}"
                ),
                inference_mode="fake",
            )
        return InferenceResult(text='{"clue":"clue"}', inference_mode="fake")

    monkeypatch.setattr(rounds, "assign_non_imposter_clue_strategy", record_non_imposter_strategy)
    monkeypatch.setattr(rounds.InferenceClient, "generate", record_prompt)
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    set_playing_order(monkeypatch, [rounds.HUMAN_PLAYER_ID, *[agent.id for agent in agents]])
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
    assert rounds.ROUNDS[round_id].prompt_technique == "meta"
    assert captured_techniques == ["meta", "meta", "meta", "meta"]


def test_opening_clues_generate_first_agent_then_batch_remaining_non_imposters(monkeypatch) -> None:
    prompts: list[str] = []
    agents = list_agent_configs()

    async def record_prompt(self, request):
        prompts.append(request.prompt)
        if '"clues"' in request.prompt:
            return InferenceResult(
                text=(
                    '{"clues":{'
                    f'"{agents[0].name}":"clue 1",'
                    f'"{agents[1].name}":"clue 2"'
                    "}}"
                ),
                inference_mode="fake",
            )
        return InferenceResult(text=f'{{"clue":"clue {len(prompts)}"}}', inference_mode="fake")

    monkeypatch.setattr(rounds.InferenceClient, "generate", record_prompt)
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    set_playing_order(monkeypatch, [agents[0].id, agents[1].id, rounds.HUMAN_PLAYER_ID, agents[2].id, agents[3].id])
    client = build_test_client()

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "generating_clues"
    assert "Previous clues: none" in prompts[0]
    assert "Word: satellite" in prompts[0]
    assert [response["agent_response"] for response in body["turns"][0]["responses"]] == [
        "clue 1",
    ]

    round_state = rounds.ROUNDS[body["id"]]
    assert round_state.status == "awaiting_human_clue"
    assert len(prompts) == 2
    assert f"- {agents[1].name}" in prompts[1]
    assert f"{agents[0].name}: clue 1" in prompts[1]
    assert [response.agent_response for response in round_state.turns[0].responses] == [
        "clue 1",
        "clue 2",
    ]


def test_submit_human_clue_resumes_and_batches_remaining_non_imposters(monkeypatch) -> None:
    prompts: list[str] = []
    agents = list_agent_configs()

    async def record_prompt(self, request):
        prompts.append(request.prompt)
        if '"clues"' in request.prompt:
            return InferenceResult(
                text=(
                    '{"clues":{'
                    f'"{agents[1].name}":"clue 2",'
                    f'"{agents[2].name}":"clue 3",'
                    f'"{agents[3].name}":"clue 4"'
                    "}}"
                ),
                inference_mode="fake",
            )
        return InferenceResult(text=f'{{"clue":"clue {len(prompts)}"}}', inference_mode="fake")

    monkeypatch.setattr(rounds.InferenceClient, "generate", record_prompt)
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    set_playing_order(
        monkeypatch,
        [agents[0].id, rounds.HUMAN_PLAYER_ID, agents[1].id, agents[2].id, agents[3].id],
    )
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(f"/rounds/{round_id}/clue", json={"clue": "human clue"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_to_vote"
    assert len(prompts) == 2
    assert "You: human clue" in prompts[1]
    assert f"- {agents[1].name}" in prompts[1]
    assert f"- {agents[2].name}" in prompts[1]
    assert f"- {agents[3].name}" in prompts[1]
    assert [response["agent_id"] for response in body["turns"][0]["responses"]] == [
        agents[0].id,
        rounds.HUMAN_PLAYER_ID,
        agents[1].id,
        agents[2].id,
        agents[3].id,
    ]


def test_opening_clues_keep_imposter_prompt_separate(monkeypatch) -> None:
    prompts: list[str] = []
    agents = list_agent_configs()
    imposter_agent = agents[0]

    async def record_prompt(self, request):
        prompts.append(request.prompt)
        return InferenceResult(text='{"clue":"space clue"}', inference_mode="fake")

    monkeypatch.setattr(rounds.InferenceClient, "generate", record_prompt)
    monkeypatch.setattr(rounds, "choice", lambda player_ids: imposter_agent.id)
    set_playing_order(monkeypatch, [agents[1].id, imposter_agent.id, rounds.HUMAN_PLAYER_ID, agents[2].id, agents[3].id])
    client = build_test_client()

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    assert len(prompts) == 2
    assert "Word: satellite" in prompts[0]
    assert f"{imposter_agent.name}:" not in prompts[0]
    assert "Hint: orbit" in prompts[1]
    assert f"{agents[1].name}: space clue" in prompts[1]
    assert "Secret word: satellite" not in prompts[1]


def test_human_first_batches_non_imposters_then_generates_imposter(monkeypatch) -> None:
    prompts: list[str] = []
    agents = list_agent_configs()
    imposter_agent = agents[1]

    async def record_prompt(self, request):
        prompts.append(request.prompt)
        if '"clues"' in request.prompt:
            return InferenceResult(
                text=(
                    '{"clues":{'
                    f'"{agents[0].name}":"clue 1",'
                    f'"{agents[2].name}":"clue 3",'
                    f'"{agents[3].name}":"clue 4"'
                    "}}"
                ),
                inference_mode="fake",
            )
        return InferenceResult(text='{"clue":"imposter clue"}', inference_mode="fake")

    monkeypatch.setattr(rounds.InferenceClient, "generate", record_prompt)
    monkeypatch.setattr(rounds, "choice", lambda player_ids: imposter_agent.id)
    set_playing_order(
        monkeypatch,
        [
            rounds.HUMAN_PLAYER_ID,
            agents[0].id,
            imposter_agent.id,
            agents[2].id,
            agents[3].id,
        ],
    )
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(f"/rounds/{round_id}/clue", json={"clue": "human clue"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_to_vote"
    assert len(prompts) == 2
    assert f"- {agents[0].name}" in prompts[0]
    assert f"- {agents[2].name}" in prompts[0]
    assert f"- {agents[3].name}" in prompts[0]
    assert "Hint: orbit" in prompts[1]
    assert f"{agents[0].name}: clue 1" in prompts[1]
    assert f"{agents[2].name}: clue 3" in prompts[1]
    assert f"{agents[3].name}: clue 4" in prompts[1]
    assert [response["agent_id"] for response in body["turns"][0]["responses"]] == [
        rounds.HUMAN_PLAYER_ID,
        agents[0].id,
        imposter_agent.id,
        agents[2].id,
        agents[3].id,
    ]


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


def test_create_round_returns_503_when_inference_fails(monkeypatch) -> None:
    async def raise_inference_error(self, request):
        raise InferenceServiceError("Model server is unavailable")

    monkeypatch.setattr(rounds.InferenceClient, "generate", raise_inference_error)
    set_playing_order(monkeypatch, [list_agent_configs()[0].id, rounds.HUMAN_PLAYER_ID])
    client = build_test_client()

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Model server is unavailable"


def test_group_vote_eliminates_imposter_when_human_vote_matches(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    first_agent_name = list_agent_configs()[0].name
    agent_ids = [agent.id for agent in list_agent_configs()]

    async def vote_for_first_agent(round_state, agents, settings, human_clue):
        return [
            voting.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=agent.name,
                voted_for=first_agent_name,
                inference_mode="embedding",
            )
            for agent in agents
        ]

    monkeypatch.setattr(rounds, "choice", lambda player_ids: first_agent_id)
    monkeypatch.setattr(voting, "_build_agent_votes", vote_for_first_agent)
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]
    clue_response = client.post(f"/rounds/{round_id}/clue", json={"clue": "circles earth"})
    assert clue_response.status_code == 200

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
    assert rounds.ROUNDS[round_id].status == "complete"


def test_agent_only_round_votes_without_human_placeholder(monkeypatch) -> None:
    agents = list_agent_configs()
    imposter_agent = agents[0]

    async def vote_for_imposter(round_state, agents, settings, human_clue):
        return [
            voting.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=agent.name,
                voted_for=imposter_agent.name,
                inference_mode="embedding",
            )
            for agent in agents
        ]

    monkeypatch.setattr(rounds, "choice", lambda player_ids: imposter_agent.id)
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
    assert body["status"] == "ready_to_vote"

    response = client.post(f"/rounds/{round_id}/vote", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["voted_agent_id"] is None
    assert body["group_voted_player_id"] == imposter_agent.id
    assert body["vote_counts"] == [
        {
            "player_id": imposter_agent.id,
            "player_name": imposter_agent.name,
            "votes": len(agents),
        }
    ]
    assert body["round_winner"] == "players"


def test_embedding_votes_reuse_cached_agent_phrase_embeddings(monkeypatch) -> None:
    agents = list_agent_configs()
    first_agent_id = agents[0].id
    agent_ids = [agent.id for agent in agents]
    embedding_calls: list[list[str]] = []

    async def record_embed(self, texts):
        embedding_calls.append(texts)
        return await original_embed(self, texts)

    original_embed = rounds.InferenceClient.embed
    monkeypatch.setattr(rounds.InferenceClient, "embed", record_embed)
    monkeypatch.setattr(rounds, "choice", lambda player_ids: first_agent_id)
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    assert embedding_calls == []

    clue_response = client.post(f"/rounds/{round_id}/clue", json={"clue": "circles earth"})
    assert clue_response.status_code == 200
    assert embedding_calls == []

    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": first_agent_id},
    )

    assert response.status_code == 200
    assert len(embedding_calls) == 1
    assert len(embedding_calls[0]) == len(agents) + 1
    assert set(voting.ROUND_EMBEDDINGS[round_id]) == {
        rounds.HUMAN_PLAYER_ID,
        *[agent.id for agent in agents],
    }


def test_group_vote_decides_round_outcome(monkeypatch) -> None:
    agents = list_agent_configs()
    imposter_agent = agents[0]
    human_vote_agent = agents[1]
    agent_ids = [agent.id for agent in agents]

    async def vote_for_imposter(round_state, agents, settings, human_clue):
        return [
            voting.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=agent.name,
                voted_for=imposter_agent.name,
                inference_mode="embedding",
            )
            for agent in agents
        ]

    monkeypatch.setattr(rounds, "choice", lambda player_ids: imposter_agent.id)
    monkeypatch.setattr(voting, "_build_agent_votes", vote_for_imposter)
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]
    clue_response = client.post(f"/rounds/{round_id}/clue", json={"clue": "circles earth"})
    assert clue_response.status_code == 200

    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": human_vote_agent.id},
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
    first_agent_name = list_agent_configs()[0].name
    agent_ids = [agent.id for agent in list_agent_configs()]

    async def vote_for_first_agent(round_state, agents, settings, human_clue):
        return [
            voting.AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=agent.name,
                voted_for=first_agent_name,
                inference_mode="embedding",
            )
            for agent in agents
        ]

    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    monkeypatch.setattr(voting, "_build_agent_votes", vote_for_first_agent)
    set_playing_order(monkeypatch, [*agent_ids, rounds.HUMAN_PLAYER_ID])
    client = build_test_client()
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]
    clue_response = client.post(f"/rounds/{round_id}/clue", json={"clue": "circles earth"})
    assert clue_response.status_code == 200

    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": first_agent_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["imposter_was"] == "You"
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

    response = client.post(
        f"/rounds/{round_id}/vote",
        json={"agent_id": "missing", "human_clue": "circles earth"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown agent"
