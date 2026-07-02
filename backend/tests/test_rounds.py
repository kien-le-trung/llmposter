from fastapi.testclient import TestClient

from app.api.routes import rounds
from app.main import create_app
from app.services.agents import list_agent_configs
from app.services.inference import InferenceServiceError


def setup_function() -> None:
    rounds.ROUNDS.clear()


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


def test_create_round_hides_word_when_human_is_imposter(monkeypatch) -> None:
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    client = TestClient(create_app())

    response = client.post("/rounds", json={"secret_word": "satellite"})

    assert response.status_code == 201
    body = response.json()
    assert body["user_role"] == "imposter"
    assert body["visible_word"] is None


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


def test_vote_correct_agent_completes_round(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "choice", lambda player_ids: first_agent_id)
    client = TestClient(create_app())
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(f"/rounds/{round_id}/vote", json={"agent_id": first_agent_id})

    assert response.status_code == 200
    body = response.json()
    assert body["voted_agent_id"] == first_agent_id
    assert body["correct"] is True
    assert rounds.ROUNDS[round_id].status == "complete"


def test_vote_agent_when_human_is_imposter_is_incorrect(monkeypatch) -> None:
    first_agent_id = list_agent_configs()[0].id
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    monkeypatch.setattr(rounds, "choice", lambda player_ids: rounds.HUMAN_PLAYER_ID)
    client = TestClient(create_app())
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(f"/rounds/{round_id}/vote", json={"agent_id": first_agent_id})

    assert response.status_code == 200
    body = response.json()
    assert body["correct"] is False
    assert body["imposter_was"] == "You"


def test_vote_unknown_round_returns_404() -> None:
    client = TestClient(create_app())

    response = client.post("/rounds/missing/vote", json={"agent_id": "agent_a"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown round"


def test_vote_unknown_agent_returns_404(monkeypatch) -> None:
    monkeypatch.setattr(rounds.settings, "inference_mode", "fake")
    client = TestClient(create_app())
    create_response = client.post("/rounds", json={"secret_word": "satellite"})
    round_id = create_response.json()["id"]

    response = client.post(f"/rounds/{round_id}/vote", json={"agent_id": "missing"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown agent"
