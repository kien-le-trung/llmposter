from fastapi.testclient import TestClient

from app.api.routes import agents
from app.main import create_app
from app.services.inference import InferenceServiceError


def test_list_agents() -> None:
    client = TestClient(create_app())

    response = client.get("/agents")

    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_generate_with_fake_inference(monkeypatch) -> None:
    monkeypatch.setattr(agents.settings, "inference_mode", "fake")
    client = TestClient(create_app())

    response = client.post(
        "/agents/generate",
        json={"agent_id": "agent_a", "prompt": "Start the game"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "agent_a"
    assert body["inference_mode"] == "fake"


def test_agent_with_bad_id() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/agents/generate",
        json={"agent_id": "nonexistent", "prompt": "Start the game"},
    )

    assert response.status_code == 404

    body = response.json()
    assert body["detail"] == "Unknown agent"


def test_generate_returns_503_when_inference_fails(monkeypatch) -> None:
    async def raise_inference_error(self, request):
        raise InferenceServiceError("Model server is unavailable")

    monkeypatch.setattr(agents.settings, "inference_mode", "remote")
    monkeypatch.setattr(agents.InferenceClient, "generate", raise_inference_error)
    client = TestClient(create_app())

    response = client.post(
        "/agents/generate",
        json={"agent_id": "agent_a", "prompt": "Start the game"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model server is unavailable"
