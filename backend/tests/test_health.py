from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_health_check() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_default_llm_config_uses_openrouter() -> None:
    settings = Settings(_env_file=None)

    llm_config = settings.load_llm_config()
    assert llm_config.provider == "openrouter"
    assert llm_config.model == "openrouter/free"


def test_cors_preflight_allows_localhost_frontend() -> None:
    client = TestClient(create_app())

    response = client.options(
        "/rounds",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200


def test_cors_preflight_allows_loopback_frontend() -> None:
    client = TestClient(create_app())

    response = client.options(
        "/rounds",
        headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
