from fastapi.testclient import TestClient

from app.main import create_app
from app.services.metrics import clear_request_metrics


def setup_function() -> None:
    clear_request_metrics()


def test_admin_metrics_records_requests() -> None:
    client = TestClient(create_app())

    client.get("/health")
    response = client.get("/admin/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["request_count"] >= 1
    assert body["summary"]["average_latency_ms"] >= 0
    assert body["requests"][0]["path"] == "/health"


def test_admin_dashboard_returns_html() -> None:
    client = TestClient(create_app())

    response = client.get("/admin/dashboard")

    assert response.status_code == 200
    assert "LLMposter Metrics" in response.text
