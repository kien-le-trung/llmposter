import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.agents.inference import InferenceClient, InferenceServiceError


def test_remote_transport_is_required_and_never_owned():
    async def run():
        settings = Settings(_env_file=None)
        with pytest.raises(InferenceServiceError, match="injected HTTP client"):
            InferenceClient(settings).require_transport()
        async with httpx.AsyncClient() as transport:
            first = InferenceClient(settings, http_client=transport)
            second = InferenceClient(settings, http_client=transport)
            assert first.require_transport() is second.require_transport() is transport
            assert first._semaphore is not second._semaphore
            assert not hasattr(first, "__aexit__")
            assert not transport.is_closed
        assert transport.is_closed
    asyncio.run(run())


def test_lifespan_owns_transport_and_clears_reference():
    settings = Settings(_env_file=None, inference_mode="fake")
    first, second = create_app(settings), create_app(settings)
    with TestClient(first):
        transport = first.state.inference_http_client
        assert not transport.is_closed
        with TestClient(second):
            assert second.state.inference_http_client is not transport
        assert not transport.is_closed
    assert transport.is_closed
    assert first.state.inference_http_client is None
