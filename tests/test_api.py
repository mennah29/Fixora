"""API-layer tests for the Fixora FastAPI application.

All tests use the TestClient fixture from conftest.py, which bypasses the
lifespan startup and injects a fully-mocked RagService. No real Chroma index,
embedding model, or LLM credentials are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# /health/live
# ---------------------------------------------------------------------------

class TestHealthLive:
    def test_returns_200_always(self, client: TestClient) -> None:
        response = client.get("/health/live")
        assert response.status_code == 200

    def test_returns_ok_when_service_ready(self, client: TestClient) -> None:
        response = client.get("/health/live")
        data = response.json()
        assert data["status"] == "ok"
        assert data["ready"] is True

    def test_returns_degraded_when_service_not_ready(self, test_settings, mock_rag_service) -> None:
        """When the service is not ready /health/live must return status=degraded (not 503)."""
        from app.main import create_app

        mock_rag_service.ready = False
        mock_rag_service.document_count = 0

        with patch("app.main.get_settings", return_value=test_settings), \
             patch("app.config.get_settings", return_value=test_settings):
            application = create_app()

        application.state.service = mock_rag_service
        degraded_client = TestClient(application, raise_server_exceptions=True)

        response = degraded_client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ready"] is False


# ---------------------------------------------------------------------------
# /health/ready
# ---------------------------------------------------------------------------

class TestHealthReady:
    def test_returns_200_when_ready(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_returns_503_when_not_ready(self, test_settings, mock_rag_service) -> None:
        mock_rag_service.ready = False
        mock_rag_service.startup_error = "No index found."

        from app.main import create_app

        with patch("app.main.get_settings", return_value=test_settings), \
             patch("app.config.get_settings", return_value=test_settings):
            application = create_app()

        application.state.service = mock_rag_service
        not_ready_client = TestClient(application, raise_server_exceptions=False)

        response = not_ready_client.get("/health/ready")
        assert response.status_code == 503
        assert "No index found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/query — authentication
# ---------------------------------------------------------------------------

class TestQueryAuth:
    def test_rejects_missing_api_key(self, client: TestClient) -> None:
        response = client.post("/v1/query", json={"query": "What is E37?"})
        assert response.status_code == 401

    def test_rejects_wrong_api_key(self, client: TestClient) -> None:
        response = client.post(
            "/v1/query",
            json={"query": "What is E37?"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401

    def test_accepts_correct_api_key(self, client: TestClient) -> None:
        response = client.post(
            "/v1/query",
            json={"query": "What is E37?"},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/query — response shape
# ---------------------------------------------------------------------------

class TestQueryResponse:
    def test_response_has_required_fields(self, client: TestClient) -> None:
        response = client.post(
            "/v1/query",
            json={"query": "What is E37?", "device_name": "Compressor X200"},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "status" in data
        assert "sources" in data
        assert "used_web_fallback" in data
        assert "validation" in data

    def test_device_name_and_top_k_forwarded(self, client: TestClient, mock_rag_service) -> None:
        """Verify that device_name and top_k from the request body reach the service."""
        received: dict = {}

        import asyncio

        async def capturing_answer(**kwargs):
            received.update(kwargs)
            from tests.conftest import _DUMMY_ANSWER
            return _DUMMY_ANSWER

        mock_rag_service.answer = capturing_answer

        client.post(
            "/v1/query",
            json={"query": "What is E37?", "device_name": "Compressor X200", "top_k": 3},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert received.get("device_name") == "Compressor X200"
        assert received.get("top_k") == 3

    def test_returns_503_when_service_not_ready(self, client: TestClient, mock_rag_service) -> None:
        from app.service import ServiceNotReadyError

        async def raise_not_ready(**kwargs):
            raise ServiceNotReadyError("Service is not ready.")

        mock_rag_service.answer = raise_not_ready

        response = client.post(
            "/v1/query",
            json={"query": "What is E37?"},
            headers={"Authorization": "Bearer test-secret"},
            # Prevent TestClient from raising the HTTPException as a Python exception.
        )
        assert response.status_code == 503

    def test_validation_rejects_too_short_query(self, client: TestClient) -> None:
        response = client.post(
            "/v1/query",
            json={"query": "X"},  # min_length=2 requires ≥2 chars
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Body-size limit middleware
# ---------------------------------------------------------------------------

class TestBodySizeLimit:
    def test_oversized_json_body_is_rejected(self, client: TestClient) -> None:
        # Send a Content-Length header that exceeds the 1 MB limit.
        oversized_body = "x" * (1 * 1024 * 1024 + 1)
        response = client.post(
            "/v1/query",
            content=oversized_body,
            headers={
                "Authorization": "Bearer test-secret",
                "Content-Type": "application/json",
                "Content-Length": str(len(oversized_body)),
            },
        )
        assert response.status_code == 413
