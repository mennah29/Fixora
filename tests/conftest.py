"""Shared pytest fixtures for all Fixora tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.service import NOT_FOUND, RagService


# ---------------------------------------------------------------------------
# Settings override — no real API keys, no filesystem access.
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_settings() -> Settings:
    """Return a minimal Settings instance suitable for unit tests."""
    return Settings(
        ENVIRONMENT="development",
        API_KEY="test-secret",
        LLM_PROVIDER="stub",
        DATA_DIR="/tmp/fixora-test",
        AUTO_INDEX=False,
    )


# ---------------------------------------------------------------------------
# A fully-mocked RagService — no Chroma, no embedding model.
# ---------------------------------------------------------------------------

_DUMMY_RESULT: dict[str, Any] = {
    "chunk_id": "chunk-1",
    "text": "E37 — Overtemperature fault. Disconnect power. [Source 1]",
    "manual_name": "compressor-x200.pdf",
    "page_number": 42,
    "section_name": "error_table",
    "device": "Compressor X200",
    "retrieval_type": "exact",
    "score": 1.0,
}

_DUMMY_ANSWER: dict[str, Any] = {
    "answer": "E37 indicates an overtemperature fault. Disconnect power immediately. [Source 1]",
    "status": "FOUND_IN_MANUAL",
    "sources": [{"manual": "compressor-x200.pdf", "page": 42, "device": "Compressor X200", "retrieval_type": "exact"}],
    "used_web_fallback": False,
    "validation": {"status": "FOUND_IN_MANUAL", "issues": []},
}


@pytest.fixture()
def mock_rag_service(test_settings: Settings) -> MagicMock:
    """Return a mock RagService that is always ready and returns *_DUMMY_ANSWER*."""
    svc = MagicMock(spec=RagService)
    svc.ready = True
    svc.startup_error = None
    svc.document_count = 10
    svc.collection = MagicMock()  # non-None → ready check passes
    svc.embedding_model = MagicMock()
    # answer() is async — return the canned dict
    import asyncio

    async def _answer(**kwargs: Any) -> dict[str, Any]:
        return _DUMMY_ANSWER

    svc.answer = _answer
    return svc


# ---------------------------------------------------------------------------
# FastAPI TestClient with the mocked service pre-wired.
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(mock_rag_service: MagicMock, test_settings: Settings) -> TestClient:
    """Return a TestClient whose app.state.service is the mock service."""
    from app.config import get_settings
    from app.main import create_app

    # Override the cached settings so create_app uses test_settings.
    with patch("app.main.get_settings", return_value=test_settings), \
         patch("app.config.get_settings", return_value=test_settings):
        application = create_app()

    # Bypass the lifespan startup so we can inject the mock directly.
    application.state.service = mock_rag_service

    return TestClient(application, raise_server_exceptions=True)
