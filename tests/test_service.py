"""Unit tests for the Fixora RAG service layer."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.service import NOT_FOUND, WEB_FALLBACK_LABEL, RagService, extract_error_codes
from app.audio import VoiceService, validate_audio_upload


# ---------------------------------------------------------------------------
# extract_error_codes
# ---------------------------------------------------------------------------

def test_extract_error_codes_normalizes_common_forms() -> None:
    assert extract_error_codes("Error code E037 and fault 42") == ["37", "42"]


def test_extract_error_codes_returns_empty_for_plain_query() -> None:
    assert extract_error_codes("How do I replace the fan belt?") == []


def test_extract_error_codes_deduplicates() -> None:
    # "ERR37" and "E037" both normalize to "37".
    codes = extract_error_codes("ERR37 is the same as E037")
    assert codes == ["37"]


def test_extract_error_codes_supports_f_and_alarm_series() -> None:
    assert extract_error_codes("F112 and alarm 004 are active") == ["4", "112"]


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

def test_context_returns_sentinel_for_no_results() -> None:
    assert RagService.build_context([]) == NOT_FOUND


def test_context_formats_multiple_sources() -> None:
    results = [
        {"device": "DevA", "manual_name": "m1.pdf", "page_number": 1, "section_name": "error_table", "text": "err1"},
        {"device": "DevB", "manual_name": "m2.pdf", "page_number": 5, "section_name": "safety", "text": "err2"},
    ]
    ctx = RagService.build_context(results)
    assert "[Source 1]" in ctx
    assert "[Source 2]" in ctx
    assert "DevA" in ctx and "DevB" in ctx


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_grounding_flags_missing_safety_notice() -> None:
    result = RagService.validate(
        "Inspect the unit. [Source 1]",
        "[Source 1]\nDANGER: Disconnect power.\nManual: demo.pdf",
        [{"manual_name": "demo.pdf"}],
    )
    assert result["status"] == "FLAGGED"
    assert any("DANGER" in issue for issue in result["issues"])


def test_grounding_flags_fabricated_error_code() -> None:
    result = RagService.validate(
        "E99 means overload. [Source 1]",
        "[Source 1]\nManual: demo.pdf\nE37 overtemperature.",
        [{"manual_name": "demo.pdf"}],
    )
    assert result["status"] == "FLAGGED"
    assert any("E99" in issue for issue in result["issues"])


def test_grounding_passes_when_source_cited_and_codes_match() -> None:
    result = RagService.validate(
        "E37 overtemperature detected. Disconnect. [Source 1]",
        "[Source 1]\nManual: demo.pdf\nE37 overtemperature. Disconnect power.",
        [{"manual_name": "demo.pdf"}],
    )
    assert result["status"] == "FOUND_IN_MANUAL"
    assert result["issues"] == []


def test_grounding_not_found_when_context_is_sentinel() -> None:
    result = RagService.validate(NOT_FOUND, NOT_FOUND, [])
    assert result["status"] == "NOT_FOUND_IN_MANUAL"


# ---------------------------------------------------------------------------
# answer() — mocked RAG service (no Chroma, no real LLM)
# ---------------------------------------------------------------------------

def _make_service_with_mocks(
    retrieve_results: list[dict[str, Any]],
    llm_response: str,
) -> RagService:
    """Build a RagService with its I/O dependencies replaced by mocks."""
    from app.config import Settings

    settings = Settings(
        ENVIRONMENT="development",
        LLM_PROVIDER="stub",
        DATA_DIR="/tmp/fixora-test",
    )
    svc = RagService(settings)
    # Fake-ready state — bypass the Chroma / SentenceTransformer path.
    svc.collection = MagicMock()
    svc.embedding_model = MagicMock()
    svc.startup_error = None
    svc.retrieve = MagicMock(return_value=retrieve_results)
    svc._call_llm = MagicMock(return_value=llm_response)
    return svc


def test_answer_happy_path() -> None:
    result = [
        {
            "chunk_id": "c1",
            "text": "E37 overtemperature. [Source 1]",
            "manual_name": "demo.pdf",
            "page_number": 3,
            "section_name": "error_table",
            "device": "DevA",
            "retrieval_type": "exact",
            "score": 1.0,
        }
    ]
    svc = _make_service_with_mocks(result, "E37 overtemperature. Disconnect power. [Source 1]")
    answer = asyncio.get_event_loop().run_until_complete(
        svc.answer("What is E37?", "DevA", None, None)
    )
    assert answer["status"] == "FOUND_IN_MANUAL"
    assert answer["used_web_fallback"] is False
    assert answer["answer"] != NOT_FOUND


def test_answer_falls_back_to_web_when_nothing_retrieved() -> None:
    svc = _make_service_with_mocks([], NOT_FOUND)
    # No Tavily key → web fallback returns the "not configured" message.
    svc.settings = svc.settings.model_copy(update={"tavily_api_key": ""})
    answer = asyncio.get_event_loop().run_until_complete(
        svc.answer("Unknown device error", None, None, None)
    )
    assert answer["used_web_fallback"] is True
    assert answer["status"] == "NOT_FOUND_IN_MANUAL"
    assert WEB_FALLBACK_LABEL in answer["answer"]


# ---------------------------------------------------------------------------
# audio validation
# ---------------------------------------------------------------------------

def test_audio_validation_rejects_an_unsupported_upload() -> None:
    try:
        validate_audio_upload("recording.txt", "text/plain", b"not audio", 1024)
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError("Unsupported audio must be rejected")


def test_audio_validation_rejects_empty_upload() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_audio_upload("recording.wav", "audio/wav", b"", 1024)


def test_audio_validation_rejects_oversized_upload() -> None:
    content = b"x" * 1025
    with pytest.raises(ValueError, match="exceeds"):
        validate_audio_upload("recording.wav", "audio/wav", content, 1024)


def test_audio_validation_accepts_known_formats() -> None:
    # Should not raise for supported MIME types.
    validate_audio_upload("clip.webm", "audio/webm", b"audio-bytes", 1024 * 1024)
    validate_audio_upload("clip.wav", "audio/wav", b"audio-bytes", 1024 * 1024)
    validate_audio_upload("clip.mp3", "audio/mpeg", b"audio-bytes", 1024 * 1024)


def test_transcription_uses_configured_stt_model() -> None:
    from app.config import Settings

    service = VoiceService(
        Settings(OPENAI_API_KEY="test-key", AUDIO_STT_PROVIDER="openai", STT_MODEL="gpt-4o-mini-transcribe"),
        MagicMock(),
    )
    client = MagicMock()
    client.audio.transcriptions.create.return_value = MagicMock(text="Check error E37")
    service._openai_client = MagicMock(return_value=client)

    assert service.transcribe(b"audio", "recording.webm", "audio/webm") == "Check error E37"
    client.audio.transcriptions.create.assert_called_once_with(
        model="gpt-4o-mini-transcribe",
        file=("recording.webm", b"audio", "audio/webm"),
    )


def test_speech_uses_configured_tts_model_and_voice() -> None:
    from app.config import Settings

    service = VoiceService(
        Settings(
            OPENAI_API_KEY="test-key",
            AUDIO_TTS_PROVIDER="openai",
            TTS_MODEL="gpt-4o-mini-tts",
            TTS_VOICE="alloy",
        ),
        MagicMock(),
    )
    client = MagicMock()
    client.audio.speech.create.return_value = MagicMock(content=b"mp3-bytes")
    service._openai_client = MagicMock(return_value=client)

    assert service.synthesize("Disconnect power before inspection.") == b"mp3-bytes"
    client.audio.speech.create.assert_called_once_with(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input="Disconnect power before inspection.",
        response_format="mp3",
    )


def test_speech_can_be_disabled() -> None:
    from app.config import Settings

    service = VoiceService(Settings(AUDIO_TTS_PROVIDER="disabled"), MagicMock())
    assert service.synthesize("A displayed answer is still available.") is None


def test_selected_whisper_model_is_used_for_local_transcription() -> None:
    from app.config import Settings

    service = VoiceService(
        Settings(AUDIO_STT_PROVIDER="transformers", STT_MODEL="openai/whisper-large-v3-turbo"),
        MagicMock(),
    )
    pipeline = MagicMock(return_value={"text": "Check fault F112"})
    with patch("app.audio._get_stt_pipeline", return_value=pipeline) as loader:
        assert service.transcribe(b"audio", "recording.wav", "audio/wav") == "Check fault F112"

    loader.assert_called_once_with("openai/whisper-large-v3-turbo")
    pipeline.assert_called_once_with(b"audio")


def test_selected_mms_model_returns_browser_playable_wav() -> None:
    from app.config import Settings

    service = VoiceService(
        Settings(AUDIO_TTS_PROVIDER="transformers", TTS_MODEL="facebook/mms-tts-eng"),
        MagicMock(),
    )
    pipeline = MagicMock(return_value={"audio": [0.0, 0.25, -0.25], "sampling_rate": 16_000})
    with patch("app.audio._get_tts_pipeline", return_value=pipeline) as loader:
        audio = service.synthesize("Disconnect power before inspection.")

    loader.assert_called_once_with("facebook/mms-tts-eng")
    pipeline.assert_called_once_with("Disconnect power before inspection.")
    assert audio is not None and audio.startswith(b"RIFF")
    assert service.audio_mime_type == "audio/wav"
