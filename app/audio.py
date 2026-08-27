"""Audio ingestion and spoken-response adapters.

Audio bytes are processed in memory only; this module never writes technician recordings
to disk or logs their content.
"""

import asyncio
import base64
import io
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import Settings
from app.service import RagService

SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/flac", "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a",
    "audio/ogg", "audio/wav", "audio/webm", "audio/x-wav",
}
SUPPORTED_AUDIO_SUFFIXES = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}


class AudioConfigurationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _get_stt_pipeline(model_name: str):
    """Load the selected Whisper pipeline once for the process."""
    import torch
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=-1,
        torch_dtype=torch.float32,
    )


@lru_cache(maxsize=1)
def _get_tts_pipeline(model_name: str):
    """Load the selected MMS text-to-speech pipeline once for the process."""
    from transformers import pipeline

    return pipeline(
        "text-to-speech",
        model=model_name,
        device=-1,
    )


def _wave_bytes(audio: object, sample_rate: int) -> bytes:
    """Convert a Transformers float waveform into a standard PCM WAV file."""
    waveform = np.asarray(audio, dtype=np.float32).squeeze()
    if waveform.ndim == 0:
        raise ValueError("The text-to-speech model returned invalid audio.")
    if waveform.ndim > 2:
        raise ValueError("The text-to-speech model returned unsupported audio channels.")
    channels = 1 if waveform.ndim == 1 else waveform.shape[0]
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())
        return buffer.getvalue()


def validate_audio_upload(filename: str | None, content_type: str | None, content: bytes, max_bytes: int) -> None:
    if not content:
        raise ValueError("The audio upload is empty.")
    if len(content) > max_bytes:
        raise ValueError(f"Audio exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    suffix = Path(filename or "").suffix.lower()
    if content_type not in SUPPORTED_AUDIO_MIME_TYPES and suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError("Unsupported audio format. Use FLAC, MP3, MP4/M4A, OGG, WAV, or WebM.")


class VoiceService:
    def __init__(self, settings: Settings, rag_service: RagService):
        self.settings = settings
        self.rag_service = rag_service

    def _openai_client(self):
        if not self.settings.openai_api_key:
            raise AudioConfigurationError("OPENAI_API_KEY is required for the configured audio providers.")
        from openai import OpenAI
        return OpenAI(api_key=self.settings.openai_api_key)

    def transcribe(self, content: bytes, filename: str, content_type: str) -> str:
        provider = self.settings.audio_stt_provider.lower()
        if provider == "openai":
            transcript = self._openai_client().audio.transcriptions.create(
                model=self.settings.stt_model,
                file=(filename, content, content_type),
            )
            text = (getattr(transcript, "text", "") or "").strip()
        elif provider == "transformers":
            import soundfile as sf
            try:
                audio_array, sr = sf.read(io.BytesIO(content), dtype="float32")
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                result = _get_stt_pipeline(self.settings.stt_model)({"raw": audio_array, "sampling_rate": sr})
            except Exception:
                result = _get_stt_pipeline(self.settings.stt_model)(content)
            text = str(result.get("text", "")).strip()
        else:
            raise AudioConfigurationError("AUDIO_STT_PROVIDER must be 'transformers' or 'openai'.")
        if not text:
            raise ValueError("No speech could be transcribed from this audio.")
        return text

    def synthesize(self, text: str) -> bytes | None:
        provider = self.settings.audio_tts_provider.lower()
        if provider == "disabled":
            return None
        spoken_text = text[: self.settings.max_tts_chars]
        if len(text) > len(spoken_text):
            spoken_text = f"{spoken_text[: self.settings.max_tts_chars - 38]} Response continues on screen."
        if provider == "openai":
            response = self._openai_client().audio.speech.create(
                model=self.settings.tts_model,
                voice=self.settings.tts_voice,
                input=spoken_text,
                response_format="mp3",
            )
            return response.content
        if provider == "transformers":
            result = _get_tts_pipeline(self.settings.tts_model)(spoken_text)
            return _wave_bytes(result["audio"], int(result["sampling_rate"]))
        raise AudioConfigurationError("AUDIO_TTS_PROVIDER must be 'transformers', 'openai', or 'disabled'.")

    @property
    def audio_mime_type(self) -> str | None:
        if self.settings.audio_tts_provider.lower() == "disabled":
            return None
        return "audio/wav" if self.settings.audio_tts_provider.lower() == "transformers" else "audio/mpeg"

    async def process(
        self,
        content: bytes,
        filename: str,
        content_type: str,
        device_name: str | None = None,
        manufacturer_domain: str | None = None,
        top_k: int | None = None,
    ) -> dict:
        validate_audio_upload(filename, content_type, content, self.settings.max_audio_bytes)
        transcript = await asyncio.to_thread(self.transcribe, content, filename, content_type)
        result = await self.rag_service.answer(transcript, device_name, manufacturer_domain, top_k)
        audio = await asyncio.to_thread(self.synthesize, result["answer"])
        return {
            "transcript": transcript,
            "result": result,
            "audio_base64": base64.b64encode(audio).decode("ascii") if audio else None,
            "audio_mime_type": self.audio_mime_type if audio else None,
        }
