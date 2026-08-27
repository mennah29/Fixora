import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pydantic reads .env for settings, but model libraries also need these values
# in the process environment before they are imported.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


class Settings(BaseSettings):
    """Runtime configuration, supplied only through environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    log_level: str = Field(default="info", validation_alias="LOG_LEVEL")
    allowed_origins: str = Field(default="", validation_alias="ALLOWED_ORIGINS")
    api_key: str = Field(default="", validation_alias="API_KEY")

    data_dir: Path = Field(default=Path("/app/data"), validation_alias="DATA_DIR")
    chunks_file: str = Field(default="all_device_fault_chunks.json", validation_alias="CHUNKS_FILE")
    chroma_dir: str = Field(default="chroma", validation_alias="CHROMA_DIR")
    chunks_path_override: Path | None = Field(default=None, validation_alias="CHUNKS_PATH")
    chroma_path_override: Path | None = Field(default=None, validation_alias="CHROMA_PATH")
    collection_name: str = Field(default="device_fault_chunks", validation_alias="COLLECTION_NAME")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL")
    embedding_local_files_only: bool = Field(default=True, validation_alias="EMBEDDING_LOCAL_FILES_ONLY")
    top_k: int = Field(default=5, ge=1, le=20, validation_alias="TOP_K")
    min_semantic_score: float = Field(default=0.30, ge=0, le=1, validation_alias="MIN_SEMANTIC_SCORE")
    auto_index: bool = Field(default=False, validation_alias="AUTO_INDEX")

    llm_provider: str = Field(default="extractive", validation_alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", validation_alias="GEMINI_MODEL")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:3b", validation_alias="OLLAMA_MODEL")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", validation_alias="ANTHROPIC_MODEL")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    qwen_model: str = Field(default="Qwen/Qwen2.5-3B-Instruct", validation_alias="QWEN_MODEL")
    qwen_model_path: Path | None = Field(default=None, validation_alias="QWEN_MODEL_PATH")
    qwen_load_in_4bit: bool = Field(default=True, validation_alias="QWEN_LOAD_IN_4BIT")
    qwen_max_new_tokens: int = Field(default=160, ge=32, le=512, validation_alias="QWEN_MAX_NEW_TOKENS")

    audio_stt_provider: str = Field(default="transformers", validation_alias="AUDIO_STT_PROVIDER")
    audio_tts_provider: str = Field(default="transformers", validation_alias="AUDIO_TTS_PROVIDER")
    stt_model: str = Field(default="openai/whisper-large-v3-turbo", validation_alias="STT_MODEL")
    tts_model: str = Field(default="facebook/mms-tts-eng", validation_alias="TTS_MODEL")
    tts_voice: str = Field(default="alloy", validation_alias="TTS_VOICE")
    max_audio_bytes: int = Field(default=25 * 1024 * 1024, ge=1_024, validation_alias="MAX_AUDIO_BYTES")
    max_tts_chars: int = Field(default=4_000, ge=100, le=4_096, validation_alias="MAX_TTS_CHARS")

    @property
    def chunks_path(self) -> Path:
        return self.chunks_path_override or (self.data_dir / self.chunks_file)

    @property
    def chroma_path(self) -> Path:
        return self.chroma_path_override or (self.data_dir / self.chroma_dir)

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
