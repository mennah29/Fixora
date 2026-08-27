from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2_000)
    device_name: str | None = Field(default=None, max_length=255)
    manufacturer_domain: str | None = Field(default=None, max_length=253)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    manual: str | None = None
    page: int | str | None = None
    device: str | None = None
    retrieval_type: str | None = None
    title: str | None = None
    url: str | None = None


class Validation(BaseModel):
    status: Literal["FOUND_IN_MANUAL", "NOT_FOUND_IN_MANUAL", "FLAGGED"]
    issues: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    status: Literal["FOUND_IN_MANUAL", "NOT_FOUND_IN_MANUAL", "FLAGGED"]
    sources: list[Source]
    used_web_fallback: bool
    validation: Validation
    has_high_priority_safety: bool | None = None
    safety_header: str | None = None
    safety_body: str | None = None
    fault_meaning: str | None = None
    checklist: list[str] = Field(default_factory=list)
    speech_text: str | None = None
    source_citation: dict[str, Any] | None = None


class VoiceResponse(BaseModel):
    transcript: str
    result: QueryResponse
    audio_base64: str | None = None
    audio_mime_type: str | None = None


class ServiceStatus(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    collection_documents: int
    detail: str | None = None
