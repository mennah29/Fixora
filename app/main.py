import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config import Settings, get_settings
from app.audio import AudioConfigurationError, VoiceService
from app.schemas import QueryRequest, QueryResponse, ServiceStatus, VoiceResponse
from app.service import RagService, ServiceNotReadyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("fixora")

# Default max body size for non-audio JSON endpoints (1 MB).
_DEFAULT_MAX_BODY_BYTES = 1 * 1024 * 1024


class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds *max_bytes* with HTTP 413.

    The voice endpoint uploads audio up to 25 MB; its own limit is checked by
    *validate_audio_upload*, so this middleware only applies a conservative cap
    for all other routes to prevent resource exhaustion on JSON endpoints.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = _DEFAULT_MAX_BODY_BYTES, exclude_paths: tuple[str, ...] = ()) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes
        self.exclude_paths = exclude_paths

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path not in self.exclude_paths:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_bytes:
                return Response(
                    content=f"Request body exceeds the {self.max_bytes // 1024} KB limit.",
                    status_code=413,  # HTTP 413 Content Too Large
                )
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.environment.lower() == "production" and not settings.api_key:
        raise RuntimeError("API_KEY must be set when ENVIRONMENT=production.")
    service = RagService(settings)
    try:
        await asyncio.to_thread(service.load)
    except Exception as exc:  # Keep liveness available; readiness reports the exact deployment fault.
        logger.exception("Service startup failed")
        service.startup_error = str(exc)
    app.state.service = service
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Fixora API", version="1.0.0", lifespan=lifespan, docs_url="/docs" if settings.environment != "production" else None)

    # Allow CORS for Streamlit and external UI clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_api_key(request: Request) -> None:
        if not settings.api_key:
            return
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, settings.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    @app.get("/health/live", response_model=ServiceStatus, tags=["operations"])
    def live() -> ServiceStatus:
        service: RagService | None = getattr(app.state, "service", None)
        is_ready = bool(service and service.ready)
        count = service.document_count if service else 0
        return ServiceStatus(status="ok" if is_ready else "degraded", ready=is_ready, collection_documents=count)


    @app.get("/health/ready", response_model=ServiceStatus, tags=["operations"])
    def ready() -> ServiceStatus:
        service: RagService | None = getattr(app.state, "service", None)
        if not service or not service.ready:
            raise HTTPException(status_code=503, detail=service.startup_error if service else "Service is starting")
        return ServiceStatus(status="ok", ready=True, collection_documents=service.document_count)

    @app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)], tags=["assistant"])
    async def query(payload: QueryRequest) -> QueryResponse:
        try:
            return QueryResponse.model_validate(await app.state.service.answer(**payload.model_dump()))
        except ServiceNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Query failed")
            raise HTTPException(status_code=502, detail="The assistant could not complete this request.") from exc

    @app.post("/v1/voice", response_model=VoiceResponse, dependencies=[Depends(require_api_key)], tags=["assistant"])
    async def voice(
        audio: UploadFile = File(..., description="Recorded technician audio in WebM, WAV, MP3, M4A, OGG, MP4, or FLAC."),
        device_name: str | None = Form(default=None),
        manufacturer_domain: str | None = Form(default=None),
        top_k: int | None = Form(default=None, ge=1, le=20),
    ) -> VoiceResponse:
        content = await audio.read()
        try:
            output = await VoiceService(settings, app.state.service).process(
                content, audio.filename or "recording.webm", audio.content_type or "audio/webm",
                device_name, manufacturer_domain, top_k,
            )
            return VoiceResponse.model_validate(output)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (AudioConfigurationError, ServiceNotReadyError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Voice request failed")
            raise HTTPException(status_code=502, detail="The voice assistant could not complete this request.") from exc

    @app.websocket("/v1/ws")
    async def websocket_query(websocket: WebSocket) -> None:
        supplied = websocket.headers.get("authorization", "").removeprefix("Bearer ") or websocket.query_params.get("api_key", "")
        if settings.api_key and not hmac.compare_digest(supplied, settings.api_key):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                payload = QueryRequest.model_validate(await websocket.receive_json())
                try:
                    await websocket.send_json(await app.state.service.answer(**payload.model_dump()))
                except ServiceNotReadyError as exc:
                    await websocket.send_json({"error": "service_not_ready", "detail": str(exc)})
        except WebSocketDisconnect:
            return

    # Root redirect → UI
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app/index.html")

    # Serve the frontend from /static on disk, exposed under /app
    _static_dir = Path(__file__).parent.parent / "static"
    if _static_dir.is_dir():
        app.mount("/app", StaticFiles(directory=str(_static_dir), html=True), name="ui")

    return app


app = create_app()
