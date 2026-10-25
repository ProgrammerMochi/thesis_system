"""FastAPI app for Increment 3 (Chapter 3, Figure 2 and Figure 3).

Run from the repo root (see server/README.md):
    SMISHING_MODE=stub python -m server      # no model, fixed valid responses
    python -m server                         # real model
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from server.app.engine import build_engine
from server.app.schemas import AnalyzeResponse, make_request_model
from server.app.timing import StageTimer
from server.config import Settings

log = logging.getLogger("smishing.server")
MAX_BODY_BYTES = 64 * 1024

# Libraries must not print inputs into logs (zero retention, Section 3.5.3).
for name in ("transformers", "shap", "urllib3", "filelock"):
    logging.getLogger(name).setLevel(logging.ERROR)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    AnalyzeRequest = make_request_model(settings.max_chars)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = build_engine(settings)
        if settings.warmup and hasattr(engine, "warm_up"):
            engine.warm_up()
        engine.ready = True
        app.state.engine = engine
        yield

    app = FastAPI(title="Taglish smishing detector", version="0.3.0", lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Body-size limit, and a catch-all that answers 500 WITHOUT re-raising,
        so no traceback (which could quote message text) reaches the logs."""
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"error": "payload_too_large"}, status_code=413)
        try:
            return await call_next(request)
        except Exception as exc:  # noqa: BLE001
            request_id = uuid.uuid4().hex[:12]
            log.error("internal_error request_id=%s type=%s", request_id, type(exc).__name__)
            return JSONResponse({"error": "internal_error", "request_id": request_id}, status_code=500)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # FastAPI's default 422 body includes each submitted "input": never echo it.
        errors = [{"field": ".".join(str(p) for p in e.get("loc", ())), "type": e.get("type", "invalid")}
                  for e in exc.errors()]
        return JSONResponse({"error": "invalid_request", "errors": errors}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": "http_error", "status": exc.status_code}, status_code=exc.status_code)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request):
        engine = getattr(request.app.state, "engine", None)
        ok = engine is not None and engine.ready
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/v1/meta")
    def meta(request: Request):
        return request.app.state.engine.meta()

    @app.post("/v1/analyze", response_model=AnalyzeResponse)
    def analyze(body: AnalyzeRequest, request: Request,
                explain: bool = Query(True, description="false skips SHAP (times classification alone)")):
        timer = StageTimer()
        result = request.app.state.engine.analyze(body.text, explain, timer)
        AnalyzeResponse.model_validate(result)   # the contract the Android client relies on
        return JSONResponse(result, headers={"Server-Timing": timer.header(), "Cache-Control": "no-store"})

    return app
