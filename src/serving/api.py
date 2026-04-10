"""FastAPI REST + WebSocket API for PulseDE.

Endpoints:
  GET  /v1/sentiment/latest         — paginated recent results
  GET  /v1/sentiment/ticker/{sym}   — ticker-level summary
  GET  /v1/sentiment/hourly         — hourly rollup for charting
  GET  /v1/sentiment/search         — full-text search over headlines
  POST /v1/sentiment/analyse        — synchronous on-demand inference
  GET  /v1/health                   — liveness + readiness probe
  GET  /v1/metrics/model            — model drift & performance summary
  WS   /ws/realtime                 — WebSocket stream of live results

Auth: Bearer JWT (HS256). Rate limit: Redis sliding window.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, AsyncGenerator

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from config.settings import settings
from src.ml.ensemble import SentimentEnsemble
from src.ml.feature_engineering import FinancialFeatureExtractor
from src.storage.redis_cache import RedisCache
from src.storage.timescale_writer import TimescaleWriter

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PulseDE API",
    description="Real-Time Financial Sentiment Intelligence — Multi-Model NLP Ensemble",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Dependency singletons ──────────────────────────────────────────────────────

_ensemble: SentimentEnsemble | None = None
_feature_extractor: FinancialFeatureExtractor | None = None
_db: TimescaleWriter | None = None
_cache: RedisCache | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _ensemble, _feature_extractor, _db, _cache
    _ensemble = SentimentEnsemble()
    _feature_extractor = FinancialFeatureExtractor()
    _db = TimescaleWriter()
    _cache = RedisCache()
    logger.info("api_startup_complete")


def get_db() -> TimescaleWriter:
    assert _db is not None
    return _db


def get_cache() -> RedisCache:
    assert _cache is not None
    return _cache


def get_ensemble() -> SentimentEnsemble:
    assert _ensemble is not None
    return _ensemble


def get_feature_extractor() -> FinancialFeatureExtractor:
    assert _feature_extractor is not None
    return _feature_extractor


# ── Auth ───────────────────────────────────────────────────────────────────────

def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.api.access_token_expire_minutes
    )
    return jwt.encode(
        {"sub": subject, "exp": expire},
        settings.api.secret_key.get_secret_value(),
        algorithm=settings.api.algorithm,
    )


async def get_current_user(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    token = auth.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(
            token,
            settings.api.secret_key.get_secret_value(),
            algorithms=[settings.api.algorithm],
        )
        return str(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def check_rate_limit(
    request: Request,
    cache: RedisCache = Depends(get_cache),
    client_id: str = Depends(get_current_user),
) -> None:
    if cache.is_rate_limited(client_id, limit=settings.api.rate_limit_per_minute):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Max 60 requests/minute.",
        )


AuthDep = Annotated[str, Depends(get_current_user)]
RateDep = Annotated[None, Depends(check_rate_limit)]

# ── Pydantic models ────────────────────────────────────────────────────────────

class SentimentResultOut(BaseModel):
    article_hash: str
    headline: str
    url: str
    source: str
    published_at: datetime
    processed_at: datetime
    ensemble_sentiment: str
    ensemble_confidence: float
    ensemble_uncertainty: float
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    tickers: list[str]
    sectors: list[str]
    is_forward_looking: bool
    has_negation: bool
    hedge_score: float
    market_impact: str
    is_uncertain: bool


class AnalyseRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50, description="Headlines to analyse")


class AnalyseResponse(BaseModel):
    results: list[dict[str, Any]]
    model_count: int
    temperature: float


class HealthResponse(BaseModel):
    status: str
    db_ok: bool
    cache_ok: bool
    model_loaded: bool
    version: str = "2.0.0"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/v1/health", response_model=HealthResponse, tags=["ops"])
async def health(
    db: TimescaleWriter = Depends(get_db),
    cache: RedisCache = Depends(get_cache),
) -> HealthResponse:
    db_ok = True
    cache_ok = True
    try:
        db.query_recent(hours=0, limit=1)
    except Exception:
        db_ok = False
    try:
        cache.get_latest(n=1)
    except Exception:
        cache_ok = False

    overall = "ok" if db_ok and cache_ok else "degraded"
    return HealthResponse(
        status=overall,
        db_ok=db_ok,
        cache_ok=cache_ok,
        model_loaded=_ensemble is not None,
    )


@app.get("/v1/sentiment/latest", response_model=list[SentimentResultOut], tags=["sentiment"])
async def get_latest(
    _: RateDep,
    n: int = Query(default=50, ge=1, le=500),
    cache: RedisCache = Depends(get_cache),
    db: TimescaleWriter = Depends(get_db),
) -> list[dict[str, Any]]:
    """Fetch the n most recent sentiment results. Served from Redis cache (< 1 ms)."""
    results = cache.get_latest(n=n)
    if not results:
        # Cache cold — fall back to DB
        results = db.query_recent(hours=24, limit=n)
    return results


@app.get("/v1/sentiment/ticker/{ticker}", tags=["sentiment"])
async def get_ticker(
    ticker: str,
    _: RateDep,
    hours: int = Query(default=24, ge=1, le=168),
    cache: RedisCache = Depends(get_cache),
    db: TimescaleWriter = Depends(get_db),
) -> dict[str, Any]:
    """Aggregated sentiment for a specific ticker symbol."""
    cached = cache.get_ticker_summary(ticker.upper())
    if cached:
        return {**cached, "source": "cache"}
    rows = db.query_by_ticker(ticker.upper(), hours=hours)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for ticker {ticker.upper()}")
    n = len(rows)
    pos = sum(1 for r in rows if r["ensemble_sentiment"] == "positive")
    neg = sum(1 for r in rows if r["ensemble_sentiment"] == "negative")
    return {
        "ticker": ticker.upper(),
        "hours": hours,
        "article_count": n,
        "positive_pct": round(pos / n, 4),
        "negative_pct": round(neg / n, 4),
        "neutral_pct": round((n - pos - neg) / n, 4),
        "avg_confidence": round(sum(r["ensemble_confidence"] for r in rows) / n, 4),
        "source": "db",
    }


@app.get("/v1/sentiment/hourly", tags=["sentiment"])
async def get_hourly(
    _: RateDep,
    hours: int = Query(default=48, ge=1, le=720),
    db: TimescaleWriter = Depends(get_db),
) -> list[dict[str, Any]]:
    """Hourly sentiment rollup — ideal for time-series charts."""
    return db.query_hourly_rollup(hours=hours)


@app.post("/v1/sentiment/analyse", response_model=AnalyseResponse, tags=["inference"])
async def analyse(
    body: AnalyseRequest,
    _: RateDep,
    ensemble: SentimentEnsemble = Depends(get_ensemble),
) -> AnalyseResponse:
    """On-demand synchronous inference on arbitrary text list."""
    results = ensemble.predict(body.texts)
    return AnalyseResponse(
        results=[
            {
                "text": text,
                "sentiment": r["sentiment"].value,
                "confidence": r["confidence"],
                "uncertainty": r["uncertainty"],
                "is_uncertain": r["is_uncertain"],
                "positive_prob": r["positive_prob"],
                "negative_prob": r["negative_prob"],
                "neutral_prob": r["neutral_prob"],
            }
            for text, r in zip(body.texts, results)
        ],
        model_count=3,
        temperature=ensemble._temperature,
    )


# ── WebSocket real-time stream ─────────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: str) -> None:
        dead: set[WebSocket] = set()
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self.active -= dead


_manager = _ConnectionManager()


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket) -> None:
    """Streams live sentiment results to connected dashboard clients via Redis Pub/Sub."""
    await _manager.connect(websocket)
    cache = get_cache()
    ps = cache.get_pubsub()

    try:
        while True:
            # Non-blocking Redis pub/sub poll
            message = ps.get_message(ignore_subscribe_messages=True, timeout=0.05)
            if message and message.get("data"):
                await websocket.send_text(message["data"])
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        _manager.disconnect(websocket)
    finally:
        ps.close()


if __name__ == "__main__":
    uvicorn.run(
        "src.serving.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.env == "development",
        workers=1 if settings.env == "development" else 4,
    )
