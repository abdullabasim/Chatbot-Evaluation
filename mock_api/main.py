"""
Mock Chatbot FastAPI Application — entry point.

Configuration is loaded from mock_api/.env via MockAPISettings.
Every value can be overridden by a shell environment variable (MOCK_* prefix).

Run directly (reads .env automatically):
    uvicorn mock_api.main:app --host 0.0.0.0 --port 8080

Or let the app self-host (reads host/port from config):
    python -m mock_api.main
"""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mock_api.core.config import settings
from mock_api.routes.chat import router as chat_router

# Logging — level driven by MOCK_LOG_LEVEL in mock_api/.env
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Application factory

app = FastAPI(
    title="Mock Chatbot API",
    description=(
        f"Non-deterministic mock chatbot simulating LLM behaviour. "
        f"Latency: {settings.min_latency_ms}–{settings.max_latency_ms} ms  |  "
        f"Hallucination rate: {settings.hallucination_rate * 100:.0f}%"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration: Restrict to internal/local origins
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://api:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route registration

app.include_router(chat_router)


# Startup / shutdown lifecycle


@app.on_event("startup")
async def on_startup() -> None:
    logger.info(
        "Mock Chatbot API started on %s:%d  "
        "(hallucination=%.0f%%  latency=%d–%dms)",
        settings.host,
        settings.port,
        settings.hallucination_rate * 100,
        settings.min_latency_ms,
        settings.max_latency_ms,
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Mock Chatbot API shutting down.")


# Self-hosting entry-point  (python -m mock_api.main)

if __name__ == "__main__":
    uvicorn.run(
        "mock_api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=settings.workers,
    )
