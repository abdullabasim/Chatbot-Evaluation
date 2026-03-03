"""
Mock API configuration — loaded from mock_api/.env via pydantic-settings.

Priority (highest → lowest):
  1. Shell environment variables  (e.g. MOCK_PORT=9090 uvicorn ...)
  2. mock_api/.env file           (committed defaults)
  3. Default values in this class (fallback)

Usage:
    from mock_api.core.config import settings
    print(settings.port)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve to  <project_root>/mock_api/.env  regardless of CWD
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class MockAPISettings(BaseSettings):
    """
    All configurable knobs for the Mock FastAPI server.
    Every field maps to a MOCK_* environment variable.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="MOCK_",
    )

    # Server
    host: str = Field("0.0.0.0", description="Bind address for uvicorn")
    port: int = Field(8080, description="Listening port")
    log_level: str = Field("info", description="Uvicorn log level")
    workers: int = Field(1, description="Number of uvicorn worker processes")

    # LLM Simulation
    hallucination_rate: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Probability (0–1) of returning a hallucinated response",
    )
    min_latency_ms: int = Field(
        200, ge=0, description="Minimum simulated latency in milliseconds"
    )
    max_latency_ms: int = Field(
        800, ge=0, description="Maximum simulated latency in milliseconds"
    )

    # Confidence scores
    confidence_min: float = Field(
        0.72, ge=0.0, le=1.0, description="Min confidence for normal responses"
    )
    confidence_max: float = Field(
        0.99, ge=0.0, le=1.0, description="Max confidence for normal responses"
    )
    hallucination_confidence_min: float = Field(
        0.10, ge=0.0, le=1.0, description="Min confidence for hallucinated responses"
    )
    hallucination_confidence_max: float = Field(
        0.45, ge=0.0, le=1.0, description="Max confidence for hallucinated responses"
    )


# Singleton — import this everywhere
settings = MockAPISettings()
