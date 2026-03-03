"""
Evaluator configuration, loaded from evaluator/.env via pydantic-settings.

Priority order (highest to lowest):
    1. Shell environment variables
    2. evaluator/.env file
    3. Hard-coded defaults in this class

CLI flags in run_tests.py override .env values when explicitly passed.

Example:
    from evaluator.config import EvaluatorConfig
    config = EvaluatorConfig()         # load from .env
    config = EvaluatorConfig(runs=5)   # override one field
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve to  <project_root>/evaluator/.env  regardless of CWD
_ENV_FILE = Path(__file__).resolve().parent / ".env"


class EvaluatorConfig(BaseSettings):
    """
    Central configuration for the chatbot evaluation pipeline.
    Every field maps to an EVAL_* environment variable.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="EVAL_",
    )

    # Target API
    base_url: str = Field(
        "http://localhost:8080",
        description="Base URL of the chatbot API (no trailing slash).",
    )

    # Dataset and output
    dataset_path: str = Field(
        "test_cases.json", description="Path to the JSON test-case dataset."
    )
    output_path: str = Field(
        "report.json", description="Path where report.json will be written."
    )

    # Evaluation strategy
    runs: int = Field(
        3, ge=3, description="Independent runs per test case (majority-vote). Must be >= 3."
    )

    @field_validator("runs")
    @classmethod
    def validate_runs(cls, v: int) -> int:
        if v < 3:
            raise ValueError("runs must be at least 3 for a valid majority vote.")
        return v

    # Response quality — semantic signal
    semantic_threshold: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="Min cosine similarity for a keyword to count as semantically matched.",
    )

    # Response quality — coverage gate
    coverage_threshold: float = Field(
        0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of expected keywords that must be covered for a turn to pass. "
            "E.g. 0.5 means at least 50% of keywords must be exact or semantic matches."
        ),
    )

    # Language support
    language: str = Field(
        "en",
        description="Language code for spaCy lemma matching (e.g. 'en', 'de', 'fr').",
    )

    # Sentence-transformer model
    model_name: str = Field(
        "all-MiniLM-L6-v2",
        description="Sentence-transformer model name for semantic similarity.",
    )
    force_onnx_export: bool = Field(
        False,
        description="If True, deletes the existing ONNX cached model and re-exports it.",
    )

    # Concurrency and networking
    max_concurrency: int = Field(
        20, ge=1, description="Max simultaneous in-flight test conversations."
    )
    request_timeout: float = Field(
        10.0, gt=0, description="Per-request HTTP timeout in seconds."
    )
    max_retries: int = Field(
        3, ge=0, description="Max retries for transient network errors per turn."
    )

    # Logging level
    log_level: str = Field(
        "WARNING",
        description="Logging verbosity: DEBUG | INFO | WARNING | ERROR",
    )

    # Derived helpers

    @property
    def chat_endpoint(self) -> str:
        """Full URL of the /chat endpoint."""
        return f"{self.base_url.rstrip('/')}/chat"

    def majority_threshold(self) -> int:
        """Minimum passing runs required for a majority-vote pass."""
        return (self.runs // 2) + 1
