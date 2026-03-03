"""
Pydantic models for the evaluation dataset and result structures.

Dataset JSON schema:
[
  {
    "test_id": "tc-001",
    "conversation": [
      {"user_id": "student-001", "message": "I want to know about tuition fees."},
      ...
    ],
    "expected_intents": ["tuition_inquiry", ...],
    "expected_response_keywords": [["tuition", "fee", "cost"], ...]
  },
  ...
]
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# Dataset models (input)


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation."""

    user_id: str = Field(..., description="User/session identifier.")
    message: str = Field(..., description="User message text.")


class TestCase(BaseModel):
    """One test case containing a full multi-turn conversation."""

    test_id: str = Field(..., description="Unique identifier for this test case.")
    conversation: list[ConversationTurn] = Field(
        ..., min_length=1, description="Ordered list of conversation turns."
    )
    expected_intents: list[str] = Field(
        ..., description="Expected intent label for each turn."
    )
    expected_response_keywords: list[list[str]] = Field(
        ...,
        description=(
            "Per-turn keyword lists. A keyword is 'covered' if it appears "
            "as an exact word (or synonym) OR semantic similarity >= threshold."
        ),
    )


# Result models (output)


class TurnResult(BaseModel):
    """Evaluation result for a single conversation turn within one run."""

    turn_index: int
    user_message: str

    # Intent evaluation
    expected_intent: str
    actual_intent: str
    intent_match: bool

    # Response quality evaluation (new per-keyword coverage model)
    response_text: str
    covered_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords covered by exact match or semantic similarity.",
    )
    missing_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords not covered by either signal.",
    )
    coverage_rate: float = Field(
        0.0,
        description="Fraction of expected keywords covered (0.0 – 1.0).",
    )
    semantic_score: float = Field(
        0.0,
        description="Highest semantic similarity score across all keywords.",
    )
    response_passed: bool

    latency_ms: float


class RunResult(BaseModel):
    """Evaluation result for one full run of a single test case."""

    run_index: int
    turns: list[TurnResult]
    intent_accuracy: float       # fraction of turns with correct intent
    response_pass_rate: float    # fraction of turns where response passed
    avg_coverage_rate: float     # mean keyword coverage across turns
    avg_latency_ms: float
    passed: bool                 # True only if ALL turns passed


class TestCaseResult(BaseModel):
    """Aggregated multi-run result for one test case."""

    test_id: str
    runs: list[RunResult]
    majority_passed: bool        # True if >50% of runs passed
    intent_accuracy: float       # mean across runs
    response_pass_rate: float    # mean across runs
    avg_coverage_rate: float     # mean keyword coverage across runs
    avg_latency_ms: float
    failure_reasons: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Top-level evaluation report written to report.json."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    # Stored as percentage strings (e.g. "92.56%") for human-readable JSON output.
    # Per-test and per-run values remain float for aggregation convenience.
    overall_intent_accuracy: str
    overall_response_pass_rate: str
    overall_avg_coverage_rate: str
    overall_avg_latency_ms: float
    failed_test_ids: list[str]
    test_results: list[TestCaseResult]
