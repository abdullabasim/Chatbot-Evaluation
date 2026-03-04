"""
Automated Chatbot Evaluation Pipeline - CLI entry point.

Usage:
    python -m evaluator.run_tests \
        --dataset test_cases.json \
        --base-url http://localhost:8080 \
        --runs 3 \
        --semantic-threshold 0.45 \
        --output report.json

How it works:
    - All test cases run in parallel via asyncio.gather.
    - Turns within a single conversation run sequentially to keep session state.
    - Each test case is run N times; majority vote decides pass/fail.
    - The sentence-transformer model is loaded once at startup and reused.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from evaluator.client import ChatAPIClient
from evaluator.config import EvaluatorConfig
from evaluator.models import (
    EvaluationReport,
    RunResult,
    TestCase,
    TestCaseResult,
    TurnResult,
)
from evaluator.validators import ResponseQualityAssessor, get_assessor

logger = logging.getLogger(__name__)


# Turn execution
async def _execute_turn(
    client: ChatAPIClient,
    assessor: ResponseQualityAssessor,
    turn_index: int,
    user_id: str,
    message: str,
    expected_intent: str,
    expected_keywords: list[str],
    config: EvaluatorConfig,
) -> TurnResult:
    """
    Send one turn to the chatbot API and evaluate the response.

    Handles network errors gracefully — returns a failed TurnResult instead
    of crashing the whole pipeline.
    """
    try:
        data, latency_ms = await client.send_message(user_id=user_id, message=message)
    except Exception as exc:
        logger.warning("Turn %d failed with exception: %s", turn_index, exc)
        return TurnResult(
            turn_index=turn_index,
            user_message=message,
            expected_intent=expected_intent,
            actual_intent="ERROR",
            intent_match=False,
            response_text=f"[ERROR] {exc}",
            covered_keywords=[],
            missing_keywords=list(expected_keywords),
            coverage_rate=0.0,
            semantic_score=0.0,
            response_passed=False,
            latency_ms=0.0,
        )

    actual_intent: str = data.get("intent", "")
    response_text: str = data.get("response", "")

    intent_match = actual_intent.strip().lower() == expected_intent.strip().lower()

    # Per-keyword assessment using the shared assessor singleton
    quality = assessor.assess(
        response=response_text,
        keywords=expected_keywords,
        semantic_threshold=config.semantic_threshold,
        coverage_threshold=config.coverage_threshold,
    )
    # Best single-keyword semantic score for reporting
    best_semantic = max(
        (r.semantic_score for r in quality.keyword_results), default=0.0
    )

    return TurnResult(
        turn_index=turn_index,
        user_message=message,
        expected_intent=expected_intent,
        actual_intent=actual_intent,
        intent_match=intent_match,
        response_text=response_text,
        # Include match stage so report shows e.g. "tuition (lemma)" or "fee (semantic)"
        covered_keywords=[
            f"{r.keyword} ({r.match_reason})"
            for r in quality.keyword_results
            if r.covered
        ],
        missing_keywords=quality.missing_keywords,
        coverage_rate=quality.coverage_rate,
        semantic_score=round(best_semantic, 4),
        response_passed=quality.response_passed,
        latency_ms=latency_ms,
    )


# Single run for one test case
async def _run_once(
    client: ChatAPIClient,
    assessor: ResponseQualityAssessor,
    test_case: TestCase,
    run_index: int,
    config: EvaluatorConfig,
) -> RunResult:
    """
    Run all turns of one test case in order, then compute per-run metrics.

    Turns must be sequential so the user_id session context is preserved
    across the conversation.
    """
    turns: list[TurnResult] = []

    for idx, (turn, intent, keywords) in enumerate(
        zip(
            test_case.conversation,
            test_case.expected_intents,
            test_case.expected_response_keywords,
        )
    ):
        turn_result = await _execute_turn(
            client=client,
            assessor=assessor,
            turn_index=idx + 1,
            user_id=turn.user_id,
            message=turn.message,
            expected_intent=intent,
            expected_keywords=keywords,
            config=config,
        )
        turns.append(turn_result)

    n = len(turns)
    intent_accuracy = sum(t.intent_match for t in turns) / n if n else 0.0
    response_pass_rate = sum(t.response_passed for t in turns) / n if n else 0.0
    avg_coverage_rate = sum(t.coverage_rate for t in turns) / n if n else 0.0
    avg_latency_ms = sum(t.latency_ms for t in turns) / n if n else 0.0
    passed = all(t.intent_match and t.response_passed for t in turns)

    return RunResult(
        run_index=run_index + 1,
        turns=turns,
        intent_accuracy=round(intent_accuracy, 4),
        response_pass_rate=round(response_pass_rate, 4),
        avg_coverage_rate=round(avg_coverage_rate, 4),
        avg_latency_ms=round(avg_latency_ms, 2),
        passed=passed,
    )


# Multi-run with majority vote
async def _evaluate_test_case(
    client: ChatAPIClient,
    assessor: ResponseQualityAssessor,
    test_case: TestCase,
    config: EvaluatorConfig,
) -> TestCaseResult:
    """
    Run the test case N times and decide pass/fail by majority vote.

    A test passes if more than half of the runs pass. This makes the result
    robust to occasional hallucinations from the LLM.
    """
    run_results: list[RunResult] = []
    for run_idx in range(config.runs):
        result = await _run_once(client, assessor, test_case, run_idx, config)
        run_results.append(result)

    passing_runs = sum(1 for r in run_results if r.passed)
    majority_passed = passing_runs > (config.runs / 2)

    mean_intent_accuracy = sum(r.intent_accuracy for r in run_results) / len(run_results)
    mean_response_pass_rate = sum(r.response_pass_rate for r in run_results) / len(run_results)
    mean_coverage_rate = sum(r.avg_coverage_rate for r in run_results) / len(run_results)
    mean_latency = sum(r.avg_latency_ms for r in run_results) / len(run_results)

    failure_reasons: list[str] = []
    if not majority_passed:
        failed_turns: set[str] = set()
        for run in run_results:
            for turn in run.turns:
                if not turn.intent_match:
                    failed_turns.add(
                        f"Turn {turn.turn_index}: intent mismatch "
                        f"({turn.actual_intent!r} ≠ {turn.expected_intent!r})"
                    )
                if not turn.response_passed:
                    failed_turns.add(
                        f"Turn {turn.turn_index}: no keywords matched in response"
                        f"  (missing: {', '.join(turn.missing_keywords) or 'none'})"
                    )
        failure_reasons = sorted(failed_turns)

    return TestCaseResult(
        test_id=test_case.test_id,
        runs=run_results,
        majority_passed=majority_passed,
        intent_accuracy=round(mean_intent_accuracy, 4),
        response_pass_rate=round(mean_response_pass_rate, 4),
        avg_coverage_rate=round(mean_coverage_rate, 4),
        avg_latency_ms=round(mean_latency, 2),
        failure_reasons=failure_reasons,
    )


# Pipeline orchestrator
async def run_pipeline(config: EvaluatorConfig) -> EvaluationReport:
    """
    Run the full evaluation pipeline end to end.

    Loads test cases, spins up the assessor (which loads the sentence-transformer
    model once), runs all tests concurrently, and aggregates everything into a
    single EvaluationReport.
    """
    # --- Load dataset ---
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        sys.exit(1)

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    test_cases = [TestCase.model_validate(tc) for tc in raw]
    logger.info("Loaded %d test cases from %s", len(test_cases), dataset_path)

    # --- Init assessor singleton (sentence-transformer loads here, once) ---
    assessor = get_assessor(
        model_name=config.model_name,
        semantic_threshold=config.semantic_threshold,
        coverage_threshold=config.coverage_threshold,
        force_onnx_export=config.force_onnx_export,
        language=config.language,
    )

    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _bounded(tc: TestCase, client: ChatAPIClient) -> TestCaseResult:
        async with semaphore:
            return await _evaluate_test_case(client, assessor, tc, config)

    # --- Execute pipeline ---
    pipeline_start = time.perf_counter()

    async with ChatAPIClient(config) as client:
        tasks = [_bounded(tc, client) for tc in test_cases]
        test_results: list[TestCaseResult] = await asyncio.gather(*tasks)

    pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
    logger.info("Pipeline completed in %.1f ms", pipeline_elapsed)

    # --- Aggregate global metrics ---
    total = len(test_results)
    passed_tests = sum(1 for r in test_results if r.majority_passed)
    failed_test_ids = [r.test_id for r in test_results if not r.majority_passed]

    overall_intent_acc = sum(r.intent_accuracy for r in test_results) / total if total else 0.0
    overall_resp_rate = sum(r.response_pass_rate for r in test_results) / total if total else 0.0
    overall_coverage = sum(r.avg_coverage_rate for r in test_results) / total if total else 0.0
    overall_latency = sum(r.avg_latency_ms for r in test_results) / total if total else 0.0

    return EvaluationReport(
        total_tests=total,
        passed_tests=passed_tests,
        failed_tests=total - passed_tests,
        # Format as percentage strings for human-readable JSON output.
        overall_intent_accuracy=f"{overall_intent_acc * 100:.2f}%",
        overall_response_pass_rate=f"{overall_resp_rate * 100:.2f}%",
        overall_avg_coverage_rate=f"{overall_coverage * 100:.2f}%",
        overall_avg_latency_ms=round(overall_latency, 2),
        failed_test_ids=failed_test_ids,
        test_results=test_results,
    )

# Console report printer
def print_report(report: EvaluationReport) -> None:
    """Print a plain-text evaluation summary to stdout."""
    print()
    print("Evaluation Summary")
    print(f"  Total Tests        : {report.total_tests}")
    print(f"  Passed             : {report.passed_tests}")
    print(f"  Failed             : {report.failed_tests}")
    print(f"  Intent Accuracy    : {report.overall_intent_accuracy}")
    print(f"  Response Pass Rate : {report.overall_response_pass_rate}")
    print(f"  Avg Coverage       : {report.overall_avg_coverage_rate}")
    print(f"  Avg Latency        : {report.overall_avg_latency_ms:.1f} ms")
    if report.failed_test_ids:
        print(f"  Failed Tests       : {', '.join(report.failed_test_ids)}")
    else:
        print("  Result             : All tests passed")
    print()


# CLI argument parser
def _build_parser() -> argparse.ArgumentParser:
    # Build the argument parser. All flags are optional and fall back to .env.
    parser = argparse.ArgumentParser(
        prog="run_tests.py",
        description="Automated Chatbot Evaluation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to the JSON test-case dataset. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL of the chatbot API. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of independent runs per test case. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=None,
        help="Min cosine similarity for semantic keyword match. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=None,
        help="Fraction of keywords that must be covered to pass a turn. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the JSON evaluation report. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum concurrent test conversations. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request HTTP timeout in seconds. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Sentence-transformer model name. (default: from evaluator/.env)",
    )
    parser.add_argument(
        "--force-onnx-export",
        action="store_true",
        help="Force re-export of the ONNX model, bypassing cache.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for spaCy lemma matching. (default: from evaluator/.env)",
    )
    return parser


# Entry point
def main() -> None:
    """
    Parse CLI args, merge with .env defaults, run the pipeline, write report.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Start from .env defaults, then override only explicitly supplied CLI flags.
    # argparse stores None for any argument not supplied on the command line
    # (we set default=None on each optional arg for exactly this purpose).
    env_config = EvaluatorConfig()  # loads evaluator/.env

    config = EvaluatorConfig(
        base_url=args.base_url or env_config.base_url,
        dataset_path=args.dataset or env_config.dataset_path,
        output_path=args.output or env_config.output_path,
        runs=args.runs if args.runs is not None else env_config.runs,
        semantic_threshold=(
            args.semantic_threshold
            if args.semantic_threshold is not None
            else env_config.semantic_threshold
        ),
        coverage_threshold=(
            args.coverage_threshold
            if args.coverage_threshold is not None
            else env_config.coverage_threshold
        ),
        request_timeout=(
            args.timeout if args.timeout is not None else env_config.request_timeout
        ),
        max_concurrency=(
            args.concurrency if args.concurrency is not None else env_config.max_concurrency
        ),
        log_level=args.log_level or env_config.log_level,
        model_name=args.model_name or env_config.model_name,
        force_onnx_export=args.force_onnx_export,
        language=args.language or env_config.language,
    )

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    print(f"\nStarting evaluation against {config.base_url}")
    print(
        f"Dataset: {config.dataset_path}  |  Runs: {config.runs}  |  "
        f"Semantic >= {config.semantic_threshold}  |  Coverage >= {config.coverage_threshold:.0%}\n"
    )

    report = asyncio.run(run_pipeline(config))

    # Write the JSON report to disk
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"Report written to: {output_path}")

    # Print console summary
    print_report(report)

    # Exit with non-zero code if any tests failed (CI-friendly)
    sys.exit(0 if report.failed_tests == 0 else 1)


if __name__ == "__main__":
    main()
