"""
Chatbot Response Quality Assessor
===================================

Keyword coverage check — two-stage per-keyword logic:

  Stage 1 — spaCy lemmatized exact match
      Load spaCy's en_core_web_sm with only the tagger + lemmatizer active
      (parser and NER disabled for speed).

      Lemmatise both the keyword and every token in the response.
      "fees" → "fee"  |  "enrolling" → "enroll"  |  "counseling" → "counsel"

      If the keyword's lemma is found among the response lemmas → ✅ covered.
      The neural model is NOT invoked.

  Stage 2 — Semantic similarity fallback (only if Stage 1 failed)
      Encode the response and the keyword with a sentence-transformer and
      compute cosine similarity. If score >= semantic_threshold → ✅ covered.

  Final verdict per keyword:
      covered = lemma_match OR semantic_match

  Final verdict per turn (pass/fail):
      response_passed = ANY keyword is covered
      ── The turn passes as soon as ONE keyword is matched (lemma or semantic).
      ── This matches the requirement: "keyword match OR semantic similarity".

  coverage_threshold (INFORMATIONAL ONLY):
      Kept as a parameter so callers can inspect what fraction of keywords
      were covered (coverage_rate), but it does NOT influence pass/fail.
      coverage_rate is reported in QualityReport for diagnostic purposes.

"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import spacy
from sentence_transformers import SentenceTransformer, util

import subprocess

# Model map: mapping from ISO language code to spaCy model name
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm",
}

# Module-level caches for spaCy Docs and single-word lemmas.
#
# A response is typically checked against MANY keywords in one assess() call.
# Without caching, _nlp(response) would be called once per keyword — O(N).
# With _DOC_CACHE we parse each unique response exactly ONCE — O(1) after
# the first call.  Similarly, _LEMMA_CACHE avoids re-parsing the same
# keyword word across repeated assess() calls (e.g. across multiple runs).

_DOC_CACHE: dict[str, object] = {}   # response text  → spaCy Doc
_LEMMA_CACHE: dict[str, str] = {}    # single word    → lemma string

_nlp_instance = None


def _get_nlp(language: str = "en"):
    """Load the spaCy model for the given language, downloading if needed."""
    global _nlp_instance
    if _nlp_instance is not None:
        return _nlp_instance

    model_name = SPACY_MODELS.get(language.lower(), "en_core_web_sm")
    try:
        _nlp_instance = spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        print(f"Downloading spaCy model '{model_name}' for language '{language}'...", flush=True)
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        _nlp_instance = spacy.load(model_name, disable=["parser", "ner"])
    
    return _nlp_instance


def _get_doc(text: str, language: str = "en"):
    """Return a cached spaCy Doc for *text*, parsing only on first call."""
    key = text.lower()
    if key not in _DOC_CACHE:
        nlp = _get_nlp(language)
        _DOC_CACHE[key] = nlp(key)
    return _DOC_CACHE[key]


def _lemmatize_word(word: str, language: str = "en") -> str:
    """Return the cached lemma of a single word."""
    key = word.lower()
    if key not in _LEMMA_CACHE:
        nlp = _get_nlp(language)
        doc = nlp(key)
        _LEMMA_CACHE[key] = doc[0].lemma_ if doc else key
    return _LEMMA_CACHE[key]


# Data classes
@dataclass
class KeywordResult:
    """Per-keyword evaluation result."""

    keyword: str

    # Stage 1 — spaCy lemmatized exact match
    lemma_match: bool        # keyword lemma found in response lemmas
    matched_word: str        # original response token that matched ("" if none)
    keyword_lemma: str       # spaCy lemma of the keyword

    # Stage 2 — semantic match (only computed if Stage 1 failed)
    semantic_score: float    # cosine similarity 0–1 (0.0 if stage 1 passed)
    semantic_match: bool

    # Final decision
    covered: bool
    match_reason: str        # "lemma" | "semantic" | "none"


@dataclass
class QualityReport:
    """Result of assessing one chatbot response against a keyword list."""

    response: str
    keyword_results: list[KeywordResult]
    covered_keywords: list[str]
    missing_keywords: list[str]
    coverage_rate: float        # covered / total  (0.0 – 1.0)
    response_passed: bool       # coverage_rate >= coverage_threshold
    elapsed_ms: float



def _load_or_export_onnx_model(model_name: str, force_export: bool = False) -> "SentenceTransformer":
    """
    Load the sentence-transformer model from a local ONNX export if it exists,
    otherwise export it first and then load.

    The ONNX model is saved to evaluator/onnx_models/<model_name>/ the first
    time this runs. On every subsequent startup it loads straight from disk,
    which is significantly faster than reloading from PyTorch weights.

    Args:
        model_name: HuggingFace model name, e.g. "all-MiniLM-L6-v2".
        force_export: If True, delete the existing ONNX cache and re-export.

    Returns:
        A SentenceTransformer instance backed by ONNX Runtime.
    """
    # Store the ONNX model next to this file, inside the evaluator package.
    base_dir = Path(__file__).resolve().parent
    onnx_dir = base_dir / "onnx_models" / model_name.replace("/", "_")

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # 3 = ERROR, suppresses warnings
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("optimum").setLevel(logging.ERROR)
    if force_export and onnx_dir.exists():
        import shutil
        print(f"Force re-export requested. Deleting existing ONNX cache for {model_name}...", flush=True)
        shutil.rmtree(onnx_dir)

    if not onnx_dir.exists():
        print(f"Converting {model_name} to ONNX...", flush=True)
        onnx_dir.mkdir(parents=True, exist_ok=True)
        # Export to ONNX. Suppress all output at the OS fd level so C-level
        # ORT / CoreML warnings don't bleed through to the terminal.
        with open(os.devnull, "w") as devnull, \
             contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            # Also dup2 the real fd-2 to /dev/null for C-level stderr
            _old_fd2 = os.dup(2)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            try:
                tmp = SentenceTransformer(model_name, backend="onnx")
                tmp.save(str(onnx_dir))
            finally:
                os.dup2(_old_fd2, 2)
                os.close(_old_fd2)
        print("ONNX model saved. Loading...", flush=True)
    else:
        print(f"Loading ONNX model from {onnx_dir.relative_to(base_dir.parent)}...", flush=True)

    # Same OS-level fd suppression when loading from the ONNX cache.
    _old_fd2 = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    try:
        with open(os.devnull, "w") as devnull, \
             contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            model = SentenceTransformer(str(onnx_dir), backend="onnx")
    finally:
        os.dup2(_old_fd2, 2)
        os.close(_old_fd2)

    return model


class ResponseQualityAssessor:
    """
    Two-stage per-keyword response quality assessor.

    Stage 1 — spaCy lemmatized exact match:
      Lemmatise the keyword and every token in the response.
      If the keyword lemma appears → covered ✅  (no neural model needed).

    Stage 2 — Semantic fallback (only if Stage 1 failed):
      Compute cosine similarity between sentence-transformer embeddings.
      If score >= semantic_threshold → covered ✅.

    Pass/Fail verdict:
      A turn PASSES if ANY keyword is covered (lemma or semantic match).
      This is a deliberate "any-match" policy — one keyword hit is enough.

    coverage_threshold [INFORMATIONAL ONLY]:
      Does NOT affect response_passed. Stored and forwarded so callers can read
      QualityReport.coverage_rate (fraction of keywords covered) for diagnostics
      and reporting, without it changing the binary pass/fail outcome.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.45,
        coverage_threshold: float = 0.30,  # informational only — see class docstring
        force_onnx_export: bool = False,
        language: str = "en",
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.coverage_threshold = coverage_threshold  # kept for reporting, not pass/fail
        self.language = language
        self._model = _load_or_export_onnx_model(model_name, force_export=force_onnx_export)

        self._embed_cache: dict[str, object] = {}
        # spaCy Doc and lemma caches are module-level (_DOC_CACHE / _LEMMA_CACHE)
        # so they are shared across all assessor instances and persist for
        # the entire process lifetime — no repeated parsing across runs.



    def _embed(self, text: str):
        if text not in self._embed_cache:
            self._embed_cache[text] = self._model.encode(
                text, convert_to_tensor=True
            )
        return self._embed_cache[text]



    def _lemma_match(self, response: str, keyword: str) -> tuple[bool, str, str]:
        """
        Return (matched, matched_word, keyword_lemma).

        Uses the module-level _DOC_CACHE so the response is parsed by spaCy
        ONCE per unique text, regardless of how many keywords are checked.
        _LEMMA_CACHE avoids re-lemmatising the same keyword across runs.

        Uses spaCy's alpha-token filter for word-boundary safety —
        'coffee' never matches 'fee'.
        """
        keyword_lemma = _lemmatize_word(keyword, language=self.language)   # cached
        response_doc = _get_doc(response, language=self.language)          # cached — no re-parse

        for token in response_doc:
            if token.is_alpha and token.lemma_ == keyword_lemma:
                return True, token.text, keyword_lemma

        return False, "", keyword_lemma



    def _semantic_match(self, response: str, keyword: str) -> tuple[float, bool]:
        """Return (score, matched). Only called when Stage 1 failed."""
        score = float(util.cos_sim(self._embed(response), self._embed(keyword)))
        return round(score, 4), score >= self.semantic_threshold



    def _check_keyword(self, response: str, keyword: str) -> KeywordResult:
        """
        Stage 1: spaCy lemma match — fast, no neural model.
        Stage 2: semantic cosine similarity — only when Stage 1 fails.
        """
        t0 = time.perf_counter()

        # Phase 1: Try lemma exact-match first (very fast, deterministic)
        has_lemma, matched_word, keyword_lemma = self._lemma_match(
            response, keyword
        )

        if has_lemma:
            return KeywordResult(
                keyword=keyword,
                lemma_match=True,
                matched_word=matched_word,
                keyword_lemma=keyword_lemma,
                semantic_score=0.0,    # not computed — fast path
                semantic_match=False,
                covered=True,
                match_reason="lemma",
            )

        # Stage 2 fallback
        sem_score, sem_hit = self._semantic_match(response, keyword)

        return KeywordResult(
            keyword=keyword,
            lemma_match=False,
            matched_word="",
            keyword_lemma=keyword_lemma,
            semantic_score=sem_score,
            semantic_match=sem_hit,
            covered=sem_hit,
            match_reason="semantic" if sem_hit else "none",
        )



    def assess(
        self,
        response: str,
        keywords: list[str],
        semantic_threshold: float | None = None,
        coverage_threshold: float | None = None,
    ) -> QualityReport:
        """
        Assess response coverage for the given keyword list.

        Args:
            response           : chatbot message to evaluate
            keywords           : e.g. ["tuition", "fee", "cost"]
            semantic_threshold : cosine similarity cutoff for Stage 2 (semantic fallback)
            coverage_threshold : [INFORMATIONAL ONLY] fraction of keywords that must be
                                 covered. Does NOT affect response_passed — pass/fail
                                 is determined by any single keyword being covered.
                                 The value is stored in the assessor and forwarded so
                                 callers can compare it against coverage_rate in reports.
        """
        sem_thr = semantic_threshold if semantic_threshold is not None else self.semantic_threshold
        cov_thr = coverage_threshold if coverage_threshold is not None else self.coverage_threshold

        # Temporarily apply thresholds for this call
        orig_sem, orig_cov = self.semantic_threshold, self.coverage_threshold
        self.semantic_threshold, self.coverage_threshold = sem_thr, cov_thr

        t0 = time.perf_counter()
        results = [self._check_keyword(response, kw) for kw in keywords]
        results.sort(key=lambda r: (r.covered, r.semantic_score), reverse=True)

        covered = [r.keyword for r in results if r.covered]
        missing = [r.keyword for r in results if not r.covered]
        # coverage_rate: fraction of keywords matched — stored for diagnostics/reporting.
        # It is NOT used for pass/fail; see response_passed below.
        coverage_rate = len(covered) / len(results) if results else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        self.semantic_threshold, self.coverage_threshold = orig_sem, orig_cov

        # Pass/fail: ANY single keyword covered (lemma OR semantic) → turn passes.
        # Requirement: "keyword match OR semantic similarity threshold" — one match is enough.
        # coverage_threshold is intentionally NOT used here; it is informational only.
        response_passed = len(covered) > 0

        return QualityReport(
            response=response,
            keyword_results=results,
            covered_keywords=covered,
            missing_keywords=missing,
            coverage_rate=round(coverage_rate, 4),  # diagnostic: what % of keywords matched
            response_passed=response_passed,
            elapsed_ms=round(elapsed, 2),
        )

    def assess_batch(
        self,
        responses: list[str],
        keywords: list[str],
        semantic_threshold: float | None = None,
        coverage_threshold: float | None = None,
    ) -> list[QualityReport]:
        """Assess multiple responses. Embeddings are cached across all calls."""
        return [
            self.assess(r, keywords, semantic_threshold, coverage_threshold)
            for r in responses
        ]



    def explain(self, report: QualityReport) -> str:
        W = 82
        pct = round(report.coverage_rate * 100)
        verdict = "✅ PASS" if report.response_passed else "❌ FAIL"
        lines = [
            "─" * W,
            f'  Response : "{report.response[:76]}{"..." if len(report.response) > 76 else ""}"',
            f"  Coverage : {len(report.covered_keywords)}/{len(report.keyword_results)} "
            f"({pct}%)  [{verdict}]  |  {report.elapsed_ms:.1f}ms",
            "",
            f"  {'Keyword':<14} {'Lemma':<14} {'Stage 1 (spaCy)':^17} {'Stage 2 (sem)':^13} {'Score':^7} Result    Reason",
            f"  {'─'*14} {'─'*14} {'─'*17} {'─'*13} {'─'*7} {'─'*8}  {'─'*13}",
        ]
        for r in report.keyword_results:
            stage1 = f"✅ ({r.matched_word})" if r.lemma_match else "❌"
            stage2 = "─ (skipped)" if r.lemma_match else ("✅" if r.semantic_match else "❌")
            score  = "  ─  " if r.lemma_match else f"{r.semantic_score:.3f}"
            result = "✅ PASS" if r.covered else "❌ FAIL"
            reason = {
                "lemma":    "spaCy lemma",
                "semantic": "semantic",
                "none":     "no match",
            }[r.match_reason]
            lines.append(
                f"  {r.keyword:<14} {r.keyword_lemma:<14} {stage1:<17} {stage2:<13} "
                f"{score:^7} {result:<8}  {reason}"
            )
        if report.missing_keywords:
            lines.append(f"\n  ⚠  Missing : {', '.join(report.missing_keywords)}")
        lines.append("─" * W)
        return "\n".join(lines)

    def explain_batch(self, reports: list[QualityReport]) -> str:
        import numpy as np
        W = 78
        lines = [
            "═" * W,
            f"  BATCH SUMMARY — {len(reports)} responses",
            f"  {'#':<4} {'Coverage':<10} {'Pass?':<8} {'Covered':<26} Preview",
            f"  {'─'*4} {'─'*10} {'─'*8} {'─'*26} {'─'*22}",
        ]
        for i, r in enumerate(reports, 1):
            pct     = f"{round(r.coverage_rate * 100)}%"
            verdict = "✅" if r.response_passed else "❌"
            covered = ", ".join(r.covered_keywords) or "none"
            preview = r.response[:28] + ("..." if len(r.response) > 28 else "")
            lines.append(f"  {i:<4} {pct:<10} {verdict:<8} {covered:<26} \"{preview}\"")
        avg = round(np.mean([r.coverage_rate for r in reports]) * 100)
        lines += [f"  {'─'*4} {'─'*10}", f"  AVG  {avg}%", "═" * W]
        return "\n".join(lines)


# Singleton factory
_assessor_instance: ResponseQualityAssessor | None = None


def get_assessor(
    model_name: str = "all-MiniLM-L6-v2",
    semantic_threshold: float = 0.45,
    coverage_threshold: float = 0.50,
    force_onnx_export: bool = False,
    language: str = "en",
) -> ResponseQualityAssessor:
    """Return the shared assessor — sentence-transformer model loaded once."""
    global _assessor_instance
    if _assessor_instance is None:
        # Load spaCy model up-front to trigger auto-download if needed,
        # otherwise the first time we use it it prints to stdout out of order.
        _get_nlp(language)
        _assessor_instance = ResponseQualityAssessor(
            model_name=model_name,
            semantic_threshold=semantic_threshold,
            coverage_threshold=coverage_threshold,
            force_onnx_export=force_onnx_export,
            language=language,
        )
    return _assessor_instance


# Standalone demo  (python -m evaluator.validators)
if __name__ == "__main__":
    KEYWORDS = ["tuition", "fee", "program", "semester", "payment", "cost"]

    assessor = ResponseQualityAssessor(
        semantic_threshold=0.40,
        coverage_threshold=0.30,
    )

    print("\n=== STAGE TRACING — Single Response ===")
    response = (
        "The tuition for our MBA program is €8,000 per semester. "
        "You can pay in three installments. Additional charges may apply."
    )
    print(assessor.explain(assessor.assess(response, KEYWORDS)))

    print("\n=== BATCH ===")
    responses = [
        # Stage 1 fires: "fees"→"fee", "programs"→"program", "semesters"→"semester"
        "The tuition for our programs ranges from €5,000 to €12,000 per semester. "
        "Payment can be made in installments. All fees are listed on our website.",
        # Stage 2 fires: different words, same meaning
        "Annual charges for our courses vary. You can settle dues in monthly "
        "tranches. Pricing depends on the academic period you enroll in.",
        # Partial
        "Our degree programs are highly ranked. Contact admissions for details.",
        # Off-topic hallucination
        "Thank you for reaching out! Our campus is in downtown Berlin.",
    ]
    reports = assessor.assess_batch(responses, KEYWORDS)
    for i, r in enumerate(reports, 1):
        print(f"\n--- Response #{i} ---")
        print(assessor.explain(r))
    print(assessor.explain_batch(reports))

    print("\n--- Interactive (type 'quit' to exit) ---\n")
    while True:
        text = input("Chatbot response: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if text:
            print(assessor.explain(assessor.assess(text, KEYWORDS)))
