# Chatbot Evaluation Pipeline

This project is a backend evaluation system built for the Chatbot Backend Engineer coding challenge.
It consists of two independent components that work together: a **Mock Chatbot API** built with FastAPI
and an **Evaluation Pipeline** that runs automated tests against it.

---

## Project Overview

The idea is straightforward , we simulate a real LLM-backed system chatbot, then evaluate it
programmatically across multiple conversations. The evaluator runs each test case several times,
scores it using two validation stages (lemma matching + semantic similarity), and produces a
structured JSON report.

### Why two components?

The mock API exists to simulate realistic chatbot behaviour, including intentional hallucinations
(~5% rate) and variable response latency, without needing a real LLM. This lets the evaluator test
its own robustness against an unpredictable backend , exactly what you'd face in production.

---

## Part 1 — Mock Chatbot API (FastAPI)

**Location:** `mock_api/`

The mock API is a FastAPI service that exposes a single `POST /chat` endpoint.
It mimics an LLM-backed system assistant for Chatbot International System of Applied Sciences.

### How it works

1. **Intent detection** — incoming messages are matched against a keyword-based intent database.
   Each intent (e.g. `tuition_inquiry`, `enrollment_inquiry`) has a list of trigger keywords.
   The intent with the most keyword hits wins. Ties are broken by position — domain-specific
   intents are listed before conversational ones (greeting, farewell), so `"Hi, how do I enroll?"`
   resolves to `enrollment_inquiry`, not `greeting`.

2. **Response generation** — once the intent is identified, a response is randomly chosen from
   a pool of template responses for that intent. This gives natural variation across runs.

3. **Hallucination simulation** — with ~5% probability, the API deliberately returns a wrong intent
   and off-topic response. This tests the evaluation pipeline's resilience through majority voting.

4. **Latency simulation** — each request includes an `asyncio.sleep` of 200–800 ms to simulate
   real LLM response times.

### Supported intents

| Intent | What it handles |
|---|---|
| `tuition_inquiry` | Tuition fees, payment plans, costs |
| `enrollment_inquiry` | How to apply, admission requirements |
| `course_inquiry` | Programme catalogue, available subjects |
| `student_support` | Academic help, student services |
| `exam_inquiry` | Exam format, grading, pass marks |
| `general_inquiry` | Anything that doesn't match a specific intent |
| `greeting` | Conversational openers , intentionally last to lose ties |
| `farewell` | Closing messages , intentionally last to lose ties |

### Running the mock API

```bash
# From the project root
python -m mock_api.main
# Server starts at http://localhost:8080
```

### Example request

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-001", "message": "How much does a program cost?"}'
```

### Example response

```json
{
  "intent": "tuition_inquiry",
  "response": "Tuition fees at Chatbot depend on the program and study model. On average, fees range from €3,000 to €20,000 per year.",
  "confidence": 0.91
}
```

---

## Part 2 — Evaluation Pipeline

**Location:** `evaluator/`

The evaluator is a CLI tool that runs structured test conversations against any chatbot API
and scores the results. It reads test cases from a JSON dataset, runs them N times each,
and applies a majority vote to decide pass or fail.

### Running the evaluation

```bash
python -m evaluator.run_tests \
  --dataset test_cases.json \
  --base-url http://localhost:8080 \
  --runs 3 \
  --output report.json
```

All flags are optional , defaults are read from `evaluator/.env`.

### How it works

```
test_cases.json
      │
      ▼
asyncio.gather ──► [testCase-001] ──► Run 1 ──► Turn 1 → Turn 2 → ...
                   [testCase-002]     Run 2         └── per-turn:
                   [testCase-N]       Run 3              ├── intent check
                                        │                └── response validation
                                    Majority vote
                                        │
                                  report.json + console summary
```

- All test cases run **in parallel** via `asyncio.gather` + `asyncio.Semaphore`
- Turns within a conversation run **sequentially** to preserve session state
- Each test case is evaluated N times; majority vote decides the final verdict

### ONNX Runtime acceleration

The semantic similarity model (`all-MiniLM-L6-v2`) is accelerated using **ONNX Runtime**.
When the evaluation pipeline starts for the first time, it automatically exports the model
from PyTorch weights to an ONNX format and saves it to `evaluator/onnx_models/`.
On every subsequent run it loads the ONNX model directly , significantly faster than
reloading from PyTorch.

```
First run:
  Converting all-MiniLM-L6-v2 to ONNX (one-time setup)...
  ONNX model saved. Loading...

All subsequent runs:
  Loading ONNX model from evaluator/onnx_models/all-MiniLM-L6-v2...
```

The `evaluator/onnx_models/` directory is gitignored , it is generated locally and not committed.

> [!TIP]
> **Changing models:** If you change `EVAL_MODEL_NAME` in `.env` or pass `--model-name <new_model>` to the CLI, the new model will automatically export to its own folder. If you want to force-rebuild the ONNX cache for the currently selected model, run the evaluator with the `--force-onnx-export` flag.

### How each turn is scored

Each turn in a conversation is evaluated on two things independently:

#### Intent matching

The API's response includes an `intent` field. The evaluator compares it to `expected_intent`
from the test case , it's a simple string match (case-insensitive). Either it matches or it doesn't.
This is what `intent_match` is in the report.

#### Keyword validation (Two-Stage)

Each keyword in `expected_response_keywords` is checked against the chatbot's response text.
The two stages run in order , Stage 2 only runs when Stage 1 fails for a specific keyword.

**Stage 1 — spaCy lemma match**
spaCy lemmatises every token in the response and the keyword itself, then checks if any of them match.
This handles inflected forms without any neural model:
- `fees` → lemma `fee` → matches keyword `fee`
- `enrolling` → lemma `enroll` → matches keyword `enroll`

> [!TIP]
> **Multiple Languages:** By default, Stage 1 uses English (`en_core_web_sm`). If you are evaluating a chatbot in a different language, you can change the spaCy model using `EVAL_LANGUAGE` in `.env` or passing `--language <code>` to the CLI. For example, pass `--language de` for German, which will download and use `de_core_news_sm`, or `--language fr` for French (`fr_core_news_sm`).

If Stage 1 matches: `semantic_score = 0.0` (Stage 2 is skipped), the keyword appears as `"fee (lemma)"` in the report.

**Stage 2 — Semantic similarity (runs only if Stage 1 failed)**
The keyword and the full response are both encoded into vectors by `all-MiniLM-L6-v2` (via ONNX Runtime).
Cosine similarity is computed between them. If the score is above `semantic_threshold`, the keyword is covered.
This catches cases that Stage 1 misses , paraphrasing, synonyms, concept overlap:
- keyword `cost` in a response that says `pricing`
- keyword `tuition` in a response about `fees` (different lemma, same concept)

If Stage 2 matches: `semantic_score` = the actual cosine score, the keyword appears as `"fee (semantic)"` in the report.

**Pass / Fail:** a turn passes response validation if at least one keyword is covered by either stage.
`coverage_rate` is stored in the report for reference but is not used for pass/fail.

---

## CLI Reference

```
python -m evaluator.run_tests [OPTIONS]

  --dataset PATH               Test dataset JSON file        [default: from .env]
  --base-url URL               Chatbot API base URL          [default: from .env]
  --runs N                     Runs per test case            [default: from .env]
  --semantic-threshold FLOAT   Min cosine similarity (0–1)   [default: from .env]
  --coverage-threshold FLOAT   Keyword coverage (informational) [default: from .env]
  --model-name NAME            Sentence-transformer model    [default: from .env]
  --language CODE              Language for lemma matching   [default: from .env]
  --output PATH                Output report path            [default: from .env]
  --concurrency N              Max parallel conversations    [default: from .env]
  --timeout FLOAT              Per-request timeout (s)       [default: from .env]
  --log-level LEVEL            DEBUG/INFO/WARNING/ERROR      [default: from .env]
  --force-onnx-export          Delete cached ONNX model and re-export
```

---

## Console Output

```
Starting evaluation against http://localhost:8080
Dataset: test_cases.json  |  Runs: 3  |  Semantic >= 0.4  |  Coverage >= 30%

Loading ONNX model from evaluator/onnx_models/all-MiniLM-L6-v2...
Report written to: report.json

Evaluation Summary
  Total Tests        : 15
  Passed             : 15
  Failed             : 0
  Intent Accuracy    : 97.22%
  Response Pass Rate : 96.85%
  Avg Coverage       : 56.00%
  Avg Latency        : 748.3 ms
  Result             : All tests passed
```

---

## Installation

See [INSTALLATION.md](INSTALLATION.md) for full setup instructions including Python environment,
dependency installation, and environment configuration.

---

## Trade-offs

The most significant design decision in the validation pipeline was **not using an LLM for keyword validation**.

An LLM (e.g. GPT-4 or a local model) could replace both stages , instead of lemma matching and
embedding similarity, you'd prompt the LLM with the chatbot response and the keyword list and ask it
to judge whether each concept is present. This would handle indirect references, negations, and
contextual paraphrasing much better than either stage does today.

The reason it wasn't used here: for a scoring pipeline that runs hundreds of keyword checks across
multiple turns and multiple runs, LLM calls would add significant latency and cost per keyword,
and introduce non-determinism into what should be a reproducible test. The two-stage approach
(fast lemma match → embedding fallback) is deterministic, cheap, and fast enough for this use case.

---

## Limitations

> [!WARNING]
> - No real conversation state — each turn is evaluated independently in the mock
> - Keyword lists are hand-crafted; a production setup would use annotated golden references
> - The embedding model and spaCy model downloads require internet access on the very first run

---

## Ideas for Future Improvements

- LLM-as-judge for validation to catch indirect or paraphrased keyword matches
- HTML report alongside `report.json` for stakeholder review


---

## Report Field Reference

Most fields in `report.json` are self-explanatory. The fields below need a bit more context.

**`overall_intent_accuracy`** — looks at every single turn across all test cases and all runs,
and returns what percentage of them got the right intent. So if you ran 15 tests × 3 runs × 2 turns = 90 turns total
and 87 of them had the correct intent, this would be `96.67%`.

**`overall_response_pass_rate`** — same idea, but for response quality instead of intent.
It's the percentage of all turns (across all tests and all runs) where at least one keyword was found in the response.

**`overall_avg_coverage_rate`** — on average, what fraction of the expected keywords were found per turn.
If a turn had 5 keywords and 3 were covered, that turn's coverage is 0.60. This averages that across everything.
Note: this is informational only — a turn can pass with just 1 keyword covered regardless of this number.

**`majority_passed`** — the final verdict for a test case. If you run 3 times and 2 pass, this is `true`.
If only 1 out of 3 passes, this is `false`. It's always the majority that decides.

**`response_pass_rate`** (per-test or per-run) — unlike `overall_response_pass_rate` which covers everything,
this one is scoped to a single test case or a single run. It tells you what fraction of that test's turns
had at least one keyword covered.

**`response_passed`** (per-turn) — the simplest level. Just `true` or `false` for one specific turn:
did the chatbot's response contain at least one of the expected keywords?

**`covered_keywords`** — the keywords that were found, each labelled with how they were matched.
`"tuition (lemma)"` means spaCy found the word (or an inflected form of it).
`"fee (semantic)"` means the embedding similarity was above the threshold.

**`semantic_score`** — the best cosine similarity score produced by Stage 2 during this turn.
If all keywords were caught by Stage 1 (lemma), this will be `0.0` because Stage 2 never ran.
If Stage 2 ran for some keywords, this is the highest score it produced — not whether it passed,
just the raw score for reference.

**`coverage_rate`** — what fraction of the expected keywords were covered in this turn (`covered / total`).
This is stored for reference only. It has no effect on whether the turn passes or fails.

**`failure_reasons`** — only filled in when a test case fails the majority vote. Lists which turns
failed and why (wrong intent, no keywords found, etc.). Empty when the test passes.
