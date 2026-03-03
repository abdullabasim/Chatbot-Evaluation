# Installation & Setup

## Prerequisites

- Python 3.10 or later
- pip

---

## 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `fastapi` + `uvicorn` — mock API server
- `httpx` — async HTTP client for the evaluator
- `pydantic` + `pydantic-settings` — config and data models
- `spacy` — lemmatizer for Stage 1 keyword validation
- `sentence-transformers[onnx]` + `onnxruntime` — semantic similarity model with ONNX Runtime acceleration
- `scikit-learn` — cosine similarity utilities

---

## 3. Download the spaCy language model

The evaluator uses spaCy for lemmatization during Stage 1 keyword matching. Download the language model that matches your configured `EVAL_LANGUAGE` (default is English).

For **English** (`EVAL_LANGUAGE=en`):
```bash
python -m spacy download en_core_web_sm
```

If you configure the evaluator to test in another language (e.g., German `EVAL_LANGUAGE=de`), you can download the respective model directly:
```bash
python -m spacy download de_core_news_sm
```

*(Note: If the evaluator runs and detects that the required model is missing, it will attempt to automatically download it for you, but it is best practice to install it upfront!)*

---

## 4. Configure environment files

## 4. Configure environment files

Both the Mock API and Evaluator have their own configuration. Copy the example files:

```bash
cp evaluator/.env.example evaluator/.env
cp mock_api/.env.example mock_api/.env
```

**`evaluator/.env` options:**

```env
EVAL_BASE_URL=http://localhost:8080   # chatbot API to evaluate
EVAL_DATASET_PATH=test_cases.json     # test cases to load
EVAL_OUTPUT_PATH=report.json          # where to write the report
EVAL_RUNS=3                           # runs per test case (majority vote)
EVAL_MAX_RETRIES=3                    # max retries for transient network errors
EVAL_SEMANTIC_THRESHOLD=0.45          # cosine similarity cutoff for Stage 2
EVAL_COVERAGE_THRESHOLD=0.50          # informational — not used for pass/fail
EVAL_MODEL_NAME=all-MiniLM-L6-v2      # huggingface sentence-transformer model
EVAL_LANGUAGE=en                      # spacy language lemma target
EVAL_MAX_CONCURRENCY=20               # parallel test conversations
EVAL_REQUEST_TIMEOUT=10.0             # per-request timeout in seconds
EVAL_LOG_LEVEL=WARNING                # DEBUG / INFO / WARNING / ERROR
```

**`mock_api/.env` options:**

```env
MOCK_API_HOST=0.0.0.0                 # host to bind the API to
MOCK_API_PORT=8080                    # port to bind the API to
MOCK_WORKERS=1                        # number of uvicorn workers for concurrency
MOCK_HALLUCINATION_RATE=0.05          # probability (0.0-1.0) of a wrong answer
MOCK_MIN_LATENCY_MS=200               # minimum simulated LLM reply latency
MOCK_MAX_LATENCY_MS=800               # maximum simulated LLM reply latency
MOCK_CONFIDENCE_MIN=0.72              # minimum confidence for correct answers
MOCK_CONFIDENCE_MAX=0.99              # maximum confidence for correct answers
MOCK_HALLUCINATION_CONFIDENCE_MIN=0.10 # min confidence for hallucinated answers
MOCK_HALLUCINATION_CONFIDENCE_MAX=0.45 # max confidence for hallucinated answers
MOCK_LOG_LEVEL=INFO                   # logging verbosity
```

---

## 5. ONNX model (auto-generated)

On the first run of the evaluation pipeline, the sentence-transformer model is automatically
exported from PyTorch to ONNX format.

You do **not** need to do anything manually — it happens at startup:

```
First run:
  Converting all-MiniLM-L6-v2 to ONNX (one-time setup)...
  ONNX model saved. Loading...

All subsequent runs:
  Loading ONNX model from evaluator/onnx_models/all-MiniLM-L6-v2...
```

**Changing models:** If you update `EVAL_MODEL_NAME` in your `.env` file to a different sentence-transformer model, the evaluator will automatically download it and export it to a new ONNX bundle on the next run. You do not need to delete the old one. If you ever need to manually force a rebuild of the currently selected model, run the evaluator with the `--force-onnx-export` flag.

> [!NOTE]
> The first run also downloads `all-MiniLM-L6-v2` (~80 MB) from Hugging Face if it is not
> already cached in `~/.cache/huggingface/`. An internet connection is required for this step only.

---

## 6. Running the project

**Start the mock API (Terminal 1):**

```bash
python -m mock_api.main
```

**Run the evaluation (Terminal 2):**

```bash
python -m evaluator.run_tests \
  --dataset test_cases.json \
  --base-url http://localhost:8080 \
  --runs 3 \
  --output report.json
```

All CLI flags are optional — they fall back to values in `evaluator/.env`.

---

## Docker Compose (optional)

If you prefer to run both the API and the evaluator simultaneously in isolated containers, you can use Docker Compose.

The `docker-compose.yml` is configured to build directly from the `main` branch of the GitHub repository. It will automatically duplicate the `.env.example` files into active `.env` files inside the running containers.

Just download the `docker-compose.yml` file to any directory on your computer, and run:

```bash
# Pull from GitHub, build, and start the system
docker compose up --build
```

This will automatically:
1. Clone the repository and build the environment.
2. Boot the API.
3. Boot the Evaluator container, run the tests against the API container (`http://api:8080`), and sink the resulting `report.json` right into your current local directory!
