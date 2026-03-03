# Installation & Setup

## Prerequisites

- Python 3.10 or later
- pip

---

## 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
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

```bash
python -m spacy download en_core_web_sm
```

This model is used by the evaluator for Stage 1 (lemma-based keyword matching).

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

The `docker-compose.yml` is configured to pull the pre-built image from GitHub Container Registry (make sure to replace the image path with your actual GitHub repository URL once uploaded).

```bash
# Start the mock API and run the evaluator
docker compose up
```

This will automatically pass `test_cases.json` to the evaluator container, run the tests against the API container (`http://api:8080`), and drop the resulting `report.json` right into your current directory via a volume mount!
