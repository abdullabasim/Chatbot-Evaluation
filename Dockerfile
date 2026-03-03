# ==========================================
# Stage 1: Build & Install Dependencies
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build tools required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a virtual environment to copy them cleanly later
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download the necessary spaCy language model
RUN python -m spacy download en_core_web_sm


# ==========================================
# Stage 2: Minimal Runtime Environment
# ==========================================
FROM python:3.10-slim

# Set environment variables to optimize Python runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user for security (Professional Docker Best Practice)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Copy the application code
COPY --chown=appuser:appuser . .

# Default command (used primarily for the Mock API, can be overridden by docker-compose)
CMD ["python", "-m", "mock_api.main"]
