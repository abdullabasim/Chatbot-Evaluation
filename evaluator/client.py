"""
Async HTTP client wrapper for the chatbot API.

Wraps httpx.AsyncClient with connection pooling, per-request latency
tracking, and structured error logging. Always use as an async context
manager so the underlying connection is properly closed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx

from evaluator.config import EvaluatorConfig

logger = logging.getLogger(__name__)


class ChatAPIClient:
    """
    Thin async wrapper around httpx.AsyncClient.

    Use as an async context manager — it opens the connection on enter
    and closes it cleanly on exit:

        async with ChatAPIClient(config) as client:
            result = await client.send_message(user_id="u1", message="hello")
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    # Lifecycle

    async def __aenter__(self) -> "ChatAPIClient":
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.request_timeout),
            # Enable HTTP/2 for multiplexed connections when available
            http2=False,
            limits=httpx.Limits(
                max_connections=self._config.max_concurrency + 10,
                max_keepalive_connections=self._config.max_concurrency,
            ),
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # Public API

    async def send_message(
        self,
        user_id: str,
        message: str,
    ) -> tuple[dict, float]:
        """
        POST one message to /chat and return (response_dict, latency_ms).

        Args:
            user_id:  Session/user identifier sent in the request body.
            message:  The user's message text.

        Returns:
            Tuple of (parsed JSON dict, round-trip latency in ms).

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses.
            httpx.RequestError:    On network or connectivity failures.
        """
        if self._client is None:
            raise RuntimeError(
                "ChatAPIClient must be used as an async context manager."
            )

        payload = {"user_id": user_id, "message": message}
        t0 = time.perf_counter()

        max_attempts = self._config.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self._client.post("/chat", json=payload)
                resp.raise_for_status()
                break  # Success
            except httpx.HTTPStatusError as exc:
                # Retry on 5xx Server Errors, but not 4xx Client Errors
                if exc.response.status_code < 500 or attempt == max_attempts:
                    logger.warning(
                        "HTTP %d from /chat for user_id=%s: %s",
                        exc.response.status_code,
                        user_id,
                        exc.response.text,
                    )
                    raise
                logger.warning(
                    "HTTP %d error for user_id=%s. Retrying (%d/%d)...",
                    exc.response.status_code,
                    user_id,
                    attempt,
                    self._config.max_retries,
                )
            except httpx.RequestError as exc:
                if attempt == max_attempts:
                    logger.error("Network error for user_id=%s: %s", user_id, exc)
                    raise
                logger.warning(
                    "Network error (%s) for user_id=%s. Retrying (%d/%d)...",
                    type(exc).__name__,
                    user_id,
                    attempt,
                    self._config.max_retries,
                )
            
            # Exponential backoff: 1s, 2s, 4s...
            await asyncio.sleep(2 ** (attempt - 1))

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "user_id=%s  latency=%.1f ms  message=%r",
            user_id,
            latency_ms,
            message[:60],
        )
        return resp.json(), latency_ms
