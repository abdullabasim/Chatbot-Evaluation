"""
Chat route — POST /chat endpoint.

Receives a ChatRequest, delegates to the mock logic layer for a probabilistic
LLM simulation, and returns a structured ChatResponse.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from mock_api.core.mock_logic import generate_mock_response
from mock_api.core.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the chatbot",
    description=(
        "Accepts a user message and returns a chatbot response with the "
        "detected intent and confidence score. Simulates LLM latency and a "
        "~10% hallucination rate to mimic real-world unpredictability."
    ),
)
async def post_chat(request: ChatRequest) -> ChatResponse:
    """
    Process a single-turn chat message.

    Args:
        request: Validated ChatRequest containing user_id and message.

    Returns:
        ChatResponse with response text, intent label, and confidence score.

    Raises:
        HTTPException 500: If the mock logic layer raises an unexpected error.
    """
    logger.info("Received message from user_id=%s", request.user_id)
    try:
        result = await generate_mock_response(request.message)
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during mock response generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while generating response.",
        ) from exc

    logger.debug(
        "Responding to user_id=%s with intent=%s confidence=%.4f",
        request.user_id,
        result.intent,
        result.confidence,
    )
    return ChatResponse(
        response=result.response,
        intent=result.intent,
        confidence=result.confidence,
    )
