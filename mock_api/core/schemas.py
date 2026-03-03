"""
Pydantic schemas for the Mock Chatbot API.
Defines the strict input/output contracts for the /chat endpoint.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming request body for the /chat endpoint."""

    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the user/session.",
        examples=["user-001"],
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The user's conversational message.",
        examples=["I want to book a flight to Berlin."],
    )


class ChatResponse(BaseModel):
    """Response payload returned by the /chat endpoint."""

    response: str = Field(
        ...,
        description="The chatbot's natural-language reply.",
    )
    intent: str = Field(
        ...,
        description="The detected intent label for the user's message.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the intent classifier (0.0–1.0).",
    )
