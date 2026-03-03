import pytest
from fastapi.testclient import TestClient
from mock_api.main import app

client = TestClient(app)

def test_chat_endpoint_success():
    """Test that a valid chat request returns the expected schema and 200 OK."""
    payload = {
        "user_id": "test-user-123",
        "message": "Hello, I am interested in computer science programs."
    }
    
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Assert expected structure
    assert "response" in data
    assert "intent" in data
    assert "confidence" in data
    
    # Assert typing
    assert isinstance(data["response"], str)
    assert isinstance(data["intent"], str)
    assert isinstance(data["confidence"], float)
    
    # Assert confidence bounds
    assert 0.0 <= data["confidence"] <= 1.0


def test_chat_endpoint_missing_fields():
    """Test that missing required fields returns a 422 Validation Error."""
    payload = {
        "user_id": "test-user-123"
        # missing "message"
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_chat_endpoint_invalid_types():
    """Test that invalid types return a 422 Validation Error."""
    payload = {
        "user_id": 12345, # should be string
        "message": ["hello"] # should be string
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
