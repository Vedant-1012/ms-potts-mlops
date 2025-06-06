import pytest
from unittest.mock import MagicMock
from ms_potts.model_gemini import GeminiModel


@pytest.fixture
def mock_gemini(monkeypatch):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text.strip.return_value = (
        "Mocked Gemini reply"
    )

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model

    monkeypatch.setattr("ms_potts.model_gemini.genai", mock_genai)
    return mock_model


@pytest.fixture
def mock_dependencies(monkeypatch):
    # Mock WHOBookRetriever
    mock_retriever = MagicMock()
    mock_retriever.embed_query.return_value = MagicMock(shape=(1, 768))
    mock_retriever.retrieve.return_value = "Mocked retrieved content"
    monkeypatch.setattr(
        "ms_potts.model_gemini.WHOBookRetriever", lambda: mock_retriever
    )

    # Mock IntentClassifier
    mock_classifier = MagicMock()
    mock_classifier.classify_from_embedding.return_value = {
        "top_intent": "General-Question"
    }
    monkeypatch.setattr(
        "ms_potts.model_gemini.IntentClassifier", lambda: mock_classifier
    )

    # Mock monitor
    monkeypatch.setattr("ms_potts.model_gemini.ModelMonitor", MagicMock())
    monkeypatch.setattr("ms_potts.model_gemini.tracer", MagicMock())


def test_get_response_empty_query(mock_dependencies):
    model = GeminiModel()
    result = model.get_response("")
    assert result["final_answer"] == "Please provide a valid query."
    assert result["detected_intent"] is None


def test_get_response_valid(mock_gemini, mock_dependencies):
    model = GeminiModel()
    result = model.get_response("What is a balanced diet?", user_context={"age": 30})
    assert isinstance(result["final_answer"], str)
    assert "final_answer" in result
    assert "detected_intent" in result
