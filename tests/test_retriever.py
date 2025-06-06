import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from ms_potts.retriever import Retriever


@pytest.fixture
def mock_retriever(monkeypatch):
    # Mock SentenceTransformer.encode()
    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
    monkeypatch.setattr(
        "ms_potts.retriever.SentenceTransformer", lambda model_name: mock_embed_model
    )

    # Mock pd.read_csv to return a fake DataFrame-like object
    with patch("ms_potts.retriever.pd.read_csv") as mock_read_csv:
        mock_df = MagicMock()

        # Mock embedding column behavior
        embedding_series = MagicMock()
        apply_mock = MagicMock()
        apply_mock.tolist.return_value = [np.array([0.1, 0.2, 0.3])]
        embedding_series.apply.return_value = apply_mock

        # Mock sentence_chunk column with .tolist()
        sentence_chunk_series = MagicMock()
        sentence_chunk_series.tolist.return_value = ["Sample nutrition content"]

        mock_df.__getitem__.side_effect = lambda key: (
            embedding_series if key == "embedding" else sentence_chunk_series
        )
        mock_read_csv.return_value = mock_df

        return Retriever()


def test_embed_query(mock_retriever):
    embedding = mock_retriever.embed_query("What should I eat?")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (3,)


def test_retrieve_in_scope(mock_retriever):
    response = mock_retriever.retrieve("Give me a healthy breakfast option")
    assert isinstance(response, str)
    assert "Sample nutrition content" in response or "nutrition" in response.lower()
