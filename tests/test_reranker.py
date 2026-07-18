import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_cross_encoder():
    with patch("backend.services.reranker.CrossEncoder") as mock_ce:
        instance = mock_ce.return_value
        # return dummy scores based on input length or just static
        instance.predict.side_effect = lambda pairs: [0.9 if "high" in p[1] else 0.1 for p in pairs]
        yield mock_ce

from backend.services.reranker import rerank, filter_by_threshold

def test_rerank(mock_cross_encoder):
    chunks = [
        {"payload": {"text": "low score text"}},
        {"payload": {"text": "high score text"}}
    ]
    reranked = rerank("query", chunks, top_k=5)
    assert len(reranked) == 2
    assert "rerank_score" in reranked[0]
    assert reranked[0]["payload"]["text"] == "high score text"
    assert reranked[1]["payload"]["text"] == "low score text"

def test_top_k_limiting(mock_cross_encoder):
    chunks = [{"payload": {"text": "text"}}] * 5
    reranked = rerank("query", chunks, top_k=2)
    assert len(reranked) == 2

def test_threshold_filtering():
    chunks = [
        {"rerank_score": 0.9, "text": "good"},
        {"rerank_score": 0.2, "text": "bad"}
    ]
    filtered = filter_by_threshold(chunks, threshold=0.5)
    assert len(filtered) == 1
    assert filtered[0]["text"] == "good"

def test_empty_input_handling(mock_cross_encoder):
    assert rerank("query", []) == []
    assert filter_by_threshold([], 0.5) == []
