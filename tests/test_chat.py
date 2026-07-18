import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_chat_success(test_client):
    with patch("backend.main.full_retrieval_pipeline") as mock_retrieval, \
         patch("backend.main.rerank") as mock_rerank, \
         patch("backend.main.generate_completion") as mock_gen:
        
        mock_retrieval.return_value = ([{"id": "1", "payload": {"text": "Context", "document_id": "1", "document_name": "A", "chunk_index": 0}}], [])
        mock_rerank.return_value = [{"id": "1", "payload": {"text": "Context", "document_id": "1", "document_name": "A", "chunk_index": 0}, "rerank_score": 0.9}]
        mock_gen.return_value = "Grounded Answer"
        
        response = test_client.post("/api/chat", json={"query": "test query", "top_k": 5})
        assert response.status_code == 200


        data = response.json()
        assert data["answer"] == "Grounded Answer"
        assert len(data["sources"]) == 1

@pytest.mark.asyncio
async def test_chat_no_context(test_client):
    with patch("backend.main.full_retrieval_pipeline") as mock_retrieval:
        mock_retrieval.return_value = ([], [])
        
        response = test_client.post("/api/chat", json={"query": "unknown query"})
        assert response.status_code == 200
        data = response.json()
        assert "I don't have enough information" in data["answer"]
        assert len(data["sources"]) == 0

@pytest.mark.asyncio
async def test_chat_history(test_client):
    with patch("backend.main.full_retrieval_pipeline") as mock_retrieval, \
         patch("backend.main.generate_completion") as mock_gen, \
         patch("backend.main.rerank") as mock_rerank:
        
        mock_retrieval.return_value = ([{"id": "1", "payload": {"text": "Context"}}], [])
        mock_rerank.return_value = [{"id": "1", "payload": {"text": "Context"}, "rerank_score": 0.9}]
        mock_gen.return_value = "Answer"
        
        response = test_client.post("/api/chat", json={
            "query": "query",
            "history": [{"role": "user", "content": "prev"}]
        })
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_empty_query(test_client):
    response = test_client.post("/api/chat", json={"query": ""})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_query_too_long(test_client):
    long_query = "a" * 5000
    response = test_client.post("/api/chat", json={"query": long_query})
    assert response.status_code == 422


