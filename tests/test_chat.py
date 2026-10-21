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


@pytest.mark.asyncio
async def test_chat_with_longllmlingua_compression(test_client):
    with patch("backend.main.full_retrieval_pipeline") as mock_retrieval, \
         patch("backend.main.rerank") as mock_rerank, \
         patch("backend.main.compress_context_with_longllmlingua") as mock_compress, \
         patch("backend.main.generate_completion") as mock_gen:

        mock_retrieval.return_value = ([{
            "id": "1",
            "payload": {
                "text": "Detailed context chunk about artificial intelligence and neural networks.",
                "document_id": "doc-1",
                "document_name": "ai.pdf",
                "chunk_index": 0
            }
        }], [])
        mock_rerank.return_value = [{
            "id": "1",
            "payload": {
                "text": "Detailed context chunk about artificial intelligence and neural networks.",
                "document_id": "doc-1",
                "document_name": "ai.pdf",
                "chunk_index": 0
            },
            "rerank_score": 0.95
        }]
        mock_compress.return_value = {
            "compressed_context": "[1] AI neural networks.",
            "compressed_prompt": "Context:\n[1] AI neural networks.\n\nQuestion: What is AI?",
            "origin_tokens": 80,
            "compressed_tokens": 35,
            "ratio": "2.3x",
            "rate": 0.44,
            "saving": "56.2%",
            "compression_time_ms": 12.5,
            "enabled": True,
            "applied": True,
            "fallback": False,
            "model_name": "gpt2"
        }
        mock_gen.return_value = "AI is artificial intelligence."

        response = test_client.post("/api/chat", json={"query": "What is AI?", "top_k": 3})
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "AI is artificial intelligence."
        assert data["compression_ratio"] == "2.3x"
        assert data["retrieval_metadata"]["longllmlingua_applied"] is True
        assert data["retrieval_metadata"]["compressed_tokens"] == 35
        assert data["retrieval_metadata"]["original_tokens"] == 80
        assert data["retrieval_metadata"]["compressor_model"] == "gpt2"

        # Verify that compressed context was fed to LLM prompt
        mock_gen.assert_called_once()
        user_prompt_arg = mock_gen.call_args[0][1]
        assert "[1] AI neural networks." in user_prompt_arg


@pytest.mark.asyncio
async def test_chat_with_longllmlingua_disabled(test_client, monkeypatch):
    monkeypatch.setattr("backend.config.settings.longllmlingua.enabled", False)

    with patch("backend.main.full_retrieval_pipeline") as mock_retrieval, \
         patch("backend.main.rerank") as mock_rerank, \
         patch("backend.main.generate_completion") as mock_gen:

        mock_retrieval.return_value = ([{
            "id": "1",
            "payload": {
                "text": "Uncompressed context.",
                "document_id": "doc-2",
                "document_name": "test.txt",
                "chunk_index": 0
            }
        }], [])
        mock_rerank.return_value = [{
            "id": "1",
            "payload": {
                "text": "Uncompressed context.",
                "document_id": "doc-2",
                "document_name": "test.txt",
                "chunk_index": 0
            },
            "rerank_score": 0.85
        }]
        mock_gen.return_value = "Answer based on uncompressed context."

        response = test_client.post("/api/chat", json={"query": "test query"})
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Answer based on uncompressed context."
        assert data["compression_ratio"] is None
        assert data["retrieval_metadata"]["longllmlingua_applied"] is False



