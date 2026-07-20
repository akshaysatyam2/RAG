import pytest
from unittest.mock import AsyncMock, MagicMock
from qdrant_client.http import models

from backend.services.vector_store import (
    initialize_collection,
    upsert_chunks,
    search_dense,
    search_sparse,
    delete_by_document
)

@pytest.mark.asyncio
async def test_initialize_collection(mock_qdrant):
    # Mock get_collections to return empty
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_qdrant.get_collections.return_value = mock_collections
    
    mock_qdrant.create_collection = AsyncMock()
    mock_qdrant.create_payload_index = AsyncMock()
    
    await initialize_collection()
    
    mock_qdrant.create_collection.assert_called_once()
    mock_qdrant.create_payload_index.assert_called_once()

@pytest.mark.asyncio
async def test_upsert_chunks(mock_qdrant):
    mock_qdrant.upsert = AsyncMock()
    chunks = [
        {
            "id": "chunk_1",
            "text": "test content",
            "dense_vector": [0.1] * 384,
            "sparse_vector": {"indices": [1], "values": [0.5]},
            "metadata": {"chunk_index": 0}
        }
    ]
    await upsert_chunks("doc_1", chunks)
    mock_qdrant.upsert.assert_called_once()

@pytest.mark.asyncio
async def test_search_dense(mock_qdrant):
    mock_result = models.ScoredPoint(id="1", score=0.9, version=1, values=None, payload={"text": "test"})
    mock_response = MagicMock()
    mock_response.points = [mock_result]
    mock_qdrant.query_points = AsyncMock(return_value=mock_response)
    
    results = await search_dense([0.1] * 384, top_k=5)
    assert results == [mock_result]
    mock_qdrant.query_points.assert_called_once()

@pytest.mark.asyncio
async def test_search_sparse(mock_qdrant):
    mock_result = models.ScoredPoint(id="1", score=0.8, version=1, values=None, payload={"text": "test"})
    mock_response = MagicMock()
    mock_response.points = [mock_result]
    mock_qdrant.query_points = AsyncMock(return_value=mock_response)
    
    results = await search_sparse({"indices": [1], "values": [0.5]}, top_k=5)
    assert results == [mock_result]
    mock_qdrant.query_points.assert_called_once()


@pytest.mark.asyncio
async def test_delete_by_document(mock_qdrant):
    mock_qdrant.delete = AsyncMock()
    await delete_by_document("doc_1")
    mock_qdrant.delete.assert_called_once()

@pytest.mark.asyncio
async def test_search_no_results(mock_qdrant):
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant.query_points = AsyncMock(return_value=mock_response)
    results = await search_dense([0.1] * 384, top_k=5)
    assert results == []


