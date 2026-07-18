import pytest
from unittest.mock import patch, AsyncMock
from qdrant_client.http import models

from backend.services.retrieval import (
    reciprocal_rank_fusion,
    hybrid_search,
    expand_with_graph,
    full_retrieval_pipeline
)

def test_reciprocal_rank_fusion():
    dense = [models.ScoredPoint(id="1", version=1, score=0.9, payload={"text": "A"})]
    sparse = [
        models.ScoredPoint(id="2", version=1, score=0.8, payload={"text": "B"}),
        models.ScoredPoint(id="1", version=1, score=0.7, payload={"text": "A"})
    ]
    
    # RRF k=60. 
    # id=1: rank 1 in dense (1/61), rank 2 in sparse (1/62). Total = 1/61 + 1/62
    # id=2: rank 1 in sparse (1/61). Total = 1/61
    results = reciprocal_rank_fusion(dense, sparse, k=60)
    
    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "2"

@pytest.mark.asyncio
@patch("backend.services.retrieval.search_dense")
@patch("backend.services.retrieval.search_sparse")
@patch("backend.services.retrieval.get_dense_embedding")
@patch("backend.services.retrieval.tokenize_for_sparse")
async def test_hybrid_search(mock_tokenize, mock_dense_emb, mock_sparse, mock_dense):
    mock_dense_emb.return_value = [0.1]
    mock_tokenize.return_value = ["test"]
    
    mock_dense.return_value = [models.ScoredPoint(id="1", version=1, score=0.9, payload={"text": "A"})]
    mock_sparse.return_value = [models.ScoredPoint(id="2", version=1, score=0.8, payload={"text": "B"})]
    
    results = await hybrid_search("test query", top_k=5)
    assert len(results) == 2

@pytest.mark.asyncio
@patch("backend.services.retrieval.expand_from_entities")
@patch("backend.services.retrieval.extract_entities_and_relations")
async def test_expand_with_graph(mock_extract, mock_expand):
    mock_extract.return_value = [{"head": "E1", "tail": "E2"}]
    mock_expand.return_value = [{"path_nodes": []}]
    
    chunks = [{"payload": {"text": "dummy"}}]
    results = await expand_with_graph(chunks)
    assert len(results) == 1

@pytest.mark.asyncio
@patch("backend.services.retrieval.hybrid_search")
@patch("backend.services.retrieval.expand_with_graph")
async def test_full_pipeline(mock_expand, mock_hybrid):
    mock_hybrid.return_value = [{"payload": {"text": "A"}}]
    mock_expand.return_value = [{"path_nodes": []}]
    
    hybrid, graph = await full_retrieval_pipeline("query", top_k=5)
    assert len(hybrid) == 1
    assert len(graph) == 1

@pytest.mark.asyncio
async def test_empty_retrieval():
    results = await expand_with_graph([])
    assert results == []
