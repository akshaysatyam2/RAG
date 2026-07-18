import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture(autouse=True)
def mock_driver():
    import backend.services.graph as graph
    graph.driver = None
    with patch("backend.services.graph.AsyncGraphDatabase.driver") as mock_d:
        driver_instance = mock_d.return_value
        session_instance = AsyncMock()
        driver_instance.session.return_value.__aenter__.return_value = session_instance
        yield session_instance
    graph.driver = None


from backend.services.graph import (
    initialize_graph,
    insert_entities_and_relations,
    expand_from_entities,
    delete_by_document
)

@pytest.mark.asyncio
async def test_initialize_graph(mock_driver):
    await initialize_graph()
    assert mock_driver.run.call_count >= 2

@pytest.mark.asyncio
async def test_insert_entities(mock_driver):
    triples = [{"head": "A", "head_type": "T1", "relation": "R", "tail": "B", "tail_type": "T2"}]
    await insert_entities_and_relations("doc_1", triples)
    assert mock_driver.run.call_count >= 1

@pytest.mark.asyncio
async def test_empty_graph_handling(mock_driver):
    await insert_entities_and_relations("doc_1", [])
    assert mock_driver.run.call_count == 0
    
    results = await expand_from_entities([])
    assert results == []

@pytest.mark.asyncio
async def test_expand_from_entities(mock_driver):
    mock_record = {"path_nodes": [{"name": "A"}], "path_rels": []}
    mock_driver.run.return_value.data = AsyncMock(return_value=[mock_record])
    
    results = await expand_from_entities(["A"], max_hops=1)
    assert results == [mock_record]

@pytest.mark.asyncio
async def test_delete_by_document_graph(mock_driver):
    await delete_by_document("doc_1")
    assert mock_driver.run.call_count >= 2
