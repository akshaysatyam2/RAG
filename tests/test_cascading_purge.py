import pytest
import os
from unittest.mock import patch, AsyncMock
from backend.database import get_document, insert_document, delete_document
from backend.config import settings


@pytest.mark.asyncio
async def test_cascading_purge(test_client, tmp_path):
    # Setup test file
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    test_file = upload_dir / "doc_1.pdf"
    test_file.write_text("dummy")
    
    # Insert metadata
    await insert_document("doc_1", "doc_1.pdf", "Original.pdf", "pdf", 100)
    
    # We patch the service functions inside backend.main
    with patch("backend.main.settings.ingestion.upload_dir", str(upload_dir)), \
         patch("backend.main.vector_delete_by_document") as mock_qdrant, \
         patch("backend.main.graph_delete_by_document") as mock_neo4j:
        
        resp = test_client.delete("/api/docs/doc_1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        
        # Verify file is deleted
        assert not test_file.exists()
        
        # Verify DB metadata is deleted
        doc = await get_document("doc_1")
        assert doc is None
        
        # Verify mock calls for external stores
        mock_qdrant.assert_called_once_with("doc_1")
        mock_neo4j.assert_called_once_with("doc_1")
