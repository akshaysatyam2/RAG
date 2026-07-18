import pytest
from unittest.mock import patch
from backend.config import settings

@pytest.mark.asyncio
async def test_successful_pdf_upload(test_client, sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        response = test_client.post(
            "/api/docs/upload",
            files={"file": ("sample.pdf", f, "application/pdf")}
        )
    assert response.status_code == 202
    data = response.json()
    assert "id" in data
    assert data["original_name"] == "sample.pdf"
    assert data["file_type"] == "pdf"

@pytest.mark.asyncio
async def test_successful_image_upload(test_client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        response = test_client.post(
            "/api/docs/upload",
            files={"file": ("sample.png", f, "image/png")}
        )
    assert response.status_code == 202
    data = response.json()
    assert data["original_name"] == "sample.png"
    assert data["file_type"] == "png"

@pytest.mark.asyncio
async def test_unsupported_file_type(test_client, tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Not supported")
    with open(txt_file, "rb") as f:
        response = test_client.post(
            "/api/docs/upload",
            files={"file": ("sample.txt", f, "text/plain")}
        )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

@pytest.mark.asyncio
async def test_oversized_file(test_client, tmp_path):
    # Temporarily set max upload size to 0MB for testing
    with patch.object(settings.ingestion, "max_upload_size_mb", 0):
        dummy_file = tmp_path / "dummy.pdf"
        dummy_file.write_text("dummy")
        with open(dummy_file, "rb") as f:
            response = test_client.post(
                "/api/docs/upload",
                files={"file": ("dummy.pdf", f, "application/pdf")}
            )
        assert response.status_code == 400
        assert "File too large" in response.json()["detail"]

@pytest.mark.asyncio
async def test_list_documents_empty(test_client):
    response = test_client.get("/api/docs")
    assert response.status_code == 200
    assert response.json() == {"documents": [], "total": 0}

@pytest.mark.asyncio
async def test_list_documents(test_client, sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        test_client.post(
            "/api/docs/upload",
            files={"file": ("sample.pdf", f, "application/pdf")}
        )
    
    response = test_client.get("/api/docs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["documents"][0]["original_name"] == "sample.pdf"

@pytest.mark.asyncio
async def test_get_single_document(test_client, sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        upload_resp = test_client.post(
            "/api/docs/upload",
            files={"file": ("sample.pdf", f, "application/pdf")}
        )
    doc_id = upload_resp.json()["id"]

    response = test_client.get(f"/api/docs/{doc_id}")
    assert response.status_code == 200
    assert response.json()["id"] == doc_id

@pytest.mark.asyncio
async def test_get_nonexistent_document(test_client):
    response = test_client.get("/api/docs/invalid_id")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_delete_document(test_client, sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        upload_resp = test_client.post(
            "/api/docs/upload",
            files={"file": ("sample.pdf", f, "application/pdf")}
        )
    doc_id = upload_resp.json()["id"]

    response = test_client.delete(f"/api/docs/{doc_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True

    get_resp = test_client.get(f"/api/docs/{doc_id}")
    assert get_resp.status_code == 404

@pytest.mark.asyncio
async def test_delete_nonexistent_document(test_client):
    response = test_client.delete("/api/docs/invalid_id")
    assert response.status_code == 404

