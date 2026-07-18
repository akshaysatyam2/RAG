import pytest
import os
import aiosqlite
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from backend.config import settings
from backend.database import initialize_schema
from backend.main import app

# Configure Celery eagerly for testing
from backend.workers.tasks import celery_app
celery_app.conf.task_always_eager = True




@pytest.fixture(autouse=True)
def test_settings(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    db_path = tmp_path / "test.db"
    
    with patch.object(settings.ingestion, "upload_dir", str(upload_dir)), \
         patch.object(settings.database, "sqlite_path", str(db_path)):
        yield settings


@pytest.fixture
def test_db(test_settings):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(initialize_schema())
    loop.close()
    yield



@pytest.fixture
def mock_qdrant():
    with patch("backend.services.vector_store.client") as mock:
        mock.get_collections = AsyncMock()
        mock.create_collection = AsyncMock()
        mock.create_payload_index = AsyncMock()
        mock.upsert = AsyncMock()
        mock.scroll = AsyncMock()
        mock.delete = AsyncMock()
        yield mock


@pytest.fixture
def mock_neo4j():
    with patch("backend.services.graph.AsyncGraphDatabase") as mock:
        driver_instance = mock.driver.return_value
        session_instance = AsyncMock()
        driver_instance.session.return_value.__aenter__.return_value = session_instance
        yield session_instance


@pytest.fixture
def mock_llm():
    with patch("backend.services.llm.client") as mock:
        yield mock


class FlaskResponseWrapper:
    def __init__(self, flask_response):
        self.resp = flask_response

    @property
    def status_code(self):
        return self.resp.status_code

    def json(self):
        return self.resp.get_json()


class FlaskTestClientWrapper:
    def __init__(self, flask_client):
        self.client = flask_client

    def post(self, url, json=None, files=None, **kwargs):
        if files:
            data = {}
            for key, val in files.items():
                if len(val) >= 2:
                    data[key] = (val[1], val[0])
                else:
                    data[key] = val
            return FlaskResponseWrapper(self.client.post(url, data=data, **kwargs))
        return FlaskResponseWrapper(self.client.post(url, json=json, **kwargs))

    def get(self, url, **kwargs):
        return FlaskResponseWrapper(self.client.get(url, **kwargs))

    def delete(self, url, **kwargs):
        return FlaskResponseWrapper(self.client.delete(url, **kwargs))


@pytest.fixture
def test_client(test_db, mock_qdrant, mock_neo4j, mock_llm):
    app.config["TESTING"] = True
    return FlaskTestClientWrapper(app.test_client())


@pytest.fixture
def sample_pdf_path(tmp_path):
    import fitz  # PyMuPDF
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test PDF Content")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_image_path(tmp_path):
    from PIL import Image
    image_path = tmp_path / "sample.png"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(image_path)
    return image_path
