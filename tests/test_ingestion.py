import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from backend.services.ingestion import (
    parse_pdf,
    parse_image,
    chunk_text,
    build_contextual_chunks
)


def test_parse_pdf():
    # Mock fitz.open (PyMuPDF)
    with patch("backend.services.ingestion.fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page text content"
        mock_doc.__iter__.return_value = [mock_page, mock_page]
        mock_open.return_value = mock_doc
        
        pages = parse_pdf("dummy.pdf")
        assert len(pages) == 2
        assert pages[0]["page_number"] == 1
        assert pages[0]["text"] == "Page text content"



def test_parse_image():
    # Mock PIL.Image.open and pytesseract.image_to_string
    with patch("backend.services.ingestion.Image.open") as mock_open, \
         patch("backend.services.ingestion.pytesseract.image_to_string") as mock_ocr:
        mock_ocr.return_value = "OCR text content"
        
        text = parse_image("dummy.png")
        assert text == "OCR text content"
        mock_ocr.assert_called_once()


def test_chunk_text():
    text = "This is a sentence for chunking. And another sentence."
    # Chunk size 20, overlap 5
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 0
    # Every chunk should be stripped
    for c in chunks:
        assert c == c.strip()


@pytest.mark.asyncio
async def test_build_contextual_chunks():
    # Mock LLM summary, LLM entity extraction, dense embeddings, sparse vector
    with patch("backend.services.ingestion.generate_context_summary") as mock_sum, \
         patch("backend.services.ingestion.extract_entities_and_relations") as mock_ext, \
         patch("backend.services.ingestion.get_dense_embeddings_batch") as mock_dense, \
         patch("backend.services.ingestion.compute_sparse_vector") as mock_sparse:
        
        mock_sum.return_value = "Document summary text"
        mock_ext.return_value = [{"head": "A", "head_type": "T1", "relation": "RELATES", "tail": "B", "tail_type": "T2"}]
        mock_dense.return_value = [[0.1] * 384]
        mock_sparse.return_value = {"indices": [1], "values": [0.5]}
        
        raw_text = "Original chunk text"
        pages = [{"page_number": 1, "text": raw_text}]
        
        # Test progress callback
        progress_calls = []
        async def progress_cb(phase, step, total, msg):
            progress_calls.append((phase, step, total, msg))
            
        chunks = await build_contextual_chunks(
            doc_id="doc_123",
            raw_text=raw_text,
            pages_metadata=pages,
            progress_callback=progress_cb
        )
        
        assert len(chunks) == 1
        assert chunks[0]["id"] == "doc_123_chunk_0"
        assert chunks[0]["text"] == raw_text
        assert "Document summary text" in chunks[0]["contextualized_text"]
        assert chunks[0]["dense_vector"] == [0.1] * 384
        assert chunks[0]["sparse_vector"] == {"indices": [1], "values": [0.5]}
        assert len(chunks[0]["triples"]) == 1
        assert chunks[0]["triples"][0]["head"] == "A"
        
        # Progress callback should be executed
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == "Complete"
