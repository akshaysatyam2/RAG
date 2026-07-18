from celery import Celery
import asyncio
import logging
from pathlib import Path

from backend.config import settings
from backend.database import update_document_status, upsert_ingestion_progress
from backend.services.ingestion import parse_pdf, parse_image, build_contextual_chunks
from backend.services.vector_store import upsert_chunks
from backend.services.graph import insert_entities_and_relations

celery_app = Celery(
    "rag_tasks",
    broker=settings.redis.url,
    backend=settings.redis.url
)

logger = logging.getLogger(__name__)


async def run_ingestion(doc_id: str, file_path: str, file_type: str):
    # Phase 1: Parsing
    await upsert_ingestion_progress(doc_id, "parsing", 0, 1, "Starting parsing")
    if file_type == "pdf":
        pages = parse_pdf(file_path)
        page_count = len(pages)
        raw_text = "\n\n".join(p["text"] for p in pages)
    elif file_type in ["png", "jpg", "jpeg"]:
        raw_text = parse_image(file_path)
        page_count = 1
        pages = [{"page_number": 1, "text": raw_text}]
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    await update_document_status(doc_id, "processing", page_count=page_count)
    await upsert_ingestion_progress(doc_id, "parsing", 1, 1, f"Parsing complete. Found {page_count} pages.")

    if not raw_text.strip():
        await update_document_status(doc_id, "ready", error_message="No text found in document")
        await upsert_ingestion_progress(doc_id, "parsing", 1, 1, "No text found in document")
        return "No text found"

    # Progress callback mapping pipeline updates to DB
    async def progress_callback(phase: str, step: int, total: int, msg: str):
        await upsert_ingestion_progress(doc_id, phase, step, total, msg)

    # Phase 2: Contextual chunking, embedding generation, entity extraction
    chunks = await build_contextual_chunks(
        doc_id=doc_id,
        raw_text=raw_text,
        pages_metadata=pages,
        progress_callback=progress_callback
    )
    chunk_count = len(chunks)
    await update_document_status(doc_id, "processing", chunk_count=chunk_count)

    if chunk_count == 0:
        await update_document_status(doc_id, "ready", error_message="No chunks created")
        return "No chunks created"

    # Phase 3: Vector Store Ingestion
    await upsert_ingestion_progress(doc_id, "vector_store", 0, 1, "Starting vector storage")
    await upsert_chunks(doc_id, chunks)
    await upsert_ingestion_progress(doc_id, "vector_store", 1, 1, "Vector storage complete")

    # Phase 4: Graph Store Ingestion
    await upsert_ingestion_progress(doc_id, "graph_store", 0, 1, "Starting graph storage")
    all_triples = []
    for c in chunks:
        all_triples.extend(c.get("triples", []))
    
    if all_triples:
        await insert_entities_and_relations(doc_id, all_triples)
    await upsert_ingestion_progress(doc_id, "graph_store", 1, 1, f"Graph storage complete. Ingested {len(all_triples)} relations.")

    await update_document_status(doc_id, "ready")
    return f"Successfully ingested document {doc_id}"


@celery_app.task(name="ingest_document", bind=True, max_retries=3)
def ingest_document(self, doc_id: str, file_path: str, file_type: str):
    try:
        return asyncio.run(run_ingestion(doc_id, file_path, file_type))
    except Exception as e:
        logger.error(f"Failed to ingest document {doc_id}: {str(e)}", exc_info=True)
        asyncio.run(update_document_status(doc_id, "error", error_message=str(e)))
        raise self.retry(exc=e, countdown=60)

