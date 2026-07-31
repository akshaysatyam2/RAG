#!/usr/bin/env python3
"""
Automated Batch Re-indexing Script for GraphRAG Pipeline.

Purges existing vector, graph, and metadata indexes for all uploaded documents
and re-runs the adaptive structure-aware chunking pipeline (TOC/Index skipping,
section breadcrumbs, adaptive chunk sizes).
"""
import sys
import os
import asyncio
import time
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reingest_all")


async def reindex_all_documents():
    from backend.database import list_documents, update_document_status
    from backend.services.ingestion import parse_pdf, parse_image, build_contextual_chunks
    from backend.services.vector_store import delete_by_document, upsert_chunks, initialize_collection
    from backend.services.graph import delete_by_document as delete_by_document_graph, insert_entities_and_relations

    logger.info("Initializing Qdrant collections...")
    await initialize_collection()

    docs = await list_documents()
    if not docs:
        logger.info("No documents found in SQLite metadata database.")
        return

    print("=" * 80)
    print(f"STARTING BATCH RE-INDEXING FOR {len(docs)} DOCUMENT(S)")
    print("=" * 80)

    stats = []

    for idx, doc in enumerate(docs, 1):
        doc_id = doc["id"]
        doc_name = doc.get("original_name") or doc.get("name") or doc_id
        filename = doc.get("filename", f"{doc_id}.pdf")
        
        file_path = PROJECT_ROOT / "uploads" / filename
        if not file_path.exists():
            file_path = PROJECT_ROOT / "uploads" / f"{doc_id}.pdf"
            
        if not file_path.exists():
            logger.warning(f"[{idx}/{len(docs)}] File not found for {doc_name} at {file_path}. Skipping.")
            continue

        print(f"\n[{idx}/{len(docs)}] Re-indexing: '{doc_name}' ({doc_id})")
        start_time = time.time()

        # Step 1: Purge old index entries
        logger.info(f"Purging old vector and graph entries for {doc_id}...")
        await delete_by_document(doc_id)
        await delete_by_document_graph(doc_id)
        await update_document_status(doc_id, "processing")

        # Step 2: Parse document
        ext = file_path.suffix.lower().lstrip(".")
        if ext == "pdf":
            pages = parse_pdf(str(file_path))
            page_count = len(pages)
            raw_text = "\n\n".join(p["text"] for p in pages if p.get("role", "BODY") == "BODY")
            if not raw_text.strip():
                raw_text = "\n\n".join(p["text"] for p in pages)
        elif ext in ["png", "jpg", "jpeg"]:
            raw_text = parse_image(str(file_path))
            page_count = 1
            pages = [{"page_number": 1, "text": raw_text, "role": "BODY"}]
        else:
            logger.warning(f"Unsupported file type '{ext}' for {doc_name}. Skipping.")
            continue

        # Step 3: Run Adaptive Structure-Aware Chunking Pipeline
        logger.info(f"Building adaptive contextual chunks for {page_count} pages ({len(raw_text):,} chars)...")
        chunks, doc_summary = await build_contextual_chunks(
            doc_id=doc_id,
            raw_text=raw_text,
            pages_metadata=pages
        )

        chunk_count = len(chunks)
        logger.info(f"Generated {chunk_count} structure-aware topic chunks.")

        # Step 4: Upsert Vectors to Qdrant
        if chunks:
            logger.info("Upserting dense + sparse vectors to Qdrant...")
            await upsert_chunks(doc_id, chunks)

            # Step 5: Insert Entity-Relation Triples into Graph DB
            all_triples = []
            for c in chunks:
                all_triples.extend(c.get("triples", []))

            if all_triples:
                logger.info(f"Inserting {len(all_triples)} entity triples into Graph database...")
                await insert_entities_and_relations(doc_id, all_triples)

        # Step 6: Mark Ready in SQLite
        elapsed = round(time.time() - start_time, 2)
        await update_document_status(
            doc_id=doc_id,
            status="ready",
            page_count=page_count,
            chunk_count=chunk_count,
            summary=doc_summary
        )

        stats.append({
            "name": doc_name,
            "pages": page_count,
            "chunks": chunk_count,
            "time_sec": elapsed
        })
        print(f"SUCCESS: Re-indexed '{doc_name}' -> {chunk_count} chunks in {elapsed}s")

    print("\n" + "=" * 80)
    print("RE-INDEXING COMPLETE — SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Document Name':<35} | {'Pages':<8} | {'New Chunks':<12} | {'Time (s)':<10}")
    print("-" * 72)
    for s in stats:
        print(f"{s['name'][:34]:<35} | {s['pages']:<8} | {s['chunks']:<12} | {s['time_sec']:<10}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(reindex_all_documents())
