import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Callable, Awaitable, Optional
from PIL import Image

import fitz
import pytesseract

from backend.config import settings
from backend.services.llm import generate_context_summary, extract_entities_and_relations
from backend.services.embeddings import get_dense_embeddings_batch, tokenize_for_sparse, compute_sparse_vector

import uuid

logger = logging.getLogger(__name__)



def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a PDF file and return a list of dictionaries with page numbers and text using PyMuPDF.
    """
    pages = []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text:
                pages.append({
                    "page_number": i + 1,
                    "text": text.strip()
                })
        doc.close()
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {e}")
        raise
    return pages



def parse_image(file_path: str) -> str:
    """
    Extract text from an image using pytesseract.
    """
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logger.error(f"Error parsing image {file_path}: {e}")
        raise


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of `chunk_size` characters with `overlap` overlap.
    Ensures start always advances to prevent infinite loops.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        # Adjust end to nearest whitespace if not at the end of the text
        if end < text_length:
            last_space = chunk.rfind(' ')
            if last_space != -1 and last_space > chunk_size // 2:
                end = start + last_space
                
        # Ensure progress is made
        if end <= start:
            end = start + chunk_size
            
        chunks.append(text[start:end].strip())
        
        if end >= text_length:
            break
            
        next_start = end - overlap
        if next_start <= start:
            start = start + 1
        else:
            start = next_start
            
    return chunks



async def build_contextual_chunks(
    doc_id: str, 
    raw_text: str, 
    pages_metadata: List[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, int, int, str], Awaitable[None]]] = None
) -> List[Dict[str, Any]]:
    """
    Full pipeline for contextual chunking.
    a. Generate document summary via LLM
    b. Chunk the text
    c. Prepend summary context to each chunk
    d. Generate dense embeddings for each chunk
    e. Extract entity-relation triples from chunks
    f. Return structured chunks ready for storage
    """
    
    async def report_progress(phase: str, step: int, total: int, msg: str):
        if progress_callback:
            await progress_callback(phase, step, total, msg)

    # 1. Document Summary
    await report_progress("Summary Generation", 1, 5, "Generating document summary...")
    doc_summary = await generate_context_summary(raw_text)
    
    # 2. Chunking
    await report_progress("Chunking", 2, 5, "Chunking document text...")
    raw_chunks = chunk_text(raw_text, settings.ingestion.chunk_size, settings.ingestion.chunk_overlap)
    
    # 3. Contextualization
    await report_progress("Contextualization", 3, 5, "Prepending context to chunks...")
    contextualized_texts = []
    for chunk in raw_chunks:
        # Prepend summary to provide global context
        ctx_text = f"Document Summary: {doc_summary}\n\nChunk Content:\n{chunk}"
        contextualized_texts.append(ctx_text)
        
    # 4. Dense Embeddings
    await report_progress("Embedding", 4, 5, "Generating dense embeddings...")
    # Process in batches to avoid OOM
    dense_embeddings = []
    batch_size = 32
    for i in range(0, len(contextualized_texts), batch_size):
        batch = contextualized_texts[i:i+batch_size]
        batch_emb = get_dense_embeddings_batch(batch)
        dense_embeddings.extend(batch_emb)
        
    # Compute corpus tokens for sparse vectors
    corpus_tokens = [tokenize_for_sparse(t) for t in contextualized_texts]
        
    # 5. Extract Entities & Finalize
    await report_progress("Entity Extraction", 5, 5, "Extracting entities and structuring data...")
    
    async def extract_triples_for_chunk(idx, text):
        try:
            return idx, await extract_entities_and_relations(text)
        except Exception as e:
            logger.error(f"Error extracting triples for chunk {idx}: {e}")
            return idx, []
            
    extraction_tasks = [extract_triples_for_chunk(i, txt) for i, txt in enumerate(raw_chunks)]
    extraction_results = await asyncio.gather(*extraction_tasks)
    triples_by_index = dict(extraction_results)
    
    structured_chunks = []
    for i, (original_text, ctx_text, embedding) in enumerate(zip(raw_chunks, contextualized_texts, dense_embeddings)):
        triples = triples_by_index.get(i, [])
        sparse_vec = compute_sparse_vector(ctx_text, corpus_tokens)
        
        # Meta info
        page_num = 1
        if pages_metadata:
            total_chunks = len(raw_chunks)
            total_pages = len(pages_metadata)
            if total_pages > 0 and total_chunks > 0:
                idx = int((i / total_chunks) * total_pages)
                page_num = pages_metadata[min(idx, total_pages-1)].get("page_number", 1)
        
        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{i}"))
        chunk_data = {
            "id": chunk_uuid,
            "text": original_text,
            "contextualized_text": ctx_text,
            "dense_vector": embedding,
            "sparse_vector": sparse_vec,
            "metadata": {
                "document_id": doc_id,
                "chunk_index": i,
                "page_number": page_num,
                "text": original_text,
                "contextualized_text": ctx_text,
            },
            "triples": triples
        }
        structured_chunks.append(chunk_data)

        
    await report_progress("Complete", 5, 5, "Ingestion pipeline complete.")
    return structured_chunks
