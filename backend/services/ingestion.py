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



def detect_page_role(page_num: int, total_pages: int, text: str) -> str:
    """
    Classifies a PDF page role: 'TOC', 'INDEX', 'EMPTY', or 'BODY'.
    TOC and INDEX pages are marked to be excluded from semantic search indexing
    so that index page listings (e.g. 'logistic regression, 6, 12, 131–137')
    don't hijack actual content retrieval.
    """
    cleaned = text.strip() if text else ""
    if not cleaned or len(cleaned) < 10:
        return "EMPTY"

    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    first_500 = cleaned[:500].lower()
    total_pages = max(total_pages, 1)

    # Table of Contents detection (front-matter)
    if page_num <= 25 and ("contents" in first_500 or "table of contents" in first_500):
        return "TOC"

    # Index page detection (back-matter)
    header = " ".join(lines[:3]).lower()
    if page_num > total_pages * 0.8 and "index" in header:
        return "INDEX"

    # Index pattern check: lines ending with page reference numbers/ranges
    import re
    index_entry_count = sum(1 for line in lines if re.search(r'[a-zA-Z\s]{3,},\s*\d+(?:\s*,\s*\d+|–\d+)*$', line))
    if len(lines) > 10 and (index_entry_count / len(lines)) > 0.35:
        return "INDEX"

    return "BODY"


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF using PyMuPDF.
    Runs Tesseract OCR on scanned/image-only pages.
    Tags each page role ('BODY', 'TOC', 'INDEX', 'EMPTY') so TOC/Index pages
    can be filtered out from semantic vector indexing.
    Returns list of {page_number, text, role} dicts.
    """
    pages = []
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if not text or len(text.strip()) < 10:
                # Page has no extractable text — try OCR
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                except Exception as ocr_err:
                    logger.warning(f"OCR fallback failed for page {i+1} of {file_path}: {ocr_err}")

            cleaned = text.strip() if text else ""
            role = detect_page_role(i + 1, total_pages, cleaned)

            # Skip EMPTY pages
            if role != "EMPTY" and len(cleaned) >= 10:
                pages.append({
                    "page_number": i + 1,
                    "text": cleaned,
                    "role": role
                })
        doc.close()
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {e}")
        raise
    return pages


def parse_image(file_path: str) -> str:
    """
    Runs Tesseract OCR on an image file and returns the extracted text.
    """
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logger.error(f"Error parsing image {file_path}: {e}")
        raise


def get_adaptive_chunk_config(total_pages: int, total_chars: int) -> Dict[str, int]:
    """
    Determines document scale and returns adaptive chunk_size and overlap:
    - Large Book (> 50 pages or > 120k chars): 1800 chars, 200 overlap (chapter/topic scope)
    - Medium Doc (15-50 pages or 30k-120k chars): 1200 chars, 150 overlap (section scope)
    - Short Doc (< 15 pages or < 30k chars): 650 chars, 100 overlap (paragraph/page scope)
    """
    if total_pages > 50 or total_chars > 120000:
        return {"chunk_size": 1800, "overlap": 200, "scale": "BOOK"}
    elif total_pages >= 15 or total_chars >= 30000:
        return {"chunk_size": 1200, "overlap": 150, "scale": "MEDIUM"}
    else:
        return {"chunk_size": 650, "overlap": 100, "scale": "SHORT"}


def chunk_text_structure_aware(
    pages_metadata: List[Dict[str, Any]], 
    chunk_size: int = 1500, 
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Document-aware & Structure-aware Hierarchical Chunker.
    - Filters out 'TOC' and 'INDEX' pages so index listing chunks don't pollute retrieval.
    - Detects Chapter and Section headers and tracks active breadcrumbs.
    - Prepends structural breadcrumbs ([Section: Chapter 4: Classification > 4.3 Logistic Regression])
      to every chunk for rich contextual vector and BM25 indexing.
    - Cleans hyphenated line-breaks and filters out noise paragraphs.
    """
    if not pages_metadata:
        return []

    import re

    # Filter out TOC and INDEX pages
    body_pages = [p for p in pages_metadata if p.get("role", "BODY") == "BODY"]
    if not body_pages:
        # Fallback to all pages if role tagging was over-aggressive
        body_pages = pages_metadata

    chunks_output = []
    active_chapter = ""
    active_section = ""

    current_chunk_paras = []
    current_length = 0
    current_chunk_pages = []

    def is_noise_paragraph(p: str) -> bool:
        p_str = p.strip()
        if len(p_str) < 15:
            return True
        tokens = p_str.split()
        if tokens and all(re.fullmatch(r'[\d.,\-+%()]+|[A-Za-z]', t) for t in tokens):
            return True
        return False

    def emit_chunk():
        nonlocal current_chunk_paras, current_length, current_chunk_pages, active_chapter, active_section
        if not current_chunk_paras:
            return
        
        raw_content = "\n\n".join(current_chunk_paras).strip()
        if not raw_content:
            return

        # Build breadcrumb context
        breadcrumbs = []
        if active_chapter:
            breadcrumbs.append(active_chapter)
        if active_section and active_section != active_chapter:
            breadcrumbs.append(active_section)

        prefix = ""
        if breadcrumbs:
            prefix = f"[Section: {' > '.join(breadcrumbs)}]\n\n"

        full_chunk_text = f"{prefix}{raw_content}"
        primary_page = current_chunk_pages[0] if current_chunk_pages else 1

        chunks_output.append({
            "text": full_chunk_text,
            "raw_text": raw_content,
            "page_number": primary_page,
            "section_context": " > ".join(breadcrumbs) if breadcrumbs else ""
        })

        # Overlap retention: keep the last paragraph for overlap context
        if len(current_chunk_paras) > 1:
            current_chunk_paras = [current_chunk_paras[-1]]
            current_length = len(current_chunk_paras[0])
            current_chunk_pages = [current_chunk_pages[-1]]
        else:
            current_chunk_paras = []
            current_length = 0
            current_chunk_pages = []

    for pdata in body_pages:
        pg_num = pdata.get("page_number", 1)
        pg_text = pdata.get("text", "").strip()

        # Fix hyphenated linebreaks: 'statisti-\ncal' -> 'statistical'
        pg_text = re.sub(r'(\w)-\n(\w)', r'\1\2', pg_text)
        # Normalize single newlines to spaces
        pg_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', pg_text)

        # Split page into raw paragraphs
        raw_paras = re.split(r'\n\s*\n|\n(?=[0-9]+\.|\b[A-Z][A-Za-z0-9\s]{2,}:)', pg_text)

        for para in raw_paras:
            para_str = para.strip()
            if not para_str or is_noise_paragraph(para_str):
                continue

            # Header detection
            # Chapter header match: e.g. "Chapter 4: Classification" or "4 Classification"
            m_chap = re.match(r'^(?:Chapter\s+(\d+)|(\d+))\s+([A-Z][A-Za-z0-9\s\-\?:,]{2,50})$', para_str)
            if m_chap:
                cnum = m_chap.group(1) or m_chap.group(2)
                active_chapter = f"Chapter {cnum}: {m_chap.group(3).strip()}"

            # Section header match: e.g. "4.3 Logistic Regression" or "4.3.1 The Logistic Model"
            m_sec = re.match(r'^(\d+\.\d+(?:\.\d+)?)\s+([A-Z][A-Za-z0-9\s\-\?:,]{2,60})$', para_str)
            if m_sec:
                active_section = f"{m_sec.group(1)} {m_sec.group(2).strip()}"

            para_len = len(para_str)

            if para_len > chunk_size:
                # Large paragraph — split by sentence
                sentences = re.split(r'(?<=[.!?])\s+', para_str)
                for sent in sentences:
                    sent_str = sent.strip()
                    if not sent_str or len(sent_str) < 10:
                        continue
                    if current_length + len(sent_str) + 1 > chunk_size:
                        emit_chunk()

                    current_chunk_paras.append(sent_str)
                    current_length += len(sent_str) + 1
                    current_chunk_pages.append(pg_num)
            else:
                if current_length + para_len + 2 > chunk_size:
                    emit_chunk()

                current_chunk_paras.append(para_str)
                current_length += para_len + 2
                current_chunk_pages.append(pg_num)

    if current_chunk_paras:
        emit_chunk()

    return chunks_output


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Legacy wrapper for string-based chunking.
    Uses chunk_text_structure_aware internally.
    """
    synthetic_page = [{"page_number": 1, "text": text, "role": "BODY"}]
    structured = chunk_text_structure_aware(synthetic_page, chunk_size, overlap)
    return [c["text"] for c in structured]




async def build_contextual_chunks(
    doc_id: str, 
    raw_text: str, 
    pages_metadata: List[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, int, int, str], Awaitable[None]]] = None
) -> List[Dict[str, Any]]:
    """
    Runs the full ingestion pipeline for a single document:
      1. Generate a document-level summary via LLM
      2. Split raw text into chunks
      3. Prepend the summary to each chunk for contextual grounding
      4. Generate dense embeddings (batched) and sparse token vectors
      5. Extract entity-relation triples from each raw chunk in parallel
      6. Return fully structured chunk dicts ready to be stored
    """
    
    async def report_progress(phase: str, step: int, total: int, msg: str):
        if progress_callback:
            await progress_callback(phase, step, total, msg)

    # 1. Document Summary
    await report_progress("Summary Generation", 1, 5, "Generating document summary...")
    doc_summary = await generate_context_summary(raw_text)
    
    # 2. Adaptive Scale Detection & Chunking
    await report_progress("Chunking", 2, 5, "Running adaptive document-aware chunking...")
    total_pages = len(pages_metadata) if pages_metadata else 1
    total_chars = len(raw_text)
    
    adaptive_cfg = get_adaptive_chunk_config(total_pages, total_chars)
    chunk_size = adaptive_cfg["chunk_size"]
    overlap = adaptive_cfg["overlap"]
    scale = adaptive_cfg["scale"]
    
    logger.info(f"Document {doc_id}: Scale={scale}, pages={total_pages}, chars={total_chars}, chunk_size={chunk_size}")

    if pages_metadata:
        structured_chunk_objs = chunk_text_structure_aware(pages_metadata, chunk_size, overlap)
    else:
        synthetic_pages = [{"page_number": 1, "text": raw_text, "role": "BODY"}]
        structured_chunk_objs = chunk_text_structure_aware(synthetic_pages, chunk_size, overlap)

    raw_chunks = [c["text"] for c in structured_chunk_objs]
    
    # 3. Contextualization
    await report_progress("Contextualization", 3, 5, "Prepending context to chunks...")
    contextualized_texts = []
    for chunk in raw_chunks:
        ctx_text = f"Document Summary: {doc_summary}\n\nChunk Content:\n{chunk}"
        contextualized_texts.append(ctx_text)
        
    # 4. Generate dense vector representations for the chunks
    await report_progress("Embedding", 4, 5, "Generating dense embeddings...")
    # Batch to avoid hitting memory limits with large documents
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
    for i, (chunk_obj, ctx_text, embedding) in enumerate(zip(structured_chunk_objs, contextualized_texts, dense_embeddings)):
        original_text = chunk_obj["text"]
        page_num = chunk_obj.get("page_number", 1)
        section_ctx = chunk_obj.get("section_context", "")

        triples = triples_by_index.get(i, [])
        sparse_vec = compute_sparse_vector(ctx_text, corpus_tokens)
        
        try:
            from backend.database import get_document
            doc_info = await get_document(doc_id)
            doc_name = doc_info.get("original_name") if doc_info else "Document"
        except Exception:
            doc_name = "Document"


        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{i}"))
        chunk_data = {
            "id": chunk_uuid,
            "text": original_text,
            "contextualized_text": ctx_text,
            "dense_vector": embedding,
            "sparse_vector": sparse_vec,
            "metadata": {
                "document_id": doc_id,
                "document_name": doc_name,
                "chunk_index": i,
                "page_number": page_num,
                "text": original_text,
                "contextualized_text": ctx_text,
            },
            "triples": triples
        }
        structured_chunks.append(chunk_data)


        
    await report_progress("Complete", 5, 5, "Ingestion pipeline complete.")
    return structured_chunks, doc_summary

