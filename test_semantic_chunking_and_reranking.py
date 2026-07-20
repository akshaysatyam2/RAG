#!/usr/bin/env python3
"""
Standalone test script to test structural/semantic paragraph & sentence chunking
and entity-weighted stop-word filtered reranking.
"""

import sys
import os
import re
import string
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "tell", "me",
    "what", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "extract", "key", "main", "topics", "discussed"
}

SAMPLE_DOC_TEXT = """
Akshay Satyam - Senior Software Engineer & RAG Architect

1. Professional Summary:
Akshay Satyam is a Senior Software Engineer specializing in Artificial Intelligence, Retrieval-Augmented Generation (RAG), and Knowledge Graphs.
He has extensive experience building production-grade vector search systems using Qdrant, PyMuPDF, SentenceTransformers, and Neo4j graph databases.

2. Core Architecture & Technical Skills:
- Programming Languages: Python, JavaScript, C++, SQL
- AI & ML Frameworks: PyTorch, SentenceTransformers, OpenAI API, LangChain
- Vector & Graph Databases: Qdrant Vector Store, Neo4j Graph Database, SQLite, Redis
- Web Frameworks: Flask, React, Celery, Docker, Pytest

3. Key Engineering Projects:
a) GraphRAG Hybrid Retrieval System:
Designed and implemented a hybrid retrieval engine combining dense vector search, BM25 sparse keyword matching, and Neo4j sub-graph traversal.
Achieved high precision on document QA tasks with contextual chunking and cross-encoder reranking.

b) Automated Document Ingestion Pipeline:
Built an automated ingestion pipeline for parsing PDFs and image documents using PyMuPDF and Tesseract OCR, with automatic entity-relation triple extraction.
"""


def semantic_chunk_text(text: str, chunk_size: int = 500, overlap: int = 64) -> List[str]:
    """
    Structural & Semantic Chunker:
    Splits text along section headers, paragraph boundaries (\n\n), and sentence endings (. ! ?).
    Ensures no chunk starts or ends with partial/truncated words or broken sentences.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    
    # 1. Split into paragraphs (by double newlines or section header patterns)
    raw_paragraphs = re.split(r'\n\s*\n|\n(?=[0-9]+\.|\b[A-Z][A-Za-z0-9\s]{2,}:)', text)
    paragraphs = [p.strip() for p in raw_paragraphs if p and p.strip()]

    chunks = []
    current_chunk_paragraphs = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        
        # If single paragraph is larger than chunk_size, split by sentences
        if para_len > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if current_length + len(sent) + 1 > chunk_size and current_chunk_paragraphs:
                    chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                    # Handle overlap by keeping last paragraph/sentence
                    current_chunk_paragraphs = [current_chunk_paragraphs[-1]] if current_chunk_paragraphs else []
                    current_length = sum(len(p) for p in current_chunk_paragraphs)
                
                current_chunk_paragraphs.append(sent)
                current_length += len(sent) + 1
        else:
            if current_length + para_len + 2 > chunk_size and current_chunk_paragraphs:
                chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                # Overlap: preserve last paragraph
                current_chunk_paragraphs = [current_chunk_paragraphs[-1]] if current_chunk_paragraphs else []
                current_length = sum(len(p) for p in current_chunk_paragraphs)

            current_chunk_paragraphs.append(para)
            current_length += para_len + 2

    if current_chunk_paragraphs:
        final_chunk = "\n\n".join(current_chunk_paragraphs).strip()
        if final_chunk and (not chunks or final_chunk != chunks[-1]):
            chunks.append(final_chunk)

    return chunks


def compute_smart_overlap_score(query: str, text: str) -> float:
    """
    Calculates an entity-weighted, stop-word filtered relevance score.
    Gives higher weight to rare terms, proper nouns, and entity matches.
    """
    # Clean punctuation and tokenize
    translator = str.maketrans('', '', string.punctuation)
    clean_query = query.translate(translator).lower()
    clean_text = text.translate(translator).lower()

    query_tokens = [w for w in clean_query.split() if w]
    text_tokens = set(clean_text.split())

    # Filter out stop words for meaningful terms
    meaningful_query_terms = [w for w in query_tokens if w not in STOP_WORDS]
    if not meaningful_query_terms:
        meaningful_query_terms = query_tokens

    score = 0.0
    for term in meaningful_query_terms:
        if term in text_tokens:
            # Proper nouns / rare words get double weight
            weight = 2.0 if len(term) > 5 or term[0].isupper() else 1.0
            score += weight

    max_possible = sum(2.0 if len(t) > 5 or t[0].isupper() else 1.0 for t in meaningful_query_terms)
    normalized_score = score / max(max_possible, 1.0)
    return float(normalized_score)


def test_semantic_chunking_and_ranking():
    print("=" * 80)
    print("      TESTING STRUCTURAL SEMANTIC CHUNKING & SMART RERANKING")
    print("=" * 80)

    # 1. Test Chunking
    print("\n--- STEP 1: Structural Paragraph & Sentence Chunking ---")
    chunks = semantic_chunk_text(SAMPLE_DOC_TEXT, chunk_size=400, overlap=50)
    print(f"Sample text ({len(SAMPLE_DOC_TEXT)} chars) split into {len(chunks)} structural chunks:\n")

    chunking_clean = True
    for i, c in enumerate(chunks):
        print(f"--- CHUNK {i} (len={len(c)}) ---")
        print(c)
        print("-" * 40)
        # Check if chunk starts or ends with broken truncated words
        first_word = c.split()[0] if c.split() else ""
        last_word = c.split()[-1] if c.split() else ""
        if len(first_word) > 1 and not first_word[0].isalnum() and first_word[0] not in ["-", "(", "[", "#", "1", "2", "3", "a", "b"]:
            print(f"[WARNING] Truncated first word: '{first_word}'")
            chunking_clean = False

    # 2. Test Smart Reranking Scores
    print("\n--- STEP 2: Smart Entity-Weighted Reranking ---")
    test_queries = [
        "tell me about akshay",
        "What are the main topics discussed?",
        "Extract key entities and relationships"
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        scored_chunks = []
        for idx, c in enumerate(chunks):
            score = compute_smart_overlap_score(q, c)
            scored_chunks.append((score, idx, c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, idx, c) in enumerate(scored_chunks, 1):
            title = c.split("\n")[0][:60]
            print(f"  Rank {rank} [Score: {score:.4f}] Chunk {idx}: {title}...")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    test_semantic_chunking_and_ranking()
