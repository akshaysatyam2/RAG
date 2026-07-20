#!/usr/bin/env python3
"""
Standalone test script to test Qdrant persistence, document chunking, saving,
re-opening Qdrant storage, and querying to ensure data persists across server restarts.
"""

import sys
import os
import asyncio
import uuid
import hashlib
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.config import settings
from backend.services.ingestion import chunk_text
from backend.services.embeddings import get_dense_embedding, tokenize_for_sparse, compute_sparse_vector
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

TEST_QDRANT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "test_qdrant"))

SAMPLE_RESUME_TEXT = """
Akshay Satyam - Senior Software Engineer & RAG Architect

Professional Summary:
Akshay Satyam is a Senior Software Engineer specializing in Artificial Intelligence, Retrieval-Augmented Generation (RAG), and Knowledge Graphs.
He has extensive experience building production-grade vector search systems using Qdrant, PyMuPDF, SentenceTransformers, and Neo4j graph databases.

Technical Skills:
- Programming Languages: Python, JavaScript, C++, SQL
- AI & ML Frameworks: PyTorch, SentenceTransformers, OpenAI API, LangChain
- Databases: Qdrant Vector Store, Neo4j Graph Database, SQLite, Redis
- Frameworks: Flask, React, Celery, Docker, Pytest

Key Projects:
1. GraphRAG Hybrid Retrieval System:
Designed and implemented a hybrid retrieval engine combining dense vector search, BM25 sparse keyword matching, and Neo4j sub-graph traversal.
Achieved 95%+ precision on document QA tasks with contextual chunking and cross-encoder reranking.

2. Enterprise Document Parsing Pipeline:
Built an automated ingestion pipeline for parsing PDFs and image documents using PyMuPDF and Tesseract OCR, with automatic entity-relation triple extraction.

Education:
Bachelor of Technology in Computer Science and Engineering.
"""

async def test_qdrant_persistence():
    print("=" * 80)
    print("      TESTING QDRANT PERSISTENCE AND CHUNKING RETRIEVAL")
    print("=" * 80)

    # 1. Clean test directory
    os.makedirs(TEST_QDRANT_DIR, exist_ok=True)

    print("\n--- STEP 1: Initializing Qdrant Client on Disk ---")
    client1 = AsyncQdrantClient(path=TEST_QDRANT_DIR)
    collection_name = "test_persistence_collection"

    # Create collection
    await client1.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    print(f"Created collection '{collection_name}' on disk at {TEST_QDRANT_DIR}")

    # 2. Chunking sample text
    print("\n--- STEP 2: Chunking Sample Document ---")
    chunks = chunk_text(SAMPLE_RESUME_TEXT, chunk_size=300, overlap=50)
    doc_id = "doc_akshay_001"
    print(f"Generated {len(chunks)} text chunks for document '{doc_id}'")

    points = []
    for i, c in enumerate(chunks):
        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{i}"))
        ctx_text = f"Document Summary: Resume of Akshay Satyam, Senior Software Engineer.\n\nChunk Content:\n{c}"
        dense_vec = get_dense_embedding(ctx_text)
        sparse_vec = compute_sparse_vector(ctx_text)

        points.append(
            models.PointStruct(
                id=chunk_uuid,
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"]
                    )
                },
                payload={
                    "document_id": doc_id,
                    "chunk_index": i,
                    "page_number": 1,
                    "text": c,
                    "contextualized_text": ctx_text
                }
            )
        )

    # Upsert points
    await client1.upsert(collection_name=collection_name, points=points)
    info1 = await client1.get_collection(collection_name)
    print(f"Saved {info1.points_count} points into Qdrant collection!")

    # Close first client instance
    await client1.close()
    print("Closed first Qdrant client connection.")

    # 3. Re-open Qdrant client on disk (Simulating Server Restart)
    print("\n--- STEP 3: Re-opening Qdrant Client from Disk (Simulating Restart) ---")
    client2 = AsyncQdrantClient(path=TEST_QDRANT_DIR)
    info2 = await client2.get_collection(collection_name)
    print(f"Re-opened Qdrant collection! Points count on disk: {info2.points_count}")

    if info2.points_count != len(chunks):
        print(f"[FAIL] Expected {len(chunks)} points after restart, got {info2.points_count}")
        await client2.close()
        return False

    # 4. Test Queries
    queries = [
        "tell me about akshay",
        "What are the main topics discussed?",
        "Extract key entities and relationships",
        "akshay"
    ]

    print("\n--- STEP 4: Executing Queries against Persisted Vector DB ---")
    all_queries_passed = True

    for q in queries:
        print(f"\nQuery: '{q}'")
        q_dense = get_dense_embedding(q)
        q_sparse = compute_sparse_vector(q)

        # Dense search
        dense_res = await client2.query_points(
            collection_name=collection_name,
            using="dense",
            query=q_dense,
            limit=3,
            with_payload=True
        )

        # Sparse search
        sparse_res = await client2.query_points(
            collection_name=collection_name,
            using="sparse",
            query=models.SparseVector(
                indices=q_sparse["indices"],
                values=q_sparse["values"]
            ),
            limit=3,
            with_payload=True
        )

        print(f"  Dense hits: {len(dense_res.points)} | Sparse hits: {len(sparse_res.points)}")
        if dense_res.points:
            top_dense = dense_res.points[0]
            print(f"  Top Dense Hit [Score: {top_dense.score:.4f}]: {top_dense.payload.get('text', '')[:70]}...")
        if sparse_res.points:
            top_sparse = sparse_res.points[0]
            print(f"  Top Sparse Hit [Score: {top_sparse.score:.4f}]: {top_sparse.payload.get('text', '')[:70]}...")

        if not dense_res.points and not sparse_res.points:
            print(f"  [FAIL] No hits for query '{q}'!")
            all_queries_passed = False

    await client2.close()

    print("\n" + "=" * 80)
    print(f"PERSISTENCE AND QUERY TEST RESULT: {'PASS' if all_queries_passed else 'FAIL'}")
    print("=" * 80)
    return all_queries_passed

if __name__ == "__main__":
    success = asyncio.run(test_qdrant_persistence())
    sys.exit(0 if success else 1)
