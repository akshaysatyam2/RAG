#!/usr/bin/env python3
"""
Integration test script to test live Flask server endpoints with clean document upload:
1. Deleting stale documents
2. Generating a fresh PDF document test_sample.pdf
3. Uploading via /api/docs/upload
4. Waiting for background ingestion to finish (/api/status/<id>)
5. Querying /api/chat with user prompts:
   - "What are the main topics discussed?"
   - "tell me about akshay"
   - "Extract key entities and relationships"
   - "akshay"
6. Verifying retrieved answer responses and sources.
"""

import sys
import os
import time
import httpx
import fitz  # PyMuPDF

SERVER_URL = "http://localhost:5000"

SAMPLE_TEXT = """
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

def create_sample_pdf(pdf_path: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), SAMPLE_TEXT)
    doc.save(pdf_path)
    doc.close()

def test_live_server():
    print("=" * 80)
    print("      LIVE SERVER END-TO-END RAG TEST (FRESH UPLOAD)")
    print("=" * 80)

    client = httpx.Client(timeout=30.0)

    # 1. Health check
    h_resp = client.get(f"{SERVER_URL}/api/health")
    print(f"Health Check Status: {h_resp.status_code} -> {h_resp.json()}")

    # 2. Delete existing stale documents
    docs_list = client.get(f"{SERVER_URL}/api/docs").json().get("documents", [])
    for d in docs_list:
        doc_id = d["id"]
        print(f"Purging old document {doc_id}...")
        client.delete(f"{SERVER_URL}/api/docs/{doc_id}")

    # 3. Create & upload fresh PDF
    pdf_path = "test_sample_resume.pdf"
    create_sample_pdf(pdf_path)
    print(f"\nCreated and uploading fresh document {pdf_path}...")

    with open(pdf_path, "rb") as f:
        files = {"file": ("akshay_resume.pdf", f, "application/pdf")}
        up_resp = client.post(f"{SERVER_URL}/api/docs/upload", files=files)
    print(f"Upload Response: {up_resp.status_code} -> {up_resp.json()}")
    new_doc_id = up_resp.json()["id"]

    # Wait for status ready
    for attempt in range(20):
        time.sleep(1)
        st_resp = client.get(f"{SERVER_URL}/api/status/{new_doc_id}")
        status = st_resp.json().get("status")
        print(f"  Ingestion status: {status}")
        if status == "ready":
            print("[SUCCESS] Fresh document ingestion completed!")
            break
        elif status == "error":
            print(f"[FAIL] Ingestion failed: {st_resp.json()}")
            return False

    # 4. Test queries
    test_queries = [
        "What are the main topics discussed?",
        "tell me about akshay",
        "Extract key entities and relationships",
        "akshay"
    ]

    all_passed = True
    print("\n--- Testing Chat Queries via /api/chat ---")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        chat_resp = client.post(f"{SERVER_URL}/api/chat", json={"query": q, "history": [], "top_k": 5})
        print(f"Status Code: {chat_resp.status_code}")
        res_data = chat_resp.json()
        answer = res_data.get("answer", "")
        sources = res_data.get("sources", [])
        print(f"Answer (length={len(answer)}): {answer[:200]}...")
        print(f"Sources Count: {len(sources)}")

        if ("don't have enough information" in answer.lower() or "no matching information" in answer.lower()) and not sources:
            print(f"[FAIL] Query '{q}' returned no information.")
            all_passed = False
        else:
            print(f"[PASS] Query '{q}' answered successfully with {len(sources)} source chunks!")

    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    print("\n" + "=" * 80)
    print(f"LIVE SERVER END-TO-END TEST: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = test_live_server()
    sys.exit(0 if success else 1)
