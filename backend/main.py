import os
import uuid
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError

from backend.config import settings
from backend.database import (
    initialize_schema,
    insert_document,
    get_document,
    list_documents,
    delete_document,
    get_ingestion_progress
)
from backend.models import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
    DeleteResponse,
    ChatRequest,
    ChatResponse,
    SourceChunk,
    IngestionStatusResponse,
    IngestionPhaseProgress
)
from backend.workers.tasks import ingest_document
from backend.services.vector_store import (
    initialize_collection,
    delete_by_document as vector_delete_by_document
)
from backend.services.graph import (
    initialize_graph,
    delete_by_document as graph_delete_by_document
)
from backend.services.retrieval import full_retrieval_pipeline
from backend.services.reranker import rerank, filter_by_threshold
from backend.services.llm import generate_completion

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

_executor = ThreadPoolExecutor(max_workers=10)


def run_async(coro):
    """
    Utility to run an async coroutine synchronously from a synchronous context,
    safely handling existing event loops (e.g. under pytest-asyncio).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop is not None and loop.is_running():
        # Run inside a separate thread to prevent loop conflict
        def _run():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return _executor.submit(_run).result()
    else:
        return asyncio.run(coro)


# Global Exception Handler
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error("Unhandled exception occurred during request:", exc_info=e)
    return jsonify({"detail": str(e)}), 500


# Health Check
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


# Document Upload Endpoint
@app.route("/api/docs/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"detail": "No file part"}), 400
    
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"detail": "No selected file"}), 400
        
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"detail": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400
    
    # Read size
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    
    max_bytes = settings.ingestion.max_upload_size_mb * 1024 * 1024
    if file_size > max_bytes:
        return jsonify({"detail": f"File too large. Max size is {settings.ingestion.max_upload_size_mb}MB"}), 400
        
    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}{ext}"
    file_path = Path(settings.ingestion.upload_dir) / filename
    
    try:
        file.save(str(file_path))
    except Exception as e:
        return jsonify({"detail": "Failed to save file"}), 500
        
    run_async(insert_document(
        doc_id=doc_id,
        filename=filename,
        original_name=file.filename,
        file_type=ext.lstrip('.'),
        file_size=file_size
    ))
    
    # Dispatch Celery task with graceful inline fallback if Redis is not running
    try:
        ingest_document.delay(doc_id, str(file_path), ext.lstrip('.'))
    except Exception as e:
        app.logger.warning(f"Failed to dispatch Celery task: {e}. Executing ingestion in background thread pool.")
        from backend.workers.tasks import run_ingestion
        _executor.submit(lambda: run_async(run_ingestion(doc_id, str(file_path), ext.lstrip('.'))))

    
    resp = DocumentUploadResponse(
        id=doc_id,
        filename=filename,
        original_name=file.filename,
        file_type=ext.lstrip('.'),
        file_size=file_size,
        status="pending",
        message="Upload successful, ingestion started."
    )
    return jsonify(resp.model_dump()), 202


# List Documents Endpoint
@app.route("/api/docs", methods=["GET"])
def get_all_documents():
    docs = run_async(list_documents())
    metadata_list = [DocumentMetadata(**doc) for doc in docs]
    resp = DocumentListResponse(documents=metadata_list, total=len(metadata_list))
    return jsonify(resp.model_dump())


# Get Single Document Endpoint
@app.route("/api/docs/<doc_id>", methods=["GET"])
def get_document_by_id(doc_id):
    doc = run_async(get_document(doc_id))
    if not doc:
        return jsonify({"detail": "Document not found"}), 404
    resp = DocumentMetadata(**doc)
    return jsonify(resp.model_dump())


# Delete Document Endpoint (Cascading Purge)
@app.route("/api/docs/<doc_id>", methods=["DELETE"])
def delete_document_endpoint(doc_id):
    doc = run_async(get_document(doc_id))
    if not doc:
        return jsonify({"detail": "Document not found"}), 404
        
    # Delete from Qdrant
    try:
        run_async(vector_delete_by_document(doc_id))
    except Exception:
        pass
        
    # Delete from Neo4j
    try:
        run_async(graph_delete_by_document(doc_id))
    except Exception:
        pass
        
    # Delete file
    file_path = Path(settings.ingestion.upload_dir) / doc["filename"]
    if file_path.exists():
        os.remove(file_path)
        
    # Delete from SQLite
    deleted = run_async(delete_document(doc_id))
    
    resp = DeleteResponse(
        id=doc_id,
        deleted=deleted,
        message="Document deleted successfully" if deleted else "Document could not be deleted from database"
    )
    return jsonify(resp.model_dump())


# Chat Retrieval Pipeline Endpoint
@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.get_json() or {}
        try:
            req = ChatRequest(**data)
        except ValidationError as e:
            return jsonify({"detail": e.errors()}), 422
            
        # 1. Retrieve chunks
        hybrid_chunks, graph_context = run_async(full_retrieval_pipeline(
            req.query, top_k=settings.retrieval.top_k_retrieval
        ))
        
        # 2. Package all retrieved payloads for re-ranking
        expanded_chunks = []
        for chunk in hybrid_chunks:
            expanded_chunks.append({
                "id": chunk.get("id"),
                "payload": chunk.get("payload", {})
            })
            
        for path in graph_context:
            nodes = path.get("path_nodes", [])
            path_str = " -> ".join([n.get("name", "") for n in nodes])
            if path_str:
                expanded_chunks.append({
                    "id": f"graph_{uuid.uuid4()}",
                    "payload": {
                        "text": f"Knowledge Graph Relationship: {path_str}",
                        "document_id": "graph_db",
                        "document_name": "Knowledge Graph"
                    }
                })
                
        if not expanded_chunks:
            resp = ChatResponse(
                answer="I don't have enough information to answer that question.",
                sources=[],
                retrieval_metadata={"status": "No context retrieved"}
            )
            return jsonify(resp.model_dump())
            
        # 3. Score chunks using cross-encoder
        reranked_chunks = rerank(req.query, expanded_chunks, top_k=req.top_k)
        
        # 4. Filter by threshold
        filtered_chunks = filter_by_threshold(reranked_chunks, settings.retrieval.similarity_threshold)
        
        if not filtered_chunks:
            resp = ChatResponse(
                answer="I don't have enough information to answer that question.",
                sources=[],
                retrieval_metadata={"status": "No context met the similarity threshold"}
            )
            return jsonify(resp.model_dump())
            
        # 5. Build prompt
        context_texts = []
        sources = []
        for i, chunk in enumerate(filtered_chunks):
            payload = chunk.get("payload", {})
            context_texts.append(f"[{i+1}] {payload.get('text', '')}")
            sources.append(
                SourceChunk(
                    document_id=payload.get("document_id", "unknown"),
                    document_name=payload.get("document_name", "unknown"),
                    chunk_index=payload.get("chunk_index", 0),
                    page_number=payload.get("page_number"),
                    content=payload.get("text", ""),
                    relevance_score=chunk.get("rerank_score", 0.0)
                )
            )
            
        context_str = "\n\n".join(context_texts)
        
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question ONLY based on the provided context. "
            "If the context does not contain sufficient information, explicitly state that you cannot answer."
        )
        
        history_str = ""
        if req.history:
            history_str = "Chat History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in req.history]) + "\n\n"
            
        user_prompt = f"{history_str}Context:\n{context_str}\n\nQuestion: {req.query}"
        
        # 6. Generate answer
        answer = run_async(generate_completion(system_prompt, user_prompt))
        
        # Grounding fallback check
        if "i don't know" in answer.lower() or "cannot answer" in answer.lower() or "don't have enough information" in answer.lower():
            answer = "I don't have enough information to answer that question."
            
        resp = ChatResponse(
            answer=answer,
            sources=sources,
            retrieval_metadata={
                "retrieved_count": len(hybrid_chunks) + len(graph_context),
                "filtered_count": len(filtered_chunks)
            }
        )
        return jsonify(resp.model_dump())
        
    except Exception as e:
        return jsonify({"detail": str(e)}), 500


# Ingestion Status Endpoint
@app.route("/api/status/<doc_id>", methods=["GET"])
def get_document_status(doc_id):
    doc = run_async(get_document(doc_id))
    if not doc:
        return jsonify({"detail": "Document not found"}), 404
        
    progress_rows = run_async(get_ingestion_progress(doc_id))
    phases = [IngestionPhaseProgress(**row) for row in progress_rows]
    
    resp = IngestionStatusResponse(
        document_id=doc_id,
        status=doc["status"],
        phases=phases
    )
    return jsonify(resp.model_dump())


# Serve Frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    if not path or not os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, "index.html")
    return send_from_directory(frontend_dir, path)


def init_app():
    settings.ensure_directories()
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        loop.create_task(initialize_schema())
        loop.create_task(initialize_collection())
        loop.create_task(initialize_graph())
    else:
        loop.run_until_complete(initialize_schema())
        try:
            loop.run_until_complete(initialize_collection())
        except Exception as e:
            app.logger.warning(f"Failed to initialize Qdrant: {e}")
        try:
            loop.run_until_complete(initialize_graph())
        except Exception as e:
            app.logger.warning(f"Failed to initialize Neo4j: {e}")


import sys

if "pytest" not in sys.modules:
    init_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

