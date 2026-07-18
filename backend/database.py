import aiosqlite
import json
from pathlib import Path
from datetime import datetime, timezone

from backend.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    original_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    page_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    message TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS graph_triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    head TEXT NOT NULL,
    head_type TEXT NOT NULL,
    relation TEXT NOT NULL,
    tail TEXT NOT NULL,
    tail_type TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_progress_doc_id ON ingestion_progress(document_id);
CREATE INDEX IF NOT EXISTS idx_graph_triples_doc_id ON graph_triples(document_id);
CREATE INDEX IF NOT EXISTS idx_graph_triples_head ON graph_triples(head);
CREATE INDEX IF NOT EXISTS idx_graph_triples_tail ON graph_triples(tail);
"""


async def get_connection() -> aiosqlite.Connection:
    db_path = settings.database.sqlite_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    return conn


async def initialize_schema():
    conn = await get_connection()
    try:
        await conn.executescript(SCHEMA)
        await conn.commit()
    finally:
        await conn.close()


async def insert_document(doc_id: str, filename: str, original_name: str,
                          file_type: str, file_size: int) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    conn = await get_connection()
    try:
        await conn.execute(
            """INSERT INTO documents (id, filename, original_name, file_type, file_size, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (doc_id, filename, original_name, file_type, file_size, now, now)
        )
        await conn.commit()
        return {
            "id": doc_id, "filename": filename, "original_name": original_name,
            "file_type": file_type, "file_size": file_size, "status": "pending",
            "page_count": 0, "chunk_count": 0, "created_at": now, "updated_at": now,
        }
    finally:
        await conn.close()


async def update_document_status(doc_id: str, status: str,
                                 page_count: int = None, chunk_count: int = None,
                                 error_message: str = None):
    now = datetime.now(timezone.utc).isoformat()
    conn = await get_connection()
    try:
        fields = ["status = ?", "updated_at = ?"]
        values = [status, now]

        if page_count is not None:
            fields.append("page_count = ?")
            values.append(page_count)
        if chunk_count is not None:
            fields.append("chunk_count = ?")
            values.append(chunk_count)
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)

        values.append(doc_id)
        query = f"UPDATE documents SET {', '.join(fields)} WHERE id = ?"
        await conn.execute(query, values)
        await conn.commit()
    finally:
        await conn.close()


async def get_document(doc_id: str) -> dict | None:
    conn = await get_connection()
    try:
        cursor = await conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        await conn.close()


async def list_documents() -> list[dict]:
    conn = await get_connection()
    try:
        cursor = await conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def delete_document(doc_id: str) -> bool:
    conn = await get_connection()
    try:
        cursor = await conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await conn.commit()
        return cursor.rowcount > 0
    finally:
        await conn.close()


async def upsert_ingestion_progress(doc_id: str, phase: str,
                                     current_step: int, total_steps: int,
                                     message: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    conn = await get_connection()
    try:
        existing = await conn.execute(
            "SELECT id FROM ingestion_progress WHERE document_id = ? AND phase = ?",
            (doc_id, phase)
        )
        row = await existing.fetchone()

        if row:
            await conn.execute(
                """UPDATE ingestion_progress
                   SET current_step = ?, total_steps = ?, message = ?, updated_at = ?
                   WHERE document_id = ? AND phase = ?""",
                (current_step, total_steps, message, now, doc_id, phase)
            )
        else:
            await conn.execute(
                """INSERT INTO ingestion_progress (document_id, phase, current_step, total_steps, message, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc_id, phase, current_step, total_steps, message, now)
            )
        await conn.commit()
    finally:
        await conn.close()


async def get_ingestion_progress(doc_id: str) -> list[dict]:
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT * FROM ingestion_progress WHERE document_id = ? ORDER BY id",
            (doc_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await conn.close()
