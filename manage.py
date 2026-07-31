#!/usr/bin/env python3
"""
GraphRAG Unified Production Management CLI.

Usage:
  python manage.py reindex     Re-indexes all uploaded documents with adaptive structure-aware chunking
  python manage.py run         Starts the Flask backend server
  python manage.py test        Runs backend unit and integration tests
  python manage.py status      Prints collection and document database status
"""
import sys
import os
import subprocess
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


def cmd_reindex():
    """Runs automated batch re-indexing pipeline."""
    print("[manage.py] Stopping background server process to release storage locks...")
    subprocess.run(["pkill", "-f", "backend/main.py"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "flask"], stderr=subprocess.DEVNULL)
    import time
    time.sleep(1)
    print("[manage.py] Executing batch re-indexing pipeline...")
    script_path = PROJECT_ROOT / "scripts" / "reingest_all.py"
    subprocess.run([sys.executable, str(script_path)], check=True)


def cmd_run():
    """Starts the Flask application server without reloader lock issues."""
    print("[manage.py] Starting Flask backend server at http://127.0.0.1:5000...")
    main_path = PROJECT_ROOT / "backend" / "main.py"
    subprocess.run([sys.executable, str(main_path)])


def cmd_test():
    """Runs backend pytest test suite."""
    print("[manage.py] Running backend test suite...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-m", "not frontend", "-v"])


async def async_status():
    from backend.database import list_documents
    from backend.services.vector_store import get_collection_info, is_qdrant_available

    docs = await list_documents()
    qdrant_ok = await is_qdrant_available()
    info = await get_collection_info() if qdrant_ok else {}

    print("=" * 70)
    print("GRAPHRAG SYSTEM STATUS REPORT")
    print("=" * 70)
    print(f"Qdrant Status     : {'🟢 ONLINE' if qdrant_ok else '🔴 OFFLINE'}")
    print(f"Total Points/Vecs : {info.get('points_count', 0)}")
    print(f"Indexed Documents : {len(docs)}")
    print("-" * 70)
    print(f"{'Filename':<35} | {'Status':<10} | {'Pages':<6} | {'Chunks':<8}")
    print("-" * 70)
    for d in docs:
        name = (d.get("original_name") or d.get("id"))[:34]
        status = d.get("status", "unknown")
        pages = d.get("page_count", 0)
        chunks = d.get("chunk_count", 0)
        print(f"{name:<35} | {status:<10} | {pages:<6} | {chunks:<8}")
    print("=" * 70)


def cmd_status():
    asyncio.run(async_status())


def print_help():
    print(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd in ["reindex", "reingest", "chunk"]:
        cmd_reindex()
    elif cmd in ["run", "start", "server"]:
        cmd_run()
    elif cmd in ["test", "tests"]:
        cmd_test()
    elif cmd in ["status", "info"]:
        cmd_status()
    else:
        print(f"Unknown command '{cmd}'")
        print_help()
