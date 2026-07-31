import uuid
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from backend.config import settings

logger = logging.getLogger(__name__)

# Use local disk storage in production and fall back to in-memory during tests
# or when the Werkzeug reloader parent process would otherwise grab the lock.
import os
import sys

qdrant_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "qdrant"))

# Don't let the Werkzeug reloader parent process lock the Qdrant directory
is_reloader_parent = (os.environ.get("WERKZEUG_RUN_MAIN") is None) and any(x in sys.argv[0].lower() for x in ["main.py", "flask"])

if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") or is_reloader_parent:
    client = AsyncQdrantClient(location=":memory:")
else:
    try:
        os.makedirs(qdrant_dir, exist_ok=True)
        client = AsyncQdrantClient(path=qdrant_dir)
    except Exception as e:
        logger.warning(f"Could not lock Qdrant storage directory: {e}. Falling back to in-memory storage.")
        client = AsyncQdrantClient(location=":memory:")



COLLECTION_NAME = settings.qdrant.collection
# Usually the dense model has a fixed dimension. all-MiniLM-L6-v2 is 384.
DENSE_DIMENSION = 384


async def is_qdrant_available() -> bool:
    """
    Quick connectivity check — returns False if the Qdrant client can't list collections.
    """
    try:
        await client.get_collections()
        return True
    except Exception:
        return False



async def initialize_collection():
    """
    Creates the Qdrant collection with separate dense and sparse vector configs.
    Also creates a keyword payload index on document_id for efficient per-doc filtering.
    Skips creation if the collection already exists.
    """
    try:
        collections_response = await client.get_collections()
        collection_names = [c.name for c in collections_response.collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creating Qdrant collection: {COLLECTION_NAME}")
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=DENSE_DIMENSION,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )
            # Create payload index for document_id filtering
            await client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Successfully created Qdrant collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {e}")
        raise


async def upsert_chunks(doc_id: str, chunks: List[Dict[str, Any]]):
    """
    Batch-upserts chunks into Qdrant with their dense and sparse vectors.
    Each chunk dict needs: id, text, contextualized_text, dense_vector, sparse_vector, metadata.
    Writes in batches of 100 to stay within Qdrant payload limits.
    """
    if not chunks:
        return

    points = []
    for chunk in chunks:
        raw_id = chunk.get("id") or str(uuid.uuid4())
        # Validate or derive a UUID for the point ID
        try:
            point_id = str(uuid.UUID(str(raw_id)))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))
        
        # Build a flat payload that always includes document_id, text, and contextualized_text
        payload = dict(chunk.get("metadata", {}))
        payload["document_id"] = doc_id
        if "text" not in payload and "text" in chunk:
            payload["text"] = chunk["text"]
        if "contextualized_text" not in payload and "contextualized_text" in chunk:
            payload["contextualized_text"] = chunk["contextualized_text"]
        
        dense_vec = chunk.get("dense_vector", [])
        sparse_vec_data = chunk.get("sparse_vector", {"indices": [], "values": []})
        
        vectors = {
            "dense": dense_vec,
            "sparse": models.SparseVector(
                indices=sparse_vec_data.get("indices", []),
                values=sparse_vec_data.get("values", [])
            )
        }
        
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            )
        )
        
    try:
        # Write in batches to avoid hitting Qdrant's per-request size limits
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        logger.info(f"Successfully upserted {len(points)} chunks for document {doc_id}")
    except Exception as e:
        logger.error(f"Error upserting chunks for document {doc_id}: {e}")
        raise


async def search_dense(query_vector: List[float], top_k: int, doc_filter: Optional[str] = None) -> List[models.ScoredPoint]:
    """
    ANN search over dense (sentence-transformer) vectors.
    Optionally scoped to a single document via document_id filter.
    """
    filter_params = None
    if doc_filter:
        filter_params = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=doc_filter)
                )
            ]
        )
        
    try:
        res = await client.query_points(
            collection_name=COLLECTION_NAME,
            using="dense",
            query=query_vector,
            query_filter=filter_params,
            limit=top_k,
            with_payload=True
        )
        return res.points
    except Exception as e:
        logger.error(f"Error searching dense vectors: {e}")
        return []


async def search_sparse(sparse_vector: Dict[str, Any], top_k: int, doc_filter: Optional[str] = None) -> List[models.ScoredPoint]:
    """
    Sparse keyword search using the BM25-derived token feature vector.
    Optionally scoped to a single document via document_id filter.
    """
    filter_params = None
    if doc_filter:
        filter_params = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=doc_filter)
                )
            ]
        )
        
    try:
        res = await client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse",
            query=models.SparseVector(
                indices=sparse_vector.get("indices", []),
                values=sparse_vector.get("values", [])
            ),
            query_filter=filter_params,
            limit=top_k,
            with_payload=True
        )
        return res.points
    except Exception as e:
        logger.error(f"Error searching sparse vectors: {e}")
        return []


async def delete_by_document(doc_id: str):
    """
    Deletes all vector points associated with a given document_id.
    """
    try:
        await client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted vectors for document {doc_id}")
    except Exception as e:
        logger.error(f"Error deleting vectors for document {doc_id}: {e}")
        raise


async def get_collection_info() -> Dict[str, Any]:
    """
    Returns status and point counts for the Qdrant collection.
    """
    try:
        collection_info = await client.get_collection(collection_name=COLLECTION_NAME)
        return {
            "status": str(collection_info.status),
            "points_count": getattr(collection_info, "points_count", 0),
            "indexed_vectors_count": getattr(collection_info, "indexed_vectors_count", 0)
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return {}

