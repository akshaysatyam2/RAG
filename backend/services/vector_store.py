import uuid
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from backend.config import settings

logger = logging.getLogger(__name__)

# Initialize Qdrant client - local file-backed storage or in-memory for tests
import os
import sys
if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
    client = AsyncQdrantClient(location=":memory:")
else:
    qdrant_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "qdrant"))
    os.makedirs(qdrant_dir, exist_ok=True)
    client = AsyncQdrantClient(path=qdrant_dir)

COLLECTION_NAME = settings.qdrant.collection
# Usually the dense model has a fixed dimension. all-MiniLM-L6-v2 is 384.
DENSE_DIMENSION = 384


async def is_qdrant_available() -> bool:
    """
    Check if the Qdrant client can connect to the database.
    """
    try:
        await client.get_collections()
        return True
    except Exception:
        return False



async def initialize_collection():
    """
    Creates the collection with dense+sparse vector config if it doesn't exist.
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
    Batch upsert chunks with metadata to Qdrant.
    chunks should be a list of dictionaries, each containing:
    - 'id': optional chunk id, else generated
    - 'dense_vector': List[float]
    - 'sparse_vector': Dict with 'indices' and 'values'
    - 'metadata': Dictionary of metadata to store
    """
    if not chunks:
        return

    points = []
    for chunk in chunks:
        point_id = chunk.get("id") or str(uuid.uuid4())
        
        # Ensure metadata contains the document_id
        payload = chunk.get("metadata", {})
        payload["document_id"] = doc_id
        
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
        # Batch upsert
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
    Search using dense vector.
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
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedVector(
                name="dense",
                vector=query_vector
            ),
            query_filter=filter_params,
            limit=top_k,
            with_payload=True
        )
        return results
    except Exception as e:
        logger.error(f"Error searching dense vectors: {e}")
        return []


async def search_sparse(sparse_vector: Dict[str, Any], top_k: int, doc_filter: Optional[str] = None) -> List[models.ScoredPoint]:
    """
    Search using sparse vector.
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
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedSparseVector(
                name="sparse",
                vector=models.SparseVector(
                    indices=sparse_vector.get("indices", []),
                    values=sparse_vector.get("values", [])
                )
            ),
            query_filter=filter_params,
            limit=top_k,
            with_payload=True
        )
        return results
    except Exception as e:
        logger.error(f"Error searching sparse vectors: {e}")
        return []


async def delete_by_document(doc_id: str):
    """
    Remove all vectors for a specific document.
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
    Get collection stats.
    """
    try:
        collection_info = await client.get_collection(collection_name=COLLECTION_NAME)
        return {
            "status": collection_info.status,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return {}
