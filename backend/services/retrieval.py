import asyncio
import logging
from typing import List, Dict, Any, Tuple

from qdrant_client.http import models

from backend.config import settings
from backend.services.embeddings import get_dense_embedding, tokenize_for_sparse, compute_sparse_vector
from backend.services.vector_store import search_dense, search_sparse
from backend.services.graph import expand_from_entities
from backend.services.llm import extract_entities_and_relations

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: List[models.ScoredPoint], 
    sparse_results: List[models.ScoredPoint], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    RRF implementation: score = sum(1/(k + rank_i))
    Returns merged ranked list of dictionaries.
    """
    scores: Dict[str, float] = {}
    payloads: Dict[str, Any] = {}
    
    # Process dense results
    for rank, point in enumerate(dense_results, 1):
        point_id = str(point.id)
        if point_id not in scores:
            scores[point_id] = 0.0
        scores[point_id] += 1.0 / (k + rank)
        payloads[point_id] = point.payload
        
    # Process sparse results
    for rank, point in enumerate(sparse_results, 1):
        point_id = str(point.id)
        if point_id not in scores:
            scores[point_id] = 0.0
        scores[point_id] += 1.0 / (k + rank)
        payloads[point_id] = point.payload
        
    # Sort by RRF score descending
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    merged_results = []
    for point_id, score in sorted_results:
        merged_results.append({
            "id": point_id,
            "score": score,
            "payload": payloads[point_id]
        })
        
    return merged_results


async def hybrid_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Runs dense+sparse in parallel, fuses with RRF.
    """
    # 1. Get query vectors
    dense_vector = get_dense_embedding(query)
    sparse_vector = compute_sparse_vector(query)
    
    dense_results, sparse_results = [], []
    
    # 2. Run searches in parallel
    try:
        dense_task = search_dense(query_vector=dense_vector, top_k=top_k)
        sparse_task = search_sparse(sparse_vector=sparse_vector, top_k=top_k)
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    except Exception as e:
        logger.warning(f"Hybrid search failed to retrieve results from Qdrant: {e}")
    
    # 3. Fuse results
    fused_results = reciprocal_rank_fusion(
        dense_results, 
        sparse_results, 
        k=settings.retrieval.rrf_k
    )
    
    # Return top_k fused
    return fused_results[:top_k]


async def expand_with_graph(pivot_chunks: List[Dict[str, Any]], max_hops: int = 2) -> List[Dict[str, Any]]:
    """
    Graph traversal from pivot entities.
    Extracts entities from the top chunks to use as pivots.
    """
    if not pivot_chunks:
        return []
        
    pivot_entities = set()
    
    for chunk in pivot_chunks:
        payload = chunk.get("payload", {})
        chunk_text = payload.get("text", "")
        if chunk_text:
            try:
                triples = await extract_entities_and_relations(chunk_text)
                for t in triples:
                    if "head" in t: pivot_entities.add(t["head"])
                    if "tail" in t: pivot_entities.add(t["tail"])
            except Exception as e:
                logger.warning(f"Failed to extract entities during graph expansion: {e}")
                
    if not pivot_entities:
        return []
        
    # Expand from pivots
    try:
        graph_context = await expand_from_entities(list(pivot_entities), max_hops=max_hops)
        return graph_context
    except Exception as e:
        logger.warning(f"Neo4j graph expansion failed: {e}")
        return []


async def full_retrieval_pipeline(query: str, top_k: int = 20) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes hybrid search + graph expansion.
    Returns: (hybrid_chunks, graph_context)
    """
    hybrid_chunks = []
    graph_context = []
    
    # 1. Hybrid Search
    try:
        hybrid_chunks = await hybrid_search(query, top_k=top_k)
    except Exception as e:
        logger.warning(f"Failed in full_retrieval_pipeline (hybrid search): {e}")
    
    # 2. Graph Expansion (using top 5 chunks as pivots to limit scope)
    try:
        pivot_chunks = hybrid_chunks[:5]
        if pivot_chunks:
            graph_context = await expand_with_graph(pivot_chunks, max_hops=2)
    except Exception as e:
        logger.warning(f"Failed in full_retrieval_pipeline (graph expansion): {e}")
    
    return hybrid_chunks, graph_context

