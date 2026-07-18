import threading
import logging
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from backend.config import settings

logger = logging.getLogger(__name__)

# Thread-safe lazy loading of the cross-encoder model
_cross_encoder = None
_model_lock = threading.Lock()


def _get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is not installed on this system.")
        with _model_lock:
            if _cross_encoder is None:
                logger.info(f"Loading CrossEncoder model: {settings.embedding.cross_encoder_model}")
                _cross_encoder = CrossEncoder(settings.embedding.cross_encoder_model)
    return _cross_encoder



def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Re-rank a list of chunks using a CrossEncoder model.
    Each chunk is expected to be a dictionary with a 'payload' containing 'text' or 'contextualized_text'.
    """
    if not chunks:
        return []

    model = _get_reranker()
    
    # Prepare pairs of (query, chunk_text)
    pairs = []
    for chunk in chunks:
        payload = chunk.get("payload", {})
        text = payload.get("contextualized_text") or payload.get("text") or ""
        pairs.append([query, text])
        
    # Predict scores
    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return chunks[:top_k]
        
    # Attach scores to chunks
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
        
    # Sort by score descending
    sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", -float("inf")), reverse=True)
    
    return sorted_chunks[:top_k]


def filter_by_threshold(chunks: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """
    Filter chunks above a given similarity/rerank score threshold.
    """
    filtered = []
    for chunk in chunks:
        score = chunk.get("rerank_score")
        if score is not None and score >= threshold:
            filtered.append(chunk)
            
    return filtered
