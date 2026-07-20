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
            return None
        with _model_lock:
            if _cross_encoder is None:
                try:
                    logger.info(f"Loading CrossEncoder model: {settings.embedding.cross_encoder_model}")
                    _cross_encoder = CrossEncoder(settings.embedding.cross_encoder_model)
                except Exception as e:
                    logger.warning(f"Could not load CrossEncoder model: {e}. Using fallback scoring.")
                    _cross_encoder = "dummy"
    return _cross_encoder


def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Re-rank a list of chunks using a CrossEncoder model, with fallback support.
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
        
    if model is None or model == "dummy":
        # Fall back to word overlap similarity score
        logger.info("Using word overlap fallback for reranking")
        query_words = set(query.lower().split())
        for chunk in chunks:
            payload = chunk.get("payload", {})
            text = (payload.get("contextualized_text") or payload.get("text") or "").lower()
            text_words = set(text.split())
            overlap = len(query_words.intersection(text_words))
            score = overlap / max(len(query_words), 1)
            chunk["rerank_score"] = float(score)
    else:
        try:
            scores = model.predict(pairs)
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
        except Exception as e:
            logger.warning(f"Error during reranking prediction: {e}. Using simple fallback scores.")
            for chunk in chunks:
                chunk["rerank_score"] = 0.5

    # Sort by score descending
    sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", -float("inf")), reverse=True)
    return sorted_chunks[:top_k]



def filter_by_threshold(chunks: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """
    Filter chunks above a given similarity/rerank score threshold.
    If no chunks meet the threshold, returns top candidate chunks as a fallback.
    """
    filtered = []
    for chunk in chunks:
        score = chunk.get("rerank_score")
        if score is not None and score >= threshold:
            filtered.append(chunk)
            
    if not filtered and chunks:
        filtered = chunks[:3]

    return filtered

