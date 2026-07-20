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


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "tell", "me",
    "what", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "extract", "key", "main", "topics", "discussed"
}

def compute_smart_overlap_score(query: str, text: str) -> float:
    """
    Calculates an entity-weighted, stop-word filtered relevance score.
    Gives higher weight to rare terms, proper nouns, and entity matches.
    """
    import string
    translator = str.maketrans('', '', string.punctuation)
    clean_query = query.translate(translator).lower()
    clean_text = text.translate(translator).lower()

    query_tokens = [w for w in clean_query.split() if w]
    text_tokens = set(clean_text.split())

    meaningful_query_terms = [w for w in query_tokens if w not in STOP_WORDS]
    if not meaningful_query_terms:
        meaningful_query_terms = query_tokens

    score = 0.0
    for term in meaningful_query_terms:
        if term in text_tokens:
            weight = 2.0 if len(term) > 5 or term[0].isupper() else 1.0
            score += weight

    max_possible = sum(2.0 if len(t) > 5 or t[0].isupper() else 1.0 for t in meaningful_query_terms)
    normalized_score = score / max(max_possible, 1.0)
    return float(normalized_score)


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
        # Fall back to entity-weighted word overlap similarity score
        for chunk in chunks:
            payload = chunk.get("payload", {})
            text = payload.get("contextualized_text") or payload.get("text") or ""
            chunk["rerank_score"] = compute_smart_overlap_score(query, text)
    else:
        try:
            scores = model.predict(pairs)
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
        except Exception as e:
            logger.warning(f"Error during reranking prediction: {e}. Using fallback scores.")
            for chunk in chunks:
                payload = chunk.get("payload", {})
                text = payload.get("contextualized_text") or payload.get("text") or ""
                chunk["rerank_score"] = compute_smart_overlap_score(query, text)

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

