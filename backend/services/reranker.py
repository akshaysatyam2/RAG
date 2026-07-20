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

import re

def stem_word(w: str) -> str:
    """Simple stemming for common English suffixes (plural 's', 'es', 'ing', 'ed')."""
    w = w.lower()
    if len(w) > 4:
        if w.endswith("ies"): return w[:-3] + "y"
        if w.endswith("es"): return w[:-2]
        if w.endswith("s") and not w.endswith("ss"): return w[:-1]
        if w.endswith("ing"): return w[:-3]
        if w.endswith("ed"): return w[:-2]
    return w

def normalize_query_text(query: str) -> str:
    """
    Normalizes query string:
    - Removes possessive suffixes ('s, ’s) e.g., "akshay's" -> "akshay"
    - Strips extra whitespace
    """
    if not query:
        return ""
    q_norm = re.sub(r"['’]s\b", "", query, flags=re.IGNORECASE)
    return q_norm.strip()

def compute_smart_overlap_score(query: str, text: str) -> float:
    """
    Calculates an entity-weighted, stop-word filtered relevance score with stemming support.
    Enforces strict matching for target query terms (email, contact, phone, etc.).
    """
    import string
    translator = str.maketrans('', '', string.punctuation)
    clean_query = normalize_query_text(query).translate(translator).lower()
    clean_text = text.translate(translator).lower()

    query_tokens = [w for w in clean_query.split() if w]
    text_tokens = [w for w in clean_text.split() if w]
    stemmed_text_tokens = {stem_word(w) for w in text_tokens}

    meaningful_query_terms = [w for w in query_tokens if w not in STOP_WORDS]
    if not meaningful_query_terms:
        meaningful_query_terms = query_tokens

    # Enforce strict matching if specific target info terms are requested
    target_terms = [t for t in meaningful_query_terms if t in {"email", "contact", "phone", "address", "location", "salary", "github", "linkedin"}]
    if target_terms:
        has_target = any(
            stem_word(t) in stemmed_text_tokens or any(stem_word(t) in st for st in stemmed_text_tokens if len(st) > 3)
            for t in target_terms
        )
        if not has_target:
            return 0.0

    score = 0.0
    for term in meaningful_query_terms:
        stemmed_term = stem_word(term)
        if stemmed_term in stemmed_text_tokens or any(stemmed_term in t for t in stemmed_text_tokens if len(t) > 3):
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

    # Post-check: Enforce target info terms (email, contact, phone, etc.)
    import string
    translator = str.maketrans('', '', string.punctuation)
    norm_q = normalize_query_text(query).translate(translator).lower()
    q_tokens = [w for w in norm_q.split() if w not in STOP_WORDS]
    target_terms = [t for t in q_tokens if t in {"email", "contact", "phone", "address", "location", "salary", "github", "linkedin"}]

    if target_terms:
        for chunk in chunks:
            payload = chunk.get("payload", {})
            text = (payload.get("contextualized_text") or payload.get("text") or "").translate(translator).lower()
            text_tokens = set(text.split())
            stemmed_text = {stem_word(w) for w in text_tokens}
            has_target = any(
                stem_word(t) in stemmed_text or any(stem_word(t) in st for st in stemmed_text if len(st) > 3)
                for t in target_terms
            )
            if not has_target:
                chunk["rerank_score"] = -100.0


    # Sort by score descending
    sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", -float("inf")), reverse=True)
    return sorted_chunks[:top_k]





def filter_by_threshold(chunks: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """
    Filter chunks above a given similarity/rerank score threshold.
    Only returns top candidate chunks as fallback if top_score > 0.0.
    """
    filtered = []
    for chunk in chunks:
        score = chunk.get("rerank_score")
        if score is not None and score >= threshold and score > 0.0:
            filtered.append(chunk)
            
    if not filtered and chunks:
        top_score = chunks[0].get("rerank_score", -float("inf"))
        if top_score > 0.0:
            filtered = chunks[:3]

    return filtered


