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

def has_target_info(term: str, raw_text: str) -> bool:
    """
    Checks if a target query term (like 'email', 'contact', 'phone') is matched
    either by literal token or by pattern matching (e.g., regex for email address or phone number).
    """
    if not raw_text or not term:
        return False
        
    t_lower = term.lower()
    raw_lower = raw_text.lower()
    
    if t_lower == "email":
        if "email" in raw_lower or "mail" in raw_lower:
            return True
        return bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_text))

    if t_lower in {"contact", "phone", "mobile"}:
        if any(k in raw_lower for k in ["contact", "phone", "mobile", "tel", "linkedin", "portfolio", "github"]):
            return True
        if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_text):
            return True
        return bool(re.search(r"\+?\d[\d\s-]{7,}\d", raw_text))

    if t_lower in {"location", "address", "city"}:
        if any(k in raw_lower for k in ["location", "address", "city", "bengaluru", "pune", "delhi", "mumbai", "india", "ka", "mh"]):
            return True

    # Fallback: token overlap or stemming
    import string
    translator = str.maketrans('', '', string.punctuation)
    clean_t = stem_word(t_lower)
    clean_text_tokens = {stem_word(w) for w in raw_lower.translate(translator).split() if w}
    return clean_t in clean_text_tokens or any(clean_t in st for st in clean_text_tokens if len(st) > 3)


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
        has_target = any(has_target_info(t, text) for t in target_terms)
        if not has_target:
            return 0.0

    score = 0.0
    for term in meaningful_query_terms:
        stemmed_term = stem_word(term)
        if has_target_info(term, text):
            weight = 2.0
            score += weight
        elif stemmed_term in stemmed_text_tokens or any(stemmed_term in t for t in stemmed_text_tokens if len(t) > 3):
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
            raw_text = payload.get("contextualized_text") or payload.get("text") or ""
            has_target = any(has_target_info(t, raw_text) for t in target_terms)
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


