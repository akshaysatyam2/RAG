import threading
import string
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
from rank_bm25 import BM25Okapi

from backend.config import settings

# Lazy-load the dense model once and reuse it across requests
_dense_model = None
_model_lock = threading.Lock()


def get_deterministic_dummy_embedding(text: str) -> List[float]:
    """
    Generates a reproducible 384-dim vector from a SHA-256 hash of the text.
    Used as a fallback when sentence-transformers isn't installed or fails to load.
    Same text always produces the same vector, so tests remain deterministic.
    """
    import hashlib
    # Seed numpy from the first 4 bytes of the SHA-256 hash
    h = hashlib.sha256(text.encode('utf-8')).digest()
    np.random.seed(int.from_bytes(h[:4], 'big'))
    vec = np.random.randn(384)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        if SentenceTransformer is None:
            return None
        with _model_lock:
            if _dense_model is None:
                try:
                    _dense_model = SentenceTransformer(settings.embedding.model_name)
                except Exception as e:
                    logger.warning(f"Could not load SentenceTransformer: {e}. Using dummy embeddings.")
                    _dense_model = "dummy"
    return _dense_model


def get_dense_embedding(text: str) -> List[float]:
    """
    Returns a dense embedding vector for a single text string.
    Falls back to the deterministic dummy if the model isn't available.
    """
    model = _get_dense_model()
    if model is None or model == "dummy":
        return get_deterministic_dummy_embedding(text)
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.warning(f"Failed to encode text with model: {e}. Using dummy embedding.")
        return get_deterministic_dummy_embedding(text)


def get_dense_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Encodes a batch of texts in one forward pass for efficiency.
    Falls back to per-text dummy embeddings if the model fails.
    """
    if not texts:
        return []
    
    model = _get_dense_model()
    if model is None or model == "dummy":
        return [get_deterministic_dummy_embedding(t) for t in texts]
    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    except Exception as e:
        logger.warning(f"Failed to encode texts with model: {e}. Using dummy embeddings.")
        return [get_deterministic_dummy_embedding(t) for t in texts]



def tokenize_for_sparse(text: str) -> List[str]:
    """
    Simple tokenizer for BM25 and sparse vector construction.
    Lowercases the input, strips punctuation, and splits on whitespace.
    """
    text = text.lower()
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = text.split()
    return tokens


def compute_sparse_vector(text: str, corpus_tokens: Optional[List[List[str]]] = None) -> Dict[str, Any]:
    """
    Builds a sparse vector using MD5-based feature hashing over token frequencies.
    The same token always maps to the same index, so indexing and querying stay consistent.
    Returns {'indices': [...], 'values': [...]} compatible with Qdrant's sparse vector format.
    """
    tokens = tokenize_for_sparse(text)
    if not tokens:
        return {"indices": [], "values": []}

    import hashlib
    counts: Dict[int, float] = {}
    for token in tokens:
        idx = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % 1000000
        counts[idx] = counts.get(idx, 0.0) + 1.0

    indices = list(counts.keys())
    values = list(counts.values())
    return {
        "indices": indices,
        "values": values
    }

