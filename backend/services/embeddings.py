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

# Thread-safe lazy loading of the dense model
_dense_model = None
_model_lock = threading.Lock()


def get_deterministic_dummy_embedding(text: str) -> List[float]:
    """
    Generate a deterministic vector of length 384 using a hash of the text.
    Acts as a fail-safe fallback when sentence-transformers is not installed.
    """
    import hashlib
    # Deterministic seed using SHA-256 hash of text
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
    Get the dense embedding for a single text.
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
    Get dense embeddings for a batch of texts.
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
    Tokenizes text for BM25 / sparse vector creation.
    Lowercases, removes punctuation, and splits by whitespace.
    """
    text = text.lower()
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = text.split()
    return tokens


def compute_sparse_vector(text: str, corpus_tokens: List[List[str]]) -> Dict[str, Any]:
    """
    Computes a sparse vector representation for the text given a background corpus.
    We'll represent the sparse vector as a dict with 'indices' (list of integers) and 'values' (list of floats).
    In Qdrant, sparse vectors require integer indices. We'll use a vocabulary built from the corpus.
    """
    if not corpus_tokens:
        return {"indices": [], "values": []}

    # Build vocabulary
    vocab = {}
    for doc in corpus_tokens:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
                
    # Initialize BM25
    bm25 = BM25Okapi(corpus_tokens)
    
    # Tokenize input text
    query_tokens = tokenize_for_sparse(text)
    
    # To get a sparse vector for the query against the corpus vocabulary,
    # we can use the IDF values from the BM25 model and term frequencies in the query.
    # Standard BM25 scoring is doc-dependent, but for a sparse query vector we can use TF*IDF.
    
    indices = []
    values = []
    
    # Calculate term frequency in the query
    tf = {}
    for token in query_tokens:
        tf[token] = tf.get(token, 0) + 1
        
    for token, count in tf.items():
        if token in vocab and token in bm25.idf:
            idx = vocab[token]
            idf = bm25.idf[token]
            # Simple tf-idf for query
            val = count * idf
            if val > 0:
                indices.append(idx)
                values.append(val)
                
    return {
        "indices": indices,
        "values": values
    }
