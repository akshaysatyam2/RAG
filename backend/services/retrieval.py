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
    k: int = 60,
    query: str = ""
) -> List[Dict[str, Any]]:
    """
    Merges dense and sparse result lists using Reciprocal Rank Fusion.
    Score formula: sum(1 / (k + rank_i)) over each result list.
    Chunks appearing in both lists get a score boost from both ranks.
    Applies entity-alignment boost for chunks matching subject entities in query.
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

    # Entity-alignment boost: boost chunks that match proper nouns/entities in query
    if query:
        q_lower = query.lower()
        named_entities = [w for w in ["akshay", "loksun", "techdome", "quadwave"] if w in q_lower]
        if named_entities:
            for point_id, payload in payloads.items():
                p_text = (str(payload.get("text", "")) + " " + str(payload.get("document_name", ""))).lower()
                if any(ne in p_text for ne in named_entities):
                    scores[point_id] += 0.05
        
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


import re
import os
import threading
from rapidfuzz import process, fuzz

_CORPUS_VOCAB_CACHE = None
_CORPUS_VOCAB_LOCK = threading.Lock()


def get_corpus_vocab() -> set[str]:
    """
    Returns unique vocabulary extracted from indexed document chunks and core domain terms.
    Cached in memory for high-performance fuzzy query auto-correction.
    """
    global _CORPUS_VOCAB_CACHE
    if _CORPUS_VOCAB_CACHE is not None:
        return _CORPUS_VOCAB_CACHE

    with _CORPUS_VOCAB_LOCK:
        if _CORPUS_VOCAB_CACHE is not None:
            return _CORPUS_VOCAB_CACHE

        vocab = {
            # Machine learning, statistical learning & ISLR vocabulary
            "logistic", "regression", "classification", "probability", "qualitative",
            "quantitative", "predictors", "coefficients", "resampling", "bootstrap",
            "cross-validation", "hyperplane", "margin", "support", "vector", "kernel",
            "variance", "regularization", "lasso", "ridge", "shrinkage", "splines",
            "clustering", "dendrogram", "entropy", "gradient", "descent", "neural",
            "network", "deep", "machine", "learning", "artificial", "intelligence",
            "algorithm", "embeddings", "tokenizer", "reranker", "compression",
            "retrieval", "synthesis", "university", "college", "degree", "bachelor",
            "cybersecurity", "engineer", "framework", "architecture", "analysis",
            "parameter", "estimate", "standard", "error", "hypothesis", "significance",
            "deviance", "confusion", "matrix", "precision", "recall", "overfitting",
            "underfitting", "validation", "multimodal", "graphrag", "per-page",
            "dimension", "dataset", "separation", "boundary", "linear", "quadratic",
            "discriminant", "likelihood", "maximum", "sigmoid", "response", "variable",
            "features", "supervised", "unsupervised", "reinforcement", "performance"
        }

        # Supplement with terms from SQLite metadata / graph triples if available
        try:
            import sqlite3
            sqlite_path = settings.database.sqlite_path
            if os.path.exists(sqlite_path):
                conn = sqlite3.connect(sqlite_path)
                cursor = conn.cursor()
                cursor.execute("SELECT head, tail FROM graph_triples LIMIT 2000")
                for row in cursor.fetchall():
                    for text_item in row:
                        if text_item:
                            words = re.findall(r'[a-zA-Z]{3,}', text_item.lower())
                            vocab.update(words)
                conn.close()
        except Exception as e:
            logger.debug(f"Could not supplement corpus vocab from DB: {e}")

        _CORPUS_VOCAB_CACHE = vocab
        return _CORPUS_VOCAB_CACHE


def normalize_and_correct_query(query: str, corpus_vocab: Optional[set[str]] = None) -> tuple[str, bool]:
    """
    Normalizes query string, auto-corrects typos (edit distance <= 2) using corpus vocabulary,
    and formats questions properly.
    - Matches user tokens against unique vocabulary of indexed corpus using rapidfuzz
    - Only triggers did_correct UI banner when the normalized lowercase tokens actually change
    Returns (corrected_query_string, did_correct_bool).
    """
    if not query or not query.strip():
        return "", False

    vocab = corpus_vocab if corpus_vocab is not None else get_corpus_vocab()

    # Common explicit typo overrides
    typo_map = {
        "logestic": "logistic",
        "logesstic": "logistic",
        "legestic": "logistic",
        "logistc": "logistic",
        "regresion": "regression",
        "regrssion": "regression",
        "clasification": "classification",
        "classifcation": "classification",
        "clasify": "classify",
        "embedings": "embeddings",
        "embeding": "embedding",
        "neurel": "neural",
        "neral": "neural",
        "suport": "support",
        "vecter": "vector",
        "vectrs": "vectors",
        "analisis": "analysis",
        "artifical": "artificial",
        "artifcial": "artificial",
        "intellegence": "intelligence",
        "inteligence": "intelligence",
        "algoritm": "algorithm",
        "algortihm": "algorithm",
        "retreival": "retrieval",
        "retrival": "retrieval",
        "reranker": "reranker",
        "tokenizor": "tokenizer",
        "tokeniser": "tokenizer",
        "compreison": "compression",
        "compresion": "compression",
        "syntesis": "synthesis",
        "whaat": "what",
        "wat": "what",
        "wht": "what",
        "wen": "when",
        "wher": "where",
        "whch": "which",
        "wich": "which",
        "explian": "explain",
        "explan": "explain",
        "sumarize": "summarize",
        "sumerize": "summarize",
        "colledge": "college",
        "collg": "college",
        "univercity": "university",
        "universiti": "university",
        "graduat": "graduate",
        "edcuation": "education",
        "projets": "projects",
        "projct": "project",
        "skils": "skills",
        "stydy": "study",
        "studie": "study",
        "studed": "studied",
    }

    question_starters = {
        "what", "who", "where", "when", "why", "how", "which",
        "can", "could", "is", "are", "do", "does", "did", "would", "will", "whom", "whose"
    }

    clean_raw = re.sub(r'\s+', ' ', query).strip()
    words = clean_raw.split()
    corrected_words = []

    for w in words:
        # Separate punctuation
        punct_match = re.search(r'([^\w\s]+)$', w)
        trailing_punct = punct_match.group(1) if punct_match else ""
        leading_match = re.match(r'^([^\w\s]+)', w)
        leading_punct = leading_match.group(1) if leading_match else ""
        clean_w = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', w).lower()

        if clean_w in typo_map:
            replacement = typo_map[clean_w]
        elif len(clean_w) < 6 or clean_w in vocab:
            corrected_words.append(w)
            continue
        else:
            match = process.extractOne(
                clean_w,
                vocab,
                scorer=fuzz.WRatio,
                score_cutoff=74
            )
            replacement = match[0] if match else clean_w

        if replacement != clean_w:
            if w.isupper():
                replacement = replacement.upper()
            elif w and w[0].isupper():
                replacement = replacement.capitalize()
            corrected_words.append(leading_punct + replacement + trailing_punct)
        else:
            corrected_words.append(w)

    result = " ".join(corrected_words).strip()

    # Auto-format question starter capitalization and punctuation
    first_word_clean = re.sub(r'[^\w]', '', result.split()[0]).lower() if result.split() else ""
    if first_word_clean in question_starters:
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        if not result.endswith(("?", ".", "!")):
            result = result.rstrip(",;:") + "?"

    # Only show the UI banner if the actual words changed (ignore pure case changes on user text)
    did_correct = (result.lower().rstrip("?") != query.strip().lower().rstrip("?")) or (
        result.lower() != query.strip().lower() and not (
            result.endswith("?") and not query.strip().endswith("?") and result[:-1].lower() == query.strip().lower()
        )
    )

    # Simple, rock-solid rule: did actual words differ from raw words?
    words_orig = [re.sub(r'[^\w]', '', x).lower() for x in query.strip().split()]
    words_corr = [re.sub(r'[^\w]', '', x).lower() for x in result.split()]
    did_correct = (words_orig != words_corr) or (result.endswith("?") and not query.strip().endswith("?"))

    return result, did_correct


def expand_query_terms(query: str) -> str:
    """
    Appends domain synonyms and performs fuzzy typo correction to improve search recall.
    - Corrects common typos (e.g. 'logestic' -> 'logistic', 'regresion' -> 'regression')
    - Appends domain synonyms (e.g. logistic -> classification, logit, odds, probability)
    """
    normalized_query, _ = normalize_and_correct_query(query)
    q_lower = normalized_query.lower()
    expansions = []

    # Domain expansions
    if any(k in q_lower for k in ["logistic", "logit", "odds", "sigmoid"]):
        expansions.extend(["logistic", "regression", "classification", "logit", "odds", "probability", "sigmoid"])

    if any(k in q_lower for k in ["college", "university", "school", "education", "degree", "studies", "academics", "graduat"]):
        expansions.extend(["education", "university", "college", "degree", "bachelor", "master", "school", "cgpa", "academics"])

    if any(k in q_lower for k in ["project", "built", "developed", "created", "architected", "system"]):
        expansions.extend(["projects", "engineered", "developed", "architected", "system", "github"])

    if any(k in q_lower for k in ["work", "job", "experience", "company", "role", "position", "career", "intern"]):
        expansions.extend(["experience", "work", "engineer", "developer", "intern", "company", "role"])

    if any(k in q_lower for k in ["skill", "stack", "tools", "languages", "frameworks", "technologies"]):
        expansions.extend(["skills", "frameworks", "technologies", "languages", "tools", "infrastructure"])

    if expansions:
        unique_exp = list(dict.fromkeys(expansions))
        return f"{normalized_query} {' '.join(unique_exp)}"
    return normalized_query


async def hybrid_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Runs dense ANN search and sparse keyword search in parallel, then fuses
    the results using RRF. Orphaned chunks from deleted documents are filtered out.
    Returns up to top_k results ranked by combined RRF score.
    """
    from backend.services.reranker import normalize_query_text
    norm_query = normalize_query_text(query)
    expanded_q = expand_query_terms(norm_query)

    # 1. Get query vectors
    dense_vector = get_dense_embedding(expanded_q)
    sparse_vector = compute_sparse_vector(expanded_q)

    
    dense_results, sparse_results = [], []
    
    # 2. Run searches in parallel
    try:
        dense_task = search_dense(query_vector=dense_vector, top_k=top_k)
        sparse_task = search_sparse(sparse_vector=sparse_vector, top_k=top_k)
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    except Exception as e:
        logger.warning(f"Hybrid search failed to retrieve results from Qdrant: {e}")

    # Filter out chunks from documents that no longer exist in SQLite
    try:
        from backend.database import list_documents
        active_docs = await list_documents()
        active_ids = {d["id"] for d in active_docs}
        if active_ids:
            dense_results = [p for p in dense_results if getattr(p, "payload", {}).get("document_id") in active_ids]
            sparse_results = [p for p in sparse_results if getattr(p, "payload", {}).get("document_id") in active_ids]
        else:
            dense_results, sparse_results = [], []
    except Exception as e:
        logger.warning(f"Failed to filter active document IDs in hybrid search: {e}")

    # 3. Fuse results
    fused_results = reciprocal_rank_fusion(
        dense_results, 
        sparse_results, 
        k=settings.retrieval.rrf_k,
        query=query
    )
    
    # Return top_k fused
    return fused_results[:top_k]




async def expand_with_graph(pivot_chunks: List[Dict[str, Any]], max_hops: int = 2, query: str = "") -> List[Dict[str, Any]]:
    """
    Extracts entities from the top retrieved chunks and uses them as
    starting nodes for a graph traversal. Returns connected context paths
    from up to max_hops away in the knowledge graph.
    """
    pivot_entities = set()

    # Seed with query entities if present
    if query:
        q_lower = query.lower()
        if "akshay" in q_lower:
            pivot_entities.update(["Akshay", "Akshay Kumar", "Akshay Satyam"])
        if "loksun" in q_lower:
            pivot_entities.add("Loksun.AI")
        if "techdome" in q_lower:
            pivot_entities.add("Techdome Solutions")

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
    Orchestrates the full two-stage retrieval:
      Stage 1: Hybrid vector search (dense + sparse, fused with RRF)
      Stage 2: Graph expansion from the top-5 chunks as entity pivots
    Returns (hybrid_chunks, graph_context) for downstream re-ranking.
    """
    hybrid_chunks = []
    graph_context = []
    
    # 1. Hybrid Search
    try:
        hybrid_chunks = await hybrid_search(query, top_k=top_k)
    except Exception as e:
        logger.warning(f"Failed in full_retrieval_pipeline (hybrid search): {e}")
    
    # Use only the top 5 chunks as pivots to keep graph traversal focused
    try:
        pivot_chunks = hybrid_chunks[:5]
        graph_context = await expand_with_graph(pivot_chunks, max_hops=2, query=query)
    except Exception as e:
        logger.warning(f"Failed in full_retrieval_pipeline (graph expansion): {e}")
    
    return hybrid_chunks, graph_context

