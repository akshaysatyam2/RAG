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
    Merges dense and sparse result lists using Reciprocal Rank Fusion.
    Score formula: sum(1 / (k + rank_i)) over each result list.
    Chunks appearing in both lists get a score boost from both ranks.
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


def normalize_and_correct_query(query: str) -> tuple[str, bool]:
    """
    Normalizes query string, auto-corrects typos, and formats questions properly.
    - Cleans redundant whitespace
    - Auto-corrects typos using domain dictionary + fuzzy matching (difflib)
    - Auto-capitalizes the first letter of sentences
    - Ensures proper question mark punctuation ('?') for interrogative questions
    Returns (corrected_query_string, was_corrected_bool).
    """
    if not query or not query.strip():
        return "", False

    import re
    import difflib

    # Common typo dictionary covering ML, RAG, domain, question, and general terms
    typo_map = {
        # Machine learning & data science
        "logestic": "logistic",
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
        "hiarchical": "hierarchical",
        "hierarcial": "hierarchical",
        "artifical": "artificial",
        "artifcial": "artificial",
        "intellegence": "intelligence",
        "inteligence": "intelligence",
        "algoritm": "algorithm",
        "algortihm": "algorithm",
        "retreival": "retrieval",
        "retrival": "retrieval",
        "reranker": "reranker",
        "rerankng": "reranking",
        "tokenizor": "tokenizer",
        "tokeniser": "tokenizer",
        "compreison": "compression",
        "compresion": "compression",
        "syntesis": "synthesis",
        "synthesize": "synthesize",
        
        # General question words & common query terms
        "whaat": "what",
        "wat": "what",
        "wht": "what",
        "wen": "when",
        "wher": "where",
        "whch": "which",
        "wich": "which",
        "hw": "how",
        "explian": "explain",
        "explan": "explain",
        "sumarize": "summarize",
        "sumerize": "summarize",
        "sumary": "summary",
        "summery": "summary",
        "documnt": "document",
        "documnets": "documents",
        "experiance": "experience",
        "experence": "experience",
        "colledge": "college",
        "collg": "college",
        "univercity": "university",
        "universiti": "university",
        "graduat": "graduate",
        "edcuation": "education",
        "educashun": "education",
        "projets": "projects",
        "projct": "project",
        "skils": "skills",
        "stydy": "study",
        "studie": "study",
        "studed": "studied",
    }

    canonical_terms = set(typo_map.values()) | {
        "machine", "learning", "deep", "transformer", "attention", "architecture",
        "database", "context", "accuracy", "precision", "recall", "threshold",
        "document", "education", "project", "experience", "overview", "language",
        "model", "framework", "knowledge", "graph", "hybrid", "rerank"
    }

    question_starters = {
        "what", "who", "where", "when", "why", "how", "which",
        "can", "could", "is", "are", "do", "does", "did", "would", "will", "whom", "whose"
    }

    clean_raw = re.sub(r'\s+', ' ', query).strip()
    words = clean_raw.split()
    corrected_words = []

    for w in words:
        # Preserve leading/trailing punctuation (e.g. quotes, brackets, question marks)
        punct_match = re.search(r'([^\w\s]+)$', w)
        trailing_punct = punct_match.group(1) if punct_match else ""
        leading_match = re.match(r'^([^\w\s]+)', w)
        leading_punct = leading_match.group(1) if leading_match else ""

        w_core = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', w)
        w_lower = w_core.lower()

        replacement = None
        if w_lower in typo_map:
            replacement = typo_map[w_lower]
        elif len(w_lower) >= 5 and w_lower not in canonical_terms and not w_lower.isdigit():
            matches = difflib.get_close_matches(w_lower, canonical_terms, n=1, cutoff=0.82)
            if matches:
                replacement = matches[0]

        if replacement and replacement != w_lower:
            if w_core.isupper():
                corrected = replacement.upper()
            elif w_core and w_core[0].isupper():
                corrected = replacement.capitalize()
            else:
                corrected = replacement
            corrected_words.append(leading_punct + corrected + trailing_punct)
        else:
            corrected_words.append(w)

    result = " ".join(corrected_words).strip()

    # Auto-format: Capitalize first character
    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    # Auto-format: Question mark punctuation for questions
    first_word = re.sub(r'[^\w]', '', result.split()[0]).lower() if result.split() else ""
    if first_word in question_starters and not result.endswith(("?", ".", "!")):
        result = result.rstrip(",;:") + "?"

    was_modified = (result != query.strip())
    return result, was_modified


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
        k=settings.retrieval.rrf_k
    )
    
    # Return top_k fused
    return fused_results[:top_k]




async def expand_with_graph(pivot_chunks: List[Dict[str, Any]], max_hops: int = 2) -> List[Dict[str, Any]]:
    """
    Extracts entities from the top retrieved chunks and uses them as
    starting nodes for a graph traversal. Returns connected context paths
    from up to max_hops away in the knowledge graph.
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
        if pivot_chunks:
            graph_context = await expand_with_graph(pivot_chunks, max_hops=2)
    except Exception as e:
        logger.warning(f"Failed in full_retrieval_pipeline (graph expansion): {e}")
    
    return hybrid_chunks, graph_context

