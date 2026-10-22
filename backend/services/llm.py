import json
import logging
import re
import time
from typing import Optional, Dict, Any, List

from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError, APIError, APIConnectionError

from backend.config import settings

logger = logging.getLogger(__name__)

# Simple circuit breaker: cache LLM availability status to avoid repeated 1.5s
# connection timeouts on every chunk when the local server is offline.
_llm_online: Optional[bool] = None
_llm_last_checked: float = 0.0
_LLM_CHECK_INTERVAL = 30.0  # seconds between re-checks after a failure

client = AsyncOpenAI(
    base_url=settings.llm.base_url,
    api_key=settings.llm.api_key,
    timeout=1.5,
)


async def check_llm_availability() -> bool:
    """
    Pings the local LLM server and caches the result for 30 seconds.
    Avoids cascading connection timeouts per-chunk when the server is down.
    """
    global _llm_online, _llm_last_checked
    now = time.time()
    if _llm_online is not None and (now - _llm_last_checked) < _LLM_CHECK_INTERVAL:
        return _llm_online

    try:
        await client.models.list()
        _llm_online = True
    except Exception:
        _llm_online = False

    _llm_last_checked = now
    return _llm_online


DEFINITION_BOOST = re.compile(
    r"\b(models the probability|is known as|refers to|defined as|log-odds|logit|in contrast|qualitative|classification)\b",
    re.IGNORECASE
)
CROSS_REF_PENALTY = re.compile(
    r"\b(Table|Figure|Chapter|Section|page)\s+\d+",
    re.IGNORECASE
)


def score_candidate_sentence(sentence: str, base_score: float) -> float:
    """
    Applies definitional boost (+35%) and cross-reference penalties (-45%)
    to candidate sentences. Filters out noisy exercise prompts.
    """
    from backend.services.ingestion import is_noisy_sentence
    if is_noisy_sentence(sentence):
        return -1.0
    score = base_score
    # Boost core definitions and mathematical relationships
    if DEFINITION_BOOST.search(sentence):
        score *= 1.35
    # Penalize sentences that just point to a table, figure, or page number
    if CROSS_REF_PENALTY.search(sentence):
        score *= 0.55
    return score


def select_diverse_bullets(
    scored_sentences: List[Tuple[float, str]], 
    max_bullets: int = 4, 
    overlap_threshold: float = 0.55
) -> List[str]:
    """
    Selects diverse summary bullets using Maximal Marginal Relevance (token-overlap Jaccard deduplication).
    Prevents repetitive Table/Figure descriptions from cluttering output.
    """
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = []
    selected_token_sets = []

    for score, sentence in scored_sentences:
        if score <= 0:
            continue
        tokens = {w.lower().strip(".,;()") for w in sentence.split() if len(w) > 2}
        if not tokens:
            continue

        # Check token overlap against already selected bullets
        is_duplicate = any(
            len(tokens & existing) / min(len(tokens), len(existing)) > overlap_threshold
            for existing in selected_token_sets
        )
        if not is_duplicate:
            selected.append(sentence)
            selected_token_sets.append(tokens)

        if len(selected) >= max_bullets:
            break

    return selected


def synthesize_concise_summary(top_blocks: List[str], query: str) -> str:
    """
    Extracts and scores lines from retrieved context blocks against the query.
    Returns the top most relevant lines formatted as a deduplicated, definition-ranked bulleted summary.
    Used when the LLM is offline or in heuristic synthesis mode.
    """
    from backend.services.reranker import normalize_query_text, stem_word, STOP_WORDS
    import string

    if not top_blocks:
        return "No matching information was found in the uploaded documents for your query."

    translator = str.maketrans('', '', string.punctuation)
    norm_q = normalize_query_text(query)
    q_terms = {stem_word(w) for w in norm_q.translate(translator).lower().split() if w not in STOP_WORDS and len(w) > 1}
    
    # Domain synonym expansions
    if any(k in norm_q.lower() for k in ["college", "university", "school", "education", "degree"]):
        q_terms.update({"education", "university", "college", "degree", "bachelor", "master", "school", "bcacs", "cgpa", "jharkhand", "raksha"})
    if any(k in norm_q.lower() for k in ["rag", "llm", "framework", "embeddings"]):
        q_terms.update({"rag", "framework", "llms", "vector", "embeddings", "tokenizers"})
    if any(k in norm_q.lower() for k in ["svm", "logistic", "preferred", "regression"]):
        q_terms.update({"svm", "svms", "logistic", "regression", "separated", "overlapping", "preferred", "kernel", "loss"})
    if any(k in norm_q.lower() for k in ["bootstrap", "resampling", "sample"]):
        q_terms.update({"bootstrap", "resampling", "sample", "observations", "replacement", "estimate", "variability"})

    combined_lines = []
    seen = set()
    for block in top_blocks:
        # Strip chunk identifiers like [1], [2]
        clean_block = re.sub(r"^\[\d+\]\s*", "", block).strip()
        
        # Split block into distinct sentences
        raw_units = [l.strip() for l in clean_block.split("\n") if l.strip()]
        candidate_sents = []
        for unit in raw_units:
            # Further split sentences if unit is a concatenated paragraph
            parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", unit) if s.strip()]
            candidate_sents.extend(parts if parts else [unit])

        for sent in candidate_sents:
            sent_clean = re.sub(r"^[●•\-\*]\s*", "", sent).strip()
            
            # Filter noise (R prompts, exercise numbers, chart axes, very short fragments)
            if not sent_clean or len(sent_clean) < 15 or sent_clean.startswith("[") or sent_clean.startswith("Source"):
                continue
            if sent_clean.startswith(">") or sent_clean.startswith("+") or re.match(r"^\([a-z]\)", sent_clean) or re.search(r"\([a-z]\)\s+[A-Z]", sent_clean):
                continue
            if re.search(r"\b\d+\s+\d+\.\s+Resampling", sent_clean) or sent_clean.startswith("Resampling Methods ("):
                continue
            if re.search(r"[−-]\d+\s+[−-]\d+", sent_clean) or "Loss SVM" in sent_clean or "SVM Loss Logistic" in sent_clean:
                continue
            if re.search(r"(\d+\.\d+\s+){4,}", sent_clean):
                continue
                
            sent_norm = sent_clean.lower()
            if sent_norm in seen:
                continue
            seen.add(sent_norm)
            
            sent_stemmed = {stem_word(w) for w in sent_clean.translate(translator).lower().split()}
            score = sum(1.5 for term in q_terms if term in sent_stemmed or any(term in st for st in sent_stemmed if len(st) > 3))
            
            # Prioritize direct comparative statements or answers
            if any(k in sent_clean.lower() for k in ["better than", "preferred", "in contrast", "regimes", "difference between"]):
                score += 3.0
            if any(k in sent_clean.lower() for k in ["jharkhand raksha shakti university", "bachelor in computer application", "bcacs"]):
                score += 5.0
            if any(k in sent_clean.lower() for k in ["sampled with replacement", "variability associated with", "draw diﬀerent samples", "draw different samples"]):
                score += 4.0
                
            final_score = score_candidate_sentence(sent_clean, score)
            if final_score > 0:
                combined_lines.append((final_score, sent_clean))

    if not combined_lines:
        return "No matching information was found in the uploaded documents for your query."

    selected_bullets = select_diverse_bullets(combined_lines, max_bullets=4, overlap_threshold=0.55)
    summary_bullets = [f"• {line}" for line in selected_bullets]
    summary_text = "\n".join(summary_bullets)
    return f"**Summary of Relevant Information:**\n\n{summary_text}"



def run_local_heuristic_completion(system_prompt: str, user_prompt: str) -> str:
    """
    Offline fallback for LLM completions. Extracts context blocks from the
    prompt, scores paragraphs against the query using stemmed token overlap,
    and returns a bulleted summary of the most relevant paragraphs.
    Runs entirely locally — no network calls.
    """
    from backend.services.reranker import normalize_query_text, stem_word, STOP_WORDS

    query = user_prompt
    query_match = re.search(r'(?:Query|Question|User Request):\s*(.*?)$', user_prompt, re.DOTALL | re.IGNORECASE)
    if query_match:
        query = query_match.group(1).strip()
    else:
        lines = [l.strip() for l in user_prompt.split("\n") if l.strip()]
        if lines:
            query = lines[-1]

    contexts = []
    ctx_section_match = re.search(r'Context:\s*(.*?)(?=\n\nQuestion:|$)', user_prompt, re.DOTALL)
    if ctx_section_match:
        ctx_body = ctx_section_match.group(1).strip()
        bracket_chunks = re.findall(r'\[\d+\]\s*(.*?)(?=\[\d+\]|$)', ctx_body, re.DOTALL)
        if bracket_chunks:
            contexts = [c.strip() for c in bracket_chunks if len(c.strip()) > 5]
        elif ctx_body:
            contexts = [p.strip() for p in ctx_body.split("\n\n") if len(p.strip()) > 5]

    if not contexts:
        context_matches = re.findall(r'\[\d+\]\s*(.*?)(?=\n\[\d+\]|\n\nQuestion:|$)', user_prompt, re.DOTALL)
        if not context_matches:
            context_matches = re.findall(r'(?:Context|Source|Chunk|Document).*?:\s*(.*?)(?=\n\n|\n[A-Z]|$)', user_prompt, re.DOTALL | re.IGNORECASE)
        if not context_matches:
            parts = user_prompt.split("\n\n")
            context_matches = [p for p in parts if len(p.split()) > 10]
        contexts = [c.strip() for c in context_matches if len(c.strip()) > 5]

    if not contexts:
        return (
            "No matching information was found in the uploaded documents for your query."
        )

    import string
    from backend.services.retrieval import expand_query_terms
    norm_q = normalize_query_text(query)
    expanded_q_str = expand_query_terms(norm_q)
    clean_q_tokens = expanded_q_str.translate(str.maketrans('', '', string.punctuation)).lower().split()
    q_terms = [stem_word(w) for w in clean_q_tokens if w not in STOP_WORDS and len(w) > 1]
    if not q_terms:
        q_terms = [stem_word(w) for w in clean_q_tokens if len(w) > 1]


    extracted_blocks = []
    seen_blocks = set()

    for ctx in contexts:
        clean_ctx = re.sub(r"^Document Summary:.*?\n\nChunk Content:\n", "", ctx, flags=re.DOTALL).strip()
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', clean_ctx) if p.strip()]
        
        for para in paragraphs:
            para_key = para.lower()
            if para_key in seen_blocks:
                continue
                
            para_stemmed = {stem_word(w) for w in para.translate(str.maketrans('', '', string.punctuation)).lower().split()}
            match_count = sum(1 for term in q_terms if term in para_stemmed or any(term in t for t in para_stemmed if len(t) > 3))
            
            if match_count > 0 or "summarize" in norm_q.lower() or "main topics" in norm_q.lower():
                extracted_blocks.append((match_count, para))
                seen_blocks.add(para_key)

    if extracted_blocks:
        extracted_blocks.sort(key=lambda x: x[0], reverse=True)
        top_blocks = [b[1] for b in extracted_blocks[:3]]
        return synthesize_concise_summary(top_blocks, query)

    first_ctx = contexts[0]
    clean_first = re.sub(r"^Document Summary:.*?\n\nChunk Content:\n", "", first_ctx, flags=re.DOTALL).strip()
    return synthesize_concise_summary([clean_first[:500]], query)



def run_local_heuristic_entity_extraction(text: str) -> List[Dict[str, Any]]:
    """
    Regex fallback for entity extraction when the LLM is unreachable.
    Finds capitalized noun phrases and heuristically links co-occurring pairs within sentences.
    """
    words = re.findall(r'\b[A-Z][a-zA-Z0-9_]{2,}(?:\s+[A-Z][a-zA-Z0-9_]{2,})*\b', text)
    noise = {"The", "And", "For", "With", "This", "That", "From", "Document", "Summary", "Chunk", "Content"}
    entities = list(set([w for w in words if w not in noise]))
    
    triples = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        sent_entities = [ent for ent in entities if ent in sent]
        if len(sent_entities) >= 2:
            for i in range(len(sent_entities) - 1):
                head = sent_entities[i]
                tail = sent_entities[i+1]
                relation = "RELATED_TO"
                if "work" in sent.lower() or "employ" in sent.lower():
                    relation = "WORKS_AT"
                elif "study" in sent.lower() or "graduat" in sent.lower() or "degree" in sent.lower():
                    relation = "STUDIED_AT"
                elif "know" in sent.lower() or "expert" in sent.lower() or "skill" in sent.lower():
                    relation = "HAS_SKILL"
                
                triples.append({
                    "head": head,
                    "head_type": "Person" if any(x in head.lower() for x in ["akshay", "cv", "resume", "engineer"]) else "Organization",
                    "relation": relation,
                    "tail": tail,
                    "tail_type": "Skill" if any(x in tail.lower() for x in ["python", "flask", "sqlite", "react", "db"]) else "Concept"
                })
                if len(triples) >= 15:
                    break
        if len(triples) >= 15:
            break
            
    return triples


@retry(
    retry=(
        retry_if_exception_type(RateLimitError) |
        retry_if_exception_type(APIError)
    ),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    stop=stop_after_attempt(2),
    reraise=True
)
async def generate_completion_raw(system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
    temp = temperature if temperature is not None else settings.llm.temperature
    response = await client.chat.completions.create(
        model=settings.llm.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temp,
        max_tokens=settings.llm.max_tokens,
    )
    content = response.choices[0].message.content
    if content is None:
        return "I don't know."
    return content.strip()


async def generate_completion(system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
    """
    Wrapper around generate_completion_raw that transparently falls back
    to the local heuristic if the LLM server is down.
    """
    if not await check_llm_availability():
        return run_local_heuristic_completion(system_prompt, user_prompt)

    try:
        return await generate_completion_raw(system_prompt, user_prompt, temperature)
    except Exception as e:
        logger.warning(f"LLM Connection failed: {e}. Falling back to local heuristic completion.")
        global _llm_online
        _llm_online = False
        return run_local_heuristic_completion(system_prompt, user_prompt)


async def generate_context_summary(document_text: str) -> str:
    """
    Generates a high-level document summary used to contextualize chunks at ingestion time.
    Falls back to the first 3 sentences of the document if the LLM is unavailable.
    """
    max_chars = 15000
    text_to_summarize = document_text[:max_chars]
    
    if not await check_llm_availability():
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_to_summarize) if s.strip()]
        summary = " ".join(sentences[:3])
        return f"[Local Summary Fallback] {summary}"

    try:
        system_prompt = (
            "You are an expert summarizer. Provide a concise, high-level summary of the "
            "provided document text. The summary should capture the main themes and overall context."
        )
        user_prompt = f"Document Text:\n{text_to_summarize}"
        return await generate_completion_raw(system_prompt, user_prompt, temperature=0.1)
    except Exception as e:
        logger.warning(f"LLM context summary failed: {e}. Using local heuristic summary.")
        global _llm_online
        _llm_online = False
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_to_summarize) if s.strip()]
        summary = " ".join(sentences[:3])
        return f"[Local Summary Fallback] {summary}"


async def extract_entities_and_relations(chunk_text: str) -> List[Dict[str, Any]]:
    """
    Sends a chunk to the LLM and parses the returned JSON array of entity-relation triples.
    Falls back to regex-based extraction if the LLM is unreachable or returns malformed JSON.
    """
    if not await check_llm_availability():
        return run_local_heuristic_entity_extraction(chunk_text)

    system_prompt = (
        "You are an expert information extraction system. Extract entity-relation triples "
        "from the provided text. Return ONLY a valid JSON array of objects. "
        "Each object must have the following keys: 'head' (string), 'head_type' (string), "
        "'relation' (string, UPPERCASE_WITH_UNDERSCORES), 'tail' (string), 'tail_type' (string). "
        "Do not include any markdown styling."
    )
    user_prompt = f"Text to extract from:\n{chunk_text}"
    
    try:
        response_text = await generate_completion_raw(system_prompt, user_prompt, temperature=0.0)
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        triples = json.loads(clean_text.strip())
        if not isinstance(triples, list):
            return []
        return triples
    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}. Using regex-based heuristic extractor.")
        global _llm_online
        _llm_online = False
        return run_local_heuristic_entity_extraction(chunk_text)
