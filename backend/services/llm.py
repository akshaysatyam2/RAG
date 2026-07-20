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

# Fast circuit breaker to prevent repeated timeouts when LLM server is offline
_llm_online: Optional[bool] = None
_llm_last_checked: float = 0.0
_LLM_CHECK_INTERVAL = 30.0  # seconds to re-test LLM connectivity after failure

client = AsyncOpenAI(
    base_url=settings.llm.base_url,
    api_key=settings.llm.api_key,
    timeout=1.5,
)


async def check_llm_availability() -> bool:
    """
    Checks if the local LLM server is reachable within a tight window.
    Caches status to avoid per-chunk HTTP connection timeouts when offline.
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


def synthesize_concise_summary(top_blocks: List[str], query: str) -> str:
    """
    Synthesizes verbatim text blocks into a clean, bulleted executive summary.
    Prioritizes lines that match the specific user query terms.
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
        q_terms.update({"education", "university", "college", "degree", "bachelor", "master", "school", "bcacs", "cgpa"})
    if any(k in norm_q.lower() for k in ["rag", "llm", "framework", "embeddings"]):
        q_terms.update({"rag", "framework", "llms", "vector", "embeddings", "tokenizers"})

    combined_lines = []
    seen = set()
    for block in top_blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        for line in lines:
            line_clean = re.sub(r"^[●•\-\*]\s*", "", line).strip()
            if not line_clean or line_clean in seen or line_clean.startswith("[") or line_clean.startswith("Source"):
                continue
            seen.add(line_clean)
            
            line_stemmed = {stem_word(w) for w in line_clean.translate(translator).lower().split()}
            score = sum(1 for term in q_terms if term in line_stemmed or any(term in st for st in line_stemmed if len(st) > 3))
            combined_lines.append((score, line_clean))

    if not combined_lines:
        return "No matching information was found in the uploaded documents for your query."

    combined_lines.sort(key=lambda x: x[0], reverse=True)

    summary_bullets = []
    for score, line in combined_lines[:5]:
        summary_bullets.append(f"• {line}")

    summary_text = "\n".join(summary_bullets)
    return f"**Summary of Relevant Information:**\n\n{summary_text}"



def run_local_heuristic_completion(system_prompt: str, user_prompt: str) -> str:
    """
    Fallback completion when local LLM server (Ollama) is offline or unresponsive.
    Uses possessive normalization, stemming, and paragraph block extraction from prompt context.
    Synthesizes concise bulleted summary instead of dumping whole verbatim documents.
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

    context_matches = re.findall(r'\[\d+\]\s*(.*?)(?=\n\[\d+\]|\n\nQuestion:|$)', user_prompt, re.DOTALL)
    if not context_matches:
        context_matches = re.findall(r'(?:Context|Source|Chunk|Document).*?:\s*(.*?)(?=\n\n|\n[A-Z]|$)', user_prompt, re.DOTALL | re.IGNORECASE)
    if not context_matches:
        parts = user_prompt.split("\n\n")
        context_matches = [p for p in parts if len(p.split()) > 10]

    contexts = [c.strip() for c in context_matches if len(c.strip()) > 10]

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
    Fallback regex-based entity extraction when LLM is unreachable.
    Finds capitalized phrases and links them using simple context rules.
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
    Generate completion with automatic local heuristic fallback if LLM is down.
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
    Generates a document-level context summary, falling back to heuristics if needed.
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
    Extracts entity-relation triples as structured JSON, falling back to regex if needed.
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
