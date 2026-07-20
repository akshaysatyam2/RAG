import json
import logging
import re
from typing import Optional, Dict, Any, List

from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError, APIError, APIConnectionError

from backend.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client using settings
client = AsyncOpenAI(
    base_url=settings.llm.base_url,
    api_key=settings.llm.api_key,
    timeout=5.0,
)


def run_local_heuristic_completion(system_prompt: str, user_prompt: str) -> str:
    """
    Fallback rule-based text generation when LLM is unreachable.
    Uses simple keyword matching and sentence extraction from the prompt context.
    """
    # Extract query/question from the prompt
    query = user_prompt
    query_match = re.search(r'(?:Query|Question|User Request):\s*(.*?)$', user_prompt, re.DOTALL | re.IGNORECASE)
    if query_match:
        query = query_match.group(1).strip()
    else:
        lines = [l.strip() for l in user_prompt.split("\n") if l.strip()]
        if lines:
            query = lines[-1]

    # Look for chunks of text or "Context:" in the prompt
    context_matches = re.findall(r'(?:Context|Source|Chunk|Document).*?:\s*(.*?)(?=\n\n|\n[A-Z]|$)', user_prompt, re.DOTALL | re.IGNORECASE)
    if not context_matches:
        # Fallback to double newline split
        parts = user_prompt.split("\n\n")
        context_matches = [p for p in parts if len(p.split()) > 15]

    contexts = [c.strip() for c in context_matches if len(c.strip()) > 10]

    if not contexts:
        return (
            "I apologize, but I could not reach the local LLM server (Ollama) "
            "and no document context was found to extract an answer."
        )

    # Keywords from query
    query_words = [w.lower().replace("?", "").replace(".", "") for w in query.split() if len(w) > 3]
    if not query_words:
        query_words = ["resume", "experience", "akshay", "skills", "education", "project"]

    matching_sentences = []
    for ctx in contexts:
        sentences = re.split(r'(?<=[.!?])\s+', ctx)
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for word in query_words if word in sent_lower)
            if score > 0:
                matching_sentences.append((score, sent.strip()))

    # Sort matching sentences by score descending
    matching_sentences = sorted(matching_sentences, key=lambda x: x[0], reverse=True)
    
    if matching_sentences:
        seen = set()
        top_sents = []
        for _, sent in matching_sentences:
            if sent not in seen:
                seen.add(sent)
                top_sents.append(sent)
                if len(top_sents) >= 5:
                    break
        answer = " ".join(top_sents)
        return f"[System: Local Fallback Mode]\nBased on the uploaded documents:\n\n{answer}"
    
    # Heuristic fallback summary if no direct matches
    summary_sents = []
    for ctx in contexts[:2]:
        sentences = re.split(r'(?<=[.!?])\s+', ctx)
        summary_sents.extend(sentences[:2])
    
    answer = " ".join(s.strip() for s in summary_sents)
    return f"[System: Local Fallback Mode]\nSummary of relevant context:\n\n{answer}"


def run_local_heuristic_entity_extraction(text: str) -> List[Dict[str, Any]]:
    """
    Fallback regex-based entity extraction when LLM is unreachable.
    Finds capitalized phrases and links them using simple context rules.
    """
    # Find capitalized words (Entities)
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
        retry_if_exception_type(APIConnectionError) |
        retry_if_exception_type(APIError)
    ),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
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
    try:
        return await generate_completion_raw(system_prompt, user_prompt, temperature)
    except Exception as e:
        logger.warning(f"LLM Connection failed: {e}. Falling back to local heuristic completion.")
        return run_local_heuristic_completion(system_prompt, user_prompt)


async def generate_context_summary(document_text: str) -> str:
    """
    Generates a document-level context summary, falling back to heuristics if needed.
    """
    max_chars = 15000
    text_to_summarize = document_text[:max_chars]
    
    try:
        system_prompt = (
            "You are an expert summarizer. Provide a concise, high-level summary of the "
            "provided document text. The summary should capture the main themes and overall context."
        )
        user_prompt = f"Document Text:\n{text_to_summarize}"
        return await generate_completion_raw(system_prompt, user_prompt, temperature=0.1)
    except Exception as e:
        logger.warning(f"LLM context summary failed: {e}. Using local heuristic summary.")
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_to_summarize) if s.strip()]
        summary = " ".join(sentences[:3])
        return f"[Local Summary Fallback] {summary}"


async def extract_entities_and_relations(chunk_text: str) -> List[Dict[str, Any]]:
    """
    Extracts entity-relation triples as structured JSON, falling back to regex if needed.
    """
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
        return run_local_heuristic_entity_extraction(chunk_text)
