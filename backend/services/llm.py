import json
import logging
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
async def generate_completion(system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
    """
    Generate a completion from the LLM.
    Strict grounding: Instructions should typically be in the system prompt.
    """
    temp = temperature if temperature is not None else settings.llm.temperature
    
    try:
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
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise


async def generate_context_summary(document_text: str) -> str:
    """
    Generates a document-level context summary.
    """
    system_prompt = (
        "You are an expert summarizer. Provide a concise, high-level summary of the "
        "provided document text. The summary should capture the main entities, themes, "
        "and overall context. If the text is empty or meaningless, reply with 'I don't know.'."
    )
    # To avoid context length issues, we might want to truncate the document_text 
    # but we'll assume the caller passes a reasonable length or we truncate simply here.
    max_chars = 15000
    text_to_summarize = document_text[:max_chars]
    
    user_prompt = f"Document Text:\n{text_to_summarize}"
    
    summary = await generate_completion(system_prompt, user_prompt, temperature=0.1)
    return summary


async def extract_entities_and_relations(chunk_text: str) -> List[Dict[str, Any]]:
    """
    Extracts entity-relation triples as structured JSON.
    Returns a list of dicts: [{"head": "Entity1", "relation": "RELATES_TO", "tail": "Entity2", "type": "Person", "description": "..."}]
    """
    system_prompt = (
        "You are an expert information extraction system. Extract entity-relation triples "
        "from the provided text. Return ONLY a valid JSON array of objects. "
        "Each object must have the following keys: 'head' (string), 'head_type' (string), "
        "'relation' (string, UPPERCASE_WITH_UNDERSCORES), 'tail' (string), 'tail_type' (string). "
        "Do not include any other text, markdown formatting like ```json, or explanations. "
        "If no relations are found, return an empty array []."
    )
    
    user_prompt = f"Text to extract from:\n{chunk_text}"
    
    response_text = await generate_completion(system_prompt, user_prompt, temperature=0.0)
    
    try:
        # Handle potential markdown formatting if the model disobeys
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        triples = json.loads(clean_text.strip())
        if not isinstance(triples, list):
            logger.warning(f"Extracted triples is not a list: {triples}")
            return []
        return triples
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {response_text}. Error: {e}")
        return []
