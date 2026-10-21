import logging
import re
import threading
import time
from typing import List, Dict, Any, Optional

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

from backend.config import settings

logger = logging.getLogger(__name__)

# Thread-safe lazy loading of the PromptCompressor model
_compressor: Optional[Any] = None
_compressor_lock = threading.Lock()


def get_compressor() -> Optional[Any]:
    """
    Returns the singleton PromptCompressor instance, lazily loaded in a thread-safe manner.
    Returns None if llmlingua is not installed or model loading fails.
    """
    global _compressor
    if _compressor is None:
        if PromptCompressor is None:
            logger.warning("llmlingua is not installed. LongLLMLingua compression is unavailable.")
            return None
        with _compressor_lock:
            if _compressor is None:
                try:
                    logger.info(
                        f"Loading LongLLMLingua PromptCompressor with model: {settings.longllmlingua.model_name} "
                        f"on device: {settings.longllmlingua.device_map}"
                    )
                    _compressor = PromptCompressor(
                        model_name=settings.longllmlingua.model_name,
                        device_map=settings.longllmlingua.device_map,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not load PromptCompressor model '{settings.longllmlingua.model_name}': {e}. "
                        "LongLLMLingua will fall back to heuristic compression."
                    )
                    _compressor = "unavailable"
    if _compressor == "unavailable":
        return None
    return _compressor


def reset_compressor() -> None:
    """Resets the singleton compressor instance (useful for testing)."""
    global _compressor
    with _compressor_lock:
        _compressor = None


def estimate_token_count(text: str) -> int:
    """Heuristic token count based on whitespace and punctuation tokens."""
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def heuristic_compress_context(
    context_chunks: List[str], query: str, rate: float = 0.55
) -> Dict[str, Any]:
    """
    Lightweight heuristic compression fallback when the neural model is unavailable.
    Filters sentences by query term overlap and reorders chunks to place highest-relevance
    sentences first, reducing token count while preserving essential facts.
    """
    from backend.services.reranker import normalize_query_text, stem_word, STOP_WORDS
    import string

    norm_q = normalize_query_text(query)
    translator = str.maketrans("", "", string.punctuation)
    q_tokens = {
        stem_word(w)
        for w in norm_q.translate(translator).lower().split()
        if w not in STOP_WORDS and len(w) > 1
    }

    scored_chunks = []
    total_orig_tokens = 0

    for idx, chunk in enumerate(context_chunks):
        orig_tokens = estimate_token_count(chunk)
        total_orig_tokens += orig_tokens

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
        scored_sentences = []
        for s in sentences:
            s_stemmed = {
                stem_word(w)
                for w in s.translate(translator).lower().split()
                if len(w) > 1
            }
            score = sum(1 for t in q_tokens if t in s_stemmed or any(t in st for st in s_stemmed if len(st) > 3))
            scored_sentences.append((score, s))

        # Keep top percentage of sentences according to rate
        if scored_sentences:
            keep_count = max(1, int(len(scored_sentences) * max(0.2, min(1.0, rate))))
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            kept = [s for _, s in scored_sentences[:keep_count]]
            chunk_score = sum(sc for sc, _ in scored_sentences[:keep_count])
            compressed_chunk_text = " ".join(kept)
        else:
            chunk_score = 0
            compressed_chunk_text = chunk

        scored_chunks.append((chunk_score, compressed_chunk_text))

    # Reorder chunks placing highest scoring first (LongLLMLingua sort behavior)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    compressed_blocks = [text for _, text in scored_chunks]
    compressed_context = "\n\n".join(compressed_blocks)
    comp_tokens = estimate_token_count(compressed_context)

    ratio_val = total_orig_tokens / max(1, comp_tokens)
    ratio_str = f"{ratio_val:.1f}x"
    saving_pct = max(0.0, (1.0 - (comp_tokens / max(1, total_orig_tokens))) * 100.0)

    return {
        "compressed_context": compressed_context,
        "compressed_prompt": f"Context:\n{compressed_context}\n\nQuestion: {query}",
        "origin_tokens": total_orig_tokens,
        "compressed_tokens": comp_tokens,
        "ratio": ratio_str,
        "rate": rate,
        "saving": f"{saving_pct:.1f}%",
        "enabled": True,
        "applied": True,
        "fallback": True,
        "model_name": "heuristic_fallback",
    }


def compress_context_with_longllmlingua(
    context_chunks: List[str],
    query: str,
    instruction: str = "",
    rate: Optional[float] = None,
    target_token: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compresses and optimizes retrieved context chunks using the LongLLMLingua algorithm.

    Key capabilities of LongLLMLingua:
      1. Question-Aware Compression: Allocates context budget dynamically based on
         perplexity conditioning on the user question.
      2. Context Reordering: Rearranges context blocks (sort) to mitigate the
         'lost-in-the-middle' phenomenon where LLMs overlook facts in long mid-prompts.
      3. Dynamic Budget Allocation: Distributes token budget to high-density informative blocks.

    Returns a dictionary with compressed context, full prompt, token savings, and metadata.
    """
    start_time = time.time()
    uncompressed_context = "\n\n".join(context_chunks) if context_chunks else ""
    orig_tokens_est = estimate_token_count(uncompressed_context)

    # 1. Check if LongLLMLingua is enabled
    if not settings.longllmlingua.enabled:
        return {
            "compressed_context": uncompressed_context,
            "compressed_prompt": f"Context:\n{uncompressed_context}\n\nQuestion: {query}",
            "origin_tokens": orig_tokens_est,
            "compressed_tokens": orig_tokens_est,
            "ratio": "1.0x",
            "rate": 1.0,
            "saving": "0.0%",
            "compression_time_ms": 0.0,
            "enabled": False,
            "applied": False,
            "model_name": settings.longllmlingua.model_name,
        }

    # 2. Return empty structure if no context chunks
    if not context_chunks:
        return {
            "compressed_context": "",
            "compressed_prompt": f"Context:\n\nQuestion: {query}",
            "origin_tokens": 0,
            "compressed_tokens": 0,
            "ratio": "1.0x",
            "rate": 1.0,
            "saving": "0.0%",
            "compression_time_ms": 0.0,
            "enabled": True,
            "applied": False,
            "model_name": settings.longllmlingua.model_name,
        }

    # 3. Retrieve compressor
    compressor = get_compressor()
    effective_rate = rate if rate is not None else settings.longllmlingua.rate
    effective_target_token = (
        target_token if target_token is not None else settings.longllmlingua.target_token
    )

    if compressor is None:
        # Fall back to heuristic compression
        fallback_res = heuristic_compress_context(context_chunks, query, rate=effective_rate)
        fallback_res["compression_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return fallback_res

    # 4. Neural LongLLMLingua compression
    try:
        compress_kwargs: Dict[str, Any] = {
            "context": context_chunks,
            "question": query,
            "rank_method": settings.longllmlingua.rank_method,
            "condition_in_question": settings.longllmlingua.condition_in_question,
            "reorder_context": settings.longllmlingua.reorder_context,
            "dynamic_context_compression_ratio": settings.longllmlingua.dynamic_context_compression_ratio,
            "condition_compare": True,
        }
        if instruction:
            compress_kwargs["instruction"] = instruction
        if effective_target_token > 0:
            compress_kwargs["target_token"] = effective_target_token
        else:
            compress_kwargs["rate"] = effective_rate

        res = compressor.compress_prompt(**compress_kwargs)

        compressed_prompt = res.get("compressed_prompt", "")
        origin_tokens = res.get("origin_tokens", orig_tokens_est)
        compressed_tokens = res.get("compressed_tokens", orig_tokens_est)
        ratio_str = str(res.get("ratio", "1.0x"))
        saving_str = str(res.get("saving", "0.0%"))
        raw_rate = res.get("rate", effective_rate)
        if isinstance(raw_rate, (int, float)):
            actual_rate = float(raw_rate)
        elif isinstance(raw_rate, str):
            try:
                actual_rate = float(raw_rate.rstrip("%")) / 100.0 if "%" in raw_rate else float(raw_rate)
            except ValueError:
                actual_rate = float(effective_rate)
        else:
            actual_rate = float(effective_rate)

        # Extract context portion from compressed_prompt
        # In LongLLMLingua with condition_in_question="after", the prompt ends with the question
        clean_context = compressed_prompt
        if clean_context.endswith(query):
            clean_context = clean_context[: -len(query)].rstrip()
        if instruction and clean_context.startswith(instruction):
            clean_context = clean_context[len(instruction) :].lstrip()

        # If clean_context is empty for any reason, fall back to compressed_prompt
        if not clean_context.strip():
            clean_context = compressed_prompt

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "compressed_context": clean_context,
            "compressed_prompt": compressed_prompt,
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": ratio_str,
            "rate": actual_rate,
            "saving": saving_str,
            "compression_time_ms": elapsed_ms,
            "enabled": True,
            "applied": True,
            "fallback": False,
            "model_name": settings.longllmlingua.model_name,
        }

    except Exception as e:
        logger.warning(
            f"LongLLMLingua prompt compression failed: {e}. Falling back to heuristic compression."
        )
        fallback_res = heuristic_compress_context(context_chunks, query, rate=effective_rate)
        fallback_res["compression_time_ms"] = round((time.time() - start_time) * 1000, 2)
        fallback_res["error"] = str(e)
        return fallback_res
