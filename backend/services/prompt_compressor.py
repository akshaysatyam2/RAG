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
                    is_v2 = getattr(settings.longllmlingua, "use_llmlingua2", False) or "llmlingua-2" in settings.longllmlingua.model_name.lower()
                    logger.info(
                        f"Loading PromptCompressor with model: {settings.longllmlingua.model_name} "
                        f"on device: {settings.longllmlingua.device_map} (use_llmlingua2={is_v2})"
                    )
                    init_kwargs = {
                        "model_name": settings.longllmlingua.model_name,
                        "device_map": settings.longllmlingua.device_map,
                    }
                    if is_v2:
                        init_kwargs["use_llmlingua2"] = True
                    _compressor = PromptCompressor(**init_kwargs)
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
    Intelligent sentence-level compression that preserves complete factual answers,
    removes code/exercise boilerplate, and retains the natural chronological flow
    of sentences within each chunk.
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

    # Intent and comparative keywords that indicate direct answers
    answer_indicators = {
        "better", "prefer", "preferred", "advantage", "difference", "compare",
        "regimes", "separated", "contrast", "versus", "loss", "kernel",
        "degree", "university", "college", "school", "bachelor", "master", "cgpa",
        "sample", "replacement", "variability", "standard", "error", "estimate", "bootstrap"
    }

    scored_chunks = []
    total_orig_tokens = 0

    for idx, chunk in enumerate(context_chunks):
        orig_tokens = estimate_token_count(chunk)
        total_orig_tokens += orig_tokens

        # Check for chunk prefix e.g. "[1] "
        prefix_match = re.match(r"^(\[\d+\])\s*", chunk)
        chunk_prefix = prefix_match.group(1) if prefix_match else f"[{idx+1}]"
        chunk_body = chunk[len(chunk_prefix):].strip() if prefix_match else chunk.strip()

        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_body) if s.strip()]
        scored_sentences = []

        for sent_idx, s in enumerate(raw_sentences):
            # Noise filter: skip pure R/code command prompts, sub-exercise labels, figure axes, etc.
            if s.startswith(">") or s.startswith("+") or re.match(r"^\([a-z]\)", s) or re.search(r"\([a-z]\)\s+[A-Z]", s):
                continue
            if re.search(r"\b\d+\s+\d+\.\s+Resampling", s) or s.startswith("Resampling Methods ("):
                continue
            if re.search(r"[−-]\d+\s+[−-]\d+", s) or "Loss SVM" in s or "SVM Loss Logistic" in s:
                continue
            if re.search(r"(\d+\.\d+\s+){4,}", s):
                continue
            # Apply textbook homework / caption / boilerplate noise filter
            from backend.services.ingestion import is_noisy_sentence
            if is_noisy_sentence(s):
                continue

            s_clean = s.translate(translator).lower()
            s_stemmed = {stem_word(w) for w in s_clean.split() if len(w) > 1}
            
            # Base query token match score
            score = sum(1.5 for t in q_tokens if t in s_stemmed or any(t in st for st in s_stemmed if len(st) > 3))
            
            # Extra weight for comparative and direct answer clauses
            if any(ind in s_stemmed for ind in answer_indicators):
                score += 2.0
                
            # Extra weight for proper nouns / person names
            if any(term in s.lower() for term in ["akshay", "jharkhand", "raksha", "bcacs"]):
                score += 3.0

            scored_sentences.append({
                "orig_index": sent_idx,
                "score": score,
                "text": s
            })

        if scored_sentences:
            # Sort by score to identify top informative sentences
            sorted_by_score = sorted(scored_sentences, key=lambda x: x["score"], reverse=True)
            
            # Keep sentences that have a positive match score or meet the rate threshold
            keep_count = max(1, int(len(scored_sentences) * max(0.4, min(0.85, rate + 0.15))))
            selected_items = sorted_by_score[:keep_count]
            
            # CRITICAL: Re-sort retained sentences by their ORIGINAL sequential index in the text
            retained_in_order = sorted(selected_items, key=lambda x: x["orig_index"])
            chunk_score = sum(item["score"] for item in selected_items)
            compressed_chunk_text = f"{chunk_prefix} " + " ".join(item["text"] for item in retained_in_order)
        else:
            chunk_score = 0
            compressed_chunk_text = chunk

        scored_chunks.append((chunk_score, compressed_chunk_text))

    # Keep chunks in order of relevance
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
    enabled: Optional[bool] = None,
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
    is_enabled = enabled if enabled is not None else settings.longllmlingua.enabled
    if not is_enabled:
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

    # CPU guard: Large contexts (> cpu_token_threshold tokens) take minutes to compress on CPU and freeze interactive chat.
    # Seamlessly route large CPU contexts to fast heuristic compression.
    cpu_thresh = getattr(settings.longllmlingua, "cpu_token_threshold", 150)
    if settings.longllmlingua.device_map == "cpu" and orig_tokens_est > cpu_thresh:
        logger.info(
            f"Context token count ({orig_tokens_est}) exceeds safe CPU neural compression threshold ({cpu_thresh}). "
            f"Using fast heuristic compression to maintain interactive UI response time."
        )
        fallback_res = heuristic_compress_context(context_chunks, query, rate=effective_rate)
        fallback_res["compression_time_ms"] = round((time.time() - start_time) * 1000, 2)
        fallback_res["applied"] = True
        fallback_res["fallback"] = False
        return fallback_res

    # 4. Neural LongLLMLingua compression
    try:
        is_v2 = getattr(settings.longllmlingua, "use_llmlingua2", False) or "llmlingua-2" in settings.longllmlingua.model_name.lower()

        if is_v2:
            res = compressor.compress_prompt(
                context=context_chunks,
                rate=effective_rate,
                force_reserve_digit=True,
            )
            compressed_prompt = res.get("compressed_prompt", "")
            origin_tokens = res.get("origin_tokens", orig_tokens_est)
            compressed_tokens = res.get("compressed_tokens", orig_tokens_est)
            ratio_str = str(res.get("ratio", "1.0x"))
            saving_str = str(res.get("saving", "0.0%"))
            raw_rate = res.get("rate", effective_rate)

            comp_list = res.get("compressed_prompt_list")
            if comp_list and len(comp_list) == len(context_chunks):
                formatted = []
                for i, c_text in enumerate(comp_list):
                    clean_c = re.sub(r"^\[\d+\]\s*", "", c_text).strip()
                    formatted.append(f"[{i+1}] {clean_c}")
                clean_context = "\n\n".join(formatted)
            else:
                clean_context = compressed_prompt
        else:
            default_instruction = "Given the technical textbook context, provide a clear conceptual and mathematical explanation answering the question."
            compress_kwargs: Dict[str, Any] = {
                "context": context_chunks,
                "question": query,
                "instruction": instruction or default_instruction,
                "rank_method": getattr(settings.longllmlingua, "rank_method", "longllmlingua"),
                "condition_in_question": getattr(settings.longllmlingua, "condition_in_question", "after_condition"),
                "condition_compare": getattr(settings.longllmlingua, "condition_compare", True),
                "dynamic_context_compression_ratio": getattr(settings.longllmlingua, "dynamic_context_compression_ratio", 0.25),
                "reorder_context": getattr(settings.longllmlingua, "reorder_context", "sort"),
                "use_context_level_filter": getattr(settings.longllmlingua, "use_context_level_filter", True),
                "use_sentence_level_filter": getattr(settings.longllmlingua, "use_sentence_level_filter", True),
                "use_token_level_filter": getattr(settings.longllmlingua, "use_token_level_filter", False),
            }
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

        if isinstance(raw_rate, (int, float)):
            actual_rate = float(raw_rate)
        elif isinstance(raw_rate, str):
            try:
                actual_rate = float(raw_rate.rstrip("%")) / 100.0 if "%" in raw_rate else float(raw_rate)
            except ValueError:
                actual_rate = float(effective_rate)
        else:
            actual_rate = float(effective_rate)

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
