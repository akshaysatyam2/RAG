import asyncio
import json
import time
from backend.config import settings
from backend.services.retrieval import full_retrieval_pipeline
from backend.services.reranker import rerank, filter_by_threshold
from backend.services.prompt_compressor import (
    compress_context_with_longllmlingua,
    estimate_token_count,
)
from backend.services.llm import generate_completion


async def evaluate_query(query: str):
    print(f"\n{'='*80}")
    print(f"QUERY: '{query}'")
    print(f"{'='*80}")

    # 1. Retrieve
    hybrid_chunks, graph_context = await full_retrieval_pipeline(query, top_k=20)
    expanded_chunks = [{"id": c.get("id"), "payload": c.get("payload", {})} for c in hybrid_chunks]

    # 2. Rerank
    reranked = rerank(query, expanded_chunks, top_k=5)
    filtered = filter_by_threshold(reranked, settings.retrieval.similarity_threshold)
    if not filtered and reranked:
        filtered = reranked[:3]

    raw_context_texts = [f"[{i+1}] {c.get('payload', {}).get('text', '')}" for i, c in enumerate(filtered)]
    raw_context_str = "\n\n".join(raw_context_texts)
    raw_tokens = estimate_token_count(raw_context_str)

    system_prompt = (
        "You are an expert AI assistant. Synthesize a concise, well-structured executive summary "
        "answering the user's question directly based ONLY on the provided context. Use bullet points and clear formatting. "
        "Do NOT dump raw unformatted text or verbatim document chunks. "
        "If the context does not contain sufficient information, explicitly state that you cannot answer."
    )

    # -------------------------------------------------------------
    # A) BASELINE (Without LongLLMLingua)
    # -------------------------------------------------------------
    baseline_user_prompt = f"Context:\n{raw_context_str}\n\nQuestion: {query}"
    t0 = time.time()
    baseline_answer = await generate_completion(system_prompt, baseline_user_prompt)
    baseline_time_ms = round((time.time() - t0) * 1000, 2)

    # -------------------------------------------------------------
    # B) WITH LongLLMLingua
    # -------------------------------------------------------------
    t1 = time.time()
    comp_res = compress_context_with_longllmlingua(
        context_chunks=raw_context_texts,
        query=query,
        instruction="Synthesize a concise executive summary answering the question directly based only on the context."
    )
    compressed_context = comp_res.get("compressed_context", raw_context_str)
    comp_tokens = comp_res.get("compressed_tokens", raw_tokens)
    ratio = comp_res.get("ratio", "1.0x")
    savings = comp_res.get("saving", "0.0%")
    comp_time_ms = comp_res.get("compression_time_ms", 0.0)

    lingua_user_prompt = f"Context:\n{compressed_context}\n\nQuestion: {query}"
    t2 = time.time()
    lingua_answer = await generate_completion(system_prompt, lingua_user_prompt)
    lingua_gen_time_ms = round((time.time() - t2) * 1000, 2)

    return {
        "query": query,
        "raw_tokens": raw_tokens,
        "compressed_tokens": comp_tokens,
        "ratio": ratio,
        "savings": savings,
        "compression_time_ms": comp_time_ms,
        "baseline": {
            "tokens": raw_tokens,
            "answer": baseline_answer,
            "latency_ms": baseline_time_ms,
            "context_sample": raw_context_str[:300] + "..."
        },
        "longllmlingua": {
            "tokens": comp_tokens,
            "answer": lingua_answer,
            "latency_ms": lingua_gen_time_ms,
            "context_sample": compressed_context[:300] + "..."
        }
    }


async def main():
    queries = [
        "What is logistic regression and when is it preferred over SVM?",
        "What university or college did Akshay graduate from?",
        "Explain the bootstrap resampling method."
    ]

    results = []
    for q in queries:
        res = await evaluate_query(q)
        results.append(res)

        print(f"\n📊 METRICS COMPARISON:")
        print(f"  • Raw Context Tokens      : {res['raw_tokens']}")
        print(f"  • Compressed Tokens       : {res['compressed_tokens']}")
        print(f"  • Compression Ratio       : {res['ratio']} ({res['savings']} token reduction)")
        print(f"  • Compression Latency     : {res['compression_time_ms']} ms")
        print(f"\n--- [BASELINE ANSWER (NO LLMLINGUA)] ---")
        print(res["baseline"]["answer"])
        print(f"\n--- [WITH LONGLLMLINGUA ANSWER] ---")
        print(res["longllmlingua"]["answer"])
        print("-" * 80)

    # Save to disk for inspection
    with open("scripts/comparison_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved comparison report to scripts/comparison_report.json")


if __name__ == "__main__":
    asyncio.run(main())
