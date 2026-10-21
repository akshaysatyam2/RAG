import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:5000/api/chat"

def query_chat(query: str, enable_compression: bool):
    payload = json.dumps({
        "query": query,
        "top_k": 3,
        "enable_compression": enable_compression
    }).encode("utf-8")
    
    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = round((time.time() - t0) * 1000, 2)
    return data, elapsed

def main():
    queries = [
        "What is logistic regression and when is it preferred over SVM?",
        "What university, degree, or college did Akshay attend?",
        "Explain the bootstrap resampling method."
    ]

    all_results = []

    for q in queries:
        print(f"\n{'='*80}")
        print(f"QUERY: '{q}'")
        print(f"{'='*80}")

        print("--> Running Baseline (NO LongLLMLingua)...")
        base_data, base_time = query_chat(q, enable_compression=False)

        print("--> Running WITH LongLLMLingua...")
        lingua_data, lingua_time = query_chat(q, enable_compression=True)

        res = {
            "query": q,
            "baseline": {
                "answer": base_data.get("answer"),
                "applied": base_data.get("retrieval_metadata", {}).get("longllmlingua_applied", False),
                "original_tokens": base_data.get("retrieval_metadata", {}).get("original_tokens", 0),
                "compressed_tokens": base_data.get("retrieval_metadata", {}).get("compressed_tokens", 0),
                "sources_count": len(base_data.get("sources", [])),
                "latency_ms": base_data.get("processing_time_ms")
            },
            "longllmlingua": {
                "answer": lingua_data.get("answer"),
                "applied": lingua_data.get("retrieval_metadata", {}).get("longllmlingua_applied", False),
                "compression_ratio": lingua_data.get("compression_ratio"),
                "original_tokens": lingua_data.get("retrieval_metadata", {}).get("original_tokens", 0),
                "compressed_tokens": lingua_data.get("retrieval_metadata", {}).get("compressed_tokens", 0),
                "savings": lingua_data.get("retrieval_metadata", {}).get("compression_savings"),
                "compression_time_ms": lingua_data.get("retrieval_metadata", {}).get("compression_time_ms"),
                "sources_count": len(lingua_data.get("sources", [])),
                "latency_ms": lingua_data.get("processing_time_ms")
            }
        }
        all_results.append(res)

        print(f"\n📊 COMPARISON METRICS:")
        print(f"  • Baseline Context Tokens : {res['baseline']['original_tokens']}")
        print(f"  • Compressed Tokens       : {res['longllmlingua']['compressed_tokens']}")
        print(f"  • Token Reduction / Ratio : {res['longllmlingua']['savings']} ({res['longllmlingua']['compression_ratio']})")
        print(f"  • Latency (Baseline vs LLMLingua): {res['baseline']['latency_ms']} ms vs {res['longllmlingua']['latency_ms']} ms")

        print(f"\n--- [BASELINE ANSWER (WITHOUT LLMLINGUA)] ---")
        print(res["baseline"]["answer"])

        print(f"\n--- [WITH LONGLLMLINGUA ANSWER] ---")
        print(res["longllmlingua"]["answer"])
        print("-" * 80)

    with open("scripts/comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults successfully saved to scripts/comparison_results.json")

if __name__ == "__main__":
    main()
