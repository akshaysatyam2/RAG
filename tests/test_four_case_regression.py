import pytest
import re
from unittest.mock import patch

from backend.services.retrieval import normalize_and_correct_query
from backend.services.reranker import rerank, filter_by_threshold
from backend.services.prompt_compressor import compress_context_with_longllmlingua
from backend.services.llm import generate_completion

REPRESENTATIVE_TEXTBOOK_CHUNKS = [
    {
        "id": "chunk_142_def",
        "payload": {
            "document_name": "ISLR_First_Printing.pdf",
            "page_number": 142,
            "text": "Rather than modeling this response Y directly, logistic regression models the probability that Y belongs to a particular category. In the Default data, logistic regression models the probability of default."
        }
    },
    {
        "id": "chunk_147_math",
        "payload": {
            "document_name": "ISLR_First_Printing.pdf",
            "page_number": 147,
            "text": "The left-hand side is called the log-odds or logit. We see that the logistic regression model (4.2) has a logit that is linear in X. Increasing X by one unit changes the log odds by beta_1, or equivalently it multiplies the odds by e^beta_1."
        }
    },
    {
        "id": "chunk_146_table",
        "payload": {
            "document_name": "ISLR_First_Printing.pdf",
            "page_number": 146,
            "text": "Table 4.1 shows the coefficient estimates of the logistic regression model that predicts the probability of default using balance."
        }
    },
    {
        "id": "chunk_146_caption",
        "payload": {
            "document_name": "ISLR_First_Printing.pdf",
            "page_number": 146,
            "text": "For the Default data, estimated coefficients of the logistic regression model that predicts the probability of default using balance."
        }
    },
    {
        "id": "chunk_hw_exercise",
        "payload": {
            "document_name": "ISLR_First_Printing.pdf",
            "page_number": 150,
            "text": "Consider the training RSS as a function of lambda for linear regression problems in chapter 3."
        }
    }
]


def compute_pairwise_max_overlap(bullets: list[str]) -> float:
    """Computes the maximum Jaccard token overlap between any pair of bullets."""
    token_sets = [{w.lower().strip(".,;()") for w in b.split() if len(w) > 2} for b in bullets]
    max_overlap = 0.0
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            s1, s2 = token_sets[i], token_sets[j]
            if s1 and s2:
                overlap = len(s1 & s2) / min(len(s1), len(s2))
                if overlap > max_overlap:
                    max_overlap = overlap
    return max_overlap


@pytest.mark.asyncio
@pytest.mark.parametrize("query, expect_correction", [
    ("tell me about logistic regression", False),
    ("tell me about logestic regression", True),
    ("tell me about logesstic regression", True),
    ("tell me about legestic regression", True),
])
async def test_four_case_regression_end_to_end(query: str, expect_correction: bool):
    """
    4-Case Regression Integration Test:
    1. 'tell me about logistic regression' (clean baseline)
    2. 'tell me about logestic regression' (edit-distance 1 typo)
    3. 'tell me about logesstic regression' (edit-distance 2 insertion typo)
    4. 'tell me about legestic regression' (edit-distance 2 substitution typo)

    Asserts:
    1. corrected_query.lower() == 'tell me about logistic regression'
    2. Pairwise token overlap between all returned summary bullets is < 0.55
    3. The first two bullets contain core domain terms ('probability', 'classification', 'logit', or 'log-odds')
       and zero homework prompts ('Consider the training RSS').
    """
    # 1. Query Normalization & Spell-Checking
    corrected_q, was_corrected = normalize_and_correct_query(query)
    effective_query = corrected_q if was_corrected else query

    # Assertion 1: corrected query resolves to 'tell me about logistic regression'
    assert effective_query.lower() == "tell me about logistic regression"
    if expect_correction:
        assert was_corrected is True
    else:
        assert was_corrected is False

    # 2. Retrieval, Rerank & Context Compression
    with patch("backend.services.retrieval.full_retrieval_pipeline", return_value=(REPRESENTATIVE_TEXTBOOK_CHUNKS, [])):
        from backend.services.retrieval import full_retrieval_pipeline
        hybrid_chunks, _ = await full_retrieval_pipeline(effective_query, top_k=15)
        expanded = [{"id": c.get("id"), "payload": c.get("payload", {})} for c in hybrid_chunks]
        reranked = rerank(effective_query, expanded, top_k=5)
        filtered = filter_by_threshold(reranked, 0.25) or reranked[:3]

        raw_context_texts = [f"[{i+1}] {c.get('payload', {}).get('text', '')}" for i, c in enumerate(filtered)]
        comp_res = compress_context_with_longllmlingua(raw_context_texts, effective_query)
        compressed_ctx = comp_res.get("compressed_context", "")

        # 3. LLM Completion / Synthesis
        system_prompt = (
            "You are an expert AI assistant. Synthesize a concise, well-structured executive summary "
            "answering the user's question directly based ONLY on the provided context. Use bullet points and clear formatting."
        )
        user_prompt = f"Context:\n{compressed_ctx}\n\nQuestion: {effective_query}"
        answer = await generate_completion(system_prompt, user_prompt)

        # Extract genuine bullet points (ignoring bold markdown headers)
        bullet_lines = [
            re.sub(r"^[•\-\*]\s*", "", line).strip()
            for line in answer.splitlines()
            if (line.strip().startswith(("•", "- ", "* ")) or line.strip().startswith("•"))
            and not line.strip().startswith("**")
            and len(line.strip()) > 10
        ]
        assert len(bullet_lines) >= 2, f"Expected at least 2 bullets, got: {bullet_lines}"

        # Assertion 2: Pairwise token overlap between all returned summary bullets is < 0.55
        max_overlap = compute_pairwise_max_overlap(bullet_lines)
        assert max_overlap < 0.55, f"Pairwise overlap {max_overlap:.2f} exceeded threshold 0.55. Bullets: {bullet_lines}"

        # Assertion 3: The first two bullets contain core domain terms and zero homework prompts
        first_two_text = " ".join(bullet_lines[:2]).lower()
        core_terms = ["probability", "classification", "logit", "log-odds"]
        assert any(term in first_two_text for term in core_terms), (
            f"First two bullets missing core domain terms ({core_terms}): {bullet_lines[:2]}"
        )
        assert "consider the training rss" not in first_two_text
        assert "using (3.4), argue" not in first_two_text
