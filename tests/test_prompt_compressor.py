import pytest
from unittest.mock import MagicMock, patch

from backend.config import Settings
from backend.services.prompt_compressor import (
    compress_context_with_longllmlingua,
    heuristic_compress_context,
    estimate_token_count,
    reset_compressor,
    get_compressor,
)


@pytest.fixture(autouse=True)
def clean_compressor():
    reset_compressor()
    yield
    reset_compressor()


def test_estimate_token_count():
    assert estimate_token_count("") == 0
    assert estimate_token_count("Hello world") == 2
    assert estimate_token_count("The capital of France is Paris.") >= 6


def test_compress_disabled(monkeypatch):
    monkeypatch.setattr("backend.config.settings.longllmlingua.enabled", False)
    chunks = ["[1] First chunk text.", "[2] Second chunk text."]
    res = compress_context_with_longllmlingua(chunks, "What is the text?")

    assert res["enabled"] is False
    assert res["applied"] is False
    assert "[1] First chunk text." in res["compressed_context"]
    assert "[2] Second chunk text." in res["compressed_context"]


def test_compress_empty_chunks():
    res = compress_context_with_longllmlingua([], "What is the capital?")
    assert res["compressed_context"] == ""
    assert res["applied"] is False


def test_heuristic_compression():
    chunks = [
        "Paragraph 1: Bananas are yellow fruits with potassium. They grow in tropical regions.",
        "Paragraph 2: Paris is the capital of France. The Eiffel Tower is in Paris. It attracts millions of visitors.",
        "Paragraph 3: Kangaroos are marsupials native to Australia. They have large tails.",
    ]
    query = "Tell me about Paris and France."

    res = heuristic_compress_context(chunks, query, rate=0.5)
    assert res["applied"] is True
    assert res["fallback"] is True
    # Paris chunk should be ranked first due to query overlap
    assert "Paris" in res["compressed_context"]
    assert res["compressed_tokens"] <= res["origin_tokens"]


def test_compress_with_mock_compressor(monkeypatch):
    monkeypatch.setattr("backend.config.settings.longllmlingua.enabled", True)
    monkeypatch.setattr("backend.config.settings.longllmlingua.rate", 0.6)

    mock_compressor = MagicMock()
    mock_compressor.compress_prompt.return_value = {
        "compressed_prompt": "[1] Compressed Paris chunk text.\n\nWhat is the capital?",
        "origin_tokens": 100,
        "compressed_tokens": 50,
        "ratio": "2.0x",
        "rate": 0.5,
        "saving": "50.0%",
    }

    with patch("backend.services.prompt_compressor.get_compressor", return_value=mock_compressor):
        chunks = ["[1] Full chunk about Paris.", "[2] Full chunk about London."]
        res = compress_context_with_longllmlingua(
            chunks, "What is the capital?", instruction="Answer based on context."
        )

        assert res["applied"] is True
        assert res["fallback"] is False
        assert res["compressed_tokens"] == 50
        assert res["ratio"] == "2.0x"
        assert "[1] Compressed Paris chunk text." in res["compressed_context"]
        mock_compressor.compress_prompt.assert_called_once()
        kwargs = mock_compressor.compress_prompt.call_args[1]
        assert kwargs["rank_method"] == "longllmlingua"
        assert kwargs["reorder_context"] == "sort"
        assert kwargs["condition_compare"] is True


def test_compress_fallback_on_error(monkeypatch):
    monkeypatch.setattr("backend.config.settings.longllmlingua.enabled", True)

    mock_compressor = MagicMock()
    mock_compressor.compress_prompt.side_effect = RuntimeError("CUDA OOM or model execution failure")

    with patch("backend.services.prompt_compressor.get_compressor", return_value=mock_compressor):
        chunks = ["[1] Important info about Paris."]
        res = compress_context_with_longllmlingua(chunks, "Paris query")

        assert res["applied"] is True
        assert res["fallback"] is True
        assert "error" in res
        assert "Paris" in res["compressed_context"]


def test_real_longllmlingua_compression_with_gpt2(monkeypatch):
    """Integration test using the locally cached open-source gpt2 model."""
    try:
        from llmlingua import PromptCompressor
    except ImportError:
        pytest.skip("llmlingua not installed")

    monkeypatch.setattr("backend.config.settings.longllmlingua.enabled", True)
    monkeypatch.setattr("backend.config.settings.longllmlingua.model_name", "gpt2")
    monkeypatch.setattr("backend.config.settings.longllmlingua.device_map", "cpu")
    monkeypatch.setattr("backend.config.settings.longllmlingua.rate", 0.6)

    chunks = [
        "[1] Apples are nutritious red or green fruits rich in fiber and vitamin C.",
        "[2] Paris is the capital and largest city of France, located on the river Seine.",
        "[3] Bananas are elongated yellow fruits that contain potassium and carbohydrates."
    ]
    query = "What is the capital of France?"

    res = compress_context_with_longllmlingua(chunks, query)
    assert res["applied"] is True
    assert res["fallback"] is False
    assert "Paris" in res["compressed_context"]
    assert res["compressed_tokens"] > 0
    assert res["model_name"] == "gpt2"

