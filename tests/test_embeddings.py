import pytest
from unittest.mock import patch, MagicMock
import numpy as np

@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    with patch("backend.services.embeddings.SentenceTransformer") as mock_st:
        instance = mock_st.return_value
        
        def encode_mock(sentences, *args, **kwargs):
            if isinstance(sentences, str):
                return np.array([0.1] * 384)
            else:
                return np.array([[0.1] * 384 for _ in sentences])
                
        instance.encode.side_effect = encode_mock
        instance.get_sentence_embedding_dimension.return_value = 384
        yield mock_st

from backend.services.embeddings import (
    get_dense_embedding,
    get_dense_embeddings_batch,
    tokenize_for_sparse,
    compute_sparse_vector
)

def test_get_dense_embedding():
    embedding = get_dense_embedding("test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 384

def test_get_dense_embeddings_batch():
    embeddings = get_dense_embeddings_batch(["test 1", "test 2"])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384

def test_tokenize_for_sparse():
    tokens = tokenize_for_sparse("Hello World, hello!")
    assert tokens == ["hello", "world", "hello"]

def test_compute_sparse_vector():
    vocab_corpus = [["hello", "world"], ["hello", "test"]]
    vector = compute_sparse_vector("hello world", vocab_corpus)
    assert "indices" in vector
    assert "values" in vector
    assert isinstance(vector["indices"], list)
    assert isinstance(vector["values"], list)

def test_empty_input_handling():
    # Model encode of empty string returns 384 dimension
    assert len(get_dense_embedding("")) == 384
    assert tokenize_for_sparse("") == []
    assert compute_sparse_vector("", []) == {"indices": [], "values": []}

