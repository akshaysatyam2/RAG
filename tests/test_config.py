import os
import pytest
from unittest.mock import patch
from backend.config import Settings, DatabaseConfig, LLMConfig


def test_default_values():
    settings = Settings()
    assert settings.database.sqlite_path == "./data/metadata.db"
    assert settings.llm.model == "llama3"
    assert settings.llm.temperature == 0.1


def test_env_overrides():
    with patch.dict(os.environ, {
        "SQLITE_PATH": "/custom/path.db",
        "LLM_MODEL": "gpt-4",
        "LLM_TEMPERATURE": "0.7"
    }):
        settings = Settings()
        assert settings.database.sqlite_path == "/custom/path.db"
        assert settings.llm.model == "gpt-4"
        assert settings.llm.temperature == 0.7


def test_ensure_directories(tmp_path):
    # Setup temporary settings
    with patch.dict(os.environ, {
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "SQLITE_PATH": str(tmp_path / "db" / "metadata.db")
    }):
        settings = Settings()
        settings.ensure_directories()
        
        assert (tmp_path / "uploads").exists()
        assert (tmp_path / "db").exists()


def test_config_mutability():
    config = LLMConfig()
    config.model = "new-model"
    assert config.model == "new-model"


def test_longllmlingua_config():
    settings = Settings()
    assert settings.longllmlingua.enabled is True
    assert settings.longllmlingua.model_name == "gpt2"
    assert settings.longllmlingua.device_map == "cpu"
    assert settings.longllmlingua.rate == 0.55
    assert settings.longllmlingua.reorder_context == "sort"

    with patch.dict(os.environ, {
        "ENABLE_LONGLLMLINGUA": "false",
        "LONGLLMLINGUA_MODEL_NAME": "Qwen/Qwen2.5-0.5B",
        "LONGLLMLINGUA_DEVICE": "cuda",
        "LONGLLMLINGUA_RATE": "0.4",
        "LONGLLMLINGUA_DYNAMIC_RATIO": "0.25",
        "LONGLLMLINGUA_REORDER_CONTEXT": "original"
    }):
        custom = Settings()
        assert custom.longllmlingua.enabled is False
        assert custom.longllmlingua.model_name == "Qwen/Qwen2.5-0.5B"
        assert custom.longllmlingua.device_map == "cuda"
        assert custom.longllmlingua.rate == 0.4
        assert custom.longllmlingua.dynamic_context_compression_ratio == 0.25
        assert custom.longllmlingua.reorder_context == "original"


