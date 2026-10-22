import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


@dataclass
class LLMConfig:
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "not-needed"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama3"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))


@dataclass
class EmbeddingConfig:
    model_name: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    cross_encoder_model: str = field(default_factory=lambda: _env("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))


@dataclass
class QdrantConfig:
    host: str = field(default_factory=lambda: _env("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("QDRANT_PORT", 6333))
    collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "rag_chunks"))


@dataclass
class Neo4jConfig:
    uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "password"))


@dataclass
class RedisConfig:
    url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))


@dataclass
class RetrievalConfig:
    similarity_threshold: float = field(default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.25))
    top_k_retrieval: int = field(default_factory=lambda: _env_int("TOP_K_RETRIEVAL", 20))
    top_k_rerank: int = field(default_factory=lambda: _env_int("TOP_K_RERANK", 5))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))


@dataclass
class IngestionConfig:
    upload_dir: str = field(default_factory=lambda: _env("UPLOAD_DIR", "./uploads"))
    max_upload_size_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_SIZE_MB", 100))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))


@dataclass
class LongLLMLinguaConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_LONGLLMLINGUA", True))
    model_name: str = field(
        default_factory=lambda: _env(
            "LONGLLMLINGUA_MODEL_NAME",
            "gpt2"
        )
    )
    use_llmlingua2: bool = field(default_factory=lambda: _env_bool("USE_LLMLINGUA2", False))
    device_map: str = field(default_factory=lambda: _env("LONGLLMLINGUA_DEVICE", "cpu"))
    rate: float = field(default_factory=lambda: _env_float("LONGLLMLINGUA_RATE", 0.65))
    target_token: int = field(default_factory=lambda: _env_int("LONGLLMLINGUA_TARGET_TOKEN", -1))
    cpu_token_threshold: int = field(default_factory=lambda: _env_int("LONGLLMLINGUA_CPU_THRESHOLD", 150))
    rank_method: str = field(default_factory=lambda: _env("LONGLLMLINGUA_RANK_METHOD", "longllmlingua"))
    dynamic_context_compression_ratio: float = field(
        default_factory=lambda: _env_float("LONGLLMLINGUA_DYNAMIC_RATIO", 0.25)
    )
    reorder_context: str = field(default_factory=lambda: _env("LONGLLMLINGUA_REORDER_CONTEXT", "sort"))
    condition_in_question: str = field(default_factory=lambda: _env("LONGLLMLINGUA_CONDITION_IN_QUESTION", "after_condition"))
    condition_compare: bool = field(default_factory=lambda: _env_bool("LONGLLMLINGUA_CONDITION_COMPARE", True))
    use_context_level_filter: bool = field(default_factory=lambda: _env_bool("LONGLLMLINGUA_CONTEXT_FILTER", True))
    use_sentence_level_filter: bool = field(default_factory=lambda: _env_bool("LONGLLMLINGUA_SENTENCE_FILTER", True))
    use_token_level_filter: bool = field(default_factory=lambda: _env_bool("LONGLLMLINGUA_TOKEN_FILTER", False))


@dataclass
class DatabaseConfig:
    sqlite_path: str = field(default_factory=lambda: _env("SQLITE_PATH", "./data/metadata.db"))


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    longllmlingua: LongLLMLinguaConfig = field(default_factory=LongLLMLinguaConfig)

    def ensure_directories(self):
        Path(self.ingestion.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.database.sqlite_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
