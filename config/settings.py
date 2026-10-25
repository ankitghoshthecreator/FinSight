"""Centralised configuration for FinSight RAG.

All tuneable parameters are sourced from environment variables (or a .env file).
Instantiated once as a module-level singleton so imports stay simple:

    from config.settings import settings
    print(settings.embedding_model)
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """FinSight runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API ────────────────────────────────────────────────────────────────
    app_name: str = Field(default="FinSight RAG API", description="Human-readable service name")
    app_version: str = Field(default="2.0.0")
    debug: bool = Field(default=False, description="Enable FastAPI debug mode")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    # ── Security ───────────────────────────────────────────────────────────
    api_key_secret: str = Field(
        default="",
        description="Static API key required in X-API-Key header. Empty = auth disabled.",
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Set to specific URLs in production.",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Max requests per minute per IP (0 = disabled).",
    )

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model name.",
    )
    embedding_dim: int = Field(default=384, description="Output dimension of the embedding model.")

    # ── Generation ─────────────────────────────────────────────────────────
    llm_model: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="Local or API LLM model identifier.",
    )
    finsight_test_mode: bool = Field(
        default=True,
        description="Use heuristic extractor instead of real LLM. Auto-set in pytest.",
    )

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_max_tokens: int = Field(default=512, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=512)

    # ── Retrieval ──────────────────────────────────────────────────────────
    default_top_k: int = Field(default=5, ge=1, le=50)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    colbert_fusion_weight: float = Field(
        default=0.65,
        description="Weight given to ColBERT score vs initial hybrid score.",
    )

    # ── Vector Store ───────────────────────────────────────────────────────
    vector_store_backend: str = Field(
        default="faiss",
        description="'faiss' for local in-memory, 'qdrant' for server mode.",
    )
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="finsight_chunks")
    faiss_persist_dir: str = Field(
        default="./data/faiss_index",
        description="Directory to save/load FAISS index across restarts.",
    )

    # ── Ingestion ──────────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    redact_pii_default: bool = Field(
        default=True,
        description="Default PII redaction behaviour for text ingestion endpoint.",
    )

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Python logging level.")
    log_format: str = Field(default="json", description="'json' or 'text'.")


# Module-level singleton — import this everywhere
settings = Settings()
