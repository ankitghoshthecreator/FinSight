"""Tests for centralised Settings configuration."""

import os
import pytest
from config.settings import Settings


def test_settings_defaults():
    s = Settings()
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.chunk_max_tokens == 512
    assert s.chunk_overlap_tokens == 64
    assert s.default_top_k == 5
    assert s.bm25_weight == 0.5
    assert s.dense_weight == 0.5
    assert s.vector_store_backend == "faiss"
    assert s.api_port == 8000


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "256")
    monkeypatch.setenv("DEFAULT_TOP_K", "10")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.chunk_max_tokens == 256
    assert s.default_top_k == 10
    assert s.log_level == "DEBUG"


def test_settings_api_key_empty_by_default():
    s = Settings()
    assert s.api_key_secret == ""


def test_settings_cors_origins_default():
    s = Settings()
    assert "*" in s.cors_origins


def test_settings_colbert_fusion_weight():
    s = Settings()
    assert 0.0 <= s.colbert_fusion_weight <= 1.0
