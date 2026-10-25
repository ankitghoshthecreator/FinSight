"""Tests for FAISS disk persistence: save/load, try_load, delete_by_doc_id, reset."""

import numpy as np
import pytest
from ingestion.models import DocumentChunk, DocumentMetadata
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore


def _make_chunk(doc_id: str, chunk_id: str, content: str) -> DocumentChunk:
    meta = DocumentMetadata(doc_id=doc_id, filename=f"{doc_id}.pdf", page_number=1)
    return DocumentChunk(chunk_id=chunk_id, content=content, metadata=meta)


def test_vector_store_auto_load_on_init(tmp_path):
    """Index saved to disk is reloaded automatically when persist_dir is supplied."""
    embedder = Embedder(dimension=384)
    store1 = VectorStore(dimension=384)
    chunk = _make_chunk("d1", "c0", "Basel III CET1 capital ratio.")
    store1.add_chunks([chunk], embedder.embed_texts([chunk.content]))
    store1.save(str(tmp_path))

    # New instance pointing at the same directory → should auto-load
    store2 = VectorStore(dimension=384, persist_dir=str(tmp_path))
    assert store2.count() == 1
    assert store2.chunks[0].chunk_id == "c0"


def test_try_load_returns_false_when_missing(tmp_path):
    store = VectorStore(dimension=384)
    result = store.try_load(str(tmp_path / "nonexistent"))
    assert result is False
    assert store.count() == 0


def test_delete_by_doc_id():
    embedder = Embedder(dimension=384)
    store = VectorStore(dimension=384)

    c1 = _make_chunk("doc_a", "ca0", "Goldman Sachs SLR 5.8%.")
    c2 = _make_chunk("doc_b", "cb0", "JPMC CET1 ratio 14.5%.")
    store.add_chunks([c1, c2], embedder.embed_texts([c1.content, c2.content]))

    removed = store.delete_by_doc_id("doc_a")
    assert removed == 1
    assert store.count() == 1
    assert store.chunks[0].metadata.doc_id == "doc_b"


def test_delete_by_doc_id_not_found():
    store = VectorStore(dimension=384)
    removed = store.delete_by_doc_id("nonexistent_doc")
    assert removed == 0


def test_reset_clears_index():
    embedder = Embedder(dimension=384)
    store = VectorStore(dimension=384)
    chunk = _make_chunk("d1", "c0", "Regulatory capital framework.")
    store.add_chunks([chunk], embedder.embed_texts([chunk.content]))
    assert store.count() == 1

    store.reset()
    assert store.count() == 0
    assert store.chunks == []


def test_get_doc_ids():
    embedder = Embedder(dimension=384)
    store = VectorStore(dimension=384)
    c1 = _make_chunk("doc_x", "cx0", "SOFR benchmark rate.")
    c2 = _make_chunk("doc_x", "cx1", "SOFR transition from LIBOR.")
    c3 = _make_chunk("doc_y", "cy0", "NSFR liquidity requirements.")
    store.add_chunks([c1, c2, c3], embedder.embed_texts([c1.content, c2.content, c3.content]))

    doc_ids = store.get_doc_ids()
    assert set(doc_ids) == {"doc_x", "doc_y"}
