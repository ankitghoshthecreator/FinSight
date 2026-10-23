"""Unit tests for retrieval: embedder, vector store, and hybrid search."""

import numpy as np
import pytest
from ingestion.models import DocumentChunk, DocumentMetadata
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridRetriever


def test_embedder_dimension():
    embedder = Embedder(dimension=384)
    texts = ["Regulation T margin requirement", "CET1 capital ratio"]
    embs = embedder.embed_texts(texts)
    assert embs.shape == (2, 384)
    # Norm check
    norm0 = np.linalg.norm(embs[0])
    assert pytest.approx(norm0, rel=1e-3) == 1.0


def test_embedder_empty():
    embedder = Embedder(dimension=384)
    embs = embedder.embed_texts([])
    assert embs.shape == (0, 384)


def test_vector_store_add_and_search():
    store = VectorStore(dimension=384)
    embedder = Embedder(dimension=384)

    meta1 = DocumentMetadata(doc_id="d1", filename="sec_10k.pdf", page_number=1, chunk_index=0)
    chunk1 = DocumentChunk(chunk_id="d1_c0", content="Goldman Sachs Q4 earnings revenue $12B.", metadata=meta1)

    meta2 = DocumentMetadata(doc_id="d2", filename="risk_report.pdf", page_number=5, chunk_index=1)
    chunk2 = DocumentChunk(chunk_id="d2_c1", content="Market risk VaR calculated at 99% confidence.", metadata=meta2)

    chunks = [chunk1, chunk2]
    embs = embedder.embed_texts([c.content for c in chunks])
    store.add_chunks(chunks, embs)

    assert store.count() == 2

    # Query search
    q_emb = embedder.embed_query("Goldman Sachs revenue")
    results = store.search(q_emb, top_k=2)

    assert len(results) == 2
    assert results[0][0].chunk_id == "d1_c0"
    assert results[0][1] >= results[1][1]


def test_vector_store_save_load(tmp_path):
    store = VectorStore(dimension=384)
    embedder = Embedder(dimension=384)
    meta = DocumentMetadata(doc_id="d1", filename="test.pdf", page_number=1, chunk_index=0)
    chunk = DocumentChunk(chunk_id="c0", content="Basel III capital accord compliance.", metadata=meta)
    store.add_chunks([chunk], embedder.embed_texts([chunk.content]))

    save_dir = str(tmp_path / "vstore")
    store.save(save_dir)

    loaded_store = VectorStore(dimension=384)
    loaded_store.load(save_dir)
    assert loaded_store.count() == 1
    assert loaded_store.chunks[0].chunk_id == "c0"


def test_hybrid_search_rrf():
    store = VectorStore(dimension=384)
    embedder = Embedder(dimension=384)
    retriever = HybridRetriever(store, embedder)

    meta1 = DocumentMetadata(doc_id="d1", filename="reg.pdf", page_number=10, chunk_index=0)
    chunk1 = DocumentChunk(
        chunk_id="c_reg",
        content="Regulation T establishes margin requirements for broker-dealers extending credit.",
        metadata=meta1
    )
    meta2 = DocumentMetadata(doc_id="d2", filename="generic.pdf", page_number=1, chunk_index=0)
    chunk2 = DocumentChunk(
        chunk_id="c_gen",
        content="General financial market conditions remained volatile during the past fiscal year.",
        metadata=meta2
    )

    chunks = [chunk1, chunk2]
    store.add_chunks(chunks, embedder.embed_texts([c.content for c in chunks]))
    retriever.index_updated()

    # Search specifically for keyword "Regulation T"
    results = retriever.search("What does Regulation T establish?", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "c_reg"
    assert results[0].score > 0.0


def test_hybrid_search_with_colbert_reranking():
    store = VectorStore(dimension=384)
    embedder = Embedder(dimension=384)
    retriever = HybridRetriever(store, embedder)

    meta1 = DocumentMetadata(doc_id="d1", filename="sec_filing.pdf", page_number=4, chunk_index=0)
    chunk1 = DocumentChunk(
        chunk_id="c_slr",
        content="The firm reported an SLR Supplementary Leverage Ratio of 5.8% for the banking subsidiary.",
        metadata=meta1,
    )
    meta2 = DocumentMetadata(doc_id="d2", filename="generic_filing.pdf", page_number=1, chunk_index=0)
    chunk2 = DocumentChunk(
        chunk_id="c_other",
        content="The executive compensation committee approved performance shares for senior management.",
        metadata=meta2,
    )

    store.add_chunks([chunk1, chunk2], embedder.embed_texts([chunk1.content, chunk2.content]))
    retriever.index_updated()

    results = retriever.search("SLR requirement", top_k=2, enable_reranking=True)
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "c_slr"
    assert results[0].score > 0.5


def test_hybrid_search_with_acronym_expansion():
    store = VectorStore(dimension=384)
    embedder = Embedder(dimension=384)
    retriever = HybridRetriever(store, embedder)

    meta = DocumentMetadata(doc_id="d1", filename="fed_stress_test.pdf", page_number=1, chunk_index=0)
    chunk = DocumentChunk(
        chunk_id="c_ccar",
        content="Under Comprehensive Capital Analysis and Review stress test guidelines, Tier 1 common capital remained resilient.",
        metadata=meta,
    )
    store.add_chunks([chunk], embedder.embed_texts([chunk.content]))
    retriever.index_updated()

    # Query using only the acronym "CCAR" - query rewriter will expand to include "Comprehensive Capital Analysis and Review"
    results = retriever.search("CCAR capital targets", top_k=1, enable_query_rewrite=True)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c_ccar"
