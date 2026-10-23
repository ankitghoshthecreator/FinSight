"""Unit tests for ColBERT late-interaction reranking."""

import pytest
from ingestion.models import DocumentChunk, DocumentMetadata, SearchResult
from retrieval.reranker import ColBERTReranker


def test_colbert_reranker_initialization():
    reranker = ColBERTReranker()
    assert reranker is not None
    assert reranker.embedding_dim == 128


def test_colbert_score_exact_match():
    reranker = ColBERTReranker()
    query = "Supplementary Leverage Ratio SLR"
    doc_match = "Goldman Sachs reported a Supplementary Leverage Ratio SLR of 5.8%."
    doc_unrelated = "Agricultural commodities harvest forecast for Midwest soybeans."

    score_match = reranker.score(query, doc_match)
    score_unrelated = reranker.score(query, doc_unrelated)

    assert score_match > score_unrelated
    assert score_match > 0.5


def test_colbert_rerank_candidates():
    reranker = ColBERTReranker()
    meta1 = DocumentMetadata(doc_id="d1", filename="sec_10k.txt", page_number=1)
    chunk1 = DocumentChunk(
        chunk_id="c1",
        content="The bank maintained a CET1 ratio of 14.5% exceeding Basel III standards.",
        metadata=meta1,
    )

    meta2 = DocumentMetadata(doc_id="d2", filename="annual.txt", page_number=2)
    chunk2 = DocumentChunk(
        chunk_id="c2",
        content="General corporate update regarding office locations and sustainability goals.",
        metadata=meta2,
    )

    candidates = [
        SearchResult(chunk=chunk2, score=0.8, dense_score=0.8, bm25_score=0.2),
        SearchResult(chunk=chunk1, score=0.6, dense_score=0.6, bm25_score=0.6),
    ]

    reranked = reranker.rerank(
        query="What is the CET1 capital ratio under Basel III?",
        candidates=candidates,
        top_k=2,
    )

    assert len(reranked) == 2
    # chunk1 should now be promoted to top rank due to token-level late interaction
    assert reranked[0].chunk.chunk_id == "c1"
    assert reranked[0].score > reranked[1].score


def test_colbert_empty_candidates():
    reranker = ColBERTReranker()
    res = reranker.rerank("any query", [])
    assert res == []


def test_colbert_compute_maxsim_empty():
    reranker = ColBERTReranker()
    assert reranker.compute_maxsim([], ["token"]) == 0.0
    assert reranker.compute_maxsim(["token"], []) == 0.0
