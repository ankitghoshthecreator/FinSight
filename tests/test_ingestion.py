"""Unit tests for ingestion: models, extractor, and semantic chunker."""

import pytest
from ingestion.models import (
    DocumentMetadata,
    DocumentChunk,
    Citation,
    QueryRequest,
    QueryResponse,
)
from ingestion.pdf_extractor import PDFExtractor, ExtractedPage
from ingestion.chunker import SemanticChunker


def test_models_pydantic_validation():
    meta = DocumentMetadata(
        doc_id="doc_123",
        filename="Q3_Earnings.pdf",
        page_number=1,
        chunk_index=0,
    )
    assert meta.doc_id == "doc_123"
    assert meta.page_number == 1

    chunk = DocumentChunk(
        chunk_id="doc_123_c0",
        content="CET1 ratio increased to 15.3%.",
        metadata=meta,
    )
    assert "CET1" in chunk.content

    citation = Citation(
        doc_name="Q3_Earnings.pdf",
        page_number=1,
        snippet="CET1 ratio increased to 15.3%.",
    )
    assert citation.doc_name == "Q3_Earnings.pdf"

    res = QueryResponse(
        answer="CET1 is 15.3% [source: Q3_Earnings.pdf, page 1].",
        sources=[citation],
        confidence=0.95,
        escalate=False,
    )
    assert res.confidence == 0.95
    assert not res.escalate


def test_semantic_chunker_splits_sentences():
    chunker = SemanticChunker(max_tokens=50, overlap_tokens=10)
    text = (
        "In Q3 2024, our Common Equity Tier 1 (CET1) ratio reached 15.3%. "
        "Net Interest Income was $22.7B across all consumer segments. "
        "Credit loss provisions stood at $1.2B under CECL framework. "
        "The board authorized a $10B share repurchase plan."
    )
    page = ExtractedPage(page_number=2, text=text)
    chunks = chunker.chunk_page(page, doc_id="test_doc", filename="test.pdf")

    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata.page_number == 2
        assert c.metadata.filename == "test.pdf"


def test_chunker_empty_page():
    chunker = SemanticChunker()
    page = ExtractedPage(page_number=1, text="   ")
    chunks = chunker.chunk_page(page, doc_id="empty", filename="empty.pdf")
    assert len(chunks) == 0


def test_chunk_document_sequential_ids():
    chunker = SemanticChunker(max_tokens=30, overlap_tokens=5)
    pages = [
        ExtractedPage(page_number=1, text="First page content regarding revenue $50B."),
        ExtractedPage(page_number=2, text="Second page content regarding expenses $30B."),
    ]
    chunks = chunker.chunk_document(pages, doc_id="report", filename="report.pdf")
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "report_c0"
    assert chunks[1].chunk_id == "report_c1"
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[1].metadata.chunk_index == 1


def test_pdf_extractor_clean_text():
    raw = "Line 1\x00\r\nLine 2\rLine 3"
    cleaned = PDFExtractor._clean_text(raw)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "Line 1\nLine 2\nLine 3" == cleaned
