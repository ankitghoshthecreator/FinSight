"""Unit tests for generation: guardrail, prompt templates, and structured answers."""

import pytest
from ingestion.models import DocumentChunk, DocumentMetadata
from generation.citation_guard import CitationGuardrail
from generation.prompt_templates import build_qa_prompt
from generation.generator import FinancialAnswerGenerator


def test_citation_guard_extracts_citations():
    guard = CitationGuardrail()
    text = "The CET1 ratio was 15.3% [source: Q3_2024.pdf, page 14] and Tier 1 leverage was 5.1% [source: Risk_Report.pdf, page 8]."
    cits = guard.extract_citations(text)
    assert len(cits) == 2
    assert cits[0] == ("Q3_2024.pdf", 14)
    assert cits[1] == ("Risk_Report.pdf", 8)


def test_citation_guard_detects_hallucination():
    guard = CitationGuardrail()
    meta = DocumentMetadata(doc_id="d1", filename="Real_Doc.pdf", page_number=1)
    chunk = DocumentChunk(chunk_id="c1", content="Approved dividend of $1.50 per share.", metadata=meta)

    # Hallucinated document
    answer = "The dividend was $1.50 per share [source: Fake_Document.pdf, page 99]."
    is_grounded, citations, sanitized = guard.verify_and_ground(answer, [chunk])

    assert not is_grounded
    assert len(citations) == 0
    assert "Insufficient context" in sanitized


def test_citation_guard_valid_grounding():
    guard = CitationGuardrail()
    meta = DocumentMetadata(doc_id="d1", filename="Real_Doc.pdf", page_number=1)
    chunk = DocumentChunk(chunk_id="c1", content="Approved dividend of $1.50 per share.", metadata=meta)

    answer = "The dividend was $1.50 per share [source: Real_Doc.pdf, page 1]."
    is_grounded, citations, sanitized = guard.verify_and_ground(answer, [chunk])

    assert is_grounded
    assert len(citations) == 1
    assert citations[0].doc_name == "Real_Doc.pdf"
    assert citations[0].page_number == 1


def test_generator_insufficient_context():
    gen = FinancialAnswerGenerator(use_stub=True)
    res = gen.generate("What is the Tier 1 ratio?", context_chunks=[])
    assert "insufficient context" in res.answer.lower()
    assert res.confidence == 0.0
    assert res.escalate is True


def test_generator_with_context():
    gen = FinancialAnswerGenerator(use_stub=True)
    meta = DocumentMetadata(doc_id="d1", filename="Earnings.pdf", page_number=4)
    chunk = DocumentChunk(
        chunk_id="c1",
        content="The net income for UBS in Q4 reached $1.4B with strong wealth management inflows.",
        metadata=meta
    )
    res = gen.generate("UBS net income Q4", context_chunks=[chunk])
    assert res.confidence >= 0.8
    assert res.escalate is False
    assert len(res.sources) == 1
    assert res.sources[0].doc_name == "Earnings.pdf"
