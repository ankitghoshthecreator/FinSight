"""Unit tests for financial query rewriting and acronym expansion."""

import pytest
from retrieval.query_rewriter import FinancialQueryRewriter


def test_query_rewriter_acronym_expansion():
    rewriter = FinancialQueryRewriter(expand_acronyms=True)
    query = "What is Goldman Sachs SLR for Q4?"
    rewritten = rewriter.rewrite(query)

    assert "Supplementary Leverage Ratio" in rewritten
    assert "Goldman Sachs" in rewritten


def test_query_rewriter_multiple_acronyms():
    rewriter = FinancialQueryRewriter(expand_acronyms=True)
    query = "Compare CET1 and CCAR requirements"
    rewritten = rewriter.rewrite(query)

    assert "Common Equity Tier 1" in rewritten
    assert "Comprehensive Capital Analysis and Review" in rewritten


def test_query_rewriter_no_expansion_when_already_present():
    rewriter = FinancialQueryRewriter(expand_acronyms=True)
    query = "Goldman Sachs Supplementary Leverage Ratio SLR report"
    rewritten = rewriter.rewrite(query)

    # Should not duplicate the expansion
    assert rewritten.count("Supplementary Leverage Ratio") == 1


def test_query_rewriter_disabled():
    rewriter = FinancialQueryRewriter(expand_acronyms=False)
    query = "What is the SLR requirement?"
    rewritten = rewriter.rewrite(query)
    assert rewritten == "What is the SLR requirement?"


def test_query_rewriter_generate_variants():
    rewriter = FinancialQueryRewriter(expand_acronyms=True)
    query = "What is Goldman Sachs SLR?"
    variants = rewriter.generate_query_variants(query)

    assert len(variants) >= 2
    assert any("Supplementary Leverage Ratio" in v for v in variants)
    assert any("Goldman Sachs" in v for v in variants)
