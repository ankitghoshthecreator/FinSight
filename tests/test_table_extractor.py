"""Unit tests for financial table extraction and markdown conversion."""

import pytest
from ingestion.table_extractor import FinancialTableExtractor


def test_table_extractor_detection():
    ext = FinancialTableExtractor()
    markdown_table = """
    | Capital Ratio | Q3 2024 | Regulatory Minimum |
    |---|---|---|
    | Common Equity Tier 1 (CET1) | 15.3% | 4.5% |
    | Supplementary Leverage Ratio (SLR) | 5.8% | 3.0% |
    """
    assert ext.is_table_markdown(markdown_table) is True

    plain_text = "This is a regular narrative paragraph without any table formatting."
    assert ext.is_table_markdown(plain_text) is False


def test_table_extractor_parsing():
    ext = FinancialTableExtractor()
    markdown_table = """
    | Metric | Value |
    |---|---|
    | Revenue | $12.7B |
    | Net Income | $3.2B |
    """
    parsed = ext.parse_markdown_table(markdown_table)
    assert parsed["headers"] == ["Metric", "Value"]
    assert len(parsed["rows"]) == 2
    assert parsed["rows"][0] == ["Revenue", "$12.7B"]
    assert parsed["rows"][1] == ["Net Income", "$3.2B"]


def test_table_extractor_chunk_generation():
    ext = FinancialTableExtractor()
    table_text = "| Ratio | Value |\n|---|---|\n| CET1 | 14.8% |"
    chunk = ext.convert_table_to_chunk(
        table_markdown=table_text,
        doc_id="sec_10k",
        filename="10k_report.pdf",
        page_number=12,
        chunk_index=3,
        table_title="Capital Ratios Schedule",
    )

    assert chunk.metadata.additional_metadata.get("is_table") is True
    assert "Capital Ratios Schedule" in chunk.content
    assert chunk.chunk_id == "sec_10k_tbl_p12_c3"
