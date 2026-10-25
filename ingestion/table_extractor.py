"""Financial table extraction and structured markdown table serialization.
Preserves financial balance sheets, capital adequacy schedules, and regulatory disclosures.
"""

import re
from typing import List, Dict, Any, Optional
from ingestion.models import DocumentChunk, DocumentMetadata


class FinancialTableExtractor:
    """Extracts and normalizes structured tables from financial filings into semantic markdown chunks."""

    @staticmethod
    def is_table_markdown(text: str) -> bool:
        """Heuristic detecting markdown-formatted table structure."""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            return False
        pipe_lines = sum(1 for line in lines if line.startswith("|") and line.endswith("|"))
        return pipe_lines >= 2

    @staticmethod
    def parse_markdown_table(table_text: str) -> Dict[str, Any]:
        """Parses markdown table into headers and rows of cells."""
        lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            return {"headers": [], "rows": []}

        # Header row
        headers = [c.strip() for c in lines[0].strip("|").split("|")]

        rows = []
        for line in lines[1:]:
            if re.match(r'^\|?[\s\-:|]+\|?$', line):
                continue  # skip separator line
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == len(headers):
                rows.append(cells)

        return {"headers": headers, "rows": rows}

    def convert_table_to_chunk(
        self,
        table_markdown: str,
        doc_id: str,
        filename: str,
        page_number: int,
        chunk_index: int,
        table_title: Optional[str] = None,
    ) -> DocumentChunk:
        """Converts extracted table into a DocumentChunk annotated with table metadata."""
        content = f"Financial Table: {table_title}\n\n{table_markdown.strip()}" if table_title else table_markdown.strip()

        meta = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            page_number=page_number,
            chunk_index=chunk_index,
            token_count=len(content.split()),
            additional_metadata={"is_table": True, "table_title": table_title or "Financial Table"}
        )

        return DocumentChunk(
            chunk_id=f"{doc_id}_tbl_p{page_number}_c{chunk_index}",
            content=content,
            metadata=meta
        )
