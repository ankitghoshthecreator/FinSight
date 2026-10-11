"""Document text and metadata extractor supporting PDF, DOCX, and TXT files."""

import os
import hashlib
from typing import List, Dict, Any, Optional


class ExtractedPage:
    """Represents text extracted from a single page of a document."""
    def __init__(self, page_number: int, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_number = page_number
        self.text = text
        self.metadata = metadata or {}


class PDFExtractor:
    """Extracts text and page-level metadata from PDF files using pdfplumber with fallback."""

    def __init__(self):
        pass

    @staticmethod
    def compute_doc_id(file_path: str) -> str:
        """Computes a SHA256 hash for document identification."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def extract(self, file_path: str) -> List[ExtractedPage]:
        """Extracts text per page from PDF, preserving page indices and table layouts where possible."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        extracted_pages: List[ExtractedPage] = []
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text(layout=True) or page.extract_text() or ""
                    # Clean control artifacts while preserving numbers & layout
                    cleaned = self._clean_text(text)
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=idx + 1,
                            text=cleaned,
                            metadata={"total_pages": len(pdf.pages), "extractor": "pdfplumber"}
                        )
                    )
            return extracted_pages
        except ImportError:
            # Fallback if pdfplumber is not yet installed (e.g. pypdf)
            pass

        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                cleaned = self._clean_text(text)
                extracted_pages.append(
                    ExtractedPage(
                        page_number=idx + 1,
                        text=cleaned,
                        metadata={"total_pages": len(reader.pages), "extractor": "pypdf"}
                    )
                )
            return extracted_pages
        except ImportError:
            raise ImportError("Neither 'pdfplumber' nor 'pypdf' is available. Please install dependencies.")

    def extract_text_file(self, file_path: str) -> List[ExtractedPage]:
        """Extracts text from a plain text or markdown file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        cleaned = self._clean_text(content)
        return [
            ExtractedPage(
                page_number=1,
                text=cleaned,
                metadata={"extractor": "plain_text"}
            )
        ]

    @staticmethod
    def _clean_text(text: str) -> str:
        """Removes null bytes and standardizes linebreaks while keeping financial formats intact."""
        if not text:
            return ""
        text = text.replace("\x00", "")
        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.strip()
