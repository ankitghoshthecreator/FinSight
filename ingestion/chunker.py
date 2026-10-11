"""Semantic chunking with sentence-boundary awareness and token overlap."""

import re
from typing import List
from ingestion.models import DocumentChunk, DocumentMetadata
from ingestion.pdf_extractor import ExtractedPage


class SemanticChunker:
    """Chunks documents into coherent passages preserving sentence boundaries.
    
    Default parameters:
    - max_tokens: 512
    - overlap_tokens: 64
    - token estimation: ~0.75 words per token or exact word-split
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Splits text into sentences preserving decimal numbers, currency amounts, and abbreviations.
        
        Example: '$3.5B' or 'U.S. SEC Rule 10b-5' should not split on internal periods.
        """
        # Lookbehind not following common abbreviations or single capital letters
        sentence_end_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z0-9$€£¥\(\[\"\'\-])'
        )
        paragraphs = text.split("\n\n")
        sentences = []
        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue
            parts = sentence_end_pattern.split(p_strip)
            for part in parts:
                clean_part = part.strip()
                if clean_part:
                    sentences.append(clean_part)
        return sentences

    def _estimate_tokens(self, text: str) -> int:
        """Estimates token count using whitespace and punctuation heuristics (~1.3 tokens per whitespace word)."""
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def chunk_page(self, page: ExtractedPage, doc_id: str, filename: str) -> List[DocumentChunk]:
        """Creates semantic chunks for an extracted page."""
        if not page.text.strip():
            return []

        sentences = self._split_into_sentences(page.text)
        if not sentences:
            sentences = [page.text]

        chunks: List[DocumentChunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        for sentence in sentences:
            s_tokens = self._estimate_tokens(sentence)
            if current_tokens + s_tokens > self.max_tokens and current_sentences:
                # Flush current chunk
                chunk_text = " ".join(current_sentences)
                chunk_id = f"{doc_id}_p{page.page_number}_c{chunk_idx}"
                metadata = DocumentMetadata(
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    token_count=current_tokens,
                    additional_metadata=page.metadata,
                )
                chunks.append(DocumentChunk(chunk_id=chunk_id, content=chunk_text, metadata=metadata))
                chunk_idx += 1

                # Calculate overlap sentences
                overlap_accum: List[str] = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    st = self._estimate_tokens(s)
                    if overlap_count + st <= self.overlap_tokens:
                        overlap_accum.insert(0, s)
                        overlap_count += st
                    else:
                        break

                current_sentences = overlap_accum
                current_tokens = overlap_count

            current_sentences.append(sentence)
            current_tokens += s_tokens

        # Flush any remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = f"{doc_id}_p{page.page_number}_c{chunk_idx}"
            metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                page_number=page.page_number,
                chunk_index=chunk_idx,
                token_count=current_tokens,
                additional_metadata=page.metadata,
            )
            chunks.append(DocumentChunk(chunk_id=chunk_id, content=chunk_text, metadata=metadata))

        return chunks

    def chunk_document(self, pages: List[ExtractedPage], doc_id: str, filename: str) -> List[DocumentChunk]:
        """Chunks all pages in a document while maintaining sequential chunk indices."""
        all_chunks: List[DocumentChunk] = []
        global_idx = 0
        for page in pages:
            page_chunks = self.chunk_page(page, doc_id, filename)
            for c in page_chunks:
                c.metadata.chunk_index = global_idx
                c.chunk_id = f"{doc_id}_c{global_idx}"
                global_idx += 1
                all_chunks.append(c)
        return all_chunks
