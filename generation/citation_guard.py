"""Citation checker and hallucination guardrail for financial answer auditability."""

import re
from typing import List, Tuple
from ingestion.models import DocumentChunk, Citation


class CitationGuardrail:
    """Verifies that citations in generated responses map to retrieved context chunks
    and enforces hallucination guardrails.
    """

    CITATION_PATTERN = re.compile(
        r'\[source:\s*([^,\]]+),\s*page\s*(\d+)\]',
        re.IGNORECASE
    )

    def extract_citations(self, text: str) -> List[Tuple[str, int]]:
        """Extracts (doc_name, page_number) pairs from text."""
        matches = self.CITATION_PATTERN.findall(text)
        results = []
        for doc_name, page_str in matches:
            results.append((doc_name.strip(), int(page_str)))
        return results

    def verify_and_ground(
        self,
        answer: str,
        retrieved_chunks: List[DocumentChunk]
    ) -> Tuple[bool, List[Citation], str]:
        """Validates citations against retrieved context.
        
        Returns:
            (is_grounded, citations_list, sanitized_answer)
        """
        # If model explicitly said insufficient context
        if "insufficient context" in answer.lower():
            return False, [], answer

        found_citations = self.extract_citations(answer)
        if not found_citations and retrieved_chunks:
            # Answer made claims without citations
            return False, [], "Insufficient context: factual claims could not be grounded in retrieved sources."

        grounded_citations: List[Citation] = []
        
        # Build map of (filename, page) -> chunks
        context_map = {}
        for chunk in retrieved_chunks:
            key = (chunk.metadata.filename.lower(), chunk.metadata.page_number)
            if key not in context_map:
                context_map[key] = []
            context_map[key].append(chunk)

        all_valid = True
        for doc_name, page in found_citations:
            key = (doc_name.lower(), page)
            matching_chunks = context_map.get(key)
            if not matching_chunks:
                # Document name or page not in retrieved chunks -> hallucinated citation!
                all_valid = False
            else:
                # Find best snippet
                best_chunk = matching_chunks[0]
                snippet = best_chunk.content[:200] + ("..." if len(best_chunk.content) > 200 else "")
                grounded_citations.append(
                    Citation(
                        doc_name=best_chunk.metadata.filename,
                        page_number=page,
                        chunk_id=best_chunk.chunk_id,
                        snippet=snippet
                    )
                )

        if not all_valid and not grounded_citations:
            return (
                False,
                [],
                "Insufficient context: ungrounded or non-existent document citations detected."
            )

        return True, grounded_citations, answer
