"""Data models and schemas for FinSight RAG pipeline."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with an ingested document or extracted chunk."""
    doc_id: str = Field(..., description="Unique document identifier or hash")
    filename: str = Field(..., description="Original filename")
    page_number: int = Field(default=1, description="1-indexed page number")
    chunk_index: int = Field(default=0, description="0-indexed chunk sequence within document")
    char_start: Optional[int] = Field(default=None, description="Starting character offset in source text")
    char_end: Optional[int] = Field(default=None, description="Ending character offset in source text")
    token_count: int = Field(default=0, description="Approximate or exact token count")
    additional_metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata (e.g. section title)")


class IngestTextRequest(BaseModel):
    """Payload for text ingestion endpoint."""
    filename: str = Field(..., description="Document filename")
    content: str = Field(..., min_length=1, description="Document body text")
    page_number: int = Field(default=1, ge=1, description="1-indexed page number")
    redact_pii: bool = Field(default=True, description="Whether to redact SSNs, PANs, emails, and bank details")




class DocumentChunk(BaseModel):
    """Represents a text chunk created during ingestion."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk, e.g. doc_id_chunk_0")
    content: str = Field(..., description="Extracted chunk text")
    metadata: DocumentMetadata = Field(..., description="Associated document metadata")


class Citation(BaseModel):
    """Grounding citation for a claim or answer."""
    doc_name: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="Source page number")
    chunk_id: Optional[str] = Field(default=None, description="Reference chunk ID")
    snippet: str = Field(..., description="Relevant text excerpt supporting the answer")


class SearchResult(BaseModel):
    """Result returned from hybrid or vector search."""
    chunk: DocumentChunk
    score: float = Field(..., description="Fused or normalized relevance score")
    dense_score: Optional[float] = Field(default=None, description="Dense cosine/L2 score")
    bm25_score: Optional[float] = Field(default=None, description="BM25 keyword score")


class QueryRequest(BaseModel):
    """Incoming query request to the API."""
    query: str = Field(..., min_length=2, description="User's natural language or financial question")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of context chunks to retrieve")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence score")
    enable_reranking: bool = Field(default=True, description="Whether to apply score fusion / reranking")


class QueryResponse(BaseModel):
    """Structured response conforming to enterprise financial requirements."""
    answer: str = Field(..., description="Source-grounded answer or 'insufficient context'")
    sources: List[Citation] = Field(default_factory=list, description="List of source citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    escalate: bool = Field(..., description="Flag true if confidence is low, context is insufficient, or conflict detected")
