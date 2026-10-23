"""FastAPI application for FinSight RAG enterprise document intelligence."""

import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ingestion.models import (
    QueryRequest,
    QueryResponse,
    DocumentChunk,
    DocumentMetadata,
    IngestTextRequest,
)
from ingestion.pii_redactor import PIIRedactor
from ingestion.pdf_extractor import PDFExtractor

from ingestion.chunker import SemanticChunker
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridRetriever
from generation.generator import FinancialAnswerGenerator


app = FastAPI(
    title="FinSight RAG API",
    description="Production-Grade Document Intelligence for Financial Workflows (JPMC, Goldman, UBS)",
    version="1.0.0",
)

# CORS middleware for integration with frontends & dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instances (singleton lifecycle)
extractor = PDFExtractor()
chunker = SemanticChunker(max_tokens=512, overlap_tokens=64)
embedder = Embedder()
vector_store = VectorStore(dimension=embedder.dimension)
retriever = HybridRetriever(vector_store, embedder)
generator = FinancialAnswerGenerator()


@app.get("/", tags=["System"])
def root():
    """Welcome endpoint with interactive documentation links and API summary."""
    return {
        "service": "FinSight RAG API",
        "status": "online",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "health_check": "/health",
        "endpoints": {
            "ingest_file": "POST /ingest/file",
            "ingest_text": "POST /ingest/text",
            "query": "POST /query",
            "documents": "GET /documents"
        }
    }


@app.get("/health", tags=["System"])
def health_check():
    """Health and readiness check endpoint."""
    return {
        "status": "healthy",
        "indexed_chunks": vector_store.count(),
        "vector_dimension": embedder.dimension,
        "dense_backend": "sentence-transformers" if not embedder.is_fallback else "hash-projection",
    }


redactor = PIIRedactor()


@app.post("/ingest/text", tags=["Ingestion"])
def ingest_text(payload: IngestTextRequest):
    """Ingests raw text or markdown document directly into index with optional PII redaction."""
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty")

    content_to_index = payload.content
    redacted_types = []
    if payload.redact_pii:
        content_to_index, redacted_types = redactor.redact(payload.content)

    from ingestion.pdf_extractor import ExtractedPage
    page = ExtractedPage(page_number=payload.page_number, text=content_to_index)
    doc_id = f"doc_{abs(hash(payload.filename + content_to_index)) % 1000000:06d}"
    chunks = chunker.chunk_document([page], doc_id=doc_id, filename=payload.filename)

    if chunks:
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(chunks, embeddings)
        retriever.index_updated()

    return {
        "doc_id": doc_id,
        "filename": payload.filename,
        "chunks_indexed": len(chunks),
        "total_store_chunks": vector_store.count(),
        "pii_redacted": redacted_types,
    }




@app.post("/ingest/file", tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Ingests and chunks an uploaded document (PDF or TXT)."""
    allowed_exts = [".pdf", ".txt", ".md", ".docx"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Supported formats: {allowed_exts}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if file_ext == ".pdf":
            pages = extractor.extract(tmp_path)
            doc_id = extractor.compute_doc_id(tmp_path)
        else:
            pages = extractor.extract_text_file(tmp_path)
            doc_id = f"doc_{abs(hash(file.filename)) % 1000000:06d}"

        chunks = chunker.chunk_document(pages, doc_id=doc_id, filename=file.filename)
        if chunks:
            texts = [c.content for c in chunks]
            embeddings = embedder.embed_texts(texts)
            vector_store.add_chunks(chunks, embeddings)
            retriever.index_updated()

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "pages_parsed": len(pages),
            "chunks_created": len(chunks),
            "total_indexed_chunks": vector_store.count(),
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


from fastapi.responses import StreamingResponse
import asyncio


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def query_pipeline(req: QueryRequest):
    """Executes the complete FinSight RAG pipeline:
    Query -> Hybrid Retrieval -> Guardrail Verification -> Structured Answer
    """
    if vector_store.count() == 0:
        return QueryResponse(
            answer="Insufficient context: no financial documents have been ingested yet.",
            sources=[],
            confidence=0.0,
            escalate=True,
        )

    # 1. Hybrid search (BM25 + Dense with RRF + ColBERT late-interaction reranking)
    results = retriever.search(req.query, top_k=req.top_k, enable_reranking=req.enable_reranking)
    context_chunks = [r.chunk for r in results]

    # 2. Generation & Guardrail
    response = generator.generate(req.query, context_chunks)
    return response


@app.post("/query/stream", tags=["Retrieval & Generation"])
async def query_pipeline_stream(req: QueryRequest):
    """Streams structured response tokens using FastAPI StreamingResponse."""
    async def token_generator():
        res = query_pipeline(req)
        # Yield metadata event
        yield f"event: metadata\ndata: {res.model_dump_json(include={'sources', 'confidence', 'escalate'})}\n\n"
        await asyncio.sleep(0.01)

        # Stream answer tokens
        words = res.answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else f" {word}"
            yield f"event: token\ndata: {chunk}\n\n"
            await asyncio.sleep(0.02)
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")



@app.get("/documents", tags=["System"])
def list_documents():
    """Lists summary of all currently indexed chunks."""
    return {
        "total_chunks": vector_store.count(),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "filename": c.metadata.filename,
                "page": c.metadata.page_number,
                "preview": c.content[:100] + "..."
            }
            for c in vector_store.chunks[:50]
        ]
    }
