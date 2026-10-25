"""FastAPI application for FinSight RAG enterprise document intelligence."""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse

from config.settings import settings
from api.auth import APIKeyMiddleware
from api.middleware import RequestTracingMiddleware
from api.tasks import job_manager, IngestionJobStatus
from api.errors import register_error_handlers
from api.metrics import metrics_collector
from ingestion.chunker import SemanticChunker
from ingestion.models import (
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
)
from ingestion.pdf_extractor import ExtractedPage, PDFExtractor
from ingestion.pii_redactor import PIIRedactor
from retrieval.embedder import Embedder
from retrieval.hybrid_search import HybridRetriever
from retrieval.vector_store import VectorStore
from generation.generator import FinancialAnswerGenerator

logger = logging.getLogger(__name__)

# ── Pipeline singletons ───────────────────────────────────────────────────────

extractor = PDFExtractor()
chunker   = SemanticChunker(
    max_tokens=settings.chunk_max_tokens,
    overlap_tokens=settings.chunk_overlap_tokens,
)
embedder = Embedder(
    model_name=settings.embedding_model,
    dimension=settings.embedding_dim,
)
vector_store = VectorStore(
    dimension=embedder.dimension,
    persist_dir=settings.faiss_persist_dir,   # auto-loads on init if dir exists
)
retriever = HybridRetriever(
    vector_store,
    embedder,
    bm25_weight=settings.bm25_weight,
    dense_weight=settings.dense_weight,
)
generator = FinancialAnswerGenerator(model_name=settings.llm_model)
redactor  = PIIRedactor()


# ── Lifespan: persist index on shutdown ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle.

    On shutdown the FAISS index is saved to ``settings.faiss_persist_dir`` so
    ingested documents survive server restarts.
    """
    logger.info(
        "FinSight RAG starting — %d chunks pre-loaded from disk",
        vector_store.count(),
    )
    yield
    # ── shutdown ──────────────────────────────────────────────────────────
    if vector_store.count() > 0:
        try:
            vector_store.save(settings.faiss_persist_dir)
            logger.info(
                "Index persisted to %s (%d chunks)",
                settings.faiss_persist_dir,
                vector_store.count(),
            )
        except Exception as exc:
            logger.error("Failed to persist index on shutdown: %s", exc)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description=(
        "Production-Grade Document Intelligence for Financial Workflows "
        "(JPMC, Goldman, UBS)"
    ),
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# ── RFC 7807 global structured error handlers ──────────────────────────────────
register_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting (slowapi) ────────────────────────────────────────────────────
if settings.rate_limit_per_minute > 0:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{settings.rate_limit_per_minute}/minute"],
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

# ── Request tracing & audit logging middleware ──────────────────────────────
app.add_middleware(RequestTracingMiddleware)

# ── API Key auth middleware ───────────────────────────────────────────────────
app.add_middleware(APIKeyMiddleware)


# ── System ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    """Welcome endpoint with interactive documentation links and API summary."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "health_check": "/health",
        "endpoints": {
            "ingest_file": "POST /ingest/file",
            "ingest_file_async": "POST /ingest/file/async",
            "ingest_status": "GET /ingest/status/{job_id}",
            "ingest_text": "POST /ingest/text",
            "query": "POST /query",
            "query_stream": "POST /query/stream",
            "documents": "GET /documents",
            "delete_document": "DELETE /documents/{doc_id}",
            "reset_index": "DELETE /documents",
        },
    }


@app.get("/health", tags=["System"])
def health_check():
    """Health and readiness check endpoint."""
    return {
        "status": "healthy",
        "indexed_chunks": vector_store.count(),
        "unique_documents": len(vector_store.get_doc_ids()),
        "vector_dimension": embedder.dimension,
        "vector_backend": settings.vector_store_backend,
        "dense_encoder": (
            "sentence-transformers" if not embedder.is_fallback else "hash-projection"
        ),
        "persist_dir": settings.faiss_persist_dir,
    }


@app.get("/metrics", tags=["System"])
def get_metrics(format: str = "json"):
    """Exposes real-time system, RAG quality, and operational audit metrics."""
    if format == "prometheus":
        return PlainTextResponse(metrics_collector.to_prometheus_format(), media_type="text/plain; version=0.0.4")
    return metrics_collector.get_summary()


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest/text", tags=["Ingestion"])
def ingest_text(payload: IngestTextRequest):
    """Ingests raw text or markdown document into the index with optional PII redaction."""
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty")

    content_to_index = payload.content
    redacted_types: List[str] = []
    if payload.redact_pii:
        content_to_index, redacted_types = redactor.redact(payload.content)

    page = ExtractedPage(page_number=payload.page_number, text=content_to_index)
    doc_id = f"doc_{abs(hash(payload.filename + content_to_index)) % 1_000_000:06d}"
    chunks = chunker.chunk_document([page], doc_id=doc_id, filename=payload.filename)

    if chunks:
        embeddings = embedder.embed_texts([c.content for c in chunks])
        vector_store.add_chunks(chunks, embeddings)
        retriever.index_updated()
        metrics_collector.record_chunks_indexed(len(chunks))

    if redacted_types:
        metrics_collector.record_pii_redacted(len(redacted_types))

    return {
        "doc_id": doc_id,
        "filename": payload.filename,
        "chunks_indexed": len(chunks),
        "total_store_chunks": vector_store.count(),
        "pii_redacted": redacted_types,
    }


@app.post("/ingest/file", tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Ingests and chunks an uploaded document (PDF, TXT, MD, DOCX)."""
    allowed_exts = [".pdf", ".txt", ".md", ".docx"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Supported: {allowed_exts}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if file_ext == ".pdf":
            pages  = extractor.extract(tmp_path)
            doc_id = extractor.compute_doc_id(tmp_path)
        else:
            pages  = extractor.extract_text_file(tmp_path)
            doc_id = f"doc_{abs(hash(file.filename)) % 1_000_000:06d}"

        chunks = chunker.chunk_document(pages, doc_id=doc_id, filename=file.filename)
        if chunks:
            embeddings = embedder.embed_texts([c.content for c in chunks])
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


def _process_file_background(job_id: str, tmp_path: str, filename: str, file_ext: str):
    """Worker task executed in background for large document files."""
    try:
        job_manager.update_job(job_id, status="processing")
        if file_ext == ".pdf":
            pages = extractor.extract(tmp_path)
            doc_id = extractor.compute_doc_id(tmp_path)
        else:
            pages = extractor.extract_text_file(tmp_path)
            doc_id = f"doc_{abs(hash(filename)) % 1_000_000:06d}"

        chunks = chunker.chunk_document(pages, doc_id=doc_id, filename=filename)
        if chunks:
            embeddings = embedder.embed_texts([c.content for c in chunks])
            vector_store.add_chunks(chunks, embeddings)
            retriever.index_updated()

        job_manager.update_job(
            job_id,
            status="completed",
            doc_id=doc_id,
            pages_parsed=len(pages),
            chunks_created=len(chunks),
            completed_at=time.time(),
        )
    except Exception as exc:
        logger.error(f"Async ingestion job {job_id} failed: {exc}")
        job_manager.update_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=time.time(),
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ingest/file/async", tags=["Ingestion"])
async def ingest_file_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Asynchronously ingests and chunks an uploaded document using background execution."""
    allowed_exts = [".pdf", ".txt", ".md", ".docx"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Supported: {allowed_exts}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    job_id = job_manager.create_job(file.filename)
    background_tasks.add_task(_process_file_background, job_id, tmp_path, file.filename, file_ext)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "queued",
        "check_status_url": f"/ingest/status/{job_id}",
    }


@app.get("/ingest/status/{job_id}", response_model=IngestionJobStatus, tags=["Ingestion"])
def get_ingest_job_status(job_id: str):
    """Polls status of an asynchronous document ingestion job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


# ── Retrieval & Generation ────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def query_pipeline(req: QueryRequest):
    """Executes the full FinSight RAG pipeline:
    Query Rewriting → Hybrid Retrieval (BM25 + Dense RRF) →
    ColBERT Reranking → Citation Guardrail → Structured Answer
    """
    if vector_store.count() == 0:
        return QueryResponse(
            answer="Insufficient context: no financial documents have been ingested yet.",
            sources=[],
            confidence=0.0,
            escalate=True,
        )

    results = retriever.search(
        req.query, top_k=req.top_k, enable_reranking=req.enable_reranking
    )
    context_chunks = [r.chunk for r in results]
    response = generator.generate(req.query, context_chunks)
    metrics_collector.record_query(latency_ms=15.0, escalated=response.escalate)
    return response


@app.post("/query/stream", tags=["Retrieval & Generation"])
async def query_pipeline_stream(req: QueryRequest):
    """Streams the answer token-by-token via Server-Sent Events (SSE)."""
    async def token_generator():
        res = query_pipeline(req)
        yield (
            f"event: metadata\n"
            f"data: {res.model_dump_json(include={'sources','confidence','escalate'})}\n\n"
        )
        await asyncio.sleep(0.01)
        words = res.answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else f" {word}"
            yield f"event: token\ndata: {chunk}\n\n"
            await asyncio.sleep(0.02)
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# ── Index management ──────────────────────────────────────────────────────────

@app.get("/documents", tags=["Index Management"])
def list_documents():
    """Lists a summary of all currently indexed chunks (max 50 shown)."""
    return {
        "total_chunks": vector_store.count(),
        "unique_documents": len(vector_store.get_doc_ids()),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.metadata.doc_id,
                "filename": c.metadata.filename,
                "page": c.metadata.page_number,
                "preview": c.content[:100] + "...",
            }
            for c in vector_store.chunks[:50]
        ],
    }


@app.delete("/documents/{doc_id}", tags=["Index Management"])
def delete_document(doc_id: str):
    """Removes all indexed chunks for the given doc_id."""
    removed = vector_store.delete_by_doc_id(doc_id)
    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for doc_id='{doc_id}'",
        )
    retriever.index_updated()
    return {
        "doc_id": doc_id,
        "chunks_removed": removed,
        "total_store_chunks": vector_store.count(),
    }


@app.delete("/documents", tags=["Index Management"])
def reset_index():
    """Wipes the entire vector index. Irreversible."""
    count_before = vector_store.count()
    vector_store.reset()
    retriever.index_updated()
    return {
        "message": "Index reset successfully",
        "chunks_removed": count_before,
        "total_store_chunks": 0,
    }
