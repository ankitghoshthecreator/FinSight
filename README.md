# FinSight RAG — Production-Grade Document Intelligence for Financial Workflows

> **Role target:** AI Engineer | JPMorgan Chase · Goldman Sachs · MUFG · UBS · BNY Mellon

---

## What It Does

FinSight RAG is a **Retrieval-Augmented Generation (RAG)** pipeline that ingests unstructured financial documents (earnings reports, regulatory filings, compliance memos) and answers structured queries with source-grounded, citation-backed responses. It is designed to operate inside the latency, cost, and auditability constraints that enterprise financial institutions impose on production AI systems.

---

## Problem

Financial analysts spend hours manually searching PDFs and policy documents for specific clauses, figures, and regulatory references. Generic LLM responses hallucinate figures and cite non-existent sections — unacceptable in regulated environments where every answer must be traceable to a source document.

---

## Solution Architecture

```
[Document Ingestion]
    PDF / DOCX → Text Extraction → Semantic Chunking (512 tokens, 64 overlap)
           ↓
[Embedding Pipeline]
    Chunks → sentence-transformers (all-MiniLM-L6-v2) → FAISS / Qdrant vector index
           ↓
[Query Pipeline]
    User Query → Query Rewriting → Hybrid Search (dense + BM25) → Top-K Reranking
           ↓
[Generation]
    Retrieved Context + Prompt Template → LLM (Qwen2.5 / Llama 3) → Grounded Answer
           ↓
[Output Validation]
    Citation checker → Hallucination guardrail → JSON-schema response
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Embedding | `sentence-transformers`, `all-MiniLM-L6-v2` |
| Vector Store | FAISS (local), Qdrant (server mode) |
| Retrieval | Hybrid BM25 + dense, `rank_bm25`, top-K reranking |
| LLM | Qwen2.5-1.5B-Instruct (local), Llama 3 (API) |
| Frameworks | LangChain, Pydantic v2, FastAPI |
| Evaluation | RAGAS (faithfulness, answer relevancy, context recall) |
| Infra | Docker, pytest (42 unit + integration tests) |

---

## Key Features

### 1. Semantic Chunking with Overlap
Fixed-size chunking loses cross-sentence context. Implemented sentence-boundary-aware chunking (512 tokens, 64-token overlap) preserving paragraph integrity for financial clauses.

### 2. Hybrid Retrieval
Dense-only retrieval misses keyword-critical queries ("Regulation T", "Tier 1 Capital Ratio"). Hybrid BM25 + dense search with score fusion improves Recall@5 on financial terminology queries.

### 3. Citation Guardrail
Every answer includes `[source: doc_name, page N]` extracted from retrieved chunks. A post-generation checker verifies every factual claim maps back to a retrieved chunk — returns `"insufficient context"` rather than hallucinating.

### 4. Pydantic Response Schema
All API responses enforce a strict JSON schema: `{answer: str, sources: list[Citation], confidence: float, escalate: bool}`. Downstream systems consume structured output without parsing.

---

## Evaluation Results

| Metric | Baseline (naive RAG) | FinSight RAG |
|---|---|---|
| Faithfulness (RAGAS) | 0.61 | 0.87 |
| Answer Relevancy | 0.58 | 0.83 |
| Context Recall@5 | 0.64 | 0.91 |
| Avg Latency (p95) | 4.2s | 1.8s |
| Hallucination Rate | ~18% | ~3% |

---

## CIRCLE Integration

The **CIRCLE** curriculum-trained LLM (see [CIRCLE repo](https://github.com/ankitghoshthecreator/CIRCLE)) is used as the local generation backbone. CIRCLE's **QLoRA-fine-tuned Qwen2.5** adapter, trained on the summarization and grammar curriculum stages, produces cleaner, better-punctuated responses than the base model — reducing post-generation cleanup passes by removing run-on sentences and fragment outputs that CIRCLE's Stage 3/4 curriculum specifically addresses.

---

## Resume Bullets (AI Engineer)

- Built a **production-grade RAG pipeline** ingesting financial PDFs with **semantic chunking** (512 tokens, 64 overlap), **hybrid BM25 + dense retrieval**, and **top-K reranking**, improving **Context Recall@5 64% → 91%** and reducing **hallucination rate ~18% → ~3%**.
- Enforced **Pydantic JSON-schema structured outputs** with **citation guardrails** that verify every factual claim against retrieved chunks, returning `"insufficient context"` rather than hallucinating — targeting regulated financial environments (JPMC, Goldman, MUFG patterns).
- Integrated **CIRCLE's QLoRA-fine-tuned Qwen2.5** as the generation backbone, reduced **answer latency p95 4.2s → 1.8s** with prompt caching, and validated end-to-end with **RAGAS** (faithfulness **0.61 → 0.87**), backed by **42 automated tests**.

---

## Project Structure

```
finsight-rag/
├── ingestion/
│   ├── pdf_extractor.py          # pdfplumber-based text extraction
│   └── chunker.py                # sentence-boundary semantic chunking
├── retrieval/
│   ├── embedder.py               # sentence-transformers embedding
│   ├── vector_store.py           # FAISS + Qdrant abstraction
│   └── hybrid_search.py          # BM25 + dense fusion
├── generation/
│   ├── prompt_templates.py       # few-shot financial QA templates
│   ├── generator.py              # LLM call + response parser
│   └── citation_guard.py         # hallucination + citation checker
├── api/
│   └── main.py                   # FastAPI endpoints
├── eval/
│   └── ragas_eval.py             # RAGAS evaluation harness
├── tests/                        # 42 unit + integration tests
└── docker/
    └── Dockerfile
```

---



## What's Next

- Upgrade to ColBERT late-interaction reranking
- Add PII redaction layer before vector indexing (MUFG/Goldman compliance pattern)
- Streaming responses via FastAPI `StreamingResponse`
