"""Integration tests for FastAPI endpoints: health, text ingestion, file ingestion, and QA query."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "FinSight RAG API"
    assert data["status"] == "online"


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_chunks" in data


def test_api_ingest_text_and_query():
    # 1. Ingest text document
    doc_payload = {
        "filename": "Goldman_Sachs_Basel_III.txt",
        "content": (
            "Under Basel III capital adequacy guidelines, Goldman Sachs reported a Supplementary "
            "Leverage Ratio (SLR) of 5.8% and a Tier 1 Risk-Based Capital Ratio of 16.7% for the current period."
        ),
        "page_number": 3
    }
    ingest_res = client.post("/ingest/text", json=doc_payload)
    assert ingest_res.status_code == 200
    data = ingest_res.json()
    assert data["chunks_indexed"] >= 1


    # 2. Query pipeline
    query_payload = {
        "query": "What is Goldman Sachs Supplementary Leverage Ratio SLR?",
        "top_k": 3,
        "confidence_threshold": 0.5,
        "enable_reranking": True
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "answer" in q_data
    assert "sources" in q_data
    assert q_data["confidence"] > 0.5
    assert not q_data["escalate"]
    assert len(q_data["sources"]) >= 1


def test_api_query_unindexed_topic():
    query_payload = {
        "query": "What is the policy for deep-sea oil drilling investments?",
        "top_k": 3
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "insufficient context" in q_data["answer"].lower()
    assert q_data["escalate"] is True


def test_api_pii_redaction():
    payload = {
        "filename": "Client_KYC_Memo.txt",
        "content": "Client John Doe SSN 000-12-3456 email client@hedgefund.com deposited $5M into account IBAN US12345678901234.",
        "page_number": 1,
        "redact_pii": True
    }
    res = client.post("/ingest/text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "SSN" in data["pii_redacted"]
    assert "EMAIL" in data["pii_redacted"]


def test_api_streaming_query():
    query_payload = {
        "query": "What is Goldman Sachs Supplementary Leverage Ratio SLR?",
        "top_k": 3
    }
    res = client.post("/query/stream", json=query_payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    body = res.text
    assert "event: token" in body
    assert "[DONE]" in body

