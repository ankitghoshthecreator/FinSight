"""Tests for observability metrics and RFC 7807 structured error responses."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_metrics_json_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "total_requests" in data
    assert "total_queries" in data
    assert "escalation_rate_pct" in data
    assert "avg_query_latency_ms" in data
    assert "endpoints" in data
    assert "status_codes" in data


def test_metrics_prometheus_endpoint():
    res = client.get("/metrics?format=prometheus")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    text = res.text
    assert "finsight_http_requests_total" in text
    assert "finsight_rag_queries_total" in text
    assert "finsight_rag_escalations_total" in text


def test_rfc7807_validation_error_format():
    # Sending invalid request payload to trigger validation error
    bad_payload = {"top_k": -5}  # missing required 'query' field, invalid top_k
    res = client.post("/query", json=bad_payload)
    assert res.status_code == 422
    data = res.json()
    assert data["error"] == "VALIDATION_ERROR"
    assert "request_id" in data
    assert "timestamp" in data
    assert "validation_errors" in data
