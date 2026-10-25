"""Tests for API key authentication middleware."""

import pytest
from fastapi.testclient import TestClient
from config.settings import settings
from api.main import app


client = TestClient(app)


def test_auth_disabled_by_default():
    """When API_KEY_SECRET is empty, all requests should pass through."""
    assert settings.api_key_secret == ""
    res = client.get("/health")
    assert res.status_code == 200


def test_auth_blocks_without_key(monkeypatch):
    """When a secret is configured, requests without the header get 401."""
    monkeypatch.setattr(settings, "api_key_secret", "supersecret123")

    # /documents is a protected route
    res = client.get("/documents")
    assert res.status_code == 401
    assert "UNAUTHORIZED" in res.json()["error"]


def test_auth_passes_with_correct_key(monkeypatch):
    """Requests with the correct X-API-Key header must be allowed through."""
    monkeypatch.setattr(settings, "api_key_secret", "supersecret123")

    res = client.get("/documents", headers={"X-API-Key": "supersecret123"})
    assert res.status_code == 200


def test_auth_exempts_public_paths(monkeypatch):
    """Public endpoints like /health and / must be accessible without any API key."""
    monkeypatch.setattr(settings, "api_key_secret", "supersecret123")

    res_root = client.get("/")
    assert res_root.status_code == 200

    res_health = client.get("/health")
    assert res_health.status_code == 200
