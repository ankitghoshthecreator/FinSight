"""Unit and integration tests for multi-turn conversation memory."""

import pytest
from generation.session import SessionManager
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_session_manager_sliding_window():
    sm = SessionManager(max_history_turns=4)
    sess_id = "test_sess_1"

    # Add 6 turns
    for i in range(6):
        sm.add_turn(sess_id, "user" if i % 2 == 0 else "assistant", f"Turn {i}")

    session = sm.get_or_create(sess_id)
    # Should only retain last 4 turns
    assert len(session.turns) == 4
    assert session.turns[0].content == "Turn 2"
    assert session.turns[-1].content == "Turn 5"


def test_session_formatted_history():
    sm = SessionManager()
    sess_id = "test_sess_2"
    sm.add_turn(sess_id, "user", "What is Goldman's SLR?")
    sm.add_turn(sess_id, "assistant", "Goldman Sachs reported an SLR of 5.8%.")

    history = sm.get_formatted_history(sess_id)
    assert "Analyst: What is Goldman's SLR?" in history
    assert "FinSight: Goldman Sachs reported an SLR of 5.8%." in history


def test_api_session_multi_turn_query():
    # Ingest document first
    client.post(
        "/ingest/text",
        json={
            "filename": "Session_Doc.txt",
            "content": "JPMorgan Chase CET1 ratio stood at 15.0% for Q4.",
            "page_number": 1,
        }
    )

    # 1. Query with new session
    res1 = client.post(
        "/query",
        json={
            "query": "What is JPMorgan Chase CET1 ratio?",
            "session_id": "analyst_session_101",
        }
    )
    assert res1.status_code == 200
    assert res1.json()["session_id"] == "analyst_session_101"

    # 2. Check session endpoint
    sess_res = client.get("/sessions/analyst_session_101")
    assert sess_res.status_code == 200
    data = sess_res.json()
    assert len(data["turns"]) == 2
    assert data["turns"][0]["role"] == "user"
    assert data["turns"][1]["role"] == "assistant"

    # 3. Delete session
    del_res = client.delete("/sessions/analyst_session_101")
    assert del_res.status_code == 200
