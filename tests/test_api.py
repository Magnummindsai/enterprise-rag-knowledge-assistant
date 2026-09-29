import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client(pipeline):
    with TestClient(app) as c:
        yield c


def test_health_reports_mock_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "mock"
    assert body["documents"] == 8


def test_ui_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Policy Assistant" in res.text


def test_ask_returns_cited_answer(client):
    res = client.post("/api/ask", json={"question": "What is the per diem for international travel?"})
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "mock"
    assert body["query_id"]
    assert "TRV-6.1" in [c["id"] for c in body["citations"]]
    assert body["trace"]


@pytest.mark.parametrize("question", ["", "hi", "x" * 1001])
def test_ask_validates_input(client, question):
    assert client.post("/api/ask", json={"question": question}).status_code == 422


def test_feedback_roundtrip(client):
    qid = client.post("/api/ask", json={"question": "minimum password length"}).json()["query_id"]
    assert client.post("/api/feedback", json={"query_id": qid, "rating": 1}).status_code == 201
    assert client.post("/api/feedback", json={"query_id": "nope", "rating": 1}).status_code == 404
    assert client.post("/api/feedback", json={"query_id": qid, "rating": 5}).status_code == 422


def test_documents_and_chunks(client):
    docs = client.get("/api/documents").json()
    assert len(docs) == 8 and all(d["sections"] > 0 for d in docs)
    chunk = client.get("/api/chunks/TRV-5.1").json()
    assert chunk["heading"] == "Hotel Rate Caps"
    assert client.get("/api/chunks/NOPE-1").status_code == 404
