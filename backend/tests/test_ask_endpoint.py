"""API-level tests for POST /documents/{id}/ask error mapping.

Search and generation are stubbed, so no live Jina/Gemini calls are made.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app import main
from app.db.session import engine
from app.models.document import Document
from app.models.user import User
from app.services.answer_service import InvalidLLMResponseError
from app.services.embedding_client import EmbeddingError
from app.services.llm_client import LLMProviderError
from sqlalchemy.orm import Session


@pytest.fixture
def ready_document_id():
    with Session(engine) as db:
        user = User(email=f"{uuid.uuid4()}@example.com", password_hash="x")
        db.add(user)
        db.flush()
        doc = Document(
            owner_id=user.id, title="t", content_hash=uuid.uuid4().hex,
            status="ready",
        )
        db.add(doc)
        db.commit()
        doc_id, user_id = doc.id, user.id
    yield doc_id
    with Session(engine) as db:
        db.execute(delete(Document).where(Document.id == doc_id))
        db.execute(delete(User).where(User.id == user_id))
        db.commit()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.settings, "jina_api_key", "test-jina")
    monkeypatch.setattr(main.settings, "llm_api_key", "test-llm")
    return TestClient(main.app)


def _stub(monkeypatch, *, search=None, answer=None):
    async def fake_search(**kwargs):
        if isinstance(search, Exception):
            raise search
        return {"question": kwargs["question"], "results": [{"x": 1}]}

    async def fake_answer(**kwargs):
        if isinstance(answer, Exception):
            raise answer
        return {"answer": "ok [S1]", "abstained": False, "citations": ["S1"]}

    monkeypatch.setattr(main, "search_question", fake_search)
    monkeypatch.setattr(main, "generate_grounded_answer", fake_answer)


def test_ask_success(client, ready_document_id, monkeypatch):
    _stub(monkeypatch)
    r = client.post(f"/documents/{ready_document_id}/ask", json={"question": "q"})
    assert r.status_code == 200
    assert r.json()["retrieved_passage_count"] == 1


def test_malformed_llm_output_is_502_not_422(client, ready_document_id, monkeypatch):
    _stub(monkeypatch, answer=InvalidLLMResponseError("bad json"))
    r = client.post(f"/documents/{ready_document_id}/ask", json={"question": "q"})
    assert r.status_code == 502


def test_embedding_failure_is_502(client, ready_document_id, monkeypatch):
    _stub(monkeypatch, search=EmbeddingError("down"))
    r = client.post(f"/documents/{ready_document_id}/ask", json={"question": "q"})
    assert r.status_code == 502


def test_llm_provider_failure_is_502(client, ready_document_id, monkeypatch):
    _stub(monkeypatch, answer=LLMProviderError("down"))
    r = client.post(f"/documents/{ready_document_id}/ask", json={"question": "q"})
    assert r.status_code == 502


def test_empty_question_is_422(client, ready_document_id, monkeypatch):
    _stub(monkeypatch, search=ValueError("Question must not be empty"))
    r = client.post(f"/documents/{ready_document_id}/ask", json={"question": " "})
    assert r.status_code == 422


def test_missing_document_is_404(client, monkeypatch):
    _stub(monkeypatch)
    r = client.post(f"/documents/{uuid.uuid4()}/ask", json={"question": "q"})
    assert r.status_code == 404
