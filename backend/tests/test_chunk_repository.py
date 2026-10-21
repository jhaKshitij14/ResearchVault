import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.services.chunk_repository import save_chunks


@pytest.fixture
def db():
    from app.core.config import settings

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    with Session(engine) as session:
        yield session
        session.rollback()
    engine.dispose()


@pytest.fixture
def document(db):
    user = User(
        email=f"{uuid.uuid4()}@example.com",
        password_hash="test-hash",
    )
    db.add(user)
    db.flush()

    doc = Document(
        owner_id=user.id,
        title="Test document",
        content_hash=uuid.uuid4().hex,
        status="processing",
    )
    db.add(doc)
    db.flush()
    return doc


def test_save_chunks_persists_text_and_embeddings(db, document):
    chunks = [
        {"chunk_index": 0, "page_number": 1, "text": "First passage"},
        {"chunk_index": 1, "page_number": 2, "text": "Second passage"},
    ]
    embeddings = [
        [0.1] * 1024,
        [0.2] * 1024,
    ]

    saved = save_chunks(db, document.id, chunks, embeddings)

    assert len(saved) == 2
    assert saved[0].chunk_index == 0
    assert saved[0].page_number == 1
    assert saved[0].text == "First passage"

    db.commit()

    rows = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
    ).all()

    assert len(rows) == 2
    assert rows[0].embedding is not None
    assert len(rows[0].embedding) == 1024
    assert rows[1].text == "Second passage"


def test_save_empty_chunks_returns_empty_list(db, document):
    assert save_chunks(db, document.id, [], []) == []


def test_save_chunks_rejects_mismatched_embedding_count(db, document):
    chunks = [
        {"chunk_index": 0, "page_number": 1, "text": "First passage"},
    ]

    with pytest.raises(ValueError, match="exactly one embedding"):
        save_chunks(db, document.id, chunks, [])


def test_save_chunks_does_not_commit_automatically(db, document):
    chunks = [
        {"chunk_index": 0, "page_number": 1, "text": "Uncommitted passage"},
    ]

    save_chunks(db, document.id, chunks, [[0.3] * 1024])

    assert db.in_transaction()
