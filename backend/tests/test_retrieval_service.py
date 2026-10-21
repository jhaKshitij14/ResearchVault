
import uuid

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.services.retrieval_service import retrieve_chunks


@pytest.fixture
def db():
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    session = Session(engine)
    session.info["test_documents"] = []

    try:
        yield session
    finally:
        session.rollback()

        for document_id, user_id in session.info["test_documents"]:
            session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == document_id
                )
            )
            session.execute(
                delete(Document).where(Document.id == document_id)
            )
            session.execute(delete(User).where(User.id == user_id))

        session.commit()
        session.close()
        engine.dispose()


def make_document(db, status="ready"):
    user = User(
        email=f"{uuid.uuid4()}@example.com",
        password_hash="test-hash",
    )
    db.add(user)
    db.flush()

    document = Document(
        owner_id=user.id,
        title="Retrieval test",
        content_hash=uuid.uuid4().hex,
        status=status,
    )
    db.add(document)
    db.commit()

    db.info["test_documents"].append((document.id, user.id))
    return user.id, document.id


def add_chunk(db, document_id, chunk_index, text, embedding, page_number=1):
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        page_number=page_number,
        text=text,
        embedding=embedding,
    )
    db.add(chunk)
    db.commit()
    return chunk


def test_returns_nearest_chunks_first(db):
    _, document_id = make_document(db)

    query = [0.0] * 1024
    query[0] = 1.0

    near = [0.0] * 1024
    near[0] = 0.9
    near[1] = 0.43589

    far = [0.0] * 1024
    far[1] = 1.0

    add_chunk(db, document_id, 0, "Far passage", far)
    add_chunk(db, document_id, 1, "Near passage", near)

    results = retrieve_chunks(
        db, query, limit=2, document_id=document_id
    )

    assert len(results) == 2
    assert results[0]["text"] == "Near passage"
    assert results[1]["text"] == "Far passage"
    assert results[0]["distance"] < results[1]["distance"]


def test_returns_empty_list_when_no_chunks_match(db):
    _, document_id = make_document(db)

    query = [0.0] * 1024
    query[0] = 1.0

    results = retrieve_chunks(
        db, query, document_id=document_id
    )

    assert results == []


def test_respects_result_limit(db):
    _, document_id = make_document(db)

    embedding = [0.0] * 1024
    embedding[0] = 1.0

    for index in range(3):
        add_chunk(
            db, document_id, index, f"Passage {index}", embedding
        )

    results = retrieve_chunks(
        db, embedding, limit=2, document_id=document_id
    )

    assert len(results) == 2


def test_filters_by_document_id(db):
    _, first_document_id = make_document(db)
    _, second_document_id = make_document(db)

    embedding = [0.0] * 1024
    embedding[0] = 1.0

    add_chunk(db, first_document_id, 0, "First document", embedding)
    add_chunk(db, second_document_id, 0, "Second document", embedding)

    results = retrieve_chunks(
        db, embedding, document_id=first_document_id
    )

    assert len(results) == 1
    assert results[0]["text"] == "First document"
    assert results[0]["document_id"] == first_document_id


def test_excludes_documents_that_are_not_ready(db):
    _, document_id = make_document(db, status="processing")

    embedding = [0.0] * 1024
    embedding[0] = 1.0

    add_chunk(db, document_id, 0, "Processing passage", embedding)

    results = retrieve_chunks(
        db, embedding, document_id=document_id
    )

    assert results == []


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_rejects_invalid_limit(db, limit):
    embedding = [0.0] * 1024

    with pytest.raises(ValueError):
        retrieve_chunks(db, embedding, limit=limit)


def test_rejects_wrong_embedding_dimension(db):
    with pytest.raises(ValueError):
        retrieve_chunks(db, [0.0] * 10)
