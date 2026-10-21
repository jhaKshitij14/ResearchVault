import asyncio
import uuid

import pymupdf
import pytest
from sqlalchemy import delete, create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.services.ingestion_service import ingest_document


class FakeEmbeddingClient:
    async def embed(self, texts, task="retrieval.passage"):
        return [[0.1] * 1024 for _ in texts]


class FailingEmbeddingClient:
    async def embed(self, texts, task="retrieval.passage"):
        raise RuntimeError("Simulated embedding failure")


@pytest.fixture
def db():
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    session = Session(engine)
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def create_document(db):
    user = User(
        email=f"{uuid.uuid4()}@example.com",
        password_hash="test-hash",
    )
    db.add(user)
    db.flush()

    document = Document(
        owner_id=user.id,
        title="Integration test PDF",
        content_hash=uuid.uuid4().hex,
        status="uploaded",
    )
    db.add(document)
    db.commit()

    return user.id, document.id


def cleanup(db, user_id, document_id):
    db.rollback()
    db.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        )
    )
    db.execute(delete(Document).where(Document.id == document_id))
    db.execute(delete(User).where(User.id == user_id))
    db.commit()


def create_pdf(path):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "PostgreSQL indexes improve query performance. "
            "Database transactions preserve data consistency.",
        )
        pdf.save(path)


def test_ingestion_saves_chunks_and_marks_document_ready(
    db, tmp_path
):
    user_id, document_id = create_document(db)
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path)

    try:
        result = asyncio.run(
            ingest_document(
                db=db,
                document_id=document_id,
                file_path=pdf_path,
                embedding_client=FakeEmbeddingClient(),
            )
        )

        assert result["status"] == "ready"
        assert result["pages_extracted"] == 1
        assert result["chunks_created"] >= 1

        document = db.get(Document, document_id)
        assert document.status == "ready"

        chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()

        assert len(chunks) == result["chunks_created"]
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert "PostgreSQL indexes" in chunks[0].text
        assert len(chunks[0].embedding) == 1024
    finally:
        cleanup(db, user_id, document_id)


def test_embedding_failure_marks_document_failed_without_chunks(
    db, tmp_path
):
    user_id, document_id = create_document(db)
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path)

    try:
        with pytest.raises(
            RuntimeError, match="Simulated embedding failure"
        ):
            asyncio.run(
                ingest_document(
                    db=db,
                    document_id=document_id,
                    file_path=pdf_path,
                    embedding_client=FailingEmbeddingClient(),
                )
            )

        db.expire_all()
        document = db.get(Document, document_id)
        assert document.status == "failed"

        chunks = db.scalars(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            )
        ).all()
        assert chunks == []
    finally:
        cleanup(db, user_id, document_id)


def test_missing_document_raises_value_error(db, tmp_path):
    pdf_path = tmp_path / "unused.pdf"

    with pytest.raises(ValueError, match="Document not found"):
        asyncio.run(
            ingest_document(
                db=db,
                document_id=uuid.uuid4(),
                file_path=pdf_path,
                embedding_client=FakeEmbeddingClient(),
            )
        )
