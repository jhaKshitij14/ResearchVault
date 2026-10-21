from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk


EMBEDDING_DIMENSIONS = 1024


def retrieve_chunks(
    db: Session,
    query_embedding: list[float],
    limit: int = 5,
    document_id: UUID | None = None,
) -> list[dict]:
    """Return the nearest chunks from ready documents, ordered by cosine distance."""
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Query embedding must have {EMBEDDING_DIMENSIONS} dimensions"
        )

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    ).label("distance")

    statement = (
        select(DocumentChunk, distance)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.status == "ready")
    )

    if document_id is not None:
        statement = statement.where(
            DocumentChunk.document_id == document_id
        )

    statement = statement.order_by(distance).limit(limit)

    rows = db.execute(statement).all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "text": chunk.text,
            "distance": float(distance_value),
        }
        for chunk, distance_value in rows
    ]
