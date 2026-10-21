from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


def save_chunks(
    db: Session,
    document_id,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> list[DocumentChunk]:
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")

    if not chunks:
        return []

    records = []

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        records.append(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk["chunk_index"],
                page_number=chunk["page_number"],
                text=chunk["text"],
                embedding=embedding,
            )
        )

    db.add_all(records)
    db.flush()

    return records
