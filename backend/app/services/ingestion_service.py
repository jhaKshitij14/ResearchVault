import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.document import Document
from app.services.chunk_repository import save_chunks
from app.services.chunker import chunk_pages
from app.services.embedding_client import JinaEmbeddingClient
from app.services.pdf_extractor import extract_pdf_pages


async def ingest_document(
    db: Session,
    document_id: uuid.UUID,
    file_path: str | Path,
    embedding_client: JinaEmbeddingClient,
) -> dict:
    document = db.get(Document, document_id)

    if document is None:
        raise ValueError(f"Document not found: {document_id}")

    document.status = "processing"
    db.commit()

    try:
        pages = extract_pdf_pages(file_path)
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError("PDF produced no text chunks")

        embeddings = await embedding_client.embed(
            [chunk["text"] for chunk in chunks],
            task="retrieval.passage",
        )

        with db.begin():
            document = db.get(Document, document_id)

            save_chunks(
                db=db,
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
            )

            document.status = "ready"

        return {
            "document_id": str(document_id),
            "status": "ready",
            "pages_extracted": len(pages),
            "chunks_created": len(chunks),
        }

    except Exception:
        db.rollback()

        document = db.get(Document, document_id)
        if document is not None:
            document.status = "failed"
            db.commit()

        raise
