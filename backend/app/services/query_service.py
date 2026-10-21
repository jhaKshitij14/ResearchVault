from uuid import UUID

from sqlalchemy.orm import Session

from app.services.embedding_client import JinaEmbeddingClient
from app.services.retrieval_service import retrieve_chunks


async def search_question(
    db: Session,
    question: str,
    embedding_client: JinaEmbeddingClient,
    limit: int = 5,
    document_id: UUID | None = None,
) -> dict:
    """Embed a question and retrieve its most relevant evidence passages."""
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty")

    query_embeddings = await embedding_client.embed(
        [question],
        task="retrieval.query",
    )

    if len(query_embeddings) != 1:
        raise ValueError("Expected exactly one query embedding")

    passages = retrieve_chunks(
        db=db,
        query_embedding=query_embeddings[0],
        limit=limit,
        document_id=document_id,
    )

    return {
        "question": question,
        "results": passages,
    }
