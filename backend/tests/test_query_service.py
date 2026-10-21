from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.services.query_service import search_question


@pytest.mark.anyio
async def test_embeds_question_and_returns_retrieved_passages(monkeypatch):
    from app.services import query_service

    embedding = [0.1] * 1024
    passages = [
        {
            "chunk_id": uuid4(),
            "document_id": uuid4(),
            "chunk_index": 2,
            "page_number": 4,
            "text": "PostgreSQL supports vector similarity search.",
            "distance": 0.12,
        }
    ]

    client = Mock()
    client.embed = AsyncMock(return_value=[embedding])

    retrieve_mock = Mock(return_value=passages)
    monkeypatch.setattr(
        query_service, "retrieve_chunks", retrieve_mock
    )

    db = Mock()
    result = await search_question(
        db=db,
        question="  How does vector search work?  ",
        embedding_client=client,
        limit=3,
    )

    assert result["question"] == "How does vector search work?"
    assert result["results"] == passages
    client.embed.assert_awaited_once_with(
        ["How does vector search work?"],
        task="retrieval.query",
    )
    retrieve_mock.assert_called_once_with(
        db=db,
        query_embedding=embedding,
        limit=3,
        document_id=None,
    )


@pytest.mark.anyio
async def test_rejects_empty_question():
    client = Mock()
    client.embed = AsyncMock()

    with pytest.raises(ValueError, match="Question must not be empty"):
        await search_question(
            db=Mock(),
            question="   ",
            embedding_client=client,
        )

    client.embed.assert_not_awaited()


@pytest.mark.anyio
async def test_rejects_unexpected_embedding_count(monkeypatch):
    from app.services import query_service

    client = Mock()
    client.embed = AsyncMock(return_value=[])

    retrieve_mock = Mock()
    monkeypatch.setattr(
        query_service, "retrieve_chunks", retrieve_mock
    )

    with pytest.raises(ValueError, match="exactly one"):
        await search_question(
            db=Mock(),
            question="What is pgvector?",
            embedding_client=client,
        )

    retrieve_mock.assert_not_called()


@pytest.mark.anyio
async def test_forwards_document_filter(monkeypatch):
    from app.services import query_service

    document_id = uuid4()
    client = Mock()
    client.embed = AsyncMock(return_value=[[0.1] * 1024])

    retrieve_mock = Mock(return_value=[])
    monkeypatch.setattr(
        query_service, "retrieve_chunks", retrieve_mock
    )

    await search_question(
        db=Mock(),
        question="Find evidence",
        embedding_client=client,
        document_id=document_id,
    )

    assert retrieve_mock.call_args.kwargs["document_id"] == document_id

@pytest.fixture
def anyio_backend():
    return "asyncio"